"""Workspace built from ANY uploaded table: every page of the site is served from this when a user
segments their own dataset (instead of the fixed retail customer schema).

It exposes the same view methods as ``SegmentationService`` (overview, profiles, explorer,
assignment, diagnostics, exports) so the API and website work unchanged on top of it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from segmentation.auto_segment import K_MAX, K_MIN, MIN_SEGMENT_SHARE, AutoSegmentation, auto_segment, fmt_value, pretty

from .file_io import read_table
from .services import SEGMENT_COLORS, _records, _round, sanitize_csv

GENERIC_FILENAME = "workspace.joblib"
MAX_PROFILE_METRICS = 6
MAX_FINGERPRINT = 12
VALUE_TOKENS = {
    "amount", "spend", "spent", "revenue", "sales", "balance", "income", "value", "price", "total", "profit",
    "salary", "payment", "payments", "cost", "fee", "premium", "deposit", "loan", "credit", "turnover", "mnt",
}
RESPONSIBLE_USE = "Use these segments to understand and serve groups better, never to make credit, pricing or eligibility decisions about individuals."


def _tokens(name: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", str(name))}


class GenericService:
    mode = "generic"

    def __init__(self, result: AutoSegmentation, table: pd.DataFrame, source: str, trained_at_utc: str):
        self.result = result
        self.source = source
        self.trained_at_utc = trained_at_utc
        self.names = dict(result.names)
        labels = result.labels
        self.table = table.reset_index(drop=True)
        ids = self.table["ID"] if "ID" in self.table.columns else pd.Series(np.arange(1, len(self.table) + 1))
        features = pd.concat([result.numeric.reset_index(drop=True), result.categorical.reset_index(drop=True)], axis=1)
        self.customers = pd.concat(
            [pd.DataFrame({"ID": ids.to_numpy(), "Segment": labels, "Segment_Name": [self.names[s] for s in labels]}), features],
            axis=1,
        )
        self.numeric_features = result.numeric_columns
        self.categorical_features = result.categorical_columns
        self.value_column = self._value_column()
        self.metrics = self._top_metrics()
        self.zprofile = self._zprofile()
        self.profiles = self._profiles()
        self.recommendations = [self._recommendation(s) for s in self.segment_ids]
        self._rec_by_id = {r["segment"]: r for r in self.recommendations}
        coords = PCA(n_components=min(3, result.X.shape[1]), random_state=0).fit(result.X)
        self._pca = coords
        self._coords = coords.transform(result.X)
        self._centers3 = coords.transform(result.centers)
        self.segmenter = SimpleNamespace(
            n_segments=result.k,
            silhouette_=float(result.silhouette),
            profiles_=self.profiles,
            recommendations_=self.recommendations,
            segment_names_=self.names,
            zprofile_=self.zprofile,
            k_selection_=SimpleNamespace(rationale=self.rationale, metrics=result.k_table),
        )

    # ------------------------------------------------------------------ building / persistence
    @classmethod
    def fit(cls, table: pd.DataFrame, n_clusters: int | None, source: str) -> "GenericService":
        result = auto_segment(table, n_clusters)
        return cls(result, table, source, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"result": self.result, "table": self.table, "source": self.source, "trained_at_utc": self.trained_at_utc}, directory / GENERIC_FILENAME)
        self.customers.to_csv(directory / "customers_segmented.csv", index=False)
        (directory / "model_card.json").write_text(json.dumps(self.model_card(), indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "GenericService":
        """Load a trusted, locally produced workspace (joblib must never load untrusted files)."""
        data = joblib.load(Path(directory) / GENERIC_FILENAME)
        return cls(data["result"], data["table"], data["source"], data["trained_at_utc"])

    # ------------------------------------------------------------------ derived tables
    @property
    def segment_ids(self) -> list[int]:
        return sorted(self.names)

    @property
    def rationale(self) -> str:
        r = self.result
        chosen = r.k_table.loc[r.k_table["k"] == r.k].iloc[0]
        if not r.auto_k:
            return f"You asked for {r.k} segments (silhouette {chosen['silhouette']:.3f})."
        return (
            f"k={r.k} was chosen automatically: I tested {K_MIN}–{K_MAX} segments with K-Means and kept the best-separated solution "
            f"with at least 3 groups, each holding ≥ {MIN_SEGMENT_SHARE:.0%} of rows (silhouette {chosen['silhouette']:.3f}, "
            f"smallest segment {chosen['min_segment_share']:.1%}). Built from {len(self.numeric_features)} numeric and "
            f"{len(self.categorical_features)} categorical columns of {self.source}."
        )

    def _value_column(self) -> str | None:
        """The column that best represents value (spend, balance, revenue...) for 'share of value' views."""
        for col in self.numeric_features:
            values = self.customers[col]
            if _tokens(col) & VALUE_TOKENS and values.min() >= 0 and values.sum() > 0:
                return col
        return None

    def _spread_ranking(self) -> list[str]:
        """Numeric features ordered by how much their averages differ between segments."""
        scores = {}
        for col in self.numeric_features:
            std = self.customers[col].std()
            if std and np.isfinite(std):
                means = self.customers.groupby("Segment")[col].mean()
                scores[col] = float((means.max() - means.min()) / std)
        return sorted(scores, key=scores.get, reverse=True)

    def _top_metrics(self) -> list[str]:
        ranked = self._spread_ranking()
        if self.value_column and self.value_column in ranked:
            ranked.remove(self.value_column)
            ranked.insert(0, self.value_column)
        return ranked[:MAX_PROFILE_METRICS]

    def _zprofile(self) -> pd.DataFrame:
        c = self.customers
        cols = [m for m in self._spread_ranking()[:MAX_FINGERPRINT] if c[m].std(ddof=0) > 0]
        means = c.groupby("Segment")[cols].mean()
        return (means - c[cols].mean()) / c[cols].std(ddof=0)

    def _profiles(self) -> pd.DataFrame:
        c = self.customers
        grouped = c.groupby("Segment")
        size = grouped.size()
        out = pd.DataFrame({"Segment_Name": pd.Series(self.names), "Customers": size, "Customer_Share_%": (100 * size / len(c)).round(1)})
        if self.value_column:
            totals = grouped[self.value_column].sum()
            out["Revenue_Share_%"] = (100 * totals / totals.sum()).round(1)
            out["Avg_Total_Spend"] = grouped[self.value_column].mean().round(2)
        else:
            out["Revenue_Share_%"] = out["Customer_Share_%"]
            out["Avg_Total_Spend"] = np.nan
        for m in self.metrics:
            out[f"Avg {m}"] = grouped[m].mean().round(2)
        for col in self.categorical_features[:3]:
            out[f"Top {col}"] = grouped[col].agg(lambda s: s.mode().iat[0])
        out.index.name = "Segment"
        return out

    def _recommendation(self, seg: int) -> dict:
        info = next(s for s in self.result.segments if s["id"] == seg)
        z = self.zprofile.loc[seg].sort_values() if seg in self.zprofile.index else pd.Series(dtype=float)
        highs = [f"{pretty(c)} (+{v:.1f} sd)" for c, v in z[z > 0.3].sort_values(ascending=False).head(4).items()]
        lows = [f"{pretty(c)} ({v:.1f} sd)" for c, v in z[z < -0.3].head(4).items()]
        share = info["share_pct"]
        traits = [t["text"] for t in info["traits"]]
        summary = f"{share:.0f}% of rows. " + ("; ".join(traits[:2]) + "." if traits else "Close to the overall average on every measure.")
        top = pretty(z.abs().sort_values(ascending=False).index[0]) if len(z) else "the key measures"
        category_hits = [t["text"].split(" (")[0] for t in info["traits"] if " mostly " in t["text"]]
        actions = [
            f"Use '{info['name']}' as a filter or tag in your own tools so this group can be reached directly.",
            f"Track {top} for this group every month to see whether it is moving towards or away from the average.",
            "Compare outcomes (sales, retention, complaints…) for this group with the rest to decide where to focus.",
            "Review a sample of rows from this segment with the business team to give it a memorable name and owner.",
        ]
        return {
            "segment": seg,
            "name": info["name"],
            "summary": summary,
            "goal": "Understand and act on what sets this group apart",
            "defining_high_traits": highs,
            "defining_low_traits": lows,
            "preferred_channel": "",
            "over_indexed_categories": category_hits,
            "actions": actions,
            "kpis": [pretty(m) for m in self.metrics[:3]],
            "responsible_use": RESPONSIBLE_USE,
            "traits": traits,
        }

    # ------------------------------------------------------------------ views (same contract as SegmentationService)
    def labels(self) -> dict:
        return {"entity": "rows", "entity_singular": "row", "value": pretty(self.value_column) if self.value_column else None, "dataset": self.source}

    def segments_meta(self) -> list[dict]:
        return [{"id": s, "name": self.names[s], "color": SEGMENT_COLORS[s % len(SEGMENT_COLORS)]} for s in self.segment_ids]

    def overview(self) -> dict:
        c = self.customers
        kpi_list = [
            {"label": "Rows segmented", "value": int(len(c)), "format": "integer"},
            {"label": "Segments", "value": self.result.k, "format": "integer"},
            {"label": "Silhouette score", "value": round(self.result.silhouette, 3), "format": "decimal3"},
            {"label": "Columns used", "value": len(self.numeric_features) + len(self.categorical_features), "format": "integer"},
        ]
        kpi_list += [{"label": f"Average {pretty(m)}", "value": _round(c[m].mean(), 2), "format": "number"} for m in self.metrics[:2]]
        value_label = pretty(self.value_column) if self.value_column else None
        profile_columns = [
            {"key": "Customers", "label": "Rows", "format": "integer"},
            {"key": "Customer_Share_%", "label": "Rows %", "format": "shareBar"},
        ]
        if value_label:
            profile_columns.append({"key": "Revenue_Share_%", "label": f"{value_label} %", "format": "shareBar"})
        profile_columns += [{"key": f"Avg {m}", "label": f"Avg {pretty(m)}", "format": "number"} for m in self.metrics]
        profile_columns += [{"key": f"Top {col}", "label": f"Most common {pretty(col)}", "format": "text"} for col in self.categorical_features[:3]]
        slide_stats = [{"key": "Customer_Share_%", "label": "Rows", "format": "percent"}]
        if value_label:
            slide_stats.append({"key": "Revenue_Share_%", "label": value_label, "format": "percent"})
        slide_stats += [{"key": f"Avg {m}", "label": f"Avg {pretty(m)}"[:18], "format": "compact"} for m in self.metrics if m != self.value_column][: 3 - len(slide_stats)]
        return {
            "mode": self.mode,
            "labels": self.labels(),
            "kpis": {"customers": int(len(c)), "segments": self.result.k, "silhouette": round(self.result.silhouette, 3)},
            "kpi_list": kpi_list,
            "segments": self.segments_meta(),
            "profiles": _records(self.profiles.reset_index()),
            "profile_columns": profile_columns,
            "slide_stats": slide_stats,
            "recommendations": self.recommendations,
            "rationale": self.rationale,
        }

    def _info(self, rows: pd.DataFrame) -> list[str]:
        cols = self.metrics[:2]
        return [" · ".join(f"{pretty(col)} {fmt_value(v)}" for col, v in zip(cols, values)) for values in rows[cols].itertuples(index=False, name=None)]

    def pca_points(self) -> dict:
        c = self.customers
        points = pd.DataFrame({"x": self._coords[:, 0].round(3), "y": self._coords[:, 1].round(3) if self._coords.shape[1] > 1 else 0.0, "segment": c["Segment"], "id": c["ID"], "info": self._info(c)})
        variance = [round(float(v), 4) for v in self._pca.explained_variance_ratio_] + [0.0, 0.0]
        return {"variance": variance[:3], "points": _records(points)}

    def pca3d(self) -> dict:
        coords = np.pad(self._coords, ((0, 0), (0, 3 - self._coords.shape[1]))).round(3)
        centers = np.pad(self._centers3, ((0, 0), (0, 3 - self._centers3.shape[1])))
        return {
            "variance": [round(float(v), 4) for v in self._pca.explained_variance_ratio_],
            "positions": coords.ravel().tolist(),
            "segments": self.customers["Segment"].astype(int).tolist(),
            "centers": [{"segment": int(s), "position": [round(float(v), 3) for v in centers[s]]} for s in self.segment_ids],
        }

    def segment_detail(self, segment: int) -> dict:
        if segment not in self.names:
            raise KeyError(segment)
        c = self.customers
        rows = c[c["Segment"] == segment]
        metrics = [{"label": "Share of rows", "value": _round(len(rows) / len(c), 4), "baseline": None, "delta_pct": None, "format": "percent"}]
        for m in self.metrics[:7]:
            value, base = rows[m].mean(), c[m].mean()
            metrics.append(
                {"label": f"Avg {pretty(m)}", "value": _round(value, 4), "baseline": _round(base, 4), "delta_pct": _round((value / base - 1) * 100, 1) if base else None, "format": "number"}
            )
        charts = []
        for col in self.categorical_features[:2]:
            mix = (rows[col].value_counts(normalize=True) * 100).round(1).head(8)
            charts.append({"title": f"{pretty(col)} mix", "caption": f"Share of this segment's rows by {pretty(col)}.", "values": {str(k): float(v) for k, v in mix.items()}, "unit": "%"})
        if len(charts) < 2:
            deltas = {pretty(m): float(d["delta_pct"]) for m, d in zip(self.metrics, metrics[1:]) if d["delta_pct"] is not None}
            charts.append({"title": "Difference vs. average", "caption": "How far this segment's averages are from all rows (%).", "values": deltas, "unit": "%"})
        profile = self.profiles.loc[segment]
        return {
            "id": segment,
            "name": self.names[segment],
            "color": SEGMENT_COLORS[segment % len(SEGMENT_COLORS)],
            "customers": int(profile["Customers"]),
            "revenue_share": float(profile["Revenue_Share_%"]),
            "metrics": metrics,
            "charts": charts[:2],
            "channel_mix": {},
            "category_mix": {},
            "zscores": {k: round(float(v), 3) for k, v in self.zprofile.loc[segment].items()},
            "recommendation": self._rec_by_id[segment],
            "labels": self.labels(),
        }

    def fingerprint(self) -> dict:
        z = self.zprofile
        return {
            "features": list(z.columns),
            "rows": [{"segment": int(s), "name": self.names[int(s)], "values": [round(float(v), 2) for v in z.loc[s]]} for s in z.index],
        }

    def distribution(self, feature: str) -> dict:
        if feature not in self.numeric_features:
            raise KeyError(feature)
        stats = []
        for s in self.segment_ids:
            values = self.customers.loc[self.customers["Segment"] == s, feature].dropna()
            q = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(3).tolist() if len(values) else [None] * 5
            stats.append({"segment": s, "min": q[0], "q1": q[1], "median": q[2], "q3": q[3], "max": q[4], "mean": _round(values.mean(), 3)})
        return {"feature": feature, "stats": stats}

    def scatter(self, x: str, y: str, segments: list[int] | None, max_points: int = 2500) -> dict:
        for feature in (x, y):
            if feature not in self.numeric_features:
                raise KeyError(feature)
        df = self.customers
        if segments:
            df = df[df["Segment"].isin(segments)]
        if len(df) > max_points:
            df = df.sample(max_points, random_state=0)
        points = pd.DataFrame({"x": df[x].round(3), "y": df[y].round(3), "segment": df["Segment"], "id": df["ID"]})
        return {"x": x, "y": y, "points": _records(points.dropna())}

    def demographics(self, attribute: str) -> dict:
        if attribute not in self.categorical_features:
            raise KeyError(attribute)
        table = pd.crosstab(self.customers["Segment"], self.customers[attribute], normalize="index") * 100
        top = self.customers[attribute].value_counts().head(8).index
        table = table[[c for c in table.columns if c in top]]
        return {
            "attribute": attribute,
            "categories": [str(c) for c in table.columns],
            "rows": [{"segment": int(s), "values": [round(float(v), 1) for v in table.loc[s]]} for s in table.index],
        }

    def explorer_features(self) -> dict:
        return {
            "numeric": self.numeric_features,
            "categorical": self.categorical_features,
            "categorical_note": "Category columns from your file (they were also used to form the segments).",
        }

    def diagnostics(self) -> dict:
        r = self.result
        table = r.k_table.copy()
        for col in ("stability_ari", "composite_score"):
            table[col] = None
        chosen = table.loc[table["k"] == r.k].iloc[0]
        enc = r.encoder
        skew = pd.DataFrame(
            {
                "feature": enc.skew_before_.index,
                "skew_before": enc.skew_before_.round(2).to_numpy(),
                "skew_after": enc.skew_after_.round(2).to_numpy(),
                "log_transformed": [c in enc.log_columns_ for c in enc.skew_before_.index],
            }
        )
        columns = [{"column": s.source, "used_as": {"number": "number", "date": "date (days before latest)", "category": "category"}[s.kind], "reason": ""} for s in enc.specs]
        columns += [{"column": col, "used_as": "not used", "reason": why} for col, why in enc.dropped.items()]
        return {
            "mode": self.mode,
            "rationale": self.rationale,
            "selected_k": r.k,
            "elbow_k": r.k,
            "best_silhouette_k": int(table.loc[table["silhouette"].idxmax(), "k"]),
            "silhouette": round(float(r.silhouette), 4),
            "k_metrics": _records(table),
            "algorithms": [],
            "cleaning": [],
            "columns": columns,
            "skewness": _records(skew),
            "proxy_check": [],
            "model_card": self.model_card(),
            "chosen": {"silhouette": float(chosen["silhouette"])},
        }

    def model_card(self) -> dict:
        return {
            "created_at_utc": self.trained_at_utc,
            "algorithm": "K-Means (k-means++, n_init=10) on all usable columns",
            "source": self.source,
            "n_segments": self.result.k,
            "k_selection_rationale": self.rationale,
            "silhouette": round(float(self.result.silhouette), 4),
            "rows": int(len(self.customers)),
            "numeric_columns": self.numeric_features,
            "categorical_columns": self.categorical_features,
            "log_transformed_columns": self.result.encoder.log_columns_,
            "columns_not_used": self.result.encoder.dropped,
            "segment_names": {str(k): v for k, v in self.names.items()},
            "intended_use": "Exploratory segmentation of an uploaded dataset.",
            "out_of_scope": "Credit, pricing or eligibility decisions about individuals.",
        }

    # ------------------------------------------------------------------ assignment
    def form_defaults(self) -> dict:
        fields = []
        enc = self.result.encoder
        for spec in enc.specs:
            if spec.kind == "category":
                values = self.customers[spec.feature]
                fields.append({"name": spec.source, "label": pretty(spec.source), "type": "select", "options": spec.levels, "value": str(values.mode().iat[0])})
            elif spec.kind == "date":
                fields.append({"name": spec.source, "label": pretty(spec.source), "type": "date", "value": (spec.ref_date - pd.Timedelta(days=float(enc.medians_[spec.feature]))).strftime("%Y-%m-%d")})
            else:
                values = self.customers[spec.feature].dropna()
                integer = bool((values % 1 == 0).all())
                fields.append(
                    {
                        "name": spec.source,
                        "label": pretty(spec.source),
                        "type": "number",
                        "value": round(float(values.median()), 0 if integer else 2),
                        "min": round(float(values.min()), 2),
                        "max": round(float(values.max()), 2),
                        "step": 1 if integer else "any",
                    }
                )
        return {"mode": self.mode, "fields": fields, "labels": self.labels()}

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        distances = self.result.distances(frame)
        segment = distances.argmin(axis=1)
        ordered = np.sort(distances, axis=1)
        second = ordered[:, 1] if ordered.shape[1] > 1 else ordered[:, 0]
        margin = 1 - ordered[:, 0] / np.where(second == 0, 1, second)
        return pd.DataFrame(
            {"Segment": segment, "Segment_Name": [self.names[s] for s in segment], "Distance_To_Centroid": ordered[:, 0].round(3), "Assignment_Margin": margin.round(3)},
            index=frame.index,
        )

    def assign(self, record: dict) -> dict:
        known = {s.source for s in self.result.encoder.specs}
        row = pd.DataFrame([{k: v for k, v in record.items() if k in known}])
        warnings = [f"{pretty(k)} was left empty and filled with the typical value." for k in known if record.get(k) in (None, "")]
        distances = self.result.distances(row)[0]
        segment = int(distances.argmin())
        ordered = np.sort(distances)
        margin = float(1 - ordered[0] / ordered[1]) if len(ordered) > 1 and ordered[1] else 1.0
        numeric, _ = self.result.encoder.features(row)
        shown = [m for m in self.metrics if m in numeric.columns]
        seg_avg = self.customers.loc[self.customers["Segment"] == segment, shown].mean()
        return {
            "segment": segment,
            "segment_name": self.names[segment],
            "assignment_margin": round(margin, 4),
            "fit": "clear" if margin >= 0.25 else "moderate" if margin >= 0.1 else "borderline",
            "distances": {self.names[i]: round(float(d), 3) for i, d in enumerate(distances)},
            "customer_features": {m: _round(numeric[m].iat[0], 3) for m in shown},
            "segment_average": {m: _round(seg_avg[m], 3) for m in shown},
            "recommendation": self._rec_by_id[segment],
            "warnings": warnings,
        }

    def score_file(self, content: bytes, filename: str, max_rows: int) -> pd.DataFrame:
        frame = read_table(content, filename, max_rows)
        return sanitize_csv(pd.concat([frame, self.predict_frame(frame)], axis=1))

    def export_customers(self, segment: int | None) -> pd.DataFrame:
        df = self.table.assign(Segment=self.customers["Segment"].to_numpy(), Segment_Name=self.customers["Segment_Name"].to_numpy())
        if segment is not None:
            if segment not in self.names:
                raise KeyError(segment)
            df = df[df["Segment"] == segment]
        return sanitize_csv(df)
