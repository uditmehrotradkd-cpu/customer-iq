"""Business-facing view models built on top of a fitted ``CustomerSegmenter``."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from segmentation import CustomerSegmenter, load_customers
from segmentation.config import (
    DEFAULT_DATA_PATH,
    MODEL_FILENAME,
    PROFILE_CATEGORICAL_FEATURES,
    PROFILE_NUMERIC_FEATURES,
    SPEND_COLUMNS,
)
from segmentation.data_loader import to_canonical_schema
from segmentation.visualization import pca_projection

from .file_io import normalize_columns, read_table

log = logging.getLogger(__name__)

CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
AGE_BAND_ORDER = ["<35", "35-44", "45-54", "55-64", "65+"]
SEGMENT_COLORS = ["#2a9d8f", "#e76f51", "#6c63ff", "#e9c46a", "#8ab17d", "#f4a261", "#264653", "#b56576", "#457b9d", "#9c6644"]


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _round(value, digits: int = 2) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), digits)


def sanitize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells of exported CSVs."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: f"'{v}" if isinstance(v, str) and v.startswith(CSV_FORMULA_PREFIXES) else v)
    return out


class SegmentationService:
    mode = "customer"

    def __init__(self, segmenter: CustomerSegmenter, trained_at_utc: str = ""):
        self.segmenter = segmenter
        self.trained_at_utc = trained_at_utc
        self.customers = segmenter.customers_
        self.names = segmenter.segment_names_
        self.numeric_features = [c for c in PROFILE_NUMERIC_FEATURES if c in self.customers.columns]
        self.categorical_features = list(PROFILE_CATEGORICAL_FEATURES)
        coords, pca = pca_projection(segmenter.X_train_)
        self._pca_coords = coords
        self._pca_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        train = np.asarray(segmenter.X_train_)
        pca3 = PCA(n_components=3, random_state=0).fit(train)
        self._pca3_coords = pca3.transform(train)
        self._pca3_centers = pca3.transform(segmenter.pipeline_.named_steps["cluster"].cluster_centers_)
        self._pca3_variance = [round(float(v), 4) for v in pca3.explained_variance_ratio_]
        self._recommendations = {r["segment"]: r for r in segmenter.recommendations_}

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, artifacts_dir: Path, auto_train: bool) -> "SegmentationService":
        model_path = artifacts_dir / MODEL_FILENAME
        if model_path.exists():
            segmenter = CustomerSegmenter.load(artifacts_dir)
        elif auto_train:
            log.warning("No model at %s; training on bundled dataset", model_path)
            segmenter = CustomerSegmenter().fit(load_customers(DEFAULT_DATA_PATH))
            segmenter.save(artifacts_dir)
        else:
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        card_path = artifacts_dir / "model_card.json"
        trained_at = json.loads(card_path.read_text(encoding="utf-8")).get("created_at_utc", "") if card_path.exists() else ""
        return cls(segmenter, trained_at)

    # ------------------------------------------------------------------ helpers
    @property
    def segment_ids(self) -> list[int]:
        return sorted(self.names)

    def segments_meta(self) -> list[dict]:
        return [
            {"id": s, "name": self.names[s], "color": SEGMENT_COLORS[s % len(SEGMENT_COLORS)]}
            for s in self.segment_ids
        ]

    # ------------------------------------------------------------------ views
    def labels(self) -> dict:
        return {"entity": "customers", "entity_singular": "customer", "value": "Revenue", "dataset": "customer data"}

    def overview(self) -> dict:
        c = self.customers
        profiles = self.segmenter.profiles_.reset_index()
        kpis = {
            "customers": int(len(c)),
            "segments": self.segmenter.n_segments,
            "silhouette": round(self.segmenter.silhouette_, 3),
            "total_spend": round(float(c["Total_Spend"].sum()), 0),
            "avg_spend": round(float(c["Total_Spend"].mean()), 0),
            "median_income": round(float(c["Income"].median()), 0),
        }
        return {
            "mode": self.mode,
            "labels": self.labels(),
            "kpis": kpis,
            "kpi_list": [
                {"label": "Customers segmented", "value": kpis["customers"], "format": "integer"},
                {"label": "Segments", "value": kpis["segments"], "format": "integer"},
                {"label": "Silhouette score", "value": kpis["silhouette"], "format": "decimal3"},
                {"label": "Total spend (2 yrs)", "value": kpis["total_spend"], "format": "compactCurrency"},
                {"label": "Average spend", "value": kpis["avg_spend"], "format": "currency"},
                {"label": "Median income", "value": kpis["median_income"], "format": "currency"},
            ],
            "segments": self.segments_meta(),
            "profiles": _records(profiles),
            "profile_columns": [
                {"key": "Customers", "label": "Customers", "format": "integer"},
                {"key": "Customer_Share_%", "label": "Customers %", "format": "shareBar"},
                {"key": "Revenue_Share_%", "label": "Revenue %", "format": "shareBar"},
                {"key": "Median_Income", "label": "Median income", "format": "currency"},
                {"key": "Avg_Total_Spend", "label": "Avg spend", "format": "currency"},
                {"key": "Avg_Purchases", "label": "Avg purchases", "format": "number"},
                {"key": "Avg_Order_Value", "label": "AOV", "format": "currency"},
                {"key": "Avg_Recency_Days", "label": "Recency (days)", "format": "number"},
                {"key": "Deal_Ratio_%", "label": "Deal ratio", "format": "percent"},
                {"key": "Campaign_Acceptance", "label": "Campaigns", "format": "number"},
                {"key": "Avg_Tenure_Days", "label": "Tenure (days)", "format": "integer"},
            ],
            "slide_stats": [
                {"key": "Customer_Share_%", "label": "Customers", "format": "percent"},
                {"key": "Revenue_Share_%", "label": "Revenue", "format": "percent"},
                {"key": "Avg_Total_Spend", "label": "Avg spend", "format": "currency"},
            ],
            "recommendations": self.segmenter.recommendations_,
            "rationale": self.segmenter.k_selection_.rationale,
        }

    def pca_points(self) -> dict:
        c = self.customers
        points = pd.DataFrame(
            {
                "x": self._pca_coords[:, 0].round(3),
                "y": self._pca_coords[:, 1].round(3),
                "segment": c["Segment"].to_numpy(),
                "id": c["ID"].to_numpy(),
                "info": [f"spend ${s:,.0f} · income ${i:,.0f}" for s, i in zip(c["Total_Spend"], c["Income"])],
            }
        )
        return {"variance": self._pca_variance, "points": _records(points)}

    def pca3d(self) -> dict:
        """Compact 3-component projection for the WebGL "customer universe" (flat arrays keep payload small)."""
        coords = self._pca3_coords.round(3)
        return {
            "variance": self._pca3_variance,
            "positions": coords.ravel().tolist(),
            "segments": self.customers["Segment"].astype(int).tolist(),
            "centers": [
                {"segment": int(s), "position": [round(float(v), 3) for v in self._pca3_centers[s]]}
                for s in self.segment_ids
            ],
        }

    def segment_detail(self, segment: int) -> dict:
        if segment not in self.names:
            raise KeyError(segment)
        c = self.customers
        rows = c[c["Segment"] == segment]
        metric_defs = [
            ("Share of customers", len(rows) / len(c), None, "percent"),
            ("Median income", rows["Income"].median(), c["Income"].median(), "currency"),
            ("Avg total spend", rows["Total_Spend"].mean(), c["Total_Spend"].mean(), "currency"),
            ("Avg purchases", rows["Total_Purchases"].mean(), c["Total_Purchases"].mean(), "number"),
            ("Avg order value", rows["Avg_Order_Value"].mean(), c["Avg_Order_Value"].mean(), "currency"),
            ("Deal ratio", rows["Deal_Ratio"].mean(), c["Deal_Ratio"].mean(), "percent"),
            ("Web visits / month", rows["NumWebVisitsMonth"].mean(), c["NumWebVisitsMonth"].mean(), "number"),
            ("Campaigns accepted", rows["Campaigns_Accepted"].mean(), c["Campaigns_Accepted"].mean(), "number"),
        ]
        metrics = [
            {
                "label": label,
                "value": _round(value, 4),
                "baseline": _round(base, 4),
                "delta_pct": _round((value / base - 1) * 100, 1) if base else None,
                "format": fmt,
            }
            for label, value, base, fmt in metric_defs
        ]
        profile = self.segmenter.profiles_.loc[segment]
        return {
            "id": segment,
            "name": self.names[segment],
            "color": SEGMENT_COLORS[segment % len(SEGMENT_COLORS)],
            "customers": int(profile["Customers"]),
            "revenue_share": float(profile["Revenue_Share_%"]),
            "metrics": metrics,
            "charts": [
                {"title": "Purchase channel mix", "caption": "Share of this segment's purchases by channel.", "values": {k: float(v) for k, v in self.segmenter.channel_mix_.loc[segment].items()}, "unit": "%"},
                {"title": "Spend by category", "caption": "Share of this segment's spend by product category.", "values": {k: float(v) for k, v in self.segmenter.category_mix_.loc[segment].items()}, "unit": "%"},
            ],
            "channel_mix": {k: float(v) for k, v in self.segmenter.channel_mix_.loc[segment].items()},
            "category_mix": {k: float(v) for k, v in self.segmenter.category_mix_.loc[segment].items()},
            "zscores": {k: round(float(v), 3) for k, v in self.segmenter.zprofile_.loc[segment].items()},
            "recommendation": self._recommendations[segment],
            "labels": self.labels(),
        }

    def fingerprint(self) -> dict:
        z = self.segmenter.zprofile_
        return {
            "features": list(z.columns),
            "rows": [
                {"segment": int(s), "name": self.names[int(s)], "values": [round(float(v), 2) for v in z.loc[s]]}
                for s in z.index
            ],
        }

    def distribution(self, feature: str) -> dict:
        if feature not in self.numeric_features:
            raise KeyError(feature)
        stats = []
        for s in self.segment_ids:
            values = self.customers.loc[self.customers["Segment"] == s, feature]
            q = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(3).tolist()
            stats.append(
                {"segment": s, "min": q[0], "q1": q[1], "median": q[2], "q3": q[3], "max": q[4], "mean": _round(values.mean(), 3)}
            )
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
        return {"x": x, "y": y, "points": _records(points)}

    def demographics(self, attribute: str) -> dict:
        if attribute not in self.categorical_features:
            raise KeyError(attribute)
        table = pd.crosstab(self.customers["Segment"], self.customers[attribute], normalize="index") * 100
        if attribute == "Age_Band":
            table = table[[c for c in AGE_BAND_ORDER if c in table.columns]]
        return {
            "attribute": attribute,
            "categories": [str(c) for c in table.columns],
            "rows": [{"segment": int(s), "values": [round(float(v), 1) for v in table.loc[s]]} for s in table.index],
        }

    def explorer_features(self) -> dict:
        return {
            "numeric": self.numeric_features,
            "categorical": self.categorical_features,
            "categorical_note": "Not used to build segments; shown only to understand who is in each group.",
        }

    def diagnostics(self) -> dict:
        s = self.segmenter
        ks = s.k_selection_
        return {
            "mode": self.mode,
            "rationale": ks.rationale,
            "selected_k": s.n_segments,
            "elbow_k": ks.elbow_k,
            "best_silhouette_k": ks.best_silhouette_k,
            "silhouette": round(s.silhouette_, 4),
            "k_metrics": _records(ks.metrics.round(4)),
            "algorithms": _records(s.algorithm_comparison_),
            "cleaning": _records(s.cleaning_report_.to_frame()),
            "skewness": _records(s.skewness_report_.reset_index(names="feature")),
            "proxy_check": _records(s.proxy_check_),
            "model_card": {**s.model_card(), "created_at_utc": self.trained_at_utc},
        }

    def form_defaults(self) -> dict:
        m = self.customers.median(numeric_only=True)
        ref = self.segmenter.reference_date_
        int_cols = [
            "Year_Birth", "Kidhome", "Teenhome", "Recency", "NumDealsPurchases", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        ]
        defaults = {col: int(m[col]) for col in int_cols}
        defaults.update({col: round(float(m[col]), 0) for col in ("Income", *SPEND_COLUMNS)})
        defaults["Kidhome"] = defaults["Teenhome"] = 0
        defaults["Dt_Customer"] = (ref - pd.Timedelta(days=int(m["Tenure_Days"]))).strftime("%Y-%m-%d")
        return {"mode": self.mode, "defaults": defaults, "reference_date": ref.strftime("%Y-%m-%d")}

    # ------------------------------------------------------------------ scoring
    def assign(self, record: dict) -> dict:
        warnings = []
        spend = sum(record[c] for c in SPEND_COLUMNS)
        purchases = record["NumWebPurchases"] + record["NumCatalogPurchases"] + record["NumStorePurchases"]
        if spend > 0 and purchases == 0:
            warnings.append("Spend was entered with zero purchases; the record is inconsistent.")
        if record["NumDealsPurchases"] > purchases:
            warnings.append("Deal purchases exceed total purchases and were capped.")
        if record.get("Income") is None:
            warnings.append("Income is missing and was imputed with the training median.")

        payload = {k: (int(v) if isinstance(v, bool) else v) for k, v in record.items()}
        payload["Income"] = np.nan if payload.get("Income") is None else payload["Income"]
        payload["Dt_Customer"] = pd.Timestamp(payload["Dt_Customer"])
        result = self.segmenter.assign_customer(payload)

        segment = result["segment"]
        features = {k: round(float(v), 3) for k, v in result["engineered_features"].items()}
        seg_avg = self.customers.loc[self.customers["Segment"] == segment, list(features)].mean()
        margin = result["assignment_margin"]
        return {
            "segment": segment,
            "segment_name": result["segment_name"],
            "assignment_margin": round(margin, 4),
            "fit": "clear" if margin >= 0.25 else "moderate" if margin >= 0.1 else "borderline",
            "distances": result["distances"],
            "customer_features": features,
            "segment_average": {k: round(float(v), 3) for k, v in seg_avg.items()},
            "recommendation": result["recommendation"],
            "warnings": warnings,
        }

    def score_file(self, content: bytes, filename: str, max_rows: int) -> pd.DataFrame:
        frame, _ = normalize_columns(read_table(content, filename, max_rows))
        canonical = to_canonical_schema(frame)
        scored = pd.concat([canonical, self.segmenter.predict(canonical)], axis=1)
        return sanitize_csv(scored)

    def export_customers(self, segment: int | None) -> pd.DataFrame:
        columns = ["ID", "Segment", "Segment_Name", *self.numeric_features, *self.categorical_features]
        df = self.customers[columns]
        if segment is not None:
            if segment not in self.names:
                raise KeyError(segment)
            df = df[df["Segment"] == segment]
        return sanitize_csv(df)
