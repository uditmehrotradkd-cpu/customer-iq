"""Schema-free segmentation: find natural segments in any table using all of its usable columns.

Unlike ``CustomerSegmenter`` (fixed retail schema), ``TableEncoder`` inspects each column, keeps
numeric, date and low-cardinality categorical fields, drops identifiers / personal data / free
text, and remembers how it encoded them so new rows can be scored the same way. ``auto_segment``
clusters every row with K-Means (k picked by silhouette) and describes each segment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from .model_selection import silhouette

K_MIN, K_MAX = 2, 8
MIN_SEGMENT_SHARE = 0.03
MAX_CATEGORY_LEVELS = 20
# One-hot columns are scaled so switching category moves a row about as far as 1 std of a numeric column.
CATEGORY_WEIGHT = 1 / np.sqrt(2)
PII_TOKENS = {
    "name", "firstname", "lastname", "fullname", "surname", "email", "mail", "phone", "mobile", "telephone",
    "address", "street", "zip", "zipcode", "postcode", "postal", "ssn", "passport", "iban", "ip",
}
ID_TOKENS = {"id", "uuid", "guid"}
DATE_TOKENS = {"date", "dt", "time", "timestamp", "since", "joined", "created", "updated", "dob", "birthday"}
OUTPUT_COLUMNS = {"Segment", "Segment_Name", "Assigned_Segment", "Intended_Segment", "Distance_To_Centroid", "Assignment_Margin"}
DATE_VALUE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
NUMBER_JUNK = r"[,$€£₹%\s]"


def _tokens(name: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", str(name))}


def pretty(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name)).replace("_", " ")
    return re.sub(r"\s+", " ", spaced).strip()


def fmt_value(value: float) -> str:
    if pd.isna(value):
        return "–"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}" if abs(value) >= 10 else f"{value:.2f}"


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    iso = pd.to_datetime(text, format="ISO8601", errors="coerce")
    rest = iso.isna()
    if rest.any():
        iso[rest] = pd.to_datetime(text[rest], format="mixed", dayfirst=True, errors="coerce")
    return iso


def _to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype("string").str.replace(NUMBER_JUNK, "", regex=True), errors="coerce").astype(float)


@dataclass
class ColumnSpec:
    source: str
    feature: str
    kind: str  # "number" | "date" | "category"
    levels: list[str] | None = None
    ref_date: pd.Timestamp | None = None


@dataclass
class TableEncoder:
    """Learns how to turn a raw table into a clustering matrix, and applies the same recipe to new rows."""

    specs: list[ColumnSpec] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    one_hot_groups: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------ planning
    def _find_one_hot_groups(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for col in frame.columns:
            name = str(col)
            if "_" in name and pd.api.types.is_numeric_dtype(frame[col]) and frame[col].dropna().isin([0, 1]).all():
                groups.setdefault(name.rsplit("_", 1)[0], []).append(col)
        keep = {}
        for prefix, cols in groups.items():
            if len(cols) >= 3 and prefix not in frame.columns and not (frame[cols].fillna(0).sum(axis=1) > 1).any():
                keep[prefix] = cols
        return keep

    def _collapse(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Turn groups of mutually exclusive 0/1 columns like ``Colour_red``/``Colour_blue`` back into one column."""
        out = frame
        for prefix, cols in self.one_hot_groups.items():
            present = [c for c in cols if c in out.columns]
            if not present or prefix in out.columns:
                continue
            block = out[present].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
            levels = np.array([str(c)[len(prefix) + 1 :] for c in present], dtype=object)
            decoded = np.where(block.sum(axis=1) == 0, "Other", levels[block.argmax(axis=1)])
            out = out.drop(columns=present).assign(**{prefix: decoded})
        return out

    def _plan(self, frame: pd.DataFrame) -> None:
        n = len(frame)
        for col in frame.columns:
            series = frame[col]
            values = series.dropna()
            tokens = _tokens(col)
            name = str(col)
            if name in OUTPUT_COLUMNS:
                self.dropped[name] = "output of an earlier segmentation"
            elif values.empty:
                self.dropped[name] = "empty"
            elif values.nunique() <= 1:
                self.dropped[name] = "same value in every row"
            elif tokens & PII_TOKENS:
                self.dropped[name] = "personal data (not used for segmentation)"
            elif tokens & ID_TOKENS:
                self.dropped[name] = "identifier"
            elif pd.api.types.is_bool_dtype(series):
                self.specs.append(ColumnSpec(name, name, "number"))
            elif pd.api.types.is_numeric_dtype(series):
                if pd.api.types.is_integer_dtype(series) and values.nunique() == n and n >= 100:
                    self.dropped[name] = "unique per row (looks like an identifier)"
                else:
                    self.specs.append(ColumnSpec(name, name, "number"))
            elif pd.api.types.is_datetime64_any_dtype(series):
                self.specs.append(ColumnSpec(name, f"{name} (days before latest)", "date", ref_date=series.max()))
            else:
                text = values.astype(str).str.strip()
                if pd.to_numeric(text.str.replace(NUMBER_JUNK, "", regex=True), errors="coerce").notna().mean() >= 0.95:
                    self.specs.append(ColumnSpec(name, name, "number"))
                    continue
                if tokens & DATE_TOKENS or text.head(200).str.match(DATE_VALUE).mean() >= 0.9:
                    dates = _parse_dates(series)
                    if dates.notna().sum() >= 0.9 * len(values):
                        self.specs.append(ColumnSpec(name, f"{name} (days before latest)", "date", ref_date=dates.max()))
                        continue
                counts = text.value_counts()
                if len(counts) <= MAX_CATEGORY_LEVELS:
                    self.specs.append(ColumnSpec(name, name, "category", levels=[str(v) for v in counts.index]))
                elif counts.head(MAX_CATEGORY_LEVELS - 1).sum() >= 0.9 * len(values):
                    self.specs.append(ColumnSpec(name, name, "category", levels=[str(v) for v in counts.head(MAX_CATEGORY_LEVELS - 1).index]))
                else:
                    self.dropped[name] = "free text / too many distinct values"

    # ------------------------------------------------------------------ features
    def features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Readable (unscaled) numeric and categorical features; missing columns become empty values."""
        frame = self._collapse(frame)
        numeric, categorical = {}, {}
        for spec in self.specs:
            series = frame[spec.source] if spec.source in frame.columns else pd.Series(np.nan, index=frame.index)
            if spec.kind == "number":
                numeric[spec.feature] = _to_number(series)
            elif spec.kind == "date":
                dates = series if pd.api.types.is_datetime64_any_dtype(series) else _parse_dates(series)
                numeric[spec.feature] = (spec.ref_date - dates).dt.days.astype(float)
            else:
                clean = series.astype("string").str.strip()
                categorical[spec.feature] = clean.where(clean.isin(spec.levels) | clean.isna(), "Other").fillna("Missing").astype(str)
        return pd.DataFrame(numeric, index=frame.index), pd.DataFrame(categorical, index=frame.index)

    @property
    def numeric_features(self) -> list[str]:
        return [s.feature for s in self.specs if s.kind != "category"]

    @property
    def categorical_features(self) -> list[str]:
        return [s.feature for s in self.specs if s.kind == "category"]

    # ------------------------------------------------------------------ fit / transform
    def fit(self, frame: pd.DataFrame) -> "TableEncoder":
        self.one_hot_groups = self._find_one_hot_groups(frame)
        self._plan(self._collapse(frame))
        if not self.specs:
            raise ValueError("No usable columns found: every column is empty, constant, an identifier, personal data or free text.")
        numeric, categorical = self.features(frame)
        self.medians_ = numeric.median()
        filled = numeric.fillna(self.medians_)
        self.log_columns_ = [c for c in filled.columns if filled[c].min() >= 0 and filled[c].skew() > 1]
        self.skew_before_ = filled.skew()
        logged = self._log(filled)
        self.lower_, self.upper_ = logged.quantile(0.01), logged.quantile(0.99)
        clipped = logged.clip(self.lower_, self.upper_, axis=1)
        self.skew_after_ = clipped.skew()
        self.mean_, self.std_ = clipped.mean(), clipped.std(ddof=0).replace(0, 1)
        self.dummy_columns_ = list(pd.get_dummies(categorical, dtype=float).columns)
        return self

    def _log(self, numeric: pd.DataFrame) -> pd.DataFrame:
        out = numeric.copy()
        if self.log_columns_:
            out[self.log_columns_] = np.log1p(out[self.log_columns_].clip(lower=0))
        return out

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric, categorical = self.features(frame)
        parts = []
        if not numeric.empty:
            scaled = self._log(numeric.fillna(self.medians_)).clip(self.lower_, self.upper_, axis=1)
            parts.append(((scaled - self.mean_) / self.std_).to_numpy(dtype=float))
        if self.dummy_columns_:
            dummies = pd.get_dummies(categorical, dtype=float).reindex(columns=self.dummy_columns_, fill_value=0.0)
            parts.append(dummies.to_numpy(dtype=float) * CATEGORY_WEIGHT)
        return np.hstack(parts)


@dataclass
class AutoSegmentation:
    encoder: TableEncoder
    kmeans: KMeans
    order: np.ndarray  # order[new_label] = original KMeans label (segments sorted by size)
    labels: np.ndarray
    names: dict[int, str]
    k: int
    auto_k: bool
    silhouette: float
    k_table: pd.DataFrame
    profiles: pd.DataFrame
    segments: list[dict]
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    X: np.ndarray

    @property
    def numeric_columns(self) -> list[str]:
        return list(self.numeric.columns)

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.categorical.columns)

    @property
    def dropped(self) -> dict[str, str]:
        return self.encoder.dropped

    @property
    def centers(self) -> np.ndarray:
        return self.kmeans.cluster_centers_[self.order]

    def distances(self, frame: pd.DataFrame) -> np.ndarray:
        """Distance from every row to every segment centre (columns follow the relabelled segment ids)."""
        return self.kmeans.transform(self.encoder.transform(frame))[:, self.order]


def _choose_k(X: np.ndarray, n_clusters: int | None, random_state: int) -> tuple[int, KMeans, pd.DataFrame]:
    ks = [n_clusters] if n_clusters else list(range(K_MIN, K_MAX + 1))
    rows, fits = [], {}
    for k in ks:
        model = KMeans(k, n_init=10, random_state=random_state).fit(X)
        labels = model.labels_
        fits[k] = model
        rows.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 2),
                "silhouette": round(silhouette(X, labels, random_state), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 2),
                "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
                "min_segment_share": round(np.bincount(labels).min() / len(X), 4),
            }
        )
    table = pd.DataFrame(rows)
    table["eligible"] = table["min_segment_share"] >= MIN_SEGMENT_SHARE
    if n_clusters:
        table["recommended"] = True
        return n_clusters, fits[n_clusters], table
    eligible = table[table["eligible"]] if table["eligible"].any() else table
    best = eligible["silhouette"].max()
    # Silhouette almost always peaks at k=2, which rarely drives different actions; prefer k>=3 unless clearly worse.
    richer = eligible[(eligible["k"] >= 3) & (eligible["silhouette"] >= 0.6 * best)]
    pool = richer if not richer.empty else eligible
    k = int(pool.loc[pool["silhouette"].idxmax(), "k"])
    table["recommended"] = table["k"] == k
    return k, fits[k], table


def _describe(numeric: pd.DataFrame, categorical: pd.DataFrame, labels: np.ndarray, k: int) -> tuple[dict[int, str], list[dict], pd.DataFrame]:
    mean_all = numeric.mean()
    std_all = numeric.std(ddof=0).replace(0, np.nan)
    names: dict[int, str] = {}
    segments, rows = [], []
    for s in range(k):
        mask = labels == s
        seg_mean = numeric[mask].mean()
        z = ((seg_mean - mean_all) / std_all).dropna()
        z = z.reindex(z.abs().sort_values(ascending=False).index)
        traits = [
            {"text": f"{'Higher' if v > 0 else 'Lower'} {pretty(c)}: {fmt_value(seg_mean[c])} vs {fmt_value(mean_all[c])} overall", "direction": "high" if v > 0 else "low"}
            for c, v in z.head(3).items()
            if abs(v) >= 0.25
        ]
        cat_hits = []
        for col in categorical.columns:
            seg_share = categorical.loc[mask, col].value_counts(normalize=True)
            lift = seg_share / categorical[col].value_counts(normalize=True).reindex(seg_share.index)
            strong = lift[(seg_share >= 0.4) & (lift >= 1.3)]
            if not strong.empty:
                level = strong.idxmax()
                overall = categorical[col].eq(level).mean()
                cat_hits.append((float(strong.max()), col, str(level), float(seg_share[level]), float(overall)))
        cat_hits.sort(reverse=True)
        traits += [
            {"text": f"{pretty(col)} mostly {level} ({share:.0%} vs {overall:.0%} overall)", "direction": "high"}
            for _, col, level, share, overall in cat_hits[:2]
        ]
        parts = [f"{'High' if v > 0 else 'Low'} {pretty(c)}" for c, v in z.head(2).items() if abs(v) >= 0.3]
        parts += [level for _, _, level, _, _ in cat_hits[: max(0, 2 - len(parts))]]
        name = " · ".join(parts) or "Typical profile"
        if name in names.values():
            name = f"{name} ({s + 1})"
        names[s] = name
        size = int(mask.sum())
        segments.append({"id": s, "name": name, "rows": size, "share_pct": round(100 * size / len(labels), 1), "traits": traits})
        row = {"Segment": s, "Segment_Name": name, "Rows": size, "Share_%": round(100 * size / len(labels), 1)}
        row.update({f"Avg {c}": round(float(v), 2) for c, v in seg_mean.items()})
        row.update({f"Top {c}": categorical.loc[mask, c].mode().iat[0] for c in categorical.columns})
        rows.append(row)
    return names, segments, pd.DataFrame(rows)


def auto_segment(frame: pd.DataFrame, n_clusters: int | None = None, random_state: int = 42) -> AutoSegmentation:
    """Segment every row of ``frame`` on all usable columns; raises ValueError with a user-safe message."""
    if n_clusters is not None and not K_MIN <= n_clusters <= 10:
        raise ValueError("Please choose between 2 and 10 segments.")
    encoder = TableEncoder().fit(frame)
    numeric, categorical = encoder.features(frame)
    X = encoder.transform(frame)
    if len(np.unique(X, axis=0)) < (n_clusters or K_MAX):
        raise ValueError("Not enough distinct rows to form segments.")
    k, kmeans, k_table = _choose_k(X, n_clusters, random_state)
    order = np.argsort(-np.bincount(kmeans.labels_, minlength=k))
    labels = np.argsort(order)[kmeans.labels_]
    names, segments, profiles = _describe(numeric, categorical, labels, k)
    return AutoSegmentation(
        encoder=encoder,
        kmeans=kmeans,
        order=order,
        labels=labels,
        names=names,
        k=k,
        auto_k=n_clusters is None,
        silhouette=silhouette(X, labels, random_state),
        k_table=k_table,
        profiles=profiles,
        segments=segments,
        numeric=numeric,
        categorical=categorical,
        X=X,
    )
