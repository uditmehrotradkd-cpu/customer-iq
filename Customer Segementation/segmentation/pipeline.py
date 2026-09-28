"""End-to-end orchestration: fit, profile, persist and score customers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from .cleaning import clean_customers
from .config import (
    ARTIFACTS_DIR,
    CHANNEL_PURCHASE_COLUMNS,
    DATE_COLUMN,
    DATE_FORMAT,
    MODEL_FILENAME,
    RAW_NUMERIC_COLUMNS,
    SegmentationConfig,
)
from .features import FeatureEngineer, engineer_features
from .model_selection import KSelectionResult, compare_algorithms, evaluate_k_range, silhouette
from .preprocessing import build_preprocessor, skewness_report
from .profiling import (
    build_recommendations,
    build_segment_profiles,
    category_mix,
    channel_mix,
    index_table,
    name_segments,
    proxy_risk_check,
    zscore_profile,
)


class CustomerSegmenter:
    """Production facade around the cleaning -> features -> preprocessing -> K-Means pipeline."""

    def __init__(self, config: SegmentationConfig | None = None, n_clusters: int | None = None):
        self.config = config or SegmentationConfig()
        self.n_clusters = n_clusters

    # ------------------------------------------------------------------ training
    def fit(self, raw: pd.DataFrame) -> "CustomerSegmenter":
        cfg = self.config
        self.n_raw_rows_ = len(raw)
        clean, self.cleaning_report_ = clean_customers(raw, cfg)

        feature_step = FeatureEngineer(cfg.clustering_features).fit(clean)
        model_features = feature_step.transform(clean)
        preprocessor = build_preprocessor(cfg).fit(model_features)
        X = preprocessor.transform(model_features)

        self.k_selection_: KSelectionResult = evaluate_k_range(X, cfg)
        k = self.n_clusters or self.k_selection_.recommended_k
        self.algorithm_comparison_ = compare_algorithms(X, k, cfg)
        kmeans = KMeans(k, n_init=cfg.n_init, random_state=cfg.random_state).fit(X)

        self.pipeline_ = Pipeline([("features", feature_step), ("preprocess", preprocessor), ("cluster", kmeans)])
        self.reference_date_ = feature_step.reference_date_
        self.skewness_report_ = skewness_report(model_features, preprocessor)
        self.X_train_ = X
        labels = kmeans.labels_

        features = engineer_features(clean, self.reference_date_)
        self.zprofile_ = zscore_profile(features, labels)
        self.segment_names_ = name_segments(self.zprofile_)
        self.index_table_ = index_table(features, labels)
        self.profiles_ = build_segment_profiles(features, labels, self.segment_names_)
        self.channel_mix_ = channel_mix(features, labels)
        self.category_mix_ = category_mix(features, labels)
        self.recommendations_ = build_recommendations(features, labels, self.segment_names_, self.zprofile_)
        self.proxy_check_ = proxy_risk_check(features, labels)
        self.silhouette_ = silhouette(X, labels, cfg.random_state)
        self.customers_ = features.assign(Segment=labels, Segment_Name=[self.segment_names_[s] for s in labels])
        return self

    @property
    def n_segments(self) -> int:
        return int(self.pipeline_.named_steps["cluster"].n_clusters)

    @property
    def labels_(self) -> np.ndarray:
        return self.customers_["Segment"].to_numpy()

    # ------------------------------------------------------------------ inference
    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in RAW_NUMERIC_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns for scoring: {missing}")
        df = raw.copy()
        if DATE_COLUMN not in df.columns:
            df[DATE_COLUMN] = pd.NaT
        for col in RAW_NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
        if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
            parsed = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
            fallback = parsed.isna() & df[DATE_COLUMN].notna()
            if fallback.any():
                parsed[fallback] = pd.to_datetime(df.loc[fallback, DATE_COLUMN].astype(str), format="ISO8601", errors="coerce")
            df[DATE_COLUMN] = parsed
        purchases = df[list(CHANNEL_PURCHASE_COLUMNS)].sum(axis=1)
        df["NumDealsPurchases"] = np.minimum(df["NumDealsPurchases"], purchases)
        return df

    def _distances(self, prepared: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.named_steps["cluster"].transform(self.pipeline_[:-1].transform(prepared))

    @staticmethod
    def _margin(distances: np.ndarray) -> np.ndarray:
        """1 - nearest/second-nearest distance: 0 = on a boundary, close to 1 = clear-cut."""
        ordered = np.sort(distances, axis=1)
        return 1 - ordered[:, 0] / np.where(ordered[:, 1] == 0, 1, ordered[:, 1])

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Assign segments; returns segment id, name, distance and separation margin per row."""
        distances = self._distances(self._prepare(raw))
        segment = distances.argmin(axis=1)
        return pd.DataFrame(
            {
                "Segment": segment,
                "Segment_Name": [self.segment_names_[s] for s in segment],
                "Distance_To_Centroid": distances.min(axis=1).round(3),
                "Assignment_Margin": self._margin(distances).round(3),
            },
            index=raw.index,
        )

    def assign_customer(self, record: dict) -> dict:
        """Score one customer and return the segment, its playbook and centroid distances."""
        prepared = self._prepare(pd.DataFrame([record]))
        distances = self._distances(prepared)
        segment = int(distances[0].argmin())
        return {
            "segment": segment,
            "segment_name": self.segment_names_[segment],
            "assignment_margin": float(self._margin(distances)[0]),
            "distances": {self.segment_names_[i]: round(float(d), 3) for i, d in enumerate(distances[0])},
            "engineered_features": self.pipeline_.named_steps["features"].transform(prepared).iloc[0].round(3).to_dict(),
            "recommendation": next(r for r in self.recommendations_ if r["segment"] == segment),
        }

    # ------------------------------------------------------------------ persistence
    def model_card(self) -> dict:
        ks = self.k_selection_
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "algorithm": "K-Means (k-means++, n_init=%d)" % self.config.n_init,
            "n_segments": self.n_segments,
            "k_selection_rationale": ks.rationale,
            "elbow_k": ks.elbow_k,
            "best_silhouette_k": ks.best_silhouette_k,
            "silhouette": round(self.silhouette_, 4),
            "clustering_features": list(self.config.clustering_features),
            "log_transformed_features": self.pipeline_.named_steps["preprocess"].named_steps["deskew"].log_columns_,
            "rows_raw": self.n_raw_rows_,
            "rows_clean": int(len(self.customers_)),
            "reference_date": self.reference_date_.strftime("%Y-%m-%d"),
            "segment_names": {str(k): v for k, v in self.segment_names_.items()},
            "sklearn_version": sklearn.__version__,
            "intended_use": "Behavioural customer segmentation for engagement planning.",
            "out_of_scope": "Credit, pricing or eligibility decisions about individuals.",
        }

    def save(self, directory: str | Path = ARTIFACTS_DIR) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / MODEL_FILENAME)
        self.customers_.to_csv(directory / "customers_segmented.csv", index=False)
        self.profiles_.to_csv(directory / "segment_profiles.csv")
        self.index_table_.to_csv(directory / "segment_index.csv")
        self.zprofile_.round(3).to_csv(directory / "segment_zscores.csv")
        self.channel_mix_.to_csv(directory / "channel_mix.csv")
        self.category_mix_.to_csv(directory / "category_mix.csv")
        self.k_selection_.metrics.round(4).to_csv(directory / "k_selection_metrics.csv", index=False)
        self.algorithm_comparison_.to_csv(directory / "algorithm_comparison.csv", index=False)
        self.cleaning_report_.to_frame().to_csv(directory / "cleaning_report.csv", index=False)
        self.skewness_report_.to_csv(directory / "skewness_report.csv")
        self.proxy_check_.to_csv(directory / "proxy_risk_check.csv", index=False)
        (directory / "recommendations.json").write_text(json.dumps(self.recommendations_, indent=2), encoding="utf-8")
        (directory / "model_card.json").write_text(json.dumps(self.model_card(), indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path = ARTIFACTS_DIR) -> "CustomerSegmenter":
        """Load a trusted, locally produced model artifact (joblib must never load untrusted files)."""
        model = joblib.load(Path(directory) / MODEL_FILENAME)
        if not isinstance(model, cls):
            raise TypeError("Artifact is not a CustomerSegmenter")
        return model
