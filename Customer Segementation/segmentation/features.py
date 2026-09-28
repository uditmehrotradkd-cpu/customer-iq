"""Feature engineering: RFM-style behavioural signals plus descriptive demographics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (
    CAMPAIGN_COLUMNS,
    CATEGORY_SHARE_MAP,
    CHANNEL_PURCHASE_COLUMNS,
    CLUSTERING_FEATURES,
    DATE_COLUMN,
    SPEND_COLUMNS,
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def engineer_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Derive spending, frequency, recency, engagement and demographic features.

    ``reference_date`` anchors age and tenure so training and scoring use the same clock.
    """
    out = df.copy()
    out["Total_Spend"] = out[list(SPEND_COLUMNS)].sum(axis=1)
    out["Total_Purchases"] = out[list(CHANNEL_PURCHASE_COLUMNS)].sum(axis=1)
    out["Avg_Order_Value"] = _safe_ratio(out["Total_Spend"], out["Total_Purchases"])
    out["Deal_Ratio"] = _safe_ratio(out["NumDealsPurchases"], out["Total_Purchases"]).clip(0, 1)
    out["Web_Share"] = _safe_ratio(out["NumWebPurchases"], out["Total_Purchases"])
    out["Catalog_Share"] = _safe_ratio(out["NumCatalogPurchases"], out["Total_Purchases"])
    out["Store_Share"] = _safe_ratio(out["NumStorePurchases"], out["Total_Purchases"])
    out["Campaigns_Accepted"] = out[list(CAMPAIGN_COLUMNS)].sum(axis=1)
    out["Spend_To_Income"] = _safe_ratio(out["Total_Spend"], out["Income"])
    for spend_col, share_col in CATEGORY_SHARE_MAP.items():
        out[share_col] = _safe_ratio(out[spend_col], out["Total_Spend"])

    out["Age"] = reference_date.year - out["Year_Birth"]
    out["Age_Band"] = pd.cut(
        out["Age"], bins=[0, 34, 44, 54, 64, 200], labels=["<35", "35-44", "45-54", "55-64", "65+"]
    ).astype(str)
    out["Children"] = out["Kidhome"] + out["Teenhome"]
    out["Is_Parent"] = np.where(out["Children"] > 0, "Parent", "No children")
    enrolled = pd.to_datetime(out[DATE_COLUMN], errors="coerce").fillna(reference_date)
    out["Tenure_Days"] = (reference_date - enrolled).dt.days.clip(lower=0)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that learns the reference date and emits clustering features."""

    def __init__(self, features: tuple[str, ...] = CLUSTERING_FEATURES):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        dates = pd.to_datetime(X[DATE_COLUMN], errors="coerce")
        self.reference_date_ = dates.max() if dates.notna().any() else pd.Timestamp.today().normalize()
        self.feature_names_out_ = np.array(self.features, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X, self.reference_date_)[list(self.features)].astype(float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
