"""Preprocessing: impute -> winsorise -> de-skew -> standardise.

Why each step matters for distance-based clustering:
* Imputation keeps every customer scoreable (no dropped rows at inference time).
* Winsorising caps extreme values so a handful of whales cannot drag centroids.
* log1p on right-skewed variables (spend, purchases, AOV) compresses long tails so
  Euclidean distance reflects relative, not absolute, differences.
* Standard scaling puts income (tens of thousands) and ratios (0-1) on the same scale;
  otherwise K-Means would cluster almost entirely on income.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SegmentationConfig


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantile bounds learned during ``fit``."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.lower)
        self.upper_bounds_ = X.quantile(self.upper)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` to non-negative columns whose skewness exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.skewness_ = X.skew()
        eligible = (self.skewness_.abs() > self.threshold) & (X.min() >= 0)
        self.log_columns_ = list(self.skewness_.index[eligible])
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        if self.log_columns_:
            X[self.log_columns_] = np.log1p(X[self.log_columns_].clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(config: SegmentationConfig | None = None) -> Pipeline:
    config = config or SegmentationConfig()
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("winsorize", Winsorizer(config.winsor_lower, config.winsor_upper)),
            ("deskew", SkewCorrector(config.skew_threshold)),
            ("scale", StandardScaler().set_output(transform="pandas")),
        ]
    )


def skewness_report(before: pd.DataFrame, preprocessor: Pipeline) -> pd.DataFrame:
    """Compare skewness of raw features vs. after winsorising and de-skewing."""
    partial = preprocessor[:-1].transform(before)
    deskew: SkewCorrector = preprocessor.named_steps["deskew"]
    return pd.DataFrame(
        {
            "skew_before": before.skew().round(2),
            "skew_after": partial.skew().round(2),
            "log_transformed": [c in deskew.log_columns_ for c in before.columns],
        }
    )
