import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from .config import Config

class FeaturePreprocessor:
    """
    Production choice:
    1. log1p reduces the influence of heavily right-skewed monetary/frequency values.
    2. 1st/99th percentile clipping limits extreme observations without deleting customers.
    3. RobustScaler uses median/IQR, making scaling less sensitive to remaining outliers.
    """
    def __init__(self, columns):
        self.columns = list(columns)
        self.bounds = {}
        self.scaler = RobustScaler()

    def fit(self, df):
        x = df[self.columns].astype(float).clip(lower=0)
        x = np.log1p(x)
        for c in self.columns:
            self.bounds[c] = (
                float(x[c].quantile(0.01)),
                float(x[c].quantile(0.99))
            )
            x[c] = x[c].clip(*self.bounds[c])
        self.scaler.fit(x)
        return self

    def transform(self, df):
        x = df[self.columns].astype(float).clip(lower=0)
        x = np.log1p(x)
        for c, (lo, hi) in self.bounds.items():
            x[c] = x[c].clip(lo, hi)
        return self.scaler.transform(x)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
