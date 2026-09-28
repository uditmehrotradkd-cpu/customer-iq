from pathlib import Path
import pandas as pd
import numpy as np
from .config import Config

def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df

def clean_dataset(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert object columns that are mostly numeric to numeric.
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.95:
                df[col] = converted

    # Numeric missing values use median; categorical values use the mode.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return df, {"rows_before": before, "rows_after": len(df), "duplicates_removed": before-len(df)}

def detect_schema(df, cfg: Config):
    cols = set(df.columns)
    monetary = [c for c in cfg.monetary_cols if c in cols]
    frequency = [c for c in cfg.frequency_cols if c in cols]
    behavior = [c for c in cfg.optional_behavior_cols if c in cols]

    if "Recency" not in cols:
        raise ValueError("This dataset must contain a Recency column or you must extend the schema detector.")

    if not monetary:
        # Generic fallback: columns beginning with Mnt are treated as monetary.
        monetary = [c for c in df.columns if c.lower().startswith("mnt")]
    if not frequency:
        frequency = [c for c in df.columns if c.lower().startswith("num") and "purchas" in c.lower()]

    if not monetary:
        raise ValueError("No monetary columns detected. Expected columns such as MntWines/MntFruits/...")
    if not frequency:
        raise ValueError("No purchase-frequency columns detected. Expected NumWebPurchases/NumStorePurchases/...")

    return {"monetary": monetary, "frequency": frequency, "behavior": behavior}

def build_feature_table(df, schema, cfg: Config):
    out = pd.DataFrame(index=df.index)
    out["Recency"] = pd.to_numeric(df["Recency"], errors="coerce").fillna(df["Recency"].median())

    out["Monetary"] = df[schema["monetary"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    out["Frequency"] = df[schema["frequency"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # Useful behavioral features available in this dataset.
    for col in schema["behavior"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derive age if Year_Birth exists. Suspicious ages are clipped rather than
    # allowing impossible birth years to dominate clustering.
    if "Year_Birth" in df.columns:
        birth = pd.to_numeric(df["Year_Birth"], errors="coerce")
        age = 2025 - birth
        out["Age"] = age.clip(18, 100).fillna(age.median())

    # AOV is a useful normalized spending behavior.
    out["AverageOrderValue"] = out["Monetary"] / out["Frequency"].clip(lower=1)

    # Retain the source customer ID for traceability, but never use it as a model feature.
    for c in cfg.id_cols:
        if c in df.columns:
            out["CustomerID"] = df[c].values
            break
    else:
        out["CustomerID"] = np.arange(len(df))

    return out
