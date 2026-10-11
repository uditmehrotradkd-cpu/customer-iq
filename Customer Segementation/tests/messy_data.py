"""Generator of realistic, uncleaned customer exports used to test the cleaning rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.config import CAMPAIGN_COLUMNS, SPEND_COLUMNS

HEADER_VARIANTS = {"ID": "Customer ID", "Income": "Annual Income", "Marital_Status": "Marital Status", "Dt_Customer": "Customer Since", "Recency": " Recency "}
EDUCATION_VARIANTS = {
    "Graduation": [" graduation", "GRADUATION", "Graduate", "Bachelor", "Bachelors"],
    "PhD": ["phd", "PHD", "Ph.D.", "Doctorate"],
    "Master": ["master", "Masters", "Master's", "MSc", "MBA"],
    "2n Cycle": ["2nd Cycle", "2n cycle", "Second Cycle"],
    "Basic": ["basic", "High School", "School"],
}
MARITAL_VARIANTS = {
    "Married": ["married", "MARRIED ", " Married"],
    "Together": ["together", "Partner", "Living Together"],
    "Single": ["single", "SINGLE", "Alone", "Unmarried"],
    "Divorced": ["divorced", "Divorcee", "Separated"],
    "Widow": ["widow", "Widowed", "Widower"],
}


def _pick(rng: np.random.Generator, n: int, frac: float) -> np.ndarray:
    return rng.choice(n, size=max(1, int(n * frac)), replace=False)


def _scatter(df: pd.DataFrame, rng: np.random.Generator, col: str, frac: float, choices) -> None:
    rows = _pick(rng, len(df), frac)
    df.loc[rows, col] = rng.choice(np.asarray(choices, dtype=object), len(rows))


def make_messy_customers(clean: pd.DataFrame, n_total: int = 100_000, seed: int = 7) -> pd.DataFrame:
    """Return ``n_total`` rows with duplicates, blanks, typos, mixed formats, text numbers and outliers."""
    rng = np.random.default_rng(seed)
    n_exact, n_id_clash, n_reregistered, n_blank = (int(n_total * f) for f in (0.03, 0.01, 0.01, 0.002))
    n_base = n_total - n_exact - n_id_clash - n_reregistered - n_blank
    base = clean.sample(n=n_base, random_state=seed).reset_index(drop=True)
    base["ID"] = np.arange(100_001, 100_001 + n_base)
    df = base.astype(object)
    n = len(df)
    df["Notes"] = rng.choice(["", "VIP", "call back", "moved city", "prefers email", None], n)

    # Text formatting problems in numeric columns.
    rows = _pick(rng, n, 0.03)
    styles = rng.integers(0, 3, len(rows))
    income = base.loc[rows, "Income"].round().astype(int).to_numpy()
    df.loc[rows, "Income"] = np.where(styles == 0, [f"${v:,}" for v in income], np.where(styles == 1, [f"{v:,}.00" for v in income], [f"{v} USD" for v in income]))
    df.loc[_pick(rng, n, 0.02), "Income"] = None
    _scatter(df, rng, "Income", 0.005, ["N/A", "unknown", "-", "?", "not disclosed"])
    _scatter(df, rng, "Income", 0.002, [666_666, 9_999_999, 1_250_000])
    for col in SPEND_COLUMNS:
        _scatter(df, rng, col, 0.004, -np.arange(1, 500))
        df.loc[_pick(rng, n, 0.004), col] = None
        rows = _pick(rng, n, 0.003)
        df.loc[rows, col] = [f" {v} " for v in base.loc[rows, col]]
    _scatter(df, rng, "Recency", 0.003, -np.arange(1, 30))
    df.loc[_pick(rng, n, 0.003), "NumWebVisitsMonth"] = None
    df.loc[_pick(rng, n, 0.003), "Kidhome"] = None

    # Implausible values.
    _scatter(df, rng, "Year_Birth", 0.002, [1893, 1899, 1900, 1905])
    _scatter(df, rng, "Year_Birth", 0.002, [2012, 2015, 2025, 2031])
    rows = _pick(rng, n, 0.003)
    df.loc[rows, ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]] = 0
    rows = _pick(rng, n, 0.005)
    df.loc[rows, "NumDealsPurchases"] = 40

    # Yes/No style and out-of-range flags.
    for col in (*CAMPAIGN_COLUMNS, "Complain"):
        rows = _pick(rng, n, 0.02)
        df.loc[rows, col] = [("Yes" if v else "No") if i % 3 == 0 else (("TRUE" if v else "FALSE") if i % 3 == 1 else ("y" if v else "n")) for i, v in enumerate(base.loc[rows, col])]
        _scatter(df, rng, col, 0.002, [2, 5, -1])

    # Inconsistent category spellings and blanks.
    for col, variants in (("Education", EDUCATION_VARIANTS), ("Marital_Status", MARITAL_VARIANTS)):
        rows = _pick(rng, n, 0.05)
        df.loc[rows, col] = [rng.choice(variants.get(v, [v])) for v in base.loc[rows, col]]
        df.loc[_pick(rng, n, 0.01), col] = None
    _scatter(df, rng, "Marital_Status", 0.002, ["YOLO", "Absurd", "???"])

    # Mixed date formats, blanks, garbage and future dates.
    dates = pd.to_datetime(base["Dt_Customer"], format="mixed", dayfirst=True)
    rows = _pick(rng, n, 0.25)
    fmts = np.array(["%Y-%m-%d", "%d/%m/%Y", "%d %b %Y", "%Y/%m/%d", "%B %d, %Y"])
    df.loc[rows, "Dt_Customer"] = [d.strftime(f) for d, f in zip(dates[rows], rng.choice(fmts, len(rows)))]
    df.loc[_pick(rng, n, 0.01), "Dt_Customer"] = None
    _scatter(df, rng, "Dt_Customer", 0.002, ["not provided", "00-00-0000", "31-02-2014"])
    df.loc[_pick(rng, n, 0.001), "Dt_Customer"] = "15-06-2031"

    # Duplicates: exact copies, re-used IDs with other data, and re-registered profiles under new IDs.
    exact = df.iloc[rng.choice(n, n_exact, replace=False)]
    clash = df.iloc[rng.choice(n, n_id_clash, replace=False)].copy()
    clash["Income"] = rng.integers(20_000, 90_000, n_id_clash)
    rereg = df.iloc[rng.choice(n, n_reregistered, replace=False)].copy()
    rereg["ID"] = np.arange(300_001, 300_001 + n_reregistered)
    blanks = pd.DataFrame(None, index=range(n_blank), columns=df.columns)
    out = pd.concat([df, exact, clash, rereg, blanks], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    out["Z_CostContact"] = 3
    out["Z_Revenue"] = 11
    return out.rename(columns=HEADER_VARIANTS)
