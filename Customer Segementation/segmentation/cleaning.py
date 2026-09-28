"""Data-quality rules: duplicates, inconsistent records, missing values and category harmonisation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    CHANNEL_PURCHASE_COLUMNS,
    CONSTANT_COLUMNS,
    DATE_COLUMN,
    EDUCATION_MAP,
    ID_COLUMN,
    MARITAL_STATUS_MAP,
    RAW_NUMERIC_COLUMNS,
    SPEND_COLUMNS,
    SegmentationConfig,
)


@dataclass
class CleaningReport:
    """Audit trail of every cleaning action, suitable for tables and slides."""

    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, rows_before: int, rows_after: int, detail: str = "") -> None:
        self.steps.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "detail": detail,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def audit_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level quality summary: type, missing, unique values and skewness."""
    numeric = df.select_dtypes(include="number")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )
    summary["skewness"] = numeric.skew().round(2)
    return summary.sort_values(["missing", "skewness"], ascending=False)


def clean_customers(df: pd.DataFrame, config: SegmentationConfig | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the full cleaning policy and return the cleaned frame plus an audit report."""
    config = config or SegmentationConfig()
    report = CleaningReport()
    out = df.copy()

    n = len(out)
    constant = [c for c in CONSTANT_COLUMNS if c in out.columns]
    schema = {*RAW_NUMERIC_COLUMNS, "Education", "Marital_Status", DATE_COLUMN}
    constant += [c for c in out.columns if c not in constant and c not in schema and out[c].nunique(dropna=False) <= 1]
    out = out.drop(columns=constant)
    report.log("Drop constant columns", n, len(out), f"Removed zero-information columns: {constant}")

    n = len(out)
    out = out.drop_duplicates()
    report.log("Drop exact duplicate rows", n, len(out))

    if ID_COLUMN in out.columns:
        n = len(out)
        out = out.drop_duplicates(subset=[ID_COLUMN], keep="first")
        report.log("Drop duplicate customer IDs", n, len(out))

    for col in RAW_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0, col] = np.nan

    # Harmonise before de-duplicating so records differing only by label spelling are caught.
    out["Marital_Status"] = out["Marital_Status"].map(MARITAL_STATUS_MAP).fillna("Unknown")
    out["Education_Level"] = out["Education"].map(EDUCATION_MAP).fillna("Unknown")
    report.log(
        "Harmonise categories",
        len(out),
        len(out),
        "Alone->Single; Absurd/YOLO->Unknown; Education grouped into Basic/Graduate/Postgraduate",
    )

    n = len(out)
    attribute_cols = [c for c in out.columns if c != ID_COLUMN]
    out = out.drop_duplicates(subset=attribute_cols, keep="first")
    report.log(
        "Drop duplicate profiles with different IDs",
        n,
        len(out),
        "Identical attributes under a new ID usually indicate a re-registered account.",
    )

    n = len(out)
    implausible_age = out["Year_Birth"] < config.min_birth_year
    out = out.loc[~implausible_age]
    report.log("Remove implausible birth years", n, len(out), f"Year_Birth < {config.min_birth_year}")

    n = len(out)
    extreme_income = out["Income"] > config.max_income
    out = out.loc[~extreme_income]
    report.log("Remove data-entry income outliers", n, len(out), f"Income > {config.max_income:,.0f}")

    n = len(out)
    total_spend = out[list(SPEND_COLUMNS)].sum(axis=1)
    total_purchases = out[list(CHANNEL_PURCHASE_COLUMNS)].sum(axis=1)
    spend_without_purchase = (total_spend > 0) & (total_purchases == 0)
    out = out.loc[~spend_without_purchase]
    report.log("Remove spend recorded with zero purchases", n, len(out), "Logically inconsistent transactions")

    n_capped = int((out["NumDealsPurchases"] > total_purchases.loc[out.index]).sum())
    out["NumDealsPurchases"] = np.minimum(out["NumDealsPurchases"], total_purchases.loc[out.index])
    report.log("Cap deal purchases at total purchases", len(out), len(out), f"{n_capped} records capped")

    n_income_missing = int(out["Income"].isna().sum())
    group_median = out.groupby("Education_Level")["Income"].transform("median")
    out["Income"] = out["Income"].fillna(group_median).fillna(out["Income"].median())
    n_date_missing = int(out[DATE_COLUMN].isna().sum())
    if out[DATE_COLUMN].notna().any():
        out[DATE_COLUMN] = out[DATE_COLUMN].fillna(out[DATE_COLUMN].median())
    other_numeric = [c for c in RAW_NUMERIC_COLUMNS if c != "Income"]
    n_other_missing = int(out[other_numeric].isna().sum().sum())
    out[other_numeric] = out[other_numeric].fillna(out[other_numeric].median())
    report.log(
        "Impute missing values",
        len(out),
        len(out),
        f"Income by education median: {n_income_missing}; enrolment date median: {n_date_missing}; "
        f"other numeric median: {n_other_missing}",
    )

    return out.reset_index(drop=True), report
