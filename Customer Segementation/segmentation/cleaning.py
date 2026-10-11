"""Data-quality rules: duplicates, inconsistent records, missing values and category harmonisation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    CAMPAIGN_COLUMNS,
    CHANNEL_PURCHASE_COLUMNS,
    CONSTANT_COLUMNS,
    DATE_COLUMN,
    EDUCATION_MAP,
    ID_COLUMN,
    MARITAL_STATUS_MAP,
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    SPEND_COLUMNS,
    SegmentationConfig,
)
from .data_loader import parse_dates, to_flag, to_number

FLAG_COLUMNS = (*CAMPAIGN_COLUMNS, "Complain")
# Common spellings in CRM exports, mapped to the dataset's own labels before grouping.
EDUCATION_SYNONYMS = {
    "graduate": "Graduation", "bachelor": "Graduation", "bachelors": "Graduation", "undergraduate": "Graduation", "bsc": "Graduation", "ba": "Graduation",
    "doctorate": "PhD", "doctoral": "PhD",
    "masters": "Master", "msc": "Master", "mba": "Master", "ma": "Master", "postgraduate": "Master",
    "2ndcycle": "2n Cycle", "secondcycle": "2n Cycle",
    "highschool": "Basic", "school": "Basic", "secondary": "Basic",
}
MARITAL_SYNONYMS = {
    "partner": "Together", "livingtogether": "Together", "cohabiting": "Together",
    "unmarried": "Single", "nevermarried": "Single",
    "divorcee": "Divorced", "separated": "Divorced",
    "widowed": "Widow", "widower": "Widow",
}


def _key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _canonical_labels(series: pd.Series, labels, synonyms: dict[str, str]) -> pd.Series:
    """Map case, spacing, punctuation and synonym variants (" phd.", "Masters") to the dataset's labels."""
    lookup = {**synonyms, **{_key(label): label for label in labels}}
    return series.map(lambda v: lookup.get(_key(v)) if pd.notna(v) else None)


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

    # Standardise values first so duplicates that differ only in formatting are still caught.
    converted = cleared = 0
    for col in RAW_NUMERIC_COLUMNS:
        raw = out[col]
        values = to_flag(raw) if col in FLAG_COLUMNS else to_number(raw)
        if not pd.api.types.is_numeric_dtype(raw):
            converted += int((values.notna() & pd.to_numeric(raw, errors="coerce").isna()).sum())
        cleared += int((raw.notna() & values.isna()).sum())
        out[col] = values
    negative = out[list(RAW_NUMERIC_COLUMNS)] < 0
    cleared += int(negative.sum().sum())
    out[list(RAW_NUMERIC_COLUMNS)] = out[list(RAW_NUMERIC_COLUMNS)].mask(negative)
    out[DATE_COLUMN] = parse_dates(out[DATE_COLUMN])
    report.log(
        "Standardise numbers and yes/no flags",
        len(out),
        len(out),
        f"{converted} text values converted (e.g. '$58,138', '58138 USD', 'Yes'); {cleared} unreadable, negative or out-of-range values cleared for imputation",
    )

    n = len(out)
    required = [*RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS]
    out = out.loc[out[required].isna().mean(axis=1) <= config.max_missing_share]
    report.log("Drop blank or mostly empty rows", n, len(out), f"More than {config.max_missing_share:.0%} of required fields missing")

    n = len(out)
    out = out.drop_duplicates()
    report.log("Drop exact duplicate rows", n, len(out))

    if ID_COLUMN in out.columns:
        n = len(out)
        ids = out[ID_COLUMN]
        out = out.loc[ids.isna() | ~ids.duplicated(keep="first")]
        missing_id = out[ID_COLUMN].isna()
        if missing_id.any():
            numeric_ids = to_number(out[ID_COLUMN])
            start = int(numeric_ids.max()) + 1 if numeric_ids.notna().any() else 1
            out.loc[missing_id, ID_COLUMN] = np.arange(start, start + int(missing_id.sum()))
        report.log("Drop duplicate customer IDs", n, len(out), f"{int(missing_id.sum())} missing IDs assigned" if missing_id.any() else "")

    # Harmonise before de-duplicating so records differing only by label spelling are caught.
    education = _canonical_labels(out["Education"], EDUCATION_MAP, EDUCATION_SYNONYMS)
    marital = _canonical_labels(out["Marital_Status"], MARITAL_STATUS_MAP, MARITAL_SYNONYMS)
    respelled = int((education.notna() & education.ne(out["Education"])).sum() + (marital.notna() & marital.ne(out["Marital_Status"])).sum())
    out["Education"] = education.fillna(out["Education"]).fillna("Unknown")
    out["Marital_Status"] = marital.map(MARITAL_STATUS_MAP).fillna("Unknown")
    out["Education_Level"] = education.map(EDUCATION_MAP).fillna("Unknown")
    report.log(
        "Harmonise categories",
        len(out),
        len(out),
        f"{respelled} spelling variants standardised (e.g. ' phd.', 'Masters', 'married '); Alone->Single; Absurd/YOLO->Unknown; "
        "Education grouped into Basic/Graduate/Postgraduate",
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

    future = out[DATE_COLUMN] > pd.Timestamp.now()
    out.loc[future, DATE_COLUMN] = pd.NaT
    report.log("Clear future enrolment dates", len(out), len(out), f"{int(future.sum())} dates after today cleared for imputation")

    n = len(out)
    latest_year = out[DATE_COLUMN].max().year if out[DATE_COLUMN].notna().any() else pd.Timestamp.now().year
    max_birth_year = latest_year - config.min_customer_age
    implausible_age = (out["Year_Birth"] < config.min_birth_year) | (out["Year_Birth"] > max_birth_year)
    out = out.loc[~implausible_age]
    report.log(
        "Remove implausible birth years",
        n,
        len(out),
        f"Year_Birth < {config.min_birth_year} or > {max_birth_year} (younger than {config.min_customer_age} in {latest_year})",
    )

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
