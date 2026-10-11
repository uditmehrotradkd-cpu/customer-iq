"""Data ingestion: read raw or one-hot-encoded customer files into one canonical schema."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATE_COLUMN,
    DATE_FORMAT,
    ID_COLUMN,
    ONE_HOT_BASELINES,
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
)

# Currency symbols/codes, thousands separators and spaces found in exported numbers ("$58,138", "58138 USD").
NUMBER_JUNK = r"(?i)usd|inr|eur|gbp|rs\.?|[,$€£₹\s]"
FLAG_WORDS = {"yes": 1.0, "y": 1.0, "true": 1.0, "t": 1.0, "no": 0.0, "n": 0.0, "false": 0.0, "f": 0.0}


def to_number(series: pd.Series) -> pd.Series:
    """Numeric values from numbers stored as text; unreadable entries ("N/A", "unknown") become NaN."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series
    text = series.astype("string").str.replace(NUMBER_JUNK, "", regex=True)
    # Plain float64 (NaN, not pd.NA) so comparisons used as row filters stay boolean.
    return pd.to_numeric(text, errors="coerce").astype(float)


def to_flag(series: pd.Series) -> pd.Series:
    """0/1 from 0/1, yes/no, true/false or y/n; anything else (e.g. 2 or -1) becomes NaN."""
    numbers = to_number(series)
    valid = numbers.isin([0, 1])
    if pd.api.types.is_numeric_dtype(series) and (valid | numbers.isna()).all():
        return series
    words = series.astype("string").str.strip().str.lower().map(FLAG_WORDS).astype(float)
    return numbers.where(valid).fillna(words)


def parse_dates(series: pd.Series) -> pd.Series:
    """Dates in the dataset's DD-MM-YYYY format, ISO, or any other common layout (day first)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format=DATE_FORMAT, errors="coerce")
    rest = parsed.isna() & text.notna()
    if rest.any():
        parsed[rest] = pd.to_datetime(text[rest], format="ISO8601", errors="coerce")
    rest = parsed.isna() & text.notna()
    if rest.any():
        parsed[rest] = pd.to_datetime(text[rest], format="mixed", dayfirst=True, errors="coerce")
    return parsed


def _detect_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def decode_one_hot(df: pd.DataFrame, prefix: str, baseline: str) -> pd.DataFrame:
    """Collapse ``prefix_<level>`` dummy columns back into a single categorical column.

    Rows with no active dummy receive the dropped ``baseline`` level; rows with more than
    one active dummy are contradictory and become missing so the cleaner can handle them.
    """
    dummy_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not dummy_cols or prefix in df.columns:
        return df

    block = df[dummy_cols].to_numpy()
    levels = np.array([c[len(prefix) + 1 :] for c in dummy_cols], dtype=object)
    active = block.sum(axis=1)
    decoded = np.where(active == 0, baseline, levels[block.argmax(axis=1)])
    decoded = np.where(active > 1, None, decoded)

    out = df.drop(columns=dummy_cols)
    out[prefix] = decoded
    return out


def to_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with raw categorical columns and a parsed ``Dt_Customer`` date."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for prefix, baseline in ONE_HOT_BASELINES.items():
        out = decode_one_hot(out, prefix, baseline)

    if DATE_COLUMN in out.columns:
        out[DATE_COLUMN] = parse_dates(out[DATE_COLUMN])

    missing = [c for c in (*RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS) if c not in out.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    # The enrolment date only describes tenure (never used to form segments), so it is optional.
    if DATE_COLUMN not in out.columns:
        out[DATE_COLUMN] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    if ID_COLUMN not in out.columns:
        out.insert(0, ID_COLUMN, np.arange(1, len(out) + 1))
    return out


def load_customers(path: str | Path) -> pd.DataFrame:
    """Load a customer CSV (comma or tab separated, raw or one-hot encoded)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Customer file not found: {path}")
    raw = pd.read_csv(path, sep=_detect_separator(path))
    return to_canonical_schema(raw)
