"""Read user-supplied tables (CSV/TSV, Excel, JSON) and map loosely named columns to the training schema."""
from __future__ import annotations

import io
import json
import re
from pathlib import PurePath

import pandas as pd

from segmentation.config import CAMPAIGN_COLUMNS, DATE_COLUMN, ID_COLUMN, RAW_CATEGORICAL_COLUMNS, RAW_NUMERIC_COLUMNS

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".jsonl", ".ndjson")
REQUIRED_COLUMNS = (*RAW_NUMERIC_COLUMNS, *RAW_CATEGORICAL_COLUMNS)
OPTIONAL_COLUMNS = (DATE_COLUMN,)


class FileFormatError(ValueError):
    """Unreadable or unsupported file; message is safe to show to users."""


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Canonical columns plus common alternative spellings found in exported CRM / e-commerce data.
_ALIASES: dict[str, str] = {_key(c): c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, ID_COLUMN)}
_ALIASES.update(
    {
        "customerid": ID_COLUMN,
        "custid": ID_COLUMN,
        "id": ID_COLUMN,
        "birthyear": "Year_Birth",
        "yearofbirth": "Year_Birth",
        "yob": "Year_Birth",
        "annualincome": "Income",
        "householdincome": "Income",
        "kids": "Kidhome",
        "kidsathome": "Kidhome",
        "teens": "Teenhome",
        "teenagers": "Teenhome",
        "dayssincelastpurchase": "Recency",
        "wine": "MntWines",
        "wines": "MntWines",
        "fruits": "MntFruits",
        "meat": "MntMeatProducts",
        "meatproducts": "MntMeatProducts",
        "fish": "MntFishProducts",
        "fishproducts": "MntFishProducts",
        "sweets": "MntSweetProducts",
        "sweetproducts": "MntSweetProducts",
        "gold": "MntGoldProds",
        "goldproducts": "MntGoldProds",
        "dealspurchases": "NumDealsPurchases",
        "dealpurchases": "NumDealsPurchases",
        "webpurchases": "NumWebPurchases",
        "catalogpurchases": "NumCatalogPurchases",
        "catalogue purchases": "NumCatalogPurchases",
        "storepurchases": "NumStorePurchases",
        "webvisits": "NumWebVisitsMonth",
        "webvisitsmonth": "NumWebVisitsMonth",
        "complaint": "Complain",
        "complained": "Complain",
        "maritalstatus": "Marital_Status",
        "marital": "Marital_Status",
        "educationlevel": "Education",
        "dtcustomer": DATE_COLUMN,
        "customersince": DATE_COLUMN,
        "enrolmentdate": DATE_COLUMN,
        "enrollmentdate": DATE_COLUMN,
        "signupdate": DATE_COLUMN,
        "joindate": DATE_COLUMN,
        "registrationdate": DATE_COLUMN,
        "onboardingdate": DATE_COLUMN,
        "accountopendate": DATE_COLUMN,
        "accountopeningdate": DATE_COLUMN,
    }
)
_ALIASES = {_key(k): v for k, v in _ALIASES.items()}


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename recognisable columns to the canonical schema; returns the frame and the renames applied."""
    renames: dict[str, str] = {}
    taken = set(frame.columns)
    for col in frame.columns:
        target = _ALIASES.get(_key(col))
        if target and target != col and target not in taken and target not in renames.values():
            renames[col] = target
    out = frame.rename(columns=renames)
    for col in CAMPAIGN_COLUMNS:
        if col in out.columns and out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip().str.lower().map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
    return out, renames


def _json_to_frame(content: bytes) -> pd.DataFrame:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FileFormatError(f"Invalid JSON: {exc}") from None
    if isinstance(payload, dict):
        for key in ("data", "customers", "records", "rows", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if payload and all(isinstance(v, (list, dict)) for v in payload.values()):
                return pd.DataFrame(payload)
            payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
        raise FileFormatError("JSON must be a list of records (objects) or an object with a 'data' list.")
    return pd.json_normalize(payload, sep="_")


def read_table(content: bytes, filename: str, max_rows: int) -> pd.DataFrame:
    """Parse an uploaded file into a DataFrame (first sheet for Excel)."""
    ext = PurePath(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileFormatError(f"Unsupported file type '{ext or 'unknown'}'. Use CSV, Excel (.xlsx/.xls) or JSON.")
    if not content:
        raise FileFormatError("The file is empty.")
    buffer = io.BytesIO(content)
    try:
        if ext in (".csv", ".tsv", ".txt"):
            frame = pd.read_csv(buffer, sep=None, engine="python", nrows=max_rows + 1, encoding_errors="replace")
        elif ext in (".xlsx", ".xlsm", ".xls"):
            frame = pd.read_excel(buffer, sheet_name=0, nrows=max_rows + 1)
        elif ext in (".jsonl", ".ndjson"):
            frame = pd.read_json(buffer, lines=True, nrows=max_rows + 1)
        else:
            frame = _json_to_frame(content)
    except FileFormatError:
        raise
    except Exception as exc:  # parsers raise many exception types for malformed input
        raise FileFormatError(f"Could not read the {ext} file: {exc}") from None
    if len(frame) > max_rows:
        raise FileFormatError(f"The file has more than {max_rows:,} rows.")
    if frame.empty:
        raise FileFormatError("The file contains no rows.")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def analyze_table(frame: pd.DataFrame, renames: dict[str, str]) -> dict:
    """Data-quality summary used by the agent's 'analyze' action."""
    one_hot_prefixes = ("Education_", "Marital_Status_", f"{DATE_COLUMN}_")
    present = set(frame.columns)
    encoded = {p.rstrip("_") for p in one_hot_prefixes if any(c.startswith(p) for c in frame.columns)}
    missing = [c for c in REQUIRED_COLUMNS if c not in present and c not in encoded]
    optional_missing = [c for c in OPTIONAL_COLUMNS if c not in present and c not in encoded]
    missing_values = frame[[c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in present]].isna().sum()
    numeric = frame[[c for c in ("Income", "MntWines", "MntMeatProducts", "Recency") if c in present]].apply(pd.to_numeric, errors="coerce")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "renamed": renames,
        "missing_columns": missing,
        "optional_missing": optional_missing,
        "missing_values": {k: int(v) for k, v in missing_values[missing_values > 0].items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_summary": numeric.describe().T[["mean", "min", "max"]].round(1).reset_index(names="column").to_dict("records") if not numeric.empty else [],
        "compatible": not missing,
    }


def table_profile(frame: pd.DataFrame, max_columns: int = 40) -> list[dict]:
    """Compact per-column profile of any table (type, missing, distinct values, range or top values)."""
    out = []
    for col in list(frame.columns)[:max_columns]:
        series = frame[col]
        info: dict = {"column": str(col)[:60], "missing": int(series.isna().sum()), "distinct": int(series.nunique(dropna=True))}
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(1, 0.9 * series.notna().sum()):
            info.update(type="number", mean=round(float(numeric.mean()), 3), min=round(float(numeric.min()), 3), max=round(float(numeric.max()), 3))
        else:
            top = series.astype(str).str.slice(0, 40).value_counts().head(4)
            info.update(type="text", top=", ".join(f"{k} ({v})" for k, v in top.items()))
        out.append(info)
    return out
