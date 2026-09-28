"""Dataset builders for the data agent: filtered extracts, synthetic customers, summaries and templates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from segmentation.config import (
    CAMPAIGN_COLUMNS,
    CHANNEL_PURCHASE_COLUMNS,
    DATE_COLUMN,
    ID_COLUMN,
    RAW_NUMERIC_COLUMNS,
    SPEND_COLUMNS,
)

from .services import SegmentationService, sanitize_csv

RAW_EXPORT_COLUMNS = [ID_COLUMN, *RAW_NUMERIC_COLUMNS, "Education", "Marital_Status", DATE_COLUMN]
# Ratios are stored 0-1; users usually type percentages ("deal ratio > 30").
RATIO_FEATURES = {"Deal_Ratio", "Web_Share", "Catalog_Share", "Store_Share", "Spend_To_Income"}
SUMMARY_TABLES = ("profiles", "index", "zscores", "channel_mix", "category_mix", "k_selection", "cleaning", "recommendations")
OPERATORS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}
# Continuous raw columns modelled jointly (log space for right-skewed amounts and counts).
SYNTH_LOG_COLUMNS = ["Income", *SPEND_COLUMNS, "NumDealsPurchases", *CHANNEL_PURCHASE_COLUMNS, "NumWebVisitsMonth"]
SYNTH_LINEAR_COLUMNS = ["Year_Birth", "Recency", "Kidhome", "Teenhome"]
SYNTH_BINARY_COLUMNS = [*CAMPAIGN_COLUMNS, "Complain"]


class DatasetError(ValueError):
    """Raised for invalid dataset requests; the message is safe to show to users."""


def _segment_filter(service: SegmentationService, segments: list[int] | None) -> list[int]:
    if not segments:
        return service.segment_ids
    unknown = [s for s in segments if s not in service.names]
    if unknown:
        raise DatasetError(f"Unknown segment id(s): {unknown}")
    return sorted(set(segments))


def filtered_customers(service: SegmentationService, segments: list[int] | None, conditions: list[dict]) -> pd.DataFrame:
    df = service.customers
    df = df[df["Segment"].isin(_segment_filter(service, segments))]
    for cond in conditions:
        feature, op, value = cond["feature"], cond["op"], cond["value"]
        if feature not in df.columns:
            raise DatasetError(f"Unknown field '{feature}'.")
        if op not in OPERATORS:
            raise DatasetError(f"Unsupported operator '{op}'.")
        column = df[feature]
        if pd.api.types.is_numeric_dtype(column):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise DatasetError(f"'{feature}' needs a numeric value.") from None
            if feature in RATIO_FEATURES and number > 1:
                number /= 100
            mask = OPERATORS[op](column.to_numpy(dtype=float), number)
        else:
            if op not in ("==", "!="):
                raise DatasetError(f"'{feature}' only supports equals / not equals.")
            mask = column.astype(str).str.lower().eq(str(value).lower()).to_numpy()
            if op == "!=":
                mask = ~mask
        df = df[mask]
    ordered = ["Segment", "Segment_Name", *RAW_EXPORT_COLUMNS]
    engineered = [c for c in [*service.numeric_features, *service.categorical_features] if c not in ordered]
    out = df[[c for c in ordered if c in df.columns] + engineered].copy()
    if DATE_COLUMN in out.columns:
        out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def synthetic_customers(service: SegmentationService, n: int, segments: list[int] | None, seed: int = 42) -> tuple[pd.DataFrame, float]:
    """Sample new customers from each segment's joint distribution (no real rows are copied).

    Continuous fields come from a multivariate normal fitted per segment (log space for skewed
    amounts), binary flags and categories from per-segment frequencies. Returns the dataset and
    the share of rows the model assigns back to their intended segment (a fidelity check).
    """
    rng = np.random.default_rng(seed)
    customers = service.customers
    chosen = _segment_filter(service, segments)
    shares = customers["Segment"].value_counts(normalize=True).reindex(chosen).fillna(0).to_numpy()
    shares = shares / shares.sum() if shares.sum() else np.full(len(chosen), 1 / len(chosen))
    counts = rng.multinomial(n, shares)

    frames = []
    for segment, count in zip(chosen, counts):
        if count == 0:
            continue
        seg = customers[customers["Segment"] == segment]
        continuous = pd.concat([np.log1p(seg[SYNTH_LOG_COLUMNS].clip(lower=0)), seg[SYNTH_LINEAR_COLUMNS]], axis=1).astype(float)
        mean = continuous.mean().to_numpy()
        cov = np.cov(continuous.to_numpy(), rowvar=False) + np.eye(len(mean)) * 1e-6
        draws = pd.DataFrame(rng.multivariate_normal(mean, cov, size=count), columns=continuous.columns)
        for col in SYNTH_LOG_COLUMNS:
            draws[col] = np.expm1(draws[col])
        lo, hi = seg[continuous.columns].min(), seg[continuous.columns].max()
        draws = draws.clip(lower=lo, upper=hi, axis=1)
        integer_cols = [c for c in draws.columns if c != "Income"]
        draws[integer_cols] = draws[integer_cols].round().astype(int)
        draws["Income"] = draws["Income"].round(-2)

        for col in SYNTH_BINARY_COLUMNS:
            draws[col] = (rng.random(count) < seg[col].mean()).astype(int)
        for col in ("Education", "Marital_Status"):
            freq = seg[col].value_counts(normalize=True)
            draws[col] = rng.choice(freq.index.to_numpy(), size=count, p=freq.to_numpy())
        dates = pd.to_datetime(seg[DATE_COLUMN]).dropna()
        if dates.empty:
            draws[DATE_COLUMN] = None
        else:
            span = max((dates.max() - dates.min()).days, 1)
            draws[DATE_COLUMN] = (dates.min() + pd.to_timedelta(rng.integers(0, span + 1, count), unit="D")).strftime("%Y-%m-%d")
        purchases = draws[list(CHANNEL_PURCHASE_COLUMNS)].sum(axis=1)
        draws["NumDealsPurchases"] = np.minimum(draws["NumDealsPurchases"], purchases)
        draws["Intended_Segment"] = service.names[segment]
        frames.append(draws)

    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    data.insert(0, ID_COLUMN, np.arange(900_000, 900_000 + len(data)))
    scored = service.segmenter.predict(data.assign(**{DATE_COLUMN: pd.to_datetime(data[DATE_COLUMN])}))
    data["Assigned_Segment"] = scored["Segment_Name"].to_numpy()
    fidelity = float((data["Assigned_Segment"] == data["Intended_Segment"]).mean())
    return data[[ID_COLUMN, *[c for c in RAW_EXPORT_COLUMNS if c != ID_COLUMN], "Intended_Segment", "Assigned_Segment"]], fidelity


def summary_table(service: SegmentationService, table: str) -> pd.DataFrame:
    s = service.segmenter
    names = service.names
    if table == "profiles":
        return s.profiles_.reset_index()
    if table == "index":
        return s.index_table_.rename(index=names).reset_index(names="Segment")
    if table == "zscores":
        return s.zprofile_.round(3).rename(index=names).reset_index(names="Segment")
    if table == "channel_mix":
        return s.channel_mix_.rename(index=names).reset_index(names="Segment")
    if table == "category_mix":
        return s.category_mix_.rename(index=names).reset_index(names="Segment")
    if table == "k_selection":
        return s.k_selection_.metrics.round(4)
    if table == "cleaning":
        return s.cleaning_report_.to_frame()
    if table == "recommendations":
        return pd.DataFrame(
            [
                {
                    "Segment": r["name"],
                    "Goal": r["goal"],
                    "Summary": r["summary"],
                    "Preferred_Channel": r["preferred_channel"],
                    "Actions": " | ".join(r["actions"]),
                    "KPIs": ", ".join(r["kpis"]),
                    "Guardrail": r["responsible_use"],
                }
                for r in s.recommendations_
            ]
        )
    raise DatasetError(f"Unknown table '{table}'. Choose one of: {', '.join(SUMMARY_TABLES)}.")


def upload_template(service: SegmentationService) -> pd.DataFrame:
    """Two illustrative rows in the exact raw schema accepted by training and batch scoring."""
    rows = service.customers.groupby("Segment").median(numeric_only=True).head(2)
    template = pd.DataFrame({c: rows[c].round().astype(int).to_numpy() for c in RAW_NUMERIC_COLUMNS if c in rows})
    template["Income"] = rows["Income"].round(-2).to_numpy()
    template.insert(0, ID_COLUMN, [1, 2])
    template["Education"] = ["Graduation", "Master"]
    template["Marital_Status"] = ["Married", "Single"]
    template[DATE_COLUMN] = ["2013-05-01", "2012-11-15"]
    return template[RAW_EXPORT_COLUMNS]


def build_dataset(service: SegmentationService, request: dict) -> tuple[pd.DataFrame, str, dict]:
    """Dispatch a dataset request; returns (frame, filename, extra info)."""
    kind = request["kind"]
    if service.mode == "generic" and kind in ("synthetic", "template") or service.mode == "generic" and kind == "summary" and request.get("table", "profiles") != "profiles":
        raise DatasetError(
            "That is only available for the customer model. Your workspace uses your own dataset: ask me for the segment report, "
            "the profiles table or the rows of a segment instead, or say \"reset to original\"."
        )
    if kind == "filtered":
        frame = filtered_customers(service, request.get("segments"), request.get("conditions", []))
        return sanitize_csv(frame), "customers_filtered.csv", {"rows": len(frame)}
    if kind == "synthetic":
        frame, fidelity = synthetic_customers(service, request.get("n", 500), request.get("segments"), request.get("seed", 42))
        return sanitize_csv(frame), f"synthetic_customers_{len(frame)}.csv", {"rows": len(frame), "fidelity": round(fidelity, 3)}
    if kind == "summary":
        table = request.get("table", "profiles")
        frame = summary_table(service, table)
        return sanitize_csv(frame), f"segment_{table}.csv", {"rows": len(frame)}
    if kind == "template":
        frame = upload_template(service)
        return frame, "customer_upload_template.csv", {"rows": len(frame)}
    raise DatasetError(f"Unknown dataset kind '{kind}'.")
