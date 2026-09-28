"""Unit, integration and dashboard smoke tests. Run from the project folder: pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation import CustomerSegmenter, SegmentationConfig, load_customers  # noqa: E402
from segmentation.cleaning import clean_customers  # noqa: E402
from segmentation.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH, MODEL_FILENAME  # noqa: E402
from segmentation.data_loader import decode_one_hot  # noqa: E402
from segmentation.features import engineer_features  # noqa: E402
from segmentation.model_selection import find_elbow  # noqa: E402
from segmentation.preprocessing import SkewCorrector, Winsorizer  # noqa: E402

FAST_CONFIG = SegmentationConfig(k_max=5, n_init=3, stability_runs=2)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_customers(DEFAULT_DATA_PATH)


@pytest.fixture(scope="module")
def fitted(raw) -> CustomerSegmenter:
    return CustomerSegmenter(FAST_CONFIG).fit(raw)


def test_decode_one_hot_uses_baseline_and_flags_conflicts():
    df = pd.DataFrame({"Edu_A": [1, 0, 1], "Edu_B": [0, 0, 1]})
    decoded = decode_one_hot(df, "Edu", "Base")
    values = decoded["Edu"].tolist()
    assert values[:2] == ["A", "Base"] and pd.isna(values[2])


def test_loader_returns_canonical_schema(raw):
    assert {"Education", "Marital_Status", "Dt_Customer"} <= set(raw.columns)
    assert not any(c.startswith("Dt_Customer_") for c in raw.columns)
    assert pd.api.types.is_datetime64_any_dtype(raw["Dt_Customer"])


def test_cleaning_removes_duplicates_and_outliers(raw):
    clean, report = clean_customers(raw)
    attrs = [c for c in clean.columns if c != "ID"]
    assert not clean.duplicated(subset=attrs).any()
    assert clean["Year_Birth"].min() >= 1920
    assert clean["Income"].max() <= 200_000
    assert clean[["Income", "Dt_Customer"]].notna().all().all()
    assert "Z_CostContact" not in clean.columns
    assert report.to_frame()["rows_removed"].sum() == len(raw) - len(clean)


def test_feature_engineering_values():
    row = {
        "Year_Birth": 1980, "Income": 50_000, "Kidhome": 1, "Teenhome": 1, "Recency": 10,
        "MntWines": 100, "MntFruits": 0, "MntMeatProducts": 100, "MntFishProducts": 0,
        "MntSweetProducts": 0, "MntGoldProds": 0, "NumDealsPurchases": 2, "NumWebPurchases": 2,
        "NumCatalogPurchases": 0, "NumStorePurchases": 2, "NumWebVisitsMonth": 5,
        "AcceptedCmp1": 1, "AcceptedCmp2": 0, "AcceptedCmp3": 0, "AcceptedCmp4": 0, "AcceptedCmp5": 0,
        "Response": 1, "Complain": 0, "Dt_Customer": pd.Timestamp("2014-01-01"),
    }
    out = engineer_features(pd.DataFrame([row]), pd.Timestamp("2014-06-29")).iloc[0]
    assert out["Total_Spend"] == 200
    assert out["Total_Purchases"] == 4
    assert out["Avg_Order_Value"] == 50
    assert out["Deal_Ratio"] == 0.5
    assert out["Campaigns_Accepted"] == 2
    assert out["Children"] == 2 and out["Age"] == 34
    assert out["Tenure_Days"] == 179


def test_transformers_reduce_skew_and_clip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"skewed": rng.lognormal(0, 1.2, 1000), "normal": rng.normal(5, 1, 1000)})
    corrector = SkewCorrector(0.75).fit(X)
    assert corrector.log_columns_ == ["skewed"]
    assert abs(corrector.transform(X)["skewed"].skew()) < abs(X["skewed"].skew())
    clipped = Winsorizer(0.05, 0.95).fit(X).transform(X)
    assert clipped["skewed"].max() <= X["skewed"].quantile(0.95) + 1e-9


def test_find_elbow():
    k = np.arange(2, 9)
    inertia = np.array([100, 60, 30, 26, 23, 21, 20], dtype=float)
    assert find_elbow(k, inertia) == 4


def test_fit_produces_consistent_outputs(fitted):
    k = fitted.n_segments
    assert FAST_CONFIG.min_business_k <= k <= FAST_CONFIG.k_max
    assert len(set(fitted.segment_names_.values())) == k
    assert fitted.profiles_["Customers"].sum() == len(fitted.customers_)
    assert np.isclose(fitted.profiles_["Revenue_Share_%"].sum(), 100, atol=0.5)
    assert len(fitted.recommendations_) == k


def test_predict_matches_training_labels(fitted, raw):
    clean, _ = clean_customers(raw)
    predicted = fitted.predict(clean.head(200))
    assert (predicted["Segment"].to_numpy() == fitted.labels_[:200]).all()
    assert predicted["Assignment_Margin"].between(0, 1).all()


def test_assign_customer_handles_missing_and_negative_values(fitted, raw):
    record = raw.iloc[0].to_dict()
    record["Income"] = np.nan
    record["MntWines"] = -5
    result = fitted.assign_customer(record)
    assert result["segment"] in fitted.segment_names_
    assert result["recommendation"]["actions"]


def test_predict_rejects_missing_columns(fitted):
    with pytest.raises(ValueError, match="Missing required columns"):
        fitted.predict(pd.DataFrame({"Income": [1]}))


def test_save_and_load_roundtrip(fitted, raw, tmp_path):
    fitted.save(tmp_path)
    loaded = CustomerSegmenter.load(tmp_path)
    sample = raw.head(50)
    pd.testing.assert_frame_equal(fitted.predict(sample), loaded.predict(sample))
    assert (tmp_path / "model_card.json").exists()


@pytest.mark.skipif(not (ARTIFACTS_DIR / MODEL_FILENAME).exists(), reason="Run `python run_pipeline.py train` first")
@pytest.mark.parametrize(
    "page",
    ["app_pages/overview.py", "app_pages/profiles.py", "app_pages/explorer.py", "app_pages/assign.py", "app_pages/diagnostics.py"],
)
def test_dashboard_pages_render(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page(page).run()
    assert not at.exception


@pytest.mark.skipif(not (ARTIFACTS_DIR / MODEL_FILENAME).exists(), reason="Run `python run_pipeline.py train` first")
def test_dashboard_assigns_new_customer():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.switch_page("app_pages/assign.py").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Assigned segment" in s.value for s in at.subheader)
