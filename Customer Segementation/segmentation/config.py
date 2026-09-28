"""Central configuration: paths, schema definitions and model hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DEFAULT_DATA_PATH = DATA_DIR / "Customer_Segmentation_Cleaned_Encoded-1.csv"
MODEL_FILENAME = "segmenter.joblib"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Raw schema
# ---------------------------------------------------------------------------
SPEND_COLUMNS = (
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
)
CHANNEL_PURCHASE_COLUMNS = ("NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases")
CAMPAIGN_COLUMNS = (
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
)
RAW_NUMERIC_COLUMNS = (
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    *SPEND_COLUMNS,
    "NumDealsPurchases",
    *CHANNEL_PURCHASE_COLUMNS,
    "NumWebVisitsMonth",
    *CAMPAIGN_COLUMNS,
    "Complain",
)
RAW_CATEGORICAL_COLUMNS = ("Education", "Marital_Status")
DATE_COLUMN = "Dt_Customer"
DATE_FORMAT = "%d-%m-%Y"
ID_COLUMN = "ID"
CONSTANT_COLUMNS = ("Z_CostContact", "Z_Revenue")

# Baseline category removed by one-hot encoding with drop_first=True (alphabetically first label).
ONE_HOT_BASELINES = {
    "Education": "2n Cycle",
    "Marital_Status": "Absurd",
    DATE_COLUMN: "01-01-2013",
}

# Harmonise inconsistent free-text categories.
MARITAL_STATUS_MAP = {
    "Married": "Married",
    "Together": "Together",
    "Single": "Single",
    "Alone": "Single",
    "Divorced": "Divorced",
    "Widow": "Widow",
    "Absurd": "Unknown",
    "YOLO": "Unknown",
}
EDUCATION_MAP = {
    "Basic": "Basic",
    "Graduation": "Graduate",
    "2n Cycle": "Postgraduate",
    "Master": "Postgraduate",
    "PhD": "Postgraduate",
}

# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------
CATEGORY_SHARE_MAP = {
    "MntWines": "Wine_Share",
    "MntFruits": "Fruit_Share",
    "MntMeatProducts": "Meat_Share",
    "MntFishProducts": "Fish_Share",
    "MntSweetProducts": "Sweet_Share",
    "MntGoldProds": "Gold_Share",
}
CATEGORY_SHARE_COLUMNS = tuple(CATEGORY_SHARE_MAP.values())

# Behavioural features used to FORM segments. Demographics (age, children, education,
# marital status) are intentionally excluded so segments reflect behaviour, not identity.
CLUSTERING_FEATURES = (
    "Income",
    "Total_Spend",
    "Total_Purchases",
    "Avg_Order_Value",
    "Recency",
    "NumWebVisitsMonth",
    "Deal_Ratio",
    "Web_Share",
    "Catalog_Share",
    "Campaigns_Accepted",
)

# Features used only to DESCRIBE segments after they are formed.
PROFILE_NUMERIC_FEATURES = (
    *CLUSTERING_FEATURES,
    "Store_Share",
    "Tenure_Days",
    "Age",
    "Children",
    "Spend_To_Income",
    *CATEGORY_SHARE_COLUMNS,
)
PROFILE_CATEGORICAL_FEATURES = ("Education_Level", "Marital_Status", "Is_Parent", "Age_Band")


@dataclass(frozen=True)
class SegmentationConfig:
    """Hyper-parameters and business rules for the segmentation workflow."""

    clustering_features: tuple[str, ...] = CLUSTERING_FEATURES
    k_min: int = 2
    k_max: int = 10
    # Fewer than 3 segments rarely yields differentiated actions for marketing teams.
    min_business_k: int = 3
    min_segment_share: float = 0.05
    min_stability_ari: float = 0.80
    # Elbow k is accepted if its silhouette is within this fraction of the best eligible silhouette.
    silhouette_tolerance: float = 0.90
    skew_threshold: float = 0.75
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    n_init: int = 20
    stability_runs: int = 10
    stability_sample_frac: float = 0.8
    min_birth_year: int = 1920
    max_income: float = 200_000
    random_state: int = RANDOM_STATE
