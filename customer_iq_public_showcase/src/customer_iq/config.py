from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class Config:
    random_state: int = 42
    min_k: int = 2
    max_k: int = 10
    artifact_dir: Path = Path(os.getenv("MODEL_DIR", "models"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))

    # This dataset is the cleaned/encoded form of the common marketing-campaign
    # customer dataset. The pipeline also auto-detects these columns when present.
    monetary_cols: tuple = (
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    )
    frequency_cols: tuple = (
        "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"
    )
    optional_behavior_cols: tuple = (
        "NumDealsPurchases", "NumWebVisitsMonth",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5", "Response", "Complain",
        "Kidhome", "Teenhome", "Income"
    )
    id_cols: tuple = ("ID", "CustomerID", "customer_id")
    date_onehot_prefixes: tuple = ("Dt_Customer_",)

    def ensure_dirs(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
