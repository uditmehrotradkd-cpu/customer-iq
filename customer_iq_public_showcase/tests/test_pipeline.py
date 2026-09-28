import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from customer_iq.data import load_csv, clean_dataset, detect_schema, build_feature_table
from customer_iq.config import Config

def test_supplied_dataset_schema():
    path = Path(__file__).parents[1] / "data" / "Customer_Segmentation_Cleaned_Encoded-1.csv"
    df = load_csv(path)
    clean, _ = clean_dataset(df)
    schema = detect_schema(clean, Config())
    features = build_feature_table(clean, schema, Config())
    assert len(features) == len(clean)
    assert {"Recency","Frequency","Monetary"}.issubset(features.columns)
