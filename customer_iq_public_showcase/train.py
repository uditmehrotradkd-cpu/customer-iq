import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from customer_iq.config import Config
from customer_iq.train import train

parser = argparse.ArgumentParser(description="Train CustomerIQ on your CSV dataset.")
parser.add_argument("--data", required=True, help="Path to the CSV dataset")
parser.add_argument("--min-k", type=int, default=2)
parser.add_argument("--max-k", type=int, default=10)
args = parser.parse_args()

cfg = Config(min_k=args.min_k, max_k=args.max_k)
artifact, clustered, metrics = train(args.data, cfg)

print("\n=== CustomerIQ Training Complete ===")
print(f"Rows: {len(clustered):,}")
print(f"Selected K: {artifact['n_clusters']}")
print(f"Silhouette: {metrics.loc[metrics.K == artifact['n_clusters'], 'Silhouette'].iloc[0]:.4f}")
print(f"Model: {cfg.artifact_dir / 'customer_iq_model.joblib'}")
print(f"Segments: {cfg.output_dir / 'customer_segments.csv'}")
