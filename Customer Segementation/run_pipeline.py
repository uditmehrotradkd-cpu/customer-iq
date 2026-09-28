"""Command-line entry point.

Train:   python run_pipeline.py train [--data PATH] [--k N]
Score:   python run_pipeline.py predict --input new_customers.csv --output scored.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from segmentation import CustomerSegmenter, load_customers  # noqa: E402
from segmentation.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH, FIGURES_DIR  # noqa: E402
from segmentation.visualization import generate_report_figures  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("segmentation")


def train(data_path: Path, k: int | None, artifacts_dir: Path) -> None:
    log.info("Loading %s", data_path)
    raw = load_customers(data_path)
    segmenter = CustomerSegmenter(n_clusters=k).fit(raw)
    log.info("Cleaning: %d raw rows -> %d clean rows", segmenter.n_raw_rows_, len(segmenter.customers_))
    log.info(segmenter.k_selection_.rationale)
    segmenter.save(artifacts_dir)
    figures = generate_report_figures(segmenter, artifacts_dir / FIGURES_DIR.name)
    log.info("Saved model, tables and %d figures to %s", len(figures), artifacts_dir)
    print(segmenter.profiles_.to_string())


def predict(input_path: Path, output_path: Path, artifacts_dir: Path) -> None:
    segmenter = CustomerSegmenter.load(artifacts_dir)
    raw = load_customers(input_path)
    scored = pd.concat([raw, segmenter.predict(raw)], axis=1)
    scored.to_csv(output_path, index=False)
    log.info("Scored %d customers -> %s", len(scored), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer segmentation pipeline")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fit the segmentation model and export artifacts")
    train_p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_p.add_argument("--k", type=int, default=None, help="Override the automatically selected k")

    predict_p = sub.add_parser("predict", help="Assign segments to new customers")
    predict_p.add_argument("--input", type=Path, required=True)
    predict_p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.k, args.artifacts)
    else:
        predict(args.input, args.output, args.artifacts)


if __name__ == "__main__":
    main()
