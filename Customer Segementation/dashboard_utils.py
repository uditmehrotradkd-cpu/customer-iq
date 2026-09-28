"""Shared, cached helpers for the Streamlit dashboard pages."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segmentation import CustomerSegmenter, load_customers  # noqa: E402
from segmentation.config import ARTIFACTS_DIR, DEFAULT_DATA_PATH, MODEL_FILENAME  # noqa: E402
from segmentation.visualization import pca_projection  # noqa: E402

SEGMENT_COLORS = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#1b9e77",
    "#d95f02",
]


@st.cache_resource(show_spinner="Loading segmentation model...")
def get_segmenter() -> CustomerSegmenter:
    """Load the trained artifact, training once on the bundled dataset if none exists."""
    if (ARTIFACTS_DIR / MODEL_FILENAME).exists():
        return CustomerSegmenter.load(ARTIFACTS_DIR)
    segmenter = CustomerSegmenter().fit(load_customers(DEFAULT_DATA_PATH))
    segmenter.save(ARTIFACTS_DIR)
    return segmenter


def segment_labels(segmenter: CustomerSegmenter) -> list[str]:
    return [segmenter.segment_names_[s] for s in sorted(segmenter.segment_names_)]


def color_scale(segmenter: CustomerSegmenter) -> alt.Scale:
    names = segment_labels(segmenter)
    return alt.Scale(domain=names, range=SEGMENT_COLORS[: len(names)])


@st.cache_data(show_spinner=False)
def pca_frame(_segmenter: CustomerSegmenter) -> tuple[pd.DataFrame, list[float]]:
    coords, pca = pca_projection(_segmenter.X_train_)
    customers = _segmenter.customers_
    frame = customers[["ID", "Segment_Name", "Income", "Total_Spend", "Total_Purchases", "Deal_Ratio"]].copy()
    frame["PC1"], frame["PC2"] = coords[:, 0], coords[:, 1]
    return frame, pca.explained_variance_ratio_.tolist()
