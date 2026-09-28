"""Customer segmentation package: cleaning, features, clustering, profiling and scoring."""
from .config import SegmentationConfig
from .data_loader import load_customers
from .pipeline import CustomerSegmenter

__all__ = ["CustomerSegmenter", "SegmentationConfig", "load_customers"]
