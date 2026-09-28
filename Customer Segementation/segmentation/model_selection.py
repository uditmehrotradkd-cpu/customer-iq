"""Choosing the number of segments and the clustering algorithm with multiple criteria."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .config import SegmentationConfig

# Silhouette and Ward are O(n^2); larger uploads are scored on a fixed random sample.
LARGE_SAMPLE = 5000


def silhouette(X: np.ndarray, labels: np.ndarray, random_state: int) -> float:
    sample = LARGE_SAMPLE if len(X) > LARGE_SAMPLE else None
    return float(silhouette_score(X, labels, sample_size=sample, random_state=random_state))


@dataclass
class KSelectionResult:
    metrics: pd.DataFrame
    elbow_k: int
    best_silhouette_k: int
    recommended_k: int
    rationale: str


def find_elbow(k_values: np.ndarray, inertia: np.ndarray) -> int:
    """Kneedle-style elbow: the point furthest from the line joining the first and last points."""
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    distance = np.abs(y - (1 - x))
    return int(k_values[np.argmax(distance)])


def _bootstrap_stability(X: np.ndarray, k: int, config: SegmentationConfig) -> float:
    """Mean ARI between a full-data solution and solutions fit on random subsamples."""
    rng = np.random.default_rng(config.random_state)
    reference = KMeans(k, n_init=config.n_init, random_state=config.random_state).fit_predict(X)
    scores = []
    for run in range(config.stability_runs):
        idx = rng.choice(len(X), size=int(len(X) * config.stability_sample_frac), replace=False)
        model = KMeans(k, n_init=5, random_state=config.random_state + run + 1).fit(X[idx])
        scores.append(adjusted_rand_score(reference, model.predict(X)))
    return float(np.mean(scores))


def _minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    span = series.max() - series.min()
    scaled = (series - series.min()) / span if span else pd.Series(1.0, index=series.index)
    return scaled if higher_is_better else 1 - scaled


def evaluate_k_range(X: pd.DataFrame | np.ndarray, config: SegmentationConfig | None = None) -> KSelectionResult:
    """Score every candidate k with inertia, silhouette, CH, DB, balance and stability.

    Decision rule:
    1. A k is *eligible* if it is actionable (k >= ``min_business_k``), every segment holds at
       least ``min_segment_share`` of customers and bootstrap stability ARI >= ``min_stability_ari``.
    2. The elbow k is recommended if eligible and its silhouette is within ``silhouette_tolerance``
       of the best eligible silhouette (diminishing returns + acceptable separation).
    3. Otherwise the eligible k with the highest composite score (silhouette, Calinski-Harabasz,
       inverted Davies-Bouldin and stability, equally weighted) is recommended.
    """
    config = config or SegmentationConfig()
    X = np.asarray(X)
    rows = []
    for k in range(config.k_min, config.k_max + 1):
        model = KMeans(k, n_init=config.n_init, random_state=config.random_state).fit(X)
        labels = model.labels_
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette(X, labels, config.random_state),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "min_segment_share": np.bincount(labels).min() / len(labels),
                "stability_ari": _bootstrap_stability(X, k, config),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["composite_score"] = (
        _minmax(metrics["silhouette"])
        + _minmax(metrics["calinski_harabasz"])
        + _minmax(metrics["davies_bouldin"], higher_is_better=False)
        + _minmax(metrics["stability_ari"])
    ) / 4
    metrics["eligible"] = (
        (metrics["k"] >= config.min_business_k)
        & (metrics["min_segment_share"] >= config.min_segment_share)
        & (metrics["stability_ari"] >= config.min_stability_ari)
    )

    elbow_k = find_elbow(metrics["k"].to_numpy(), metrics["inertia"].to_numpy())
    best_silhouette_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    candidates = metrics[metrics["eligible"]] if metrics["eligible"].any() else metrics
    best_composite = candidates.loc[candidates["composite_score"].idxmax()]
    elbow_row = metrics.loc[metrics["k"] == elbow_k].iloc[0]
    silhouette_floor = config.silhouette_tolerance * candidates["silhouette"].max()
    elbow_accepted = bool(elbow_row["eligible"]) and elbow_row["silhouette"] >= silhouette_floor
    best = elbow_row if elbow_accepted else best_composite
    recommended_k = int(best["k"])

    reason = (
        f"it is the elbow of the inertia curve and passes every validation gate (silhouette "
        f"{best['silhouette']:.3f} >= {silhouette_floor:.3f})"
        if elbow_accepted
        else f"the elbow (k={elbow_k}) failed validation, so the highest composite score "
        f"({best['composite_score']:.2f}) among eligible solutions was used"
    )
    rationale = (
        f"k={recommended_k} selected because {reason}. Gates: k>={config.min_business_k}, smallest segment "
        f">= {config.min_segment_share:.0%}, stability ARI >= {config.min_stability_ari:.2f}. "
        f"At k={recommended_k}: silhouette={best['silhouette']:.3f}, Davies-Bouldin={best['davies_bouldin']:.2f}, "
        f"stability ARI={best['stability_ari']:.2f}, smallest segment={best['min_segment_share']:.1%}. "
        f"Raw silhouette peaks at k={best_silhouette_k}; best eligible composite is k={int(best_composite['k'])}."
    )
    metrics["recommended"] = metrics["k"] == recommended_k
    return KSelectionResult(metrics, elbow_k, best_silhouette_k, recommended_k, rationale)


def compare_algorithms(X: pd.DataFrame | np.ndarray, k: int, config: SegmentationConfig | None = None) -> pd.DataFrame:
    """Benchmark K-Means against GMM and Ward hierarchical clustering at the chosen k."""
    config = config or SegmentationConfig()
    X = np.asarray(X)
    if len(X) > LARGE_SAMPLE:
        X = X[np.random.default_rng(config.random_state).choice(len(X), LARGE_SAMPLE, replace=False)]
    candidates = {
        "K-Means": (KMeans(k, n_init=config.n_init, random_state=config.random_state), True),
        "Gaussian Mixture": (
            GaussianMixture(k, covariance_type="full", n_init=3, random_state=config.random_state),
            True,
        ),
        "Agglomerative (Ward)": (AgglomerativeClustering(n_clusters=k, linkage="ward"), False),
    }
    rows = []
    for name, (model, can_predict) in candidates.items():
        labels = model.fit_predict(X)
        rows.append(
            {
                "algorithm": name,
                "silhouette": silhouette(X, labels, config.random_state),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "min_segment_share": np.bincount(labels).min() / len(labels),
                "assigns_new_customers": can_predict,
            }
        )
    return pd.DataFrame(rows).round(4)
