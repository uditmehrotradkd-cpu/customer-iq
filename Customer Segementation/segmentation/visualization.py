"""Matplotlib/Seaborn figures for EDA, model selection and segment storytelling."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from .config import SPEND_COLUMNS

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "Set2"
RADAR_FEATURES = (
    "Income",
    "Total_Spend",
    "Total_Purchases",
    "Avg_Order_Value",
    "Deal_Ratio",
    "NumWebVisitsMonth",
    "Web_Share",
    "Catalog_Share",
    "Campaigns_Accepted",
    "Recency",
)


def segment_colors(n: int) -> list:
    return sns.color_palette(PALETTE, n)


def _labels_to_names(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    return np.array([f"{s}: {names[s]}" for s in labels])


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------- EDA
def plot_behaviour_overview(features: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    spend = features[list(SPEND_COLUMNS)].sum().sort_values()
    spend.index = [c.replace("Mnt", "") for c in spend.index]
    axes[0, 0].barh(spend.index, spend.values, color=sns.color_palette(PALETTE, len(spend)))
    axes[0, 0].set_title("Total spend by product category")
    channels = features[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]].sum()
    channels.index = ["Web", "Catalog", "Store", "Deals (any channel)"]
    axes[0, 1].bar(channels.index, channels.values, color=sns.color_palette(PALETTE, 4))
    axes[0, 1].set_title("Purchases by channel")
    sns.scatterplot(data=features, x="Income", y="Total_Spend", hue="Is_Parent", alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Income vs. total spend")
    sns.histplot(features["Recency"], bins=30, ax=axes[1, 1], color="#66c2a5")
    axes[1, 1].set_title("Days since last purchase (recency)")
    fig.suptitle("Customer behaviour overview", fontsize=16, weight="bold")
    fig.tight_layout()
    return fig


def plot_distributions(features: pd.DataFrame, columns: list[str]) -> plt.Figure:
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    for ax, col in zip(axes.ravel(), columns):
        sns.histplot(features[col], bins=30, kde=True, ax=ax, color="#8da0cb")
        ax.set_title(f"{col} (skew={features[col].skew():.2f})", fontsize=10)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Feature distributions", fontsize=15, weight="bold")
    fig.tight_layout()
    return fig


def plot_skew_correction(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(len(columns), 2, figsize=(11, 2.8 * len(columns)), squeeze=False)
    for i, col in enumerate(columns):
        sns.histplot(before[col], bins=30, ax=axes[i, 0], color="#fc8d62")
        axes[i, 0].set_title(f"{col} raw (skew={before[col].skew():.2f})")
        sns.histplot(after[col], bins=30, ax=axes[i, 1], color="#66c2a5")
        axes[i, 1].set_title(f"{col} winsorised + log1p (skew={after[col].skew():.2f})")
        axes[i, 0].set_xlabel("")
        axes[i, 1].set_xlabel("")
    fig.suptitle("Why we transform: skew before vs. after", fontsize=15, weight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(features: pd.DataFrame, columns: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = features[columns].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlation of behavioural features", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- model selection
def plot_k_selection(metrics: pd.DataFrame, recommended_k: int, elbow_k: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        ("inertia", "Elbow method: within-cluster SSE", elbow_k),
        ("silhouette", "Silhouette score (higher = better)", None),
        ("davies_bouldin", "Davies-Bouldin index (lower = better)", None),
        ("stability_ari", "Bootstrap stability ARI (higher = better)", None),
    ]
    for ax, (col, title, marker_k) in zip(axes.ravel(), panels):
        ax.plot(metrics["k"], metrics[col], marker="o", color="#4c72b0")
        ax.axvline(recommended_k, color="#dd8452", linestyle="--", label=f"Selected k={recommended_k}")
        if marker_k is not None:
            y = metrics.loc[metrics["k"] == marker_k, col].iloc[0]
            ax.scatter([marker_k], [y], s=180, facecolors="none", edgecolors="red", linewidths=2, label=f"Elbow k={marker_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of segments (k)")
        ax.set_xticks(metrics["k"])
        ax.legend()
    fig.suptitle("Choosing the number of segments", fontsize=16, weight="bold")
    fig.tight_layout()
    return fig


def plot_silhouette_diagram(X: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> plt.Figure:
    X = np.asarray(X)
    values = silhouette_samples(X, labels)
    colors = segment_colors(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for seg in sorted(names):
        seg_values = np.sort(values[labels == seg])
        y_upper = y_lower + len(seg_values)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, seg_values, color=colors[seg], alpha=0.8)
        ax.text(-0.08, y_lower + len(seg_values) / 2, str(seg))
        y_lower = y_upper + 10
    ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean silhouette = {values.mean():.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Customers grouped by segment")
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Silhouette diagram of the final segmentation", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- segments
def pca_projection(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(np.asarray(X)), pca


def plot_pca_segments(X: np.ndarray, labels: np.ndarray, names: dict[int, str], centers: np.ndarray) -> plt.Figure:
    coords, pca = pca_projection(X)
    centers_2d = pca.transform(centers)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=_labels_to_names(labels, names),
        hue_order=[f"{s}: {names[s]}" for s in sorted(names)],
        palette=segment_colors(len(names)),
        alpha=0.6,
        s=25,
        ax=ax,
    )
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="X", s=300, c="black", label="Centroids")
    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title("Customer segments in PCA space", fontsize=14, weight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_segment_sizes(profiles: pd.DataFrame) -> plt.Figure:
    data = profiles.reset_index()
    data["Label"] = data["Segment"].astype(str) + ": " + data["Segment_Name"]
    melted = data.melt(id_vars="Label", value_vars=["Customer_Share_%", "Revenue_Share_%"], var_name="Metric", value_name="Percent")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=melted, y="Label", x="Percent", hue="Metric", palette=["#8da0cb", "#fc8d62"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylabel("")
    ax.set_title("Share of customers vs. share of revenue", fontsize=14, weight="bold")
    fig.tight_layout()
    return fig


def plot_zscore_heatmap(zprofile: pd.DataFrame, names: dict[int, str]) -> plt.Figure:
    data = zprofile.copy()
    data.index = [f"{s}: {names[s]}" for s in data.index]
    fig, ax = plt.subplots(figsize=(max(12, 0.6 * data.shape[1]), 1.1 * len(data) + 2))
    sns.heatmap(data, cmap="RdBu_r", center=0, annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "Std. dev. vs. average"})
    ax.set_title("Segment fingerprint (z-score of segment mean vs. all customers)", fontsize=14, weight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_radar(zprofile: pd.DataFrame, names: dict[int, str], features=RADAR_FEATURES) -> plt.Figure:
    features = [f for f in features if f in zprofile.columns]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    colors = segment_colors(len(names))
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for seg in zprofile.index:
        values = zprofile.loc[seg, features].clip(-2, 2).tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[seg], linewidth=2, label=f"{seg}: {names[seg]}")
        ax.fill(angles, values, color=colors[seg], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    ax.set_title("Segment behavioural radar (z-scores, clipped to +/-2)", fontsize=13, weight="bold", pad=25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_segment_boxplots(customers: pd.DataFrame, names: dict[int, str], columns: list[str]) -> plt.Figure:
    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    order = sorted(names)
    for ax, col in zip(axes.ravel(), columns):
        sns.boxplot(
            data=customers,
            x="Segment",
            y=col,
            hue="Segment",
            order=order,
            palette=segment_colors(len(names)),
            legend=False,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(col)
    for ax in axes.ravel()[len(columns) :]:
        ax.axis("off")
    legend = " | ".join(f"{s} = {names[s]}" for s in order)
    fig.suptitle(f"Key metrics by segment\n{legend}", fontsize=13, weight="bold")
    fig.tight_layout()
    return fig


def plot_channel_category_mix(channels: pd.DataFrame, categories: pd.DataFrame, names: dict[int, str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, data, title in ((axes[0], channels, "Purchase channel mix (%)"), (axes[1], categories, "Spend category mix (%)")):
        plot = data.copy()
        plot.index = [f"{s}: {names[s]}" for s in plot.index]
        plot.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def plot_demographics(customers: pd.DataFrame, names: dict[int, str]) -> plt.Figure:
    columns = ["Age_Band", "Is_Parent", "Education_Level", "Marital_Status"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), columns):
        table = pd.crosstab(customers["Segment"], customers[col], normalize="index") * 100
        if col == "Age_Band":
            table = table[[b for b in ("<35", "35-44", "45-54", "55-64", "65+") if b in table.columns]]
        table.index = [f"{s}: {names[s]}" for s in table.index]
        table.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", edgecolor="white")
        ax.set_title(f"{col.replace('_', ' ')} mix (%)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_xlim(0, 100)
    fig.suptitle("Descriptive demographics (NOT used to form segments)", fontsize=15, weight="bold")
    fig.tight_layout()
    return fig


def generate_report_figures(segmenter, out_dir: str | Path) -> dict[str, Path]:
    """Render every presentation figure for a fitted ``CustomerSegmenter``."""
    out_dir = Path(out_dir)
    customers = segmenter.customers_
    names = segmenter.segment_names_
    labels = segmenter.labels_
    X = segmenter.X_train_
    kmeans = segmenter.pipeline_.named_steps["cluster"]
    preprocess = segmenter.pipeline_.named_steps["preprocess"]
    model_features = segmenter.pipeline_.named_steps["features"].transform(customers)
    deskewed = preprocess[:-1].transform(model_features)
    log_cols = preprocess.named_steps["deskew"].log_columns_
    behaviour_cols = list(segmenter.config.clustering_features) + ["Store_Share", "Tenure_Days"]

    figures = {
        "01_behaviour_overview": plot_behaviour_overview(customers),
        "02_feature_distributions": plot_distributions(customers, behaviour_cols),
        "03_skew_correction": plot_skew_correction(model_features, deskewed, log_cols),
        "04_correlation_heatmap": plot_correlation_heatmap(customers, behaviour_cols),
        "05_k_selection": plot_k_selection(
            segmenter.k_selection_.metrics, segmenter.n_segments, segmenter.k_selection_.elbow_k
        ),
        "06_silhouette_diagram": plot_silhouette_diagram(X, labels, names),
        "07_pca_segments": plot_pca_segments(X, labels, names, kmeans.cluster_centers_),
        "08_segment_sizes": plot_segment_sizes(segmenter.profiles_),
        "09_segment_fingerprint": plot_zscore_heatmap(segmenter.zprofile_, names),
        "10_segment_radar": plot_radar(segmenter.zprofile_, names),
        "11_segment_boxplots": plot_segment_boxplots(
            customers,
            names,
            ["Income", "Total_Spend", "Total_Purchases", "Avg_Order_Value", "Deal_Ratio", "NumWebVisitsMonth"],
        ),
        "12_channel_category_mix": plot_channel_category_mix(segmenter.channel_mix_, segmenter.category_mix_, names),
        "13_demographics": plot_demographics(customers, names),
    }
    paths = {}
    for name, fig in figures.items():
        paths[name] = save_figure(fig, out_dir / f"{name}.png")
        plt.close(fig)
    return paths
