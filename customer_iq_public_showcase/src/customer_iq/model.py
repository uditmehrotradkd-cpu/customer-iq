from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

def evaluate_k(X, min_k=2, max_k=10, random_state=42):
    rows = []
    max_k = min(max_k, len(X)-1)
    for k in range(min_k, max_k+1):
        model = KMeans(n_clusters=k, n_init=30, random_state=random_state)
        labels = model.fit_predict(X)
        rows.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette": float(silhouette_score(X, labels)),
            "DaviesBouldin": float(davies_bouldin_score(X, labels)),
            "CalinskiHarabasz": float(calinski_harabasz_score(X, labels))
        })
    return pd.DataFrame(rows)

def choose_k(metrics):
    # Primary criterion: silhouette. DB and CH are retained as supporting diagnostics.
    return int(metrics.sort_values(["Silhouette", "DaviesBouldin"],
                                   ascending=[False, True]).iloc[0]["K"])

def fit_kmeans(X, k, random_state=42):
    model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels

def fit_pca(X):
    n = min(3, X.shape[1])
    return PCA(n_components=n, random_state=42).fit(X)

def save_artifact(path, artifact):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)

def load_artifact(path):
    return joblib.load(path)
