import pandas as pd
from .model import load_artifact

def predict_customer(recency, frequency, monetary, artifact):
    row = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    # Fill non-RFM training features with their training medians where available.
    cols = artifact["model_columns"]
    for c in cols:
        if c not in row.columns:
            row[c] = artifact["training_feature_medians"].get(c, 0.0)

    X = artifact["preprocessor"].transform(row[cols])
    cluster = int(artifact["model"].predict(X)[0])
    pca = artifact["pca"].transform(X)[0]

    return {
        "cluster": cluster,
        "persona": artifact["personas"].get(cluster, f"Cluster {cluster}"),
        "pca": [float(x) for x in pca]
    }
