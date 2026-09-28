import pandas as pd

def build_personas(clustered):
    means = clustered.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    # Persona labels are deterministic business descriptions based on relative
    # cluster behavior; they are not learned labels.
    score = means["Frequency"].rank(pct=True) + means["Monetary"].rank(pct=True) - means["Recency"].rank(pct=True)

    labels = {}
    remaining = set(means.index)

    vip = int(score.idxmax())
    labels[vip] = "VIP Heavy Spenders"
    remaining.remove(vip)

    if remaining:
        risk = int(means.loc[list(remaining), "Recency"].idxmax())
        labels[risk] = "At-Risk Customers"
        remaining.remove(risk)

    for c in remaining:
        if means.loc[c, "Frequency"] >= means["Frequency"].median():
            labels[int(c)] = "Loyal Frequent Buyers"
        elif means.loc[c, "Monetary"] <= means["Monetary"].median():
            labels[int(c)] = "Low-Value / New Customers"
        else:
            labels[int(c)] = "Casual Shoppers"
    return labels
