"""Segment profiling, persona naming, recommendations and responsible-use checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency

from .config import CATEGORY_SHARE_MAP, PROFILE_NUMERIC_FEATURES

# Each archetype is scored against a segment's standardised mean profile (z-scores vs. the
# whole customer base). Names are then assigned one-to-one with the Hungarian algorithm so
# every segment receives the best-fitting unique persona.
ARCHETYPES: dict[str, dict] = {
    "Premium Campaign Responders": {
        "weights": {"Total_Spend": 1.0, "Income": 0.7, "Campaigns_Accepted": 1.2, "Deal_Ratio": -0.3},
        "summary": "High-income, high-spend customers who actively respond to marketing campaigns.",
        "goal": "Retain and grow share of wallet",
        "actions": [
            "Invite to a tiered loyalty / VIP programme with early access to new premium ranges.",
            "Use them as the priority audience for new-campaign pilots; they convert best.",
            "Personalise offers around their top categories rather than discounting.",
            "Offer concierge-style service on catalog and web to protect satisfaction.",
        ],
        "kpis": ["Retention rate", "Campaign conversion rate", "Share of wallet"],
        "responsible_use": "Cap contact frequency; high responsiveness is not consent to unlimited messaging.",
    },
    "Affluent Quiet Loyalists": {
        "weights": {
            "Total_Spend": 1.0,
            "Income": 0.8,
            "Avg_Order_Value": 0.5,
            "Campaigns_Accepted": -0.8,
            "NumWebVisitsMonth": -0.5,
        },
        "summary": "High-value customers who buy steadily but ignore promotional campaigns.",
        "goal": "Protect value without over-marketing",
        "actions": [
            "Reduce generic campaign volume; switch to service and quality-led communication.",
            "Offer curated bundles and new-arrival notes in their dominant categories.",
            "Recognise loyalty with non-monetary perks (priority delivery, member events).",
            "Monitor recency closely: silent high-value customers churn without warning.",
        ],
        "kpis": ["Revenue retention", "Average order value", "Unsubscribe rate"],
        "responsible_use": "Respect low engagement with campaigns as a preference; avoid pressure tactics.",
    },
    "Digital Deal-Seekers": {
        "weights": {"Web_Share": 1.0, "Deal_Ratio": 0.8, "NumWebVisitsMonth": 0.6, "Total_Spend": 0.2},
        "summary": "Mid-value, web-first shoppers who browse often and buy heavily on promotion.",
        "goal": "Convert browsing into full-price, higher-basket purchases",
        "actions": [
            "Trigger personalised web/app offers based on browsing, bundled to raise basket size.",
            "Replace blanket discounts with loyalty points that reward repeat purchases.",
            "Use cart-abandonment and price-drop alerts with clear, honest expiry dates.",
            "Test free-shipping thresholds just above current average order value.",
        ],
        "kpis": ["Web conversion rate", "Average order value", "Full-price sales share"],
        "responsible_use": "No fake scarcity or countdown timers; show genuine savings transparently.",
    },
    "Budget-Conscious Browsers": {
        "weights": {"Total_Spend": -1.0, "Total_Purchases": -1.0, "Income": -0.7, "NumWebVisitsMonth": 0.4},
        "summary": "Lower-income, low-spend customers who visit frequently but purchase rarely.",
        "goal": "Deliver everyday value and build habit",
        "actions": [
            "Promote value packs and essentials rather than premium categories.",
            "Reach them in-store and via low-cost email/app; avoid expensive printed catalogs.",
            "Introduce an entry-level loyalty tier that rewards visit-to-purchase conversion.",
            "Simplify the web journey; high visits with few purchases signal friction.",
        ],
        "kpis": ["Visit-to-purchase conversion", "Purchase frequency", "Cost to serve"],
        "responsible_use": "Never push credit, debt or upsell beyond affordability; keep pricing fair.",
    },
    "Lapsing Customers": {
        "weights": {"Recency": 1.5, "Total_Purchases": -0.2},
        "summary": "Customers whose last purchase was long ago and who are at risk of churning.",
        "goal": "Win back before churn",
        "actions": [
            "Send a win-back sequence with a relevant, time-bound incentive.",
            "Run a short feedback survey to learn why they stopped buying.",
            "Suppress from full-price campaigns until re-engaged.",
        ],
        "kpis": ["Reactivation rate", "Days since last purchase"],
        "responsible_use": "Offer an easy opt-out; respect customers who chose to leave.",
    },
    "Recently Active Starters": {
        "weights": {"Recency": -1.5, "Total_Spend": -0.5},
        "summary": "Recently active customers with low spend so far.",
        "goal": "Accelerate second and third purchases",
        "actions": [
            "Onboarding journey highlighting best-selling categories.",
            "Second-purchase incentive within 30 days.",
        ],
        "kpis": ["Second-purchase rate", "90-day spend"],
        "responsible_use": "Keep onboarding frequency modest and relevant.",
    },
    "Steady Mid-Value Regulars": {
        "weights": {"Total_Purchases": 0.8, "Total_Spend": 0.3, "Store_Share": 0.6},
        "summary": "Frequent, store-oriented shoppers with moderate spend.",
        "goal": "Increase basket size",
        "actions": [
            "In-store cross-sell of complementary categories.",
            "Loyalty points multipliers on larger baskets.",
        ],
        "kpis": ["Basket size", "Purchase frequency"],
        "responsible_use": "Base offers on purchase behaviour, not personal attributes.",
    },
    "Catalog Connoisseurs": {
        "weights": {"Catalog_Share": 1.2, "Avg_Order_Value": 0.5},
        "summary": "Customers who prefer the catalog channel and place large orders.",
        "goal": "Modernise the channel without losing value",
        "actions": [
            "Curated catalog editions around top categories.",
            "Offer a digital catalog with one-click reorder.",
        ],
        "kpis": ["Catalog revenue", "Average order value"],
        "responsible_use": "Provide paper-free options for customers who prefer them.",
    },
    "Newly Acquired Customers": {
        "weights": {"Tenure_Days": -1.5},
        "summary": "Recently enrolled customers still forming habits.",
        "goal": "Build early loyalty",
        "actions": ["Welcome series with category discovery.", "Early feedback request."],
        "kpis": ["90-day retention", "Second-purchase rate"],
        "responsible_use": "Explain data use clearly during onboarding.",
    },
    "Emerging Value Growers": {
        "weights": {"Total_Spend": 0.3, "Deal_Ratio": 0.5, "Web_Share": 0.5, "Tenure_Days": 0.5},
        "summary": "Long-tenured customers with growing spend and promotion affinity.",
        "goal": "Grow into high-value customers",
        "actions": ["Progressive loyalty milestones.", "Personalised category recommendations."],
        "kpis": ["Spend growth", "Tier upgrades"],
        "responsible_use": "Avoid over-reliance on discounting that erodes trust.",
    },
}

CHANNEL_COLUMNS = {"Web_Share": "Web", "Catalog_Share": "Catalog", "Store_Share": "Store"}


def zscore_profile(features: pd.DataFrame, labels: np.ndarray, columns=PROFILE_NUMERIC_FEATURES) -> pd.DataFrame:
    """Segment means expressed as standard deviations from the overall customer mean."""
    columns = [c for c in columns if c in features.columns]
    data = features[columns]
    std = data.std(ddof=0).replace(0, 1)
    return (data.groupby(labels).mean() - data.mean()) / std


def index_table(features: pd.DataFrame, labels: np.ndarray, columns=PROFILE_NUMERIC_FEATURES) -> pd.DataFrame:
    """Segment mean / population mean x 100 (100 = average customer)."""
    columns = [c for c in columns if c in features.columns]
    data = features[columns]
    return (data.groupby(labels).mean() / data.mean().replace(0, np.nan) * 100).round(0)


def name_segments(zprofile: pd.DataFrame) -> dict[int, str]:
    """Assign a unique, best-fitting archetype name to every segment."""
    names = list(ARCHETYPES)
    scores = np.zeros((len(zprofile), len(names)))
    for j, name in enumerate(names):
        for feature, weight in ARCHETYPES[name]["weights"].items():
            if feature in zprofile.columns:
                scores[:, j] += weight * zprofile[feature].to_numpy()
    rows, cols = linear_sum_assignment(-scores)
    mapping = {int(zprofile.index[r]): names[c] for r, c in zip(rows, cols)}
    for seg in zprofile.index:
        mapping.setdefault(int(seg), f"Segment {int(seg)}")
    return mapping


def top_traits(zprofile: pd.DataFrame, segment: int, n: int = 3) -> tuple[list[str], list[str]]:
    row = zprofile.loc[segment].drop(labels=list(CATEGORY_SHARE_MAP.values()), errors="ignore")
    high = [f"{k} (+{v:.1f} sd)" for k, v in row.sort_values(ascending=False).head(n).items() if v > 0.2]
    low = [f"{k} ({v:.1f} sd)" for k, v in row.sort_values().head(n).items() if v < -0.2]
    return high, low


def build_segment_profiles(features: pd.DataFrame, labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """Business summary table: size, revenue share and key behavioural medians/means."""
    df = features.assign(Segment=labels)
    grouped = df.groupby("Segment")
    total_revenue = df["Total_Spend"].sum()
    profile = pd.DataFrame(
        {
            "Segment_Name": pd.Series(names),
            "Customers": grouped.size(),
            "Customer_Share_%": (grouped.size() / len(df) * 100).round(1),
            "Revenue_Share_%": (grouped["Total_Spend"].sum() / total_revenue * 100).round(1),
            "Median_Income": grouped["Income"].median().round(0),
            "Avg_Total_Spend": grouped["Total_Spend"].mean().round(0),
            "Avg_Purchases": grouped["Total_Purchases"].mean().round(1),
            "Avg_Order_Value": grouped["Avg_Order_Value"].mean().round(1),
            "Avg_Recency_Days": grouped["Recency"].mean().round(1),
            "Avg_Web_Visits": grouped["NumWebVisitsMonth"].mean().round(1),
            "Deal_Ratio_%": (grouped["Deal_Ratio"].mean() * 100).round(1),
            "Campaign_Acceptance": grouped["Campaigns_Accepted"].mean().round(2),
            "Avg_Age": grouped["Age"].mean().round(1),
            "Avg_Children": grouped["Children"].mean().round(2),
            "Avg_Tenure_Days": grouped["Tenure_Days"].mean().round(0),
        }
    )
    profile.index.name = "Segment"
    return profile.sort_values("Avg_Total_Spend", ascending=False)


def channel_mix(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    return (features[list(CHANNEL_COLUMNS)].groupby(labels).mean() * 100).rename(columns=CHANNEL_COLUMNS).round(1)


def category_mix(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    shares = list(CATEGORY_SHARE_MAP.values())
    renamed = {c: c.replace("_Share", "") for c in shares}
    return (features[shares].groupby(labels).mean() * 100).rename(columns=renamed).round(1)


def demographic_mix(features: pd.DataFrame, labels: np.ndarray, column: str) -> pd.DataFrame:
    return (pd.crosstab(labels, features[column], normalize="index") * 100).round(1)


def build_recommendations(
    features: pd.DataFrame, labels: np.ndarray, names: dict[int, str], zprofile: pd.DataFrame
) -> list[dict]:
    """Combine archetype playbooks with segment-specific evidence (channel, categories, traits)."""
    channels = channel_mix(features, labels)
    categories = category_mix(features, labels)
    base_categories = category_mix(features, np.zeros(len(features), dtype=int)).iloc[0]
    output = []
    for seg, name in sorted(names.items()):
        playbook = ARCHETYPES.get(name, {})
        cat_index = (categories.loc[seg] / base_categories.replace(0, np.nan)).sort_values(ascending=False)
        high, low = top_traits(zprofile, seg)
        output.append(
            {
                "segment": int(seg),
                "name": name,
                "summary": playbook.get("summary", ""),
                "goal": playbook.get("goal", ""),
                "defining_high_traits": high,
                "defining_low_traits": low,
                "preferred_channel": str(channels.loc[seg].idxmax()),
                "channel_mix_%": channels.loc[seg].to_dict(),
                "over_indexed_categories": list(cat_index.index[:2]),
                "actions": playbook.get("actions", []),
                "kpis": playbook.get("kpis", []),
                "responsible_use": playbook.get("responsible_use", ""),
            }
        )
    return output


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def proxy_risk_check(features: pd.DataFrame, labels: np.ndarray, columns=("Age_Band", "Is_Parent", "Education_Level", "Marital_Status")) -> pd.DataFrame:
    """Measure how strongly segments align with demographic attributes (Cramer's V).

    Segments are built from behaviour only, but they can still act as a proxy for sensitive
    attributes. V >= 0.3 is flagged for human review before targeting decisions are made.
    """
    rows = []
    for col in columns:
        if col in features.columns:
            v = cramers_v(pd.Series(labels, index=features.index), features[col])
            rows.append({"attribute": col, "cramers_v": round(v, 3), "review_needed": v >= 0.3})
    return pd.DataFrame(rows)
