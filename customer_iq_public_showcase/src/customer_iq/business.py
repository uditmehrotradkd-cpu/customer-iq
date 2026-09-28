STRATEGIES = {
    "VIP Heavy Spenders": {
        "objective": "Protect and expand high-value relationships.",
        "actions": ["VIP access", "Premium bundles", "Early access", "Concierge support"]
    },
    "Loyal Frequent Buyers": {
        "objective": "Increase repeat value and loyalty.",
        "actions": ["Cross-sell", "Subscriptions", "Points", "Referral rewards"]
    },
    "Casual Shoppers": {
        "objective": "Increase purchase frequency.",
        "actions": ["Personalized recommendations", "Bundles", "Second-purchase campaigns"]
    },
    "At-Risk Customers": {
        "objective": "Recover customers before inactivity.",
        "actions": ["Win-back campaigns", "Past-product recommendations", "Feedback surveys"]
    },
    "Low-Value / New Customers": {
        "objective": "Drive second and third purchases.",
        "actions": ["Starter bundles", "Onboarding", "Discovery offers"]
    }
}

def segment_summary(df):
    cols = ["Recency", "Frequency", "Monetary", "AverageOrderValue"]
    available = [c for c in cols if c in df.columns]
    return df.groupby("Persona")[available].agg(["mean", "median", "count"]).round(2)
