"""Question answering for the data agent: live facts from the model plus a site knowledge base.

Answers come from two places, both fully local:
1. Dynamic handlers that read the current model (segment profiles, superlatives, counts, rationale).
2. A curated knowledge base about every page, metric, method and feature, matched with TF-IDF.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .services import SegmentationService

MIN_SIMILARITY = 0.1
# When a general AI model is available, weak site matches are handed to it instead of guessing.
MIN_SIMILARITY_WITH_AI = 0.2
STOPWORDS = set(
    "a an the is are was were be been do does did what whats which who how why when where can could should would will i me my you your "
    "we our it its this that these those of in on at to for from with by about as and or not no tell please show explain describe "
    "there here get use used using mean means meaning".split()
)


def _stem(token: str) -> str:
    for suffix in ("ation", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def analyze_text(text: str) -> list[str]:
    """Lower-case, drop stop words, light stemming, then unigrams + bigrams."""
    tokens = [_stem(t) for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS]
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


@dataclass
class Entry:
    title: str
    keywords: str
    answer: Callable[[SegmentationService], str] | str
    links: list[tuple[str, str]] = field(default_factory=list)

    def text(self, service: SegmentationService) -> str:
        return self.answer(service) if callable(self.answer) else self.answer


def _money(v: float) -> str:
    return f"${v:,.0f}"


def _pct(v: float) -> str:
    return f"{v:.1f}%"


# ---------------------------------------------------------------- dynamic answer builders
def _k_answer(s: SegmentationService) -> str:
    ks = s.segmenter.k_selection_
    return (
        f"The model uses {s.segmenter.n_segments} segments. {ks.rationale} "
        "The candidate k values were scored on inertia (elbow), silhouette, Davies-Bouldin, Calinski-Harabasz and bootstrap "
        "stability; the full table is on the Model & methodology page."
    )


def _cleaning_answer(s: SegmentationService) -> str:
    steps = s.segmenter.cleaning_report_.to_frame()
    removed = steps[steps["rows_removed"] > 0]
    detail = "; ".join(f"{r.step}: {r.rows_removed}" for r in removed.itertuples())
    return (
        f"The raw file had {s.segmenter.n_raw_rows_:,} rows and {len(s.customers):,} remain after cleaning. Rows removed — {detail}. "
        "Constant columns were dropped, categories harmonised (Alone→Single, Absurd/YOLO→Unknown), deal purchases capped at total "
        "purchases and missing values imputed (income by education median)."
    )


def _algorithms_answer(s: SegmentationService) -> str:
    rows = s.segmenter.algorithm_comparison_
    parts = ", ".join(f"{r.algorithm} silhouette {r.silhouette:.3f}" for r in rows.itertuples())
    return (
        f"Three algorithms were compared at the chosen k: {parts}. K-Means was deployed because it separates segments best, "
        "is easy to explain (each centroid is the 'typical' customer) and can assign new customers instantly; Ward hierarchical "
        "clustering cannot score new customers and the Gaussian Mixture produced overlapping, unbalanced groups."
    )


def _fairness_answer(s: SegmentationService) -> str:
    rows = s.segmenter.proxy_check_
    parts = ", ".join(f"{r.attribute.replace('_', ' ')} V={r.cramers_v:.2f}{' (review)' if r.review_needed else ''}" for r in rows.itertuples())
    return (
        "Segments are built from behaviour only; age, children, education and marital status are used just to describe them. "
        f"A Cramér's V proxy check measures how strongly segments still correlate with demographics: {parts}. "
        "Values of 0.3 or more are flagged for human review. The model must not be used for credit, pricing or eligibility decisions, "
        "and every playbook has a guardrail (contact caps, no fake scarcity, affordability limits)."
    )


def _overview_answer(s: SegmentationService) -> str:
    p = s.segmenter.profiles_
    top = p.sort_values("Revenue_Share_%", ascending=False).head(2)
    return (
        f"{len(s.customers):,} customers are grouped into {s.segmenter.n_segments} behavioural segments: "
        + ", ".join(f"{r['Segment_Name']} ({_pct(r['Customer_Share_%'])} of customers, {_pct(r['Revenue_Share_%'])} of revenue)" for _, r in p.iterrows())
        + f". The top two segments by revenue hold {_pct(top['Customer_Share_%'].sum())} of customers but {_pct(top['Revenue_Share_%'].sum())} of revenue."
    )


def _features_answer(s: SegmentationService) -> str:
    return (
        "The model forms segments from these behavioural features: "
        + ", ".join(f.replace("_", " ") for f in s.segmenter.config.clustering_features)
        + ". Right-skewed features (" + ", ".join(s.segmenter.pipeline_.named_steps["preprocess"].named_steps["deskew"].log_columns_)
        + ") are log-transformed; all are winsorised and standardised first."
    )


def _data_answer(s: SegmentationService) -> str:
    return (
        f"The built-in dataset is a retail marketing file with {s.segmenter.n_raw_rows_:,} customers and two years of history: spend on wine, "
        "fruit, meat, fish, sweets and gold; web, catalog, store and discounted purchases; monthly web visits; days since last purchase; "
        "responses to 6 campaigns; complaints; birth year, income, children, education, marital status and enrolment date. "
        "It arrived one-hot encoded (699 columns) and is decoded back to one row per customer before modelling."
    )


def _model_answer(s: SegmentationService) -> str:
    card = s.segmenter.model_card()
    return (
        f"The live model is {card['algorithm']} with {card['n_segments']} segments (silhouette {card['silhouette']:.3f}), trained on "
        f"{card['rows_clean']:,} clean customers. Pipeline: clean → engineer features → impute → winsorise → log1p skewed features → "
        "standard-scale → K-Means. Intended use: behavioural segmentation for engagement planning; out of scope: credit, pricing or "
        "eligibility decisions about individuals."
    )


# ---------------------------------------------------------------- knowledge base
GLOSSARY = {
    "Income": "annual household income in dollars.",
    "Total Spend": "total amount spent across the six product categories over two years (monetary value).",
    "Total Purchases": "web + catalog + store purchases (purchase frequency).",
    "Average Order Value (AOV)": "total spend divided by total purchases — the typical basket size.",
    "Recency": "days since the customer's last purchase; lower means more recently active.",
    "Web visits per month": "number of visits to the website in the last month (digital engagement).",
    "Deal ratio": "share of purchases made with a discount (promotion sensitivity), from 0% to 100%.",
    "Web / Catalog / Store share": "share of a customer's purchases made in each channel (channel preference).",
    "Campaigns accepted": "how many of the 6 marketing campaigns the customer accepted (0–6).",
    "Tenure": "days since the customer enrolled.",
    "Spend to income": "total spend divided by income (share of wallet).",
    "Category shares": "share of spend on wine, fruit, meat, fish, sweets and gold.",
}

METRICS = [
    Entry("Silhouette score", "silhouette score separation cohesion quality cluster -1 1 good bad",
          lambda s: f"The silhouette score (−1 to 1) measures how much closer each customer is to its own segment than to the nearest other one. Higher is better; values around 0.2–0.35 are typical for real customer data. The live model scores {s.segmenter.silhouette_:.3f}.",
          [("Model & methodology", "#/diagnostics")]),
    Entry("Davies-Bouldin index", "davies bouldin db index lower better similarity", "The Davies-Bouldin index averages how similar each segment is to its most similar neighbour. Lower is better; it is one of the four criteria used to choose k."),
    Entry("Calinski-Harabasz", "calinski harabasz ch variance ratio between within dispersion", "Calinski-Harabasz is the ratio of between-segment to within-segment dispersion. Higher is better; it naturally favours small k, so it is used alongside other metrics."),
    Entry("Elbow method and inertia", "elbow inertia sse within cluster sum squares kneedle", "Inertia is the within-segment sum of squared distances. The elbow method looks for the k where adding segments stops reducing inertia much; here it is found automatically with a Kneedle-style distance test."),
    Entry("Bootstrap stability (ARI)", "stability ari adjusted rand index bootstrap subsample robust", "Stability re-fits K-Means on random 80% subsamples and compares the result with the full solution using the Adjusted Rand Index (1 = identical). Solutions below 0.80 are rejected as unstable."),
    Entry("Composite score", "composite score combined ranking k selection", "The composite score is the average of min-max-scaled silhouette, Calinski-Harabasz, inverted Davies-Bouldin and stability. It is the fallback rule when the elbow k fails a validation gate."),
    Entry("Cramér's V", "cramers v cramer proxy association demographic", "Cramér's V (0–1) measures association between two categorical variables. The site uses it to check whether behavioural segments act as a proxy for demographics; 0.3 or more is flagged for review."),
    Entry("Assignment margin and fit", "assignment margin fit clear moderate borderline confidence distance centroid", "When a customer is assigned, the margin is 1 − (distance to nearest segment ÷ distance to second-nearest). ≥25% is a clear fit, 10–25% moderate, below 10% borderline (the customer sits between two segments)."),
    Entry("Z-score fingerprint", "z-score zscore fingerprint heatmap standard deviations above below average", "The fingerprint shows each segment's average for every feature in standard deviations from the overall average: red is above average, blue below. It is how segment names and traits are derived."),
    Entry("Index table", "index table 100 average relative", "In the index table 100 equals the average customer: 150 means the segment is 50% above average on that feature, 50 means half the average."),
    Entry("Customer share vs revenue share", "customer share revenue share percent value concentration", "Customer share is the percentage of customers in a segment; revenue share is its percentage of total spend. Comparing them shows where value is concentrated."),
    Entry("PCA projection", "pca principal component analysis 2d 3d map projection variance galaxy", "PCA compresses the 10 scaled features into 2 (segment map) or 3 (Galaxy view) dimensions for visualisation. Some overlap is expected because the plot hides the remaining variance; clustering itself uses all features."),
]

METHODS = [
    Entry("How the number of segments was chosen", "why k number of segments clusters choose chosen four 4 how many decide", _k_answer, [("Model & methodology", "#/diagnostics")]),
    Entry("Data cleaning", "cleaning clean cleaned duplicates outliers missing values inconsistent records quality removed prepared", _cleaning_answer, [("Model & methodology", "#/diagnostics")]),
    Entry("Algorithm comparison", "algorithm algorithms method technique kmeans k-means clustering gmm gaussian mixture ward hierarchical agglomerative why compare which", _algorithms_answer, [("Model & methodology", "#/diagnostics")]),
    Entry("Preprocessing", "preprocessing transform winsorise winsorize log1p skew scaling standard scaler impute why",
          "Before clustering, features are median-imputed (so any customer can be scored), winsorised at the 1st–99th percentile (extreme spenders can't drag centroids), log1p-transformed when skewness exceeds 0.75 (long tails compressed) and standard-scaled (income and ratios on the same scale). All steps live in one sklearn pipeline, so training and scoring are identical."),
    Entry("Features used for clustering", "features variables inputs columns clustering behavioural used", _features_answer),
    Entry("Segment naming", "names personas naming archetype hungarian how named", "Names are assigned automatically: each segment's z-score fingerprint is scored against a library of business archetypes (e.g. Premium Campaign Responders, Digital Deal-Seekers) and the Hungarian algorithm gives every segment its best-fitting unique name."),
    Entry("Recommendations and playbooks", "recommendations actions playbook next best action kpi guardrail engagement", "Every segment has a goal, 3–4 recommended actions, KPIs to track and a responsible-use guardrail, enriched with its preferred channel and over-indexed product categories. Ask me about a specific segment, or download the recommendations table.", [("Segment profiles", "#/segments")]),
    Entry("Responsible AI and fairness", "responsible ai fair fairness bias biased ethics ethical proxy demographic privacy guardrail discrimination trust", _fairness_answer, [("Model & methodology", "#/diagnostics")]),
    Entry("The dataset", "dataset data source origin come from where columns rows file kaggle raw one-hot built-in", _data_answer),
    Entry("The model", "model pipeline trained how works kmeans live model card", _model_answer, [("Model & methodology", "#/diagnostics")]),
    Entry("Segments overview", "segments overview summary all groups what are the segments list", _overview_answer, [("Overview", "#/overview"), ("Segment profiles", "#/segments")]),
    Entry("Recency insight", "recency uniform lifecycle win-back trigger", "Recency (days since last purchase) is almost uniformly distributed from 0 to 99 days, so it barely separates segments. It works better as a lifecycle trigger inside each segment, e.g. start a win-back journey when recency exceeds 60 days."),
    Entry("Synthetic data", "synthetic fake generated sample data privacy how made fidelity", "Synthetic customers are sampled from each segment's statistical profile: a multivariate normal on (log-)amounts and counts, plus per-segment frequencies for flags and categories. No real rows are copied. The model then re-scores them and reports fidelity — the share assigned back to their intended segment."),
]

SITE = [
    Entry("Overview page", "overview home page 3d skyline galaxy towers hero kpi coverflow cards", "The Overview page shows headline KPIs, a 3D Skyline (tower height = revenue share, width = customer share; tap a tower to open it) with a Galaxy view of every customer, swipeable segment cards, customer-vs-revenue chart, 2D segment map and the summary table.", [("Open Overview", "#/overview")]),
    Entry("Segment profiles page", "segment profiles page tabs traits channel category radar heatmap download", "Segment profiles shows one segment at a time: metrics vs. the average, defining traits, channel and category mix, recommended actions, KPIs, guardrail, a radar comparing all segments, the z-score fingerprint and a CSV download of that segment's customers.", [("Open Segment profiles", "#/segments")]),
    Entry("Explorer page", "explorer page filter compare distribution scatter demographics download", "The Explorer lets you pick segments, compare the distribution of any feature, plot any two features against each other, see the demographic mix and download all customers.", [("Open Explorer", "#/explorer")]),
    Entry("Assign a customer page", "assign customer page form new customer score predict batch scoring upload", "Assign a customer: fill in a profile to get the nearest segment, fit margin, distance to every segment and next best actions. Batch scoring there accepts a CSV, Excel or JSON file and returns every row with its segment.", [("Open Assign", "#/assign")]),
    Entry("Model & methodology page", "methodology diagnostics page model card metrics cleaning skewness", "Model & methodology shows the k-selection charts and table, algorithm comparison, skewness before/after, the cleaning audit, the fairness proxy check and the full model card.", [("Open Methodology", "#/diagnostics")]),
    Entry("Data agent", "agent chatbot assistant what can you do capabilities help", "I answer questions about the site, segments, methods and metrics; build filtered customer lists, synthetic data, summary tables and an upload template; analyse, score or train on files you attach (CSV, Excel or JSON); find segments in any dataset using all of its columns; and publish or reset the live model. When the site owner connects an AI model, I also answer general questions like any AI assistant, using your workspace as context."),
    Entry("Chat history", "chat history previous chats old conversations past data saved datasets downloads earlier", "Press History in the chat header to reopen any earlier conversation (you can continue it) or switch to the Data tab to re-download every dataset I produced. History is saved only in this browser; delete single chats or clear everything there. Scored and segmented files stay downloadable for 24 hours."),
    Entry("Training your own model", "train retrain own data upload file draft how to train new model", "Attach a CSV, Excel or JSON file with 📎, choose 'Train model' (optionally a number of segments) and send. I train a draft and compare it with the live model; nothing changes on the site until you press Publish. You can also type 'train with 5 segments' to retrain on the built-in data. Limits: 20,000 rows, 5 MB, a few runs per 10 minutes."),
    Entry("Publishing and resetting", "publish reset restore rollback original live draft make live undo", "Publish replaces the live model for the whole website (every page updates). 'Reset to original' restores the original model at any time. Drafts and scored files expire after 24 hours; on Railway a redeploy also returns to the original model unless a volume is attached. Your chat history stays in your browser (open it with the History button)."),
    Entry("Accepted file formats", "file formats csv excel xlsx xls json jsonl upload columns template schema accepted", "I accept CSV/TSV, Excel (.xlsx, .xls — first sheet) and JSON (a list of records, an object with a 'data' list, or JSON Lines), up to 5 MB. Common column names are mapped automatically (e.g. 'customer_id'→ID, 'birth year'→Year_Birth). Ask for the 'upload template' to get the exact schema."),
    Entry("File actions", "analyze analyse score assign segments uploaded file what happens", "After attaching a file choose: Analyze (data-quality report: rows, missing columns and values, duplicates, compatibility), Assign segments (scores every row and gives you a download with Segment and Segment_Name), or Train model (creates a draft you can publish), or Find segments (all columns): works on any dataset, uses every usable column, picks the number of segments from the data and labels every row. The enrolment date (Dt_Customer) is optional."),
    Entry("Theme and mobile", "dark mode light theme auto mobile phone responsive", "Use the Light / Dark / Auto switch in the top bar; Auto follows your device setting. The site is designed mobile-first: menu collapses, cards swipe and forms stack on phones."),
    Entry("API", "api rest endpoints docs developers integrate crm openapi swagger", "Everything on the site is available through a versioned REST API under /api/v1 (health, overview, segments, explorer, assign, batch scoring, datasets, agent). Interactive docs are at /docs.", [("API docs", "/docs")]),
    Entry("Security and privacy", "security privacy safe data stored uploaded files https rate limit", "The site runs over HTTPS with strict security headers, input validation, upload size limits, rate limiting and CSV-injection protection. Uploaded files are processed in memory; only trained drafts, published models and scored results are stored temporarily on the server."),
    Entry("Deployment and tech stack", "deploy deployment railway docker hosting stack technology fastapi three.js python", "The site is a FastAPI (Python) app with scikit-learn, served from a Docker image on Railway. The front end is plain HTML/CSS/JavaScript with Chart.js and Three.js, all self-hosted. The model is trained during the Docker build."),
]

GLOSSARY_ENTRIES = [Entry(f"{term}", f"{term.lower()} meaning definition what is", f"{term}: {definition}") for term, definition in GLOSSARY.items()]
ALL_ENTRIES = [*METRICS, *METHODS, *SITE, *GLOSSARY_ENTRIES]

SUPERLATIVE_METRICS = {
    "income": ("Median_Income", _money, "median income"),
    "rich": ("Median_Income", _money, "median income"),
    "spend": ("Avg_Total_Spend", _money, "average spend"),
    "revenue": ("Revenue_Share_%", _pct, "share of revenue"),
    "valuable": ("Revenue_Share_%", _pct, "share of revenue"),
    "customers": ("Customer_Share_%", _pct, "share of customers"),
    "biggest": ("Customer_Share_%", _pct, "share of customers"),
    "largest": ("Customer_Share_%", _pct, "share of customers"),
    "smallest": ("Customer_Share_%", _pct, "share of customers"),
    "purchase": ("Avg_Purchases", lambda v: f"{v:.1f}", "average purchases"),
    "frequent": ("Avg_Purchases", lambda v: f"{v:.1f}", "average purchases"),
    "order value": ("Avg_Order_Value", _money, "average order value"),
    "aov": ("Avg_Order_Value", _money, "average order value"),
    "basket": ("Avg_Order_Value", _money, "average order value"),
    "deal": ("Deal_Ratio_%", _pct, "deal ratio"),
    "discount": ("Deal_Ratio_%", _pct, "deal ratio"),
    "campaign": ("Campaign_Acceptance", lambda v: f"{v:.2f}", "campaigns accepted"),
    "web visit": ("Avg_Web_Visits", lambda v: f"{v:.1f}", "web visits per month"),
    "recen": ("Avg_Recency_Days", lambda v: f"{v:.0f} days", "days since last purchase"),
    "age": ("Avg_Age", lambda v: f"{v:.0f}", "average age"),
    "older": ("Avg_Age", lambda v: f"{v:.0f}", "average age"),
    "younger": ("Avg_Age", lambda v: f"{v:.0f}", "average age"),
    "children": ("Avg_Children", lambda v: f"{v:.2f}", "children per household"),
    "kids": ("Avg_Children", lambda v: f"{v:.2f}", "children per household"),
    "tenure": ("Avg_Tenure_Days", lambda v: f"{v:.0f} days", "tenure"),
}
LOW_WORDS = ("lowest", "least", "smallest", "fewest", "minimum", "poorest", "cheapest", "younger", "youngest", "worst")
HIGH_WORDS = ("highest", "most", "largest", "biggest", "maximum", "richest", "best", "top", "older", "oldest", "greatest", "more")


# ---------------------------------------------------------------- dynamic handlers
def describe_segment(service: SegmentationService, seg: int) -> str:
    p = service.segmenter.profiles_.loc[seg]
    rec = next(r for r in service.segmenter.recommendations_ if r["segment"] == seg)
    return (
        f"{rec['name']}: {rec['summary']} They are {_pct(p['Customer_Share_%'])} of customers and generate {_pct(p['Revenue_Share_%'])} of revenue. "
        f"Median income {_money(p['Median_Income'])}, average spend {_money(p['Avg_Total_Spend'])}, {p['Avg_Purchases']:.1f} purchases, "
        f"deal ratio {_pct(p['Deal_Ratio_%'])}, {p['Campaign_Acceptance']:.2f} campaigns accepted, preferred channel {rec['preferred_channel']}.\n"
        f"Stands out on: {', '.join(rec['defining_high_traits']) or '—'}. Below average on: {', '.join(rec['defining_low_traits']) or '—'}.\n"
        f"Goal: {rec['goal']}. Top actions: " + " ".join(f"({i}) {a}" for i, a in enumerate(rec["actions"][:3], 1))
        + f"\nGuardrail: {rec['responsible_use']}"
    )


def compare_segments(service: SegmentationService, segs: list[int]) -> str:
    p = service.segmenter.profiles_
    lines = []
    for col, label, fmt in (
        ("Customer_Share_%", "Customers", _pct),
        ("Revenue_Share_%", "Revenue", _pct),
        ("Median_Income", "Median income", _money),
        ("Avg_Total_Spend", "Avg spend", _money),
        ("Avg_Purchases", "Purchases", lambda v: f"{v:.1f}"),
        ("Deal_Ratio_%", "Deal ratio", _pct),
        ("Campaign_Acceptance", "Campaigns", lambda v: f"{v:.2f}"),
        ("Avg_Web_Visits", "Web visits", lambda v: f"{v:.1f}"),
    ):
        lines.append(f"• {label}: " + " vs ".join(f"{service.names[s]} {fmt(p.loc[s, col])}" for s in segs))
    return "Comparison:\n" + "\n".join(lines)


def superlative(text: str, service: SegmentationService) -> str | None:
    if not re.search(r"\b(which|what|who)\b.*\b(segment|group|cluster|customers)\b|\b(segment|group)\b.*\b(highest|lowest|most|least)", text):
        return None
    metric = next((v for k, v in SUPERLATIVE_METRICS.items() if k in text), None)
    if metric is None:
        return None
    low = any(w in text for w in LOW_WORDS)
    high = any(w in text for w in HIGH_WORDS)
    if not (low or high):
        return None
    col, fmt, label = metric
    p = service.segmenter.profiles_
    seg = p[col].idxmin() if low else p[col].idxmax()
    ranking = p[col].sort_values(ascending=low)
    ranked = "; ".join(f"{service.names[s]} {fmt(v)}" for s, v in ranking.items())
    return f"{service.names[seg]} has the {'lowest' if low else 'highest'} {label}: {fmt(p.loc[seg, col])}. Full ranking: {ranked}."


def segment_metric(text: str, service: SegmentationService, seg: int) -> str | None:
    metric = next((v for k, v in SUPERLATIVE_METRICS.items() if k in text and k not in ("biggest", "largest", "smallest", "customers")), None)
    if metric is None or not re.search(r"\b(average|avg|mean|median|how much|how many|what is|what's|typical)\b", text):
        return None
    col, fmt, label = metric
    p = service.segmenter.profiles_
    value = p.loc[seg, col]
    overall = (p[col] * p["Customer_Share_%"]).sum() / p["Customer_Share_%"].sum()
    return f"{service.names[seg]}: {label} is {fmt(value)} (customer-weighted average across all segments: {fmt(overall)})."


def counts(text: str, service: SegmentationService, segs: list[int]) -> str | None:
    if not re.search(r"\bhow many\b", text):
        return None
    if re.search(r"\b(segments|clusters|groups)\b", text) and not segs:
        return f"There are {service.segmenter.n_segments} segments: " + ", ".join(service.names[s] for s in service.segment_ids) + "."
    if re.search(r"\b(customers|people|rows|records)\b", text):
        if segs:
            p = service.segmenter.profiles_
            return " ".join(f"{service.names[s]} has {int(p.loc[s, 'Customers']):,} customers ({_pct(p.loc[s, 'Customer_Share_%'])})." for s in segs)
        return f"The model covers {len(service.customers):,} customers after cleaning ({service.segmenter.n_raw_rows_:,} rows in the raw file)."
    return None


# ---------------------------------------------------------------- retrieval
class KnowledgeBase:
    def __init__(self, service: SegmentationService):
        self.service = service
        self.entries = ALL_ENTRIES
        docs = [f"{e.title} {e.keywords} " * 3 + e.text(service) for e in self.entries]
        self.vectorizer = TfidfVectorizer(analyzer=analyze_text, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top: int = 3) -> list[tuple[Entry, float]]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(scores)[::-1][:top]
        return [(self.entries[i], float(scores[i])) for i in order]


_KB_CACHE: dict[int, KnowledgeBase] = {}


def knowledge_base(service: SegmentationService) -> KnowledgeBase:
    kb = _KB_CACHE.get(id(service))
    if kb is None or kb.service is not service:
        kb = KnowledgeBase(service)
        _KB_CACHE.clear()
        _KB_CACHE[id(service)] = kb
    return kb


def answer_question(text: str, service: SegmentationService, segments: list[int], min_score: float = MIN_SIMILARITY) -> dict | None:
    """Return {'reply', 'links', 'suggestions'} or None when no confident answer exists."""
    lowered = text.lower()
    if re.search(r"\bwhy\b", lowered) and re.search(r"\b(\d+|k|segments?|clusters?|groups?)\b", lowered) and not segments:
        return {"reply": _k_answer(service), "links": [("Model & methodology", "#/diagnostics")], "suggestions": ["What is the silhouette score?", "What is bootstrap stability?"]}
    definition = re.search(r"\b(mean|meaning|define|definition|stand for|stands for)\b", lowered)
    dynamic = None if definition else superlative(lowered, service) or counts(lowered, service, segments)
    if dynamic is None and segments and not definition:
        if len(segments) >= 2 and re.search(r"\b(compare|comparison|vs|versus|difference|differ|between)\b", lowered):
            dynamic = compare_segments(service, segments[:4])
        else:
            dynamic = segment_metric(lowered, service, segments[0]) or describe_segment(service, segments[0])
    if dynamic is not None:
        links = [("Segment profiles", f"#/segments/{segments[0]}")] if segments else [("Overview", "#/overview")]
        return {"reply": dynamic, "links": links, "suggestions": [f"Give me {service.names[segments[0]].split()[0].lower()} customers"] if segments else []}

    hits = knowledge_base(service).search(lowered)
    best, score = hits[0]
    if score < min_score:
        return None
    related = [e.title for e, s in hits[1:] if s >= MIN_SIMILARITY * 0.8]
    return {
        "reply": best.text(service),
        "links": best.links,
        "suggestions": [f"Tell me about {t.lower()}" for t in related],
    }
