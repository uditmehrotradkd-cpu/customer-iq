import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from dashboard_utils import color_scale, get_segmenter
from segmentation.config import CAMPAIGN_COLUMNS, EDUCATION_MAP, MARITAL_STATUS_MAP, SPEND_COLUMNS
from segmentation.data_loader import to_canonical_schema

MAX_UPLOAD_ROWS = 50_000

segmenter = get_segmenter()
customers = segmenter.customers_
median = customers.median(numeric_only=True)
reference_date = segmenter.reference_date_.date()

st.caption("Enter a customer's profile and transactions to place them in the closest segment.")

with st.form("assign_form"):
    profile_col, spend_col, activity_col = st.columns(3)
    with profile_col:
        st.markdown("**Profile**")
        year_birth = st.number_input("Year of birth", 1920, reference_date.year - 16, int(median["Year_Birth"]))
        income = st.number_input("Annual household income ($)", 0, 200_000, int(median["Income"]), step=1_000)
        kidhome = st.number_input("Children at home", 0, 5, 0)
        teenhome = st.number_input("Teenagers at home", 0, 5, 0)
        education = st.selectbox("Education", list(EDUCATION_MAP), index=list(EDUCATION_MAP).index("Graduation"))
        marital_options = sorted(set(MARITAL_STATUS_MAP.values()))
        marital = st.selectbox("Marital status", marital_options, index=marital_options.index("Married"))
        enrolled = st.date_input(
            "Customer since",
            value=reference_date - dt.timedelta(days=int(median["Tenure_Days"])),
            max_value=reference_date,
        )
    with spend_col:
        st.markdown("**Spend in last 2 years ($)**")
        spend = {
            col: st.number_input(col.replace("Mnt", "").replace("Prods", "").replace("Products", ""), 0, 5_000, int(median[col]))
            for col in SPEND_COLUMNS
        }
    with activity_col:
        st.markdown("**Activity**")
        web = st.number_input("Web purchases", 0, 50, int(median["NumWebPurchases"]))
        catalog = st.number_input("Catalog purchases", 0, 50, int(median["NumCatalogPurchases"]))
        store = st.number_input("Store purchases", 0, 50, int(median["NumStorePurchases"]))
        deals = st.number_input("Purchases made with a discount", 0, 50, int(median["NumDealsPurchases"]))
        visits = st.number_input("Web visits last month", 0, 30, int(median["NumWebVisitsMonth"]))
        recency = st.slider("Days since last purchase", 0, 120, int(median["Recency"]))
        accepted = st.pills("Campaigns accepted", list(CAMPAIGN_COLUMNS), selection_mode="multi", default=[])
        complain = st.checkbox("Complained in last 2 years")
    submitted = st.form_submit_button("Assign segment", type="primary", icon=":material/person_search:")

if submitted:
    record = {
        "Year_Birth": year_birth,
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Recency": recency,
        **spend,
        "NumDealsPurchases": deals,
        "NumWebPurchases": web,
        "NumCatalogPurchases": catalog,
        "NumStorePurchases": store,
        "NumWebVisitsMonth": visits,
        **{c: int(c in (accepted or [])) for c in CAMPAIGN_COLUMNS},
        "Complain": int(complain),
        "Education": education,
        "Marital_Status": marital,
        "Dt_Customer": pd.Timestamp(enrolled),
    }
    if web + catalog + store == 0 and sum(spend.values()) > 0:
        st.warning("Spend was entered with zero purchases; the record is inconsistent.", icon=":material/warning:")

    result = segmenter.assign_customer(record)
    rec = result["recommendation"]
    margin = result["assignment_margin"]
    strength = "clear" if margin >= 0.25 else "moderate" if margin >= 0.1 else "borderline"

    with st.container(border=True):
        st.subheader(f"Assigned segment: {result['segment_name']}", anchor=False)
        st.markdown(rec["summary"])
        with st.container(horizontal=True):
            st.metric("Assignment margin", f"{margin:.0%}", border=True, help="1 - nearest / second-nearest centroid distance")
            st.metric("Fit", strength.capitalize(), border=True)
            st.metric("Preferred channel (segment)", rec["preferred_channel"], border=True)

    dist_col, feat_col = st.columns(2)
    with dist_col:
        with st.container(border=True):
            st.markdown("**Distance to each segment centroid** (lower = closer)")
            distances = pd.DataFrame(result["distances"].items(), columns=["Segment_Name", "Distance"])
            st.altair_chart(
                alt.Chart(distances)
                .mark_bar()
                .encode(
                    x="Distance:Q",
                    y=alt.Y("Segment_Name:N", title=None, sort="x"),
                    color=alt.Color("Segment_Name:N", scale=color_scale(segmenter), legend=None),
                )
                .properties(height=220)
            )
    with feat_col:
        with st.container(border=True):
            st.markdown("**This customer vs. segment average**")
            seg_avg = customers.loc[customers["Segment"] == result["segment"], list(result["engineered_features"])].mean()
            comparison = pd.DataFrame({"Customer": pd.Series(result["engineered_features"]), "Segment average": seg_avg.round(2)})
            st.dataframe(comparison)

    with st.container(border=True):
        st.markdown("**Next best actions**")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(rec["actions"], start=1)))
        st.warning(rec["responsible_use"], icon=":material/shield:")

st.divider()
st.subheader("Batch scoring", anchor=False)
st.caption("Upload a CSV in the raw or one-hot-encoded training schema to assign segments in bulk.")
upload = st.file_uploader("Customer CSV", type=["csv"])
if upload is not None:
    try:
        batch = pd.read_csv(upload, nrows=MAX_UPLOAD_ROWS + 1)
        if len(batch) > MAX_UPLOAD_ROWS:
            st.error(f"Please upload at most {MAX_UPLOAD_ROWS:,} rows.")
        else:
            batch = to_canonical_schema(batch)
            scored = pd.concat([batch, segmenter.predict(batch)], axis=1)
            st.dataframe(scored.head(200), hide_index=True)
            st.download_button(
                "Download scored customers",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_customers.csv",
                mime="text/csv",
                icon=":material/download:",
            )
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not score file: {exc}")
