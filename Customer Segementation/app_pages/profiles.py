import altair as alt
import pandas as pd
import streamlit as st

from dashboard_utils import SEGMENT_COLORS, get_segmenter

segmenter = get_segmenter()
names = segmenter.segment_names_
recommendations = {r["segment"]: r for r in segmenter.recommendations_}
customers = segmenter.customers_
options = sorted(names)

selected = st.segmented_control(
    "Segment",
    options,
    format_func=lambda s: names[s],
    default=options[0],
    key="profile_segment",
)
if selected is None:
    selected = options[0]

rec = recommendations[selected]
segment_rows = customers[customers["Segment"] == selected]

with st.container(border=True):
    st.subheader(names[selected], anchor=False)
    st.markdown(rec["summary"])
    st.badge(f"Goal: {rec['goal']}", icon=":material/flag:", color="blue")

metrics = [
    ("Share of customers", len(segment_rows) / len(customers), None, "{:.1%}"),
    ("Median income", segment_rows["Income"].median(), customers["Income"].median(), "${:,.0f}"),
    ("Avg spend", segment_rows["Total_Spend"].mean(), customers["Total_Spend"].mean(), "${:,.0f}"),
    ("Avg purchases", segment_rows["Total_Purchases"].mean(), customers["Total_Purchases"].mean(), "{:.1f}"),
    ("Avg order value", segment_rows["Avg_Order_Value"].mean(), customers["Avg_Order_Value"].mean(), "${:,.1f}"),
    ("Deal ratio", segment_rows["Deal_Ratio"].mean(), customers["Deal_Ratio"].mean(), "{:.0%}"),
    ("Campaigns accepted", segment_rows["Campaigns_Accepted"].mean(), customers["Campaigns_Accepted"].mean(), "{:.2f}"),
]
with st.container(horizontal=True):
    for label, value, baseline, fmt in metrics:
        delta = f"{value / baseline - 1:+.0%} vs. avg" if baseline else None
        st.metric(label, fmt.format(value), delta, delta_color="off", border=True)

traits_col, channel_col, category_col = st.columns(3)
with traits_col:
    with st.container(border=True, height="stretch"):
        st.markdown("**What defines this segment**")
        for trait in rec["defining_high_traits"]:
            st.markdown(f":material/arrow_upward: :green[{trait}]")
        for trait in rec["defining_low_traits"]:
            st.markdown(f":material/arrow_downward: :red[{trait}]")
        st.caption(f"Preferred channel: **{rec['preferred_channel']}** · Over-indexed categories: "
                   f"**{', '.join(rec['over_indexed_categories'])}**")

color = SEGMENT_COLORS[selected]
with channel_col:
    with st.container(border=True):
        st.markdown("**Purchase channel mix**")
        channels = segmenter.channel_mix_.loc[selected].rename("Percent").rename_axis("Channel").reset_index()
        st.altair_chart(
            alt.Chart(channels)
            .mark_bar(color=color)
            .encode(x=alt.X("Percent:Q", title="% of purchases"), y=alt.Y("Channel:N", title=None, sort="-x"))
            .properties(height=180)
        )
with category_col:
    with st.container(border=True):
        st.markdown("**Spend by category**")
        categories = segmenter.category_mix_.loc[selected].rename("Percent").rename_axis("Category").reset_index()
        st.altair_chart(
            alt.Chart(categories)
            .mark_bar(color=color)
            .encode(x=alt.X("Percent:Q", title="% of spend"), y=alt.Y("Category:N", title=None, sort="-x"))
            .properties(height=180)
        )

actions_col, guard_col = st.columns([2, 1])
with actions_col:
    with st.container(border=True, height="stretch"):
        st.markdown("**Recommended engagement actions**")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(rec["actions"], start=1)))
        st.markdown("**KPIs to track:** " + ", ".join(rec["kpis"]))
with guard_col:
    with st.container(border=True, height="stretch"):
        st.markdown("**Responsible-use guardrail**")
        st.warning(rec["responsible_use"], icon=":material/shield:")

with st.container(border=True):
    st.subheader("Segment fingerprint comparison", anchor=False)
    st.caption("Standard deviations above (red) or below (blue) the average customer.")
    z = segmenter.zprofile_.copy()
    z.index = [names[s] for s in z.index]
    long = z.reset_index(names="Segment").melt(id_vars="Segment", var_name="Feature", value_name="Z")
    heatmap = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("Feature:N", sort=list(z.columns), title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Segment:N", title=None),
            color=alt.Color("Z:Q", scale=alt.Scale(scheme="redblue", reverse=True, domainMid=0), title="z-score"),
            tooltip=["Segment", "Feature", alt.Tooltip("Z:Q", format=".2f")],
        )
        .properties(height=60 * len(z) + 40)
    )
    text = heatmap.mark_text(fontSize=10).encode(text=alt.Text("Z:Q", format=".1f"), color=alt.value("black"))
    st.altair_chart(heatmap + text)

with st.expander("Segment index table (100 = average customer)"):
    index_table = segmenter.index_table_.copy()
    index_table.index = pd.Index([names[s] for s in index_table.index], name="Segment")
    st.dataframe(index_table)
