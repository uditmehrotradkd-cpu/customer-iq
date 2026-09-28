import altair as alt
import streamlit as st

from dashboard_utils import color_scale, get_segmenter, pca_frame

segmenter = get_segmenter()
profiles = segmenter.profiles_.reset_index()
customers = segmenter.customers_

st.caption(
    "Behavioural segments discovered with K-Means on spending, frequency, recency, channel and "
    "campaign-engagement features. Demographics are used only to describe segments."
)

with st.container(horizontal=True):
    st.metric("Customers segmented", f"{len(customers):,}", border=True)
    st.metric("Segments", segmenter.n_segments, border=True)
    st.metric("Silhouette score", f"{segmenter.silhouette_:.3f}", border=True)
    st.metric("Total spend (2 yrs)", f"${customers['Total_Spend'].sum():,.0f}", border=True)

left, right = st.columns([1, 1.3])
with left:
    with st.container(border=True):
        st.subheader("Customers vs. revenue share", anchor=False)
        melted = profiles.melt(
            id_vars="Segment_Name",
            value_vars=["Customer_Share_%", "Revenue_Share_%"],
            var_name="Metric",
            value_name="Percent",
        )
        melted["Metric"] = melted["Metric"].map({"Customer_Share_%": "Customers", "Revenue_Share_%": "Revenue"})
        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                y=alt.Y("Segment_Name:N", title=None, sort=list(profiles["Segment_Name"])),
                x=alt.X("Percent:Q", title="% of total"),
                yOffset="Metric:N",
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#8da0cb", "#fc8d62"])),
                tooltip=["Segment_Name", "Metric", alt.Tooltip("Percent:Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart)

with right:
    with st.container(border=True):
        st.subheader("Segments in PCA space", anchor=False)
        frame, variance = pca_frame(segmenter)
        scatter = (
            alt.Chart(frame)
            .mark_circle(size=35, opacity=0.6)
            .encode(
                x=alt.X("PC1:Q", title=f"PC1 ({variance[0]:.0%} variance)"),
                y=alt.Y("PC2:Q", title=f"PC2 ({variance[1]:.0%} variance)"),
                color=alt.Color("Segment_Name:N", title="Segment", scale=color_scale(segmenter)),
                tooltip=[
                    "ID",
                    "Segment_Name",
                    alt.Tooltip("Income:Q", format=",.0f"),
                    alt.Tooltip("Total_Spend:Q", format=",.0f"),
                    "Total_Purchases",
                    alt.Tooltip("Deal_Ratio:Q", format=".0%"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(scatter)

with st.container(border=True):
    st.subheader("Segment summary", anchor=False)
    st.dataframe(
        profiles,
        hide_index=True,
        column_config={
            "Segment": st.column_config.NumberColumn("ID", width="small"),
            "Segment_Name": st.column_config.TextColumn("Segment", width="medium"),
            "Customer_Share_%": st.column_config.ProgressColumn("Customers %", format="%.1f%%", min_value=0, max_value=100),
            "Revenue_Share_%": st.column_config.ProgressColumn("Revenue %", format="%.1f%%", min_value=0, max_value=100),
            "Median_Income": st.column_config.NumberColumn("Median income", format="dollar"),
            "Avg_Total_Spend": st.column_config.NumberColumn("Avg spend", format="dollar"),
            "Avg_Order_Value": st.column_config.NumberColumn("AOV", format="dollar"),
        },
    )
