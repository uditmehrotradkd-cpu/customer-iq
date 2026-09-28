import altair as alt
import streamlit as st

from dashboard_utils import color_scale, get_segmenter, segment_labels
from segmentation.config import PROFILE_CATEGORICAL_FEATURES, PROFILE_NUMERIC_FEATURES

segmenter = get_segmenter()
customers = segmenter.customers_
all_segments = segment_labels(segmenter)
numeric_features = [c for c in PROFILE_NUMERIC_FEATURES if c in customers.columns]

with st.container(border=True):
    chosen = st.pills("Segments", all_segments, selection_mode="multi", default=all_segments, key="explorer_segments")
    chosen = chosen or all_segments
    filtered = customers[customers["Segment_Name"].isin(chosen)]
    st.caption(f"{len(filtered):,} customers selected")

dist_col, scatter_col = st.columns(2)
with dist_col:
    with st.container(border=True):
        feature = st.selectbox("Distribution of", numeric_features, index=numeric_features.index("Total_Spend"))
        box = (
            alt.Chart(filtered)
            .mark_boxplot(extent="min-max", size=40)
            .encode(
                x=alt.X("Segment_Name:N", title=None, sort=all_segments, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y(f"{feature}:Q", title=feature),
                color=alt.Color("Segment_Name:N", scale=color_scale(segmenter), legend=None),
            )
            .properties(height=360)
        )
        st.altair_chart(box)

with scatter_col:
    with st.container(border=True):
        x_col, y_col = st.columns(2)
        x_feature = x_col.selectbox("X axis", numeric_features, index=numeric_features.index("Income"))
        y_feature = y_col.selectbox("Y axis", numeric_features, index=numeric_features.index("Total_Spend"))
        scatter = (
            alt.Chart(filtered)
            .mark_circle(size=30, opacity=0.55)
            .encode(
                x=alt.X(f"{x_feature}:Q"),
                y=alt.Y(f"{y_feature}:Q"),
                color=alt.Color("Segment_Name:N", title="Segment", scale=color_scale(segmenter)),
                tooltip=["ID", "Segment_Name", x_feature, y_feature],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(scatter)

with st.container(border=True):
    st.subheader("Descriptive demographics", anchor=False)
    st.caption("These attributes were not used to build segments; shown only to understand who is in each group.")
    attribute = st.segmented_control(
        "Attribute", PROFILE_CATEGORICAL_FEATURES, default="Age_Band", key="explorer_attribute"
    ) or "Age_Band"
    mix = (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            y=alt.Y("Segment_Name:N", title=None, sort=all_segments),
            x=alt.X("count():Q", stack="normalize", title="Share of segment", axis=alt.Axis(format="%")),
            color=alt.Color(f"{attribute}:N", title=attribute),
            tooltip=["Segment_Name", f"{attribute}:N", "count():Q"],
        )
        .properties(height=50 * len(chosen) + 40)
    )
    st.altair_chart(mix)

with st.container(border=True):
    st.subheader("Customer records", anchor=False)
    columns = ["ID", "Segment", "Segment_Name", *numeric_features, *PROFILE_CATEGORICAL_FEATURES]
    st.dataframe(filtered[columns], hide_index=True, height=320)
    st.download_button(
        "Download selection as CSV",
        filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="segmented_customers.csv",
        mime="text/csv",
        icon=":material/download:",
    )
