import altair as alt
import streamlit as st

from dashboard_utils import get_segmenter

segmenter = get_segmenter()
ks = segmenter.k_selection_
metrics = ks.metrics

st.info(ks.rationale, icon=":material/rule:")

with st.container(horizontal=True):
    st.metric("Selected k", segmenter.n_segments, border=True)
    st.metric("Elbow k", ks.elbow_k, border=True)
    st.metric("Best raw silhouette k", ks.best_silhouette_k, border=True)
    st.metric("Final silhouette", f"{segmenter.silhouette_:.3f}", border=True)

panels = [
    ("inertia", "Elbow: within-cluster SSE"),
    ("silhouette", "Silhouette (higher = better)"),
    ("davies_bouldin", "Davies-Bouldin (lower = better)"),
    ("stability_ari", "Bootstrap stability ARI"),
]
rule = alt.Chart(metrics[metrics["recommended"]]).mark_rule(color="#fc8d62", strokeDash=[6, 4]).encode(x="k:Q")
for row in (panels[:2], panels[2:]):
    cols = st.columns(2)
    for col, (metric, title) in zip(cols, row):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                line = (
                    alt.Chart(metrics)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("k:Q", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)),
                        tooltip=["k", alt.Tooltip(f"{metric}:Q", format=".3f")],
                    )
                )
                st.altair_chart((line + rule).properties(height=220))

with st.container(border=True):
    st.subheader("Candidate k scores", anchor=False)
    st.dataframe(metrics.round(3), hide_index=True)

with st.container(border=True):
    st.subheader("Algorithm comparison at selected k", anchor=False)
    st.caption("K-Means is deployed: best separation metrics and it can assign new customers to a segment.")
    st.dataframe(segmenter.algorithm_comparison_, hide_index=True)

clean_col, skew_col = st.columns(2)
with clean_col:
    with st.container(border=True):
        st.subheader("Cleaning audit", anchor=False)
        st.dataframe(segmenter.cleaning_report_.to_frame(), hide_index=True)
with skew_col:
    with st.container(border=True):
        st.subheader("Skewness before / after", anchor=False)
        st.dataframe(segmenter.skewness_report_)

with st.container(border=True):
    st.subheader("Responsible AI: demographic proxy check", anchor=False)
    st.caption(
        "Cramér's V between segment membership and demographics that were NOT used for clustering. "
        "V >= 0.3 means the segment could act as a proxy; review messaging to avoid indirect targeting."
    )
    st.dataframe(segmenter.proxy_check_, hide_index=True)

with st.expander("Model card"):
    st.json(segmenter.model_card())
