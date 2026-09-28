"""Customer segmentation dashboard. Run with: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Customer segmentation", page_icon=":material/groups:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:"),
        st.Page("app_pages/profiles.py", title="Segment profiles", icon=":material/groups:"),
        st.Page("app_pages/explorer.py", title="Explorer", icon=":material/query_stats:"),
        st.Page("app_pages/assign.py", title="Assign a customer", icon=":material/person_add:"),
        st.Page("app_pages/diagnostics.py", title="Model diagnostics", icon=":material/science:"),
    ],
    position="top",
)
st.title(page.title, anchor=False)
page.run()
