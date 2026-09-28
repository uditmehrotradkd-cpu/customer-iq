import sys, json, html, tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from customer_iq.config import Config
from customer_iq.sources import load_source, sqlite_tables
from customer_iq.train import train_dataframe
from customer_iq.predict import predict_customer
from customer_iq.business import STRATEGIES
from customer_iq.visuals import pca_scatter, segment_box, k_diagnostics

st.set_page_config(
    page_title="CustomerIQ — Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme / CSS -----------------------------
THEMES = {
    "Aurora": {
        "bg": "#070B16", "panel": "rgba(16,24,45,.72)", "text": "#F5F7FF",
        "muted": "#9AA8C7", "accent": "#7C5CFC", "accent2": "#00D4FF",
        "glow": "rgba(124,92,252,.45)"
    },
    "Cyber": {
        "bg": "#050807", "panel": "rgba(8,25,18,.74)", "text": "#E9FFF4",
        "muted": "#86A999", "accent": "#00F5A0", "accent2": "#00E5FF",
        "glow": "rgba(0,245,160,.38)"
    },
    "Sunset": {
        "bg": "#120812", "panel": "rgba(35,17,35,.76)", "text": "#FFF4F8",
        "muted": "#C6A6B7", "accent": "#FF5EA8", "accent2": "#FFB86B",
        "glow": "rgba(255,94,168,.42)"
    }
}
theme_name = st.sidebar.selectbox("🎨 Experience theme", list(THEMES), index=0)
T = THEMES[theme_name]

st.markdown(f"""
<style>
:root {{
 --bg:{T['bg']}; --panel:{T['panel']}; --text:{T['text']};
 --muted:{T['muted']}; --accent:{T['accent']}; --accent2:{T['accent2']};
}}
.stApp {{
 background:
 radial-gradient(circle at 10% 5%, {T['glow']} 0%, transparent 28%),
 radial-gradient(circle at 90% 15%, rgba(0,212,255,.18) 0%, transparent 30%),
 linear-gradient(135deg,var(--bg),#02040B 75%);
 color:var(--text);
}}
.block-container {{max-width:1450px;padding-top:1.5rem;}}
section[data-testid="stSidebar"] {{
 background:linear-gradient(180deg,rgba(5,8,18,.98),rgba(10,14,28,.96));
 border-right:1px solid rgba(255,255,255,.08);
}}
.hero {{
 padding:2.2rem 2.4rem; border-radius:28px;
 background:linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
 border:1px solid rgba(255,255,255,.12);
 box-shadow:0 25px 90px rgba(0,0,0,.35), inset 0 1px rgba(255,255,255,.08);
 backdrop-filter:blur(18px);
}}
.eyebrow {{letter-spacing:.18em;text-transform:uppercase;color:var(--accent2);font-size:.78rem;font-weight:800;}}
.hero h1 {{font-size:clamp(2.7rem,6vw,5.8rem);line-height:.95;margin:.45rem 0 1rem;
 background:linear-gradient(90deg,var(--text),var(--accent2),var(--accent));
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.hero p {{font-size:1.12rem;color:var(--muted);max-width:850px;}}
.pill {{
 display:inline-block;padding:.42rem .78rem;border-radius:999px;margin:.2rem;
 border:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.045);
 color:var(--text);font-size:.82rem;
}}
.card {{
 padding:1.1rem 1.25rem;border-radius:20px;background:var(--panel);
 border:1px solid rgba(255,255,255,.1);box-shadow:0 16px 50px rgba(0,0,0,.20);
}}
.kicker {{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.12em;}}
.value {{font-size:1.8rem;font-weight:800;color:var(--text);}}
.small {{color:var(--muted);font-size:.86rem;}}
.stButton > button {{
 border-radius:12px;border:1px solid rgba(255,255,255,.12);
 background:linear-gradient(90deg,var(--accent),var(--accent2)); color:white;font-weight:800;
}}
[data-testid="stMetric"] {{
 background:var(--panel);border:1px solid rgba(255,255,255,.1);
 padding:1rem;border-radius:18px;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------- State -----------------------------
cfg = Config()
cfg.ensure_dirs()
for key in ["raw_df", "artifact", "clustered", "metrics", "source_name"]:
    st.session_state.setdefault(key, None)

# ----------------------------- 3D hero -----------------------------
hero_html = f"""
<div class="hero">
  <div class="eyebrow">CUSTOMER INTELLIGENCE PLATFORM · {theme_name.upper()}</div>
  <h1>Turn customers<br/>into decisions.</h1>
  <p>
    A cinematic, interactive customer-segmentation cockpit combining RFM intelligence,
    unsupervised ML, 3D customer maps and real-time business recommendations.
  </p>
  <div>
    <span class="pill">RFM + Behavioral ML</span>
    <span class="pill">K-Means Intelligence</span>
    <span class="pill">3D PCA Explorer</span>
    <span class="pill">What-If Simulator</span>
    <span class="pill">CSV · Excel · SQLite</span>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Animated visual with no backend dependency.
st.components.v1.html(f"""
<!doctype html><html><body style="margin:0;background:transparent;overflow:hidden">
<div id="scene" style="height:250px;border-radius:26px;overflow:hidden"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script>
const box=document.getElementById('scene');
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(55,box.clientWidth/250,.1,100);
camera.position.z=6;
const renderer=new THREE.WebGLRenderer({{antialias:true,alpha:true}});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(box.clientWidth,250);
box.appendChild(renderer.domElement);
const group=new THREE.Group(); scene.add(group);
const geo=new THREE.SphereGeometry(.035,8,8);
const colors=['{T["accent"]}','{T["accent2"]}','#FFFFFF'];
for(let i=0;i<480;i++){{
 const m=new THREE.MeshBasicMaterial({{color:colors[i%3],transparent:true,opacity:.65}});
 const p=new THREE.Mesh(geo,m);
 const r=1.1+Math.random()*2.4, a=Math.random()*Math.PI*2, y=(Math.random()-.5)*3;
 p.position.set(Math.cos(a)*r,y,Math.sin(a)*r);
 group.add(p);
}}
const ring=new THREE.Mesh(
 new THREE.TorusGeometry(1.65,.012,8,160),
 new THREE.MeshBasicMaterial({{color:'{T["accent2"]}',transparent:true,opacity:.35}})
);
ring.rotation.x=Math.PI/2.4; group.add(ring);
function animate(t){{
 requestAnimationFrame(animate);
 group.rotation.y=t*.00012; group.rotation.x=Math.sin(t*.00025)*.08;
 ring.rotation.z=t*.0002; renderer.render(scene,camera);
}}
animate(0);
window.addEventListener('resize',()=>{{renderer.setSize(box.clientWidth,250);camera.aspect=box.clientWidth/250;camera.updateProjectionMatrix();}});
</script></body></html>
""", height=265)

# ----------------------------- Data source -----------------------------
with st.sidebar:
    st.header("DATA LAB")
    source = st.radio("Source", ["CSV", "Excel", "SQLite"])
    raw = None
    source_name = None

    if source == "CSV":
        f = st.file_uploader("Upload customer CSV", type=["csv"])
        if f:
            raw = load_source("CSV", file_bytes=f.getvalue())
            source_name = f.name
    elif source == "Excel":
        f = st.file_uploader("Upload Excel workbook", type=["xlsx","xls"])
        sheet = st.text_input("Sheet", "0")
        if f:
            sheet_arg = int(sheet) if sheet.isdigit() else sheet
            raw = load_source("Excel", file_bytes=f.getvalue(), sheet_name=sheet_arg)
            source_name = f"{f.name} / {sheet}"
    else:
        f = st.file_uploader("Upload SQLite database", type=["db","sqlite","sqlite3"])
        if f:
            tmp = Path(tempfile.NamedTemporaryFile(delete=False,suffix=".db").name)
            tmp.write_bytes(f.getvalue())
            tables = sqlite_tables(tmp)
            table = st.selectbox("Table", tables)
            query = st.text_area("Optional SQL query", "")
            raw = load_source("SQLite", path=tmp, table=table, query=query or None)
            source_name = f"{f.name} / {table}"

    if raw is not None:
        st.session_state.raw_df = raw
        st.session_state.source_name = source_name
        st.success(f"{len(raw):,} rows loaded")

    if st.session_state.raw_df is not None:
        st.divider()
        st.header("MODEL LAB")
        min_k = st.slider("Min K", 2, 12, 2)
        max_k = st.slider("Max K", 2, 12, 8)
        if st.button("⚡ TRAIN INTELLIGENCE", use_container_width=True):
            if min_k > max_k:
                st.error("Min K must be ≤ Max K")
            else:
                with st.spinner("Learning customer behavior..."):
                    try:
                        cfg.min_k, cfg.max_k = min_k, max_k
                        art, cl, met = train_dataframe(
                            st.session_state.raw_df,
                            st.session_state.source_name or "dataset", cfg
                        )
                        st.session_state.artifact = art
                        st.session_state.clustered = cl
                        st.session_state.metrics = met
                        st.success("Model trained.")
                    except Exception as e:
                        st.exception(e)

clustered = st.session_state.clustered
artifact = st.session_state.artifact
metrics = st.session_state.metrics

if clustered is None:
    st.markdown("### 🚀 Your mission")
    cols = st.columns(3)
    for col, title, body in zip(cols,
        ["1 · Connect", "2 · Discover", "3 · Activate"],
        ["Upload CSV, Excel or SQLite.", "Train ML and explore 3D customer clusters.", "Turn segments into targeted actions."]):
        col.markdown(f'<div class="card"><div class="kicker">{title}</div><h3>{body}</h3><div class="small">Designed for live hackathon demos.</div></div>', unsafe_allow_html=True)
    st.stop()

# ----------------------------- KPIs -----------------------------
k = artifact["n_clusters"]
sil = float(metrics.loc[metrics.K == k, "Silhouette"].iloc[0])
c = st.columns(5)
c[0].metric("CUSTOMERS", f"{len(clustered):,}")
c[1].metric("SEGMENTS", k)
c[2].metric("SILHOUETTE", f"{sil:.3f}")
c[3].metric("TOTAL VALUE", f"{clustered.Monetary.sum():,.0f}")
c[4].metric("SOURCE", st.session_state.source_name or "Dataset")

tabs = st.tabs(["🌌 3D COMMAND CENTER", "🧬 SEGMENT DNA", "🎯 LIVE CLASSIFIER", "📈 MODEL PROOF"])

with tabs[0]:
    st.subheader("Customer Galaxy")
    st.caption("Every point is a customer. Distance reflects learned behavioral similarity.")
    st.plotly_chart(pca_scatter(clustered), use_container_width=True)
    st.subheader("Segment scale")
    st.bar_chart(clustered.Persona.value_counts())

with tabs[1]:
    st.subheader("Segment DNA")
    profile_cols = [c for c in ["Recency","Frequency","Monetary","AverageOrderValue"] if c in clustered]
    st.dataframe(clustered.groupby("Persona")[profile_cols].agg(["mean","median","count"]).round(2), use_container_width=True)
    persona = st.selectbox("Choose a persona", sorted(clustered.Persona.unique()))
    st.markdown(f'<div class="card"><div class="kicker">PLAYBOOK</div><h2>{html.escape(persona)}</h2><p>{html.escape(str(STRATEGIES.get(persona, {})))}</p></div>', unsafe_allow_html=True)
    for feature in ["Recency","Frequency","Monetary"]:
        if feature in clustered:
            st.plotly_chart(segment_box(clustered, feature), use_container_width=True)

with tabs[2]:
    st.subheader("Live Customer Simulator")
    st.caption("Enter a customer's behavior and watch the model classify them instantly.")
    a,b,c = st.columns(3)
    rec = a.number_input("Recency (days)", 0.0, 5000.0, 30.0)
    freq = b.number_input("Purchase frequency", 1.0, 1000.0, 10.0)
    mon = c.number_input("Monetary value", 0.0, 1e9, 1000.0)
    if st.button("🎯 CLASSIFY CUSTOMER", use_container_width=True):
        result = predict_customer(rec, freq, mon, artifact)
        st.success(f"Assigned persona: {result['persona']}")
        st.json(result)

with tabs[3]:
    st.subheader("Why should I trust the clustering?")
    st.dataframe(metrics.round(4), use_container_width=True)
    e,s = k_diagnostics(metrics)
    st.plotly_chart(e, use_container_width=True)
    st.plotly_chart(s, use_container_width=True)
    st.download_button("Download customer segments", clustered.to_csv(index=False).encode(), "customer_segments.csv", "text/csv")
    st.download_button("Download diagnostics", metrics.to_csv(index=False).encode(), "cluster_metrics.csv", "text/csv")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#8B98B5">CustomerIQ · Built for Customer Segmentation Problem Statement 5 · ML first, business second, AI on top.</div>',
    unsafe_allow_html=True
)
