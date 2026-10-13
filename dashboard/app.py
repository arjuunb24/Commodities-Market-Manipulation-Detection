"""
dashboard/app.py
================
Veridex — Market Manipulation Detection Dashboard

Four-tab analytics suite:
  Tab 1 — Market Overview    : OHLCV, volume, microstructure metrics per round
  Tab 2 — Flagged Trades     : Illegal windows with confidence, SHAP, LLM narratives
  Tab 3 — Model Performance  : Per-round Precision / Recall / F1 / FPR, pre vs post retrain
  Tab 4 — Analyst Chat       : Tool-grounded Q&A over the run's logs (Module 4C)

Run:
    .venv\\Scripts\\streamlit.exe run dashboard/app.py
"""

import sys
from pathlib import Path

# Allow local imports from project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Veridex | Market Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Interpreter check: the dashboard needs the project's virtual environment ──
_missing = []
for _mod in ("pandera", "xgboost", "shap", "litellm", "pyarrow"):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)
if _missing:
    st.error(
        f"This Streamlit is running on `{sys.executable}`, which is missing: {', '.join(_missing)}. "
        "Start the dashboard with the project's virtual environment instead."
    )
    st.code(r".\.venv\Scripts\python.exe -m streamlit run dashboard/app.py", language="powershell")
    st.stop()

from dashboard.components.sidebar import render_sidebar
from dashboard.components.tab_market   import render_market_tab
from dashboard.components.tab_flags    import render_flags_tab
from dashboard.components.tab_metrics  import render_metrics_tab
from dashboard.components.tab_chat     import render_chat_tab
from dashboard.styles import inject_styles

# ── Global styles ──────────────────────────────────────────────────────────
inject_styles()

# ── Sidebar (run selector + round selector) ────────────────────────────────
run_dir, selected_round, selected_market = render_sidebar()

# ── Header ─────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("## 🛡️")
with col_title:
    st.markdown(
        "<h1 style='margin:0; padding-top:6px; font-size:1.85rem; font-weight:700; color:#ffffff !important; letter-spacing:-0.02em;'>Veridex — Market Surveillance</h1>"
        "<p style='margin:0; color:#94a3b8 !important; font-size:0.88rem; font-weight:400;'>Commodities Market Manipulation Detection · Adversarial ABM</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Active Tab Navigation (Lazy Loading for Maximum Performance) ───────────
TAB_OPTIONS = [
    "📈  Market Overview",
    "🚨  Flagged Trades",
    "📊  Model Performance",
    "💬  Analyst Chat",
]

selected_tab = st.segmented_control(
    "Navigation",
    options=TAB_OPTIONS,
    default=TAB_OPTIONS[0],
    label_visibility="collapsed",
    key="dashboard_nav_tab",
)

if not selected_tab:
    selected_tab = TAB_OPTIONS[0]

if selected_tab == "📈  Market Overview":
    render_market_tab(run_dir, selected_round, selected_market)
elif selected_tab == "🚨  Flagged Trades":
    render_flags_tab(run_dir, selected_round, selected_market)
elif selected_tab == "📊  Model Performance":
    render_metrics_tab(run_dir)
elif selected_tab == "💬  Analyst Chat":
    render_chat_tab(run_dir)

st.markdown("---")
st.caption(
    "All data is synthetic. No real trading, real accounts, or real regulatory findings are "
    "represented in this application."
)
