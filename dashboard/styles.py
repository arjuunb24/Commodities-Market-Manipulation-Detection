"""
dashboard/styles.py
===================
Global CSS injection for the Veridex dashboard.
Finance-grade dark aesthetic with high-contrast typography and glassmorphism.
"""

import streamlit as st

GLOBAL_CSS = """
<style>
/* ── Fonts ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Root palette ───────────────────────────────────────── */
:root {
    --bg-base:       #0a0e1a;
    --bg-card:       rgba(15, 23, 42, 0.88);
    --bg-card-alt:   rgba(20, 30, 55, 0.75);
    --border:        rgba(99, 130, 200, 0.22);
    --border-hover:  rgba(99, 130, 200, 0.45);
    --accent-blue:   #3b82f6;
    --accent-cyan:   #06b6d4;
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-amber:  #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary:  #ffffff;
    --text-body:     #e2e8f0;
    --text-muted:    #94a3b8;
    --text-dim:      #64748b;
}

/* ── Streamlit Header Bar ───────────────────────────────── */
header[data-testid="stHeader"], .stAppHeader {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Base Typography & High Visibility ─────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-body) !important;
}

/* Ensure all headers are bright and clear */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Markdown paragraphs and body text */
p, [data-testid="stMarkdownContainer"] p {
    color: var(--text-body) !important;
}

/* Labels for all form inputs */
label, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

/* ── App background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1629 45%, #081020 100%) !important;
    min-height: 100vh;
}

/* ── Main content container ─────────────────────────────── */
.main .block-container {
    padding: 1.25rem 2rem 3rem !important;
    max-width: 1650px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1629 0%, #070d18 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}

/* ── Tabs & Segmented Controls ───────────────────────────── */
.stTabs [data-baseweb="tab-list"],
[data-testid="stSegmentedControl"] {
    background: rgba(15, 23, 42, 0.9) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid var(--border) !important;
    gap: 8px !important;
    margin-bottom: 1rem !important;
    display: inline-flex !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}
.stTabs [data-baseweb="tab"],
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="option"] {
    border-radius: 8px !important;
    padding: 8px 22px !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover,
[data-testid="stSegmentedControl"] button:hover,
[data-testid="stSegmentedControl"] [role="option"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.08) !important;
}
.stTabs [aria-selected="true"],
[data-testid="stSegmentedControl"] button[aria-selected="true"],
[data-testid="stSegmentedControl"] [aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-checked="true"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(59, 130, 246, 0.6) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 1.25rem 0 !important;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(14px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.15) !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] * {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ── Selectbox & inputs ──────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
[data-baseweb="select"] * {
    color: #ffffff !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background: #0f172a !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
}
li[role="option"] {
    color: #e2e8f0 !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: rgba(37, 99, 235, 0.3) !important;
    color: #ffffff !important;
}

/* ── Slider ──────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"] summary {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #3b82f6 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #94a3b8 !important;
}

/* ── Dataframes / tables ─────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: #0f172a !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 0.85rem 0 !important;
}

/* ── Custom card helper ──────────────────────────────────── */
.vx-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1rem;
}
.vx-card-header {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

/* ── Persona badges ──────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-spoofing      { background: rgba(239,68,68,0.2);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.45); }
.badge-wash_trading  { background: rgba(245,158,11,0.2);  color: #fcd34d; border: 1px solid rgba(245,158,11,0.45); }
.badge-pump_and_dump { background: rgba(139,92,246,0.2); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.45); }
.badge-layering      { background: rgba(6,182,212,0.2);   color: #67e8f9; border: 1px solid rgba(6,182,212,0.45); }

/* ── Confidence bar ──────────────────────────────────────── */
.conf-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 99px; height: 6px; width: 100%; }
.conf-bar { height: 6px; border-radius: 99px; }

/* ── Alert / info boxes ──────────────────────────────────── */
.stAlert {
    background: rgba(59, 130, 246, 0.1) !important;
    border: 1px solid rgba(59, 130, 246, 0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Plotly chart frames ─────────────────────────────────── */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Button ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-color: rgba(59, 130, 246, 0.8) !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.4) !important;
    color: #ffffff !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99, 130, 200, 0.35); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 130, 200, 0.6); }

/* ── Section titles ──────────────────────────────────────── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.02em;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
</style>
"""


def inject_styles() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
