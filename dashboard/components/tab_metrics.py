"""
dashboard/components/tab_metrics.py
=====================================
Tab 3 — Model Performance

Sections:
  • Per-round P/R/F1/FPR evolution line charts (all personas)
  • Radar chart: current vs baseline per-persona performance
  • Precision–Recall trade-off scatter
  • Parameter evolution table (from persona_config.yaml)
  • LLM strategist reasoning log viewer
  • F1 heatmap: persona × round
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from dashboard.data_loader import load_cumulative_metrics, load_llm_reasoning, load_persona_config

# ── Design tokens ────────────────────────────────────────────────────────────
CHART_BG    = "rgba(10,14,26,0)"
PAPER_BG    = "rgba(10,14,26,0)"
GRID_COLOR  = "rgba(99,130,200,0.1)"
FONT_COLOR  = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#06b6d4"
ACCENT_GREEN= "#10b981"
ACCENT_RED  = "#ef4444"
ACCENT_AMBER= "#f59e0b"
ACCENT_PURPLE="#8b5cf6"

PERSONA_LINE_COLORS = {
    "spoofing":      ACCENT_RED,
    "wash_trading":  ACCENT_AMBER,
    "pump_and_dump": ACCENT_PURPLE,
    "layering":      ACCENT_CYAN,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(99,130,200,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True, dtick=1),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
)


# ── Chart builders ───────────────────────────────────────────────────────────

def _build_metric_evolution(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Line chart of a metric (F1/precision/recall/fpr) across rounds per persona."""
    fig = go.Figure()

    for persona in sorted(df["persona"].unique()):
        p_df = df[df["persona"] == persona].sort_values("round")
        color = PERSONA_LINE_COLORS.get(persona, FONT_COLOR)
        label = persona.replace("_", " ").title()

        fig.add_trace(go.Scatter(
            x=p_df["round"],
            y=p_df[metric],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=8, symbol="circle", line=dict(width=2, color=color),
                        color="rgba(10,14,26,1)"),
            hovertemplate=f"<b>{label}</b><br>Round %{{x}}<br>{metric.upper()}=%{{y:.3f}}<extra></extra>",
        ))

    # Add reference lines
    if metric in ["f1", "precision", "recall"]:
        fig.add_hline(y=0.9, line=dict(color="rgba(16,185,129,0.3)", dash="dash", width=1),
                      annotation_text="0.90", annotation_font_size=10)
        fig.add_hline(y=0.7, line=dict(color="rgba(245,158,11,0.3)", dash="dash", width=1),
                      annotation_text="0.70", annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title_text=title,
        xaxis_title="Adversarial Round",
        yaxis_title=metric.upper(),
        yaxis_range=[0, 1.05],
    )
    return fig


def _build_fpr_chart(df: pd.DataFrame) -> go.Figure:
    """FPR evolution (lower is better) — distinct styling."""
    fig = go.Figure()

    for persona in sorted(df["persona"].unique()):
        p_df = df[df["persona"] == persona].sort_values("round")
        color = PERSONA_LINE_COLORS.get(persona, FONT_COLOR)
        label = persona.replace("_", " ").title()

        fig.add_trace(go.Scatter(
            x=p_df["round"],
            y=p_df["fpr"],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5, dash="dot"),
            marker=dict(size=8, symbol="diamond",
                        line=dict(width=2, color=color), color="rgba(10,14,26,1)"),
            fill="tozeroy",
            fillcolor=f"{color}14",
            hovertemplate=f"<b>{label}</b><br>Round %{{x}}<br>FPR=%{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0.1, line=dict(color="rgba(245,158,11,0.4)", dash="dash", width=1),
                  annotation_text="FPR=10%", annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        title_text="False Positive Rate (lower is better)",
        xaxis_title="Adversarial Round",
        yaxis_title="FPR",
        yaxis_range=[0, 1.05],
    )
    return fig


def _build_radar_chart(df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing round 0 (baseline) vs latest round."""
    rounds = sorted(df["round"].unique())
    if len(rounds) < 2:
        # Single round — just draw it
        rounds_to_show = [rounds[0]]
    else:
        rounds_to_show = [rounds[0], rounds[-1]]

    personas = sorted(df["persona"].unique())
    metrics_to_show = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    fig = go.Figure()

    colors_per_round = [ACCENT_BLUE, ACCENT_GREEN]
    for round_idx, rnd in enumerate(rounds_to_show):
        r_df = df[df["round"] == rnd]
        # Average across personas
        vals = [r_df[m].mean() for m in metrics_to_show]
        vals.append(vals[0])  # close the loop
        labels = metric_labels + [metric_labels[0]]
        c = colors_per_round[round_idx % len(colors_per_round)]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels,
            name=f"Round {rnd}",
            line=dict(color=c, width=2.5),
            fill="toself",
            fillcolor=f"{c}22",
            marker=dict(size=8, color=c),
        ))

    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=11),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=9, color=FONT_COLOR),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#e2e8f0"),
                gridcolor=GRID_COLOR,
                linecolor=GRID_COLOR,
            ),
        ),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(99,130,200,0.2)", borderwidth=1),
        title_text="Baseline vs Latest Round (Average across Personas)",
    )
    return fig


def _build_f1_heatmap(df: pd.DataFrame) -> go.Figure:
    """F1 heatmap: personas (rows) × rounds (cols)."""
    personas = sorted(df["persona"].unique())
    rounds   = sorted(df["round"].unique())

    matrix = []
    for p in personas:
        row = []
        for r in rounds:
            cell = df[(df["persona"] == p) & (df["round"] == r)]["f1"]
            row.append(float(cell.values[0]) if not cell.empty else np.nan)
        matrix.append(row)

    text_matrix = [[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in matrix]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"R{r}" for r in rounds],
        y=[p.replace("_", " ").title() for p in personas],
        colorscale=[
            [0.0,  "rgba(239,68,68,0.8)"],
            [0.5,  "rgba(245,158,11,0.8)"],
            [0.75, "rgba(59,130,246,0.8)"],
            [1.0,  "rgba(16,185,129,0.9)"],
        ],
        zmin=0, zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        showscale=True,
        colorbar=dict(
            title="F1 Score",
            tickfont=dict(size=10),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0", "0.25", "0.50", "0.75", "1.0"],
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>F1=%{text}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(200, len(personas) * 70 + 80),
        title_text="F1 Score Heatmap — Persona × Round",
        xaxis=dict(side="top"),
    )
    return fig


def _build_pr_scatter(df: pd.DataFrame) -> go.Figure:
    """Precision-Recall scatter, coloured by persona, sized by support, marker per round."""
    fig = go.Figure()

    for persona in sorted(df["persona"].unique()):
        p_df = df[df["persona"] == persona].sort_values("round")
        color = PERSONA_LINE_COLORS.get(persona, FONT_COLOR)
        label = persona.replace("_", " ").title()

        support = p_df.get("support", pd.Series(50, index=p_df.index))
        sizes = (support / (support.max() + 1e-9) * 25 + 8).clip(8, 35)

        fig.add_trace(go.Scatter(
            x=p_df["precision"],
            y=p_df["recall"],
            mode="markers+lines+text",
            name=label,
            text=[f"R{r}" for r in p_df["round"]],
            textposition="top center",
            textfont=dict(size=9, color=color),
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.85,
                line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
            ),
            line=dict(color=f"{color}55", width=1.5, dash="dot"),
            hovertemplate=(
                f"<b>{label}</b><br>Round %{{text}}<br>"
                "Precision=%{x:.3f}<br>Recall=%{y:.3f}<extra></extra>"
            ),
        ))

    # ISO-F1 contours
    for f1_val in [0.5, 0.7, 0.8, 0.9]:
        p_range = np.linspace(0.01, 1.0, 200)
        r_vals  = f1_val * p_range / (2 * p_range - f1_val)
        mask    = (r_vals > 0) & (r_vals <= 1)
        fig.add_trace(go.Scatter(
            x=p_range[mask], y=r_vals[mask],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            name=f"F1={f1_val}",
        ))
        fig.add_annotation(
            x=0.95, y=f1_val * 0.95 / (2 * 0.95 - f1_val),
            text=f"F1={f1_val}",
            font=dict(size=9, color="rgba(255,255,255,0.3)"),
            showarrow=False,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        title_text="Precision–Recall Trade-off (marker size = support)",
        xaxis_title="Precision",
        yaxis_title="Recall",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
    )
    return fig


def _build_parameter_table(persona_cfg: dict) -> pd.DataFrame | None:
    """
    Builds a tidy DataFrame showing how each persona's parameters evolved.
    Columns: persona, parameter, round_0, round_1, …
    """
    personas = ["spoofing", "wash_trading", "pump_and_dump"]
    ignore_keys = {"enabled", "start_tick", "end_tick"}

    rows = []
    rounds_found = sorted([
        int(k.split("_")[1]) for k in persona_cfg if k.startswith("round_")
    ])

    for persona in personas:
        # Collect all params across rounds
        all_params: set[str] = set()
        for r in rounds_found:
            cfg_node = persona_cfg.get(f"round_{r}", {})
            if persona in cfg_node:
                for p in cfg_node[persona]:
                    if p not in ignore_keys:
                        all_params.add(p)

        for param in sorted(all_params):
            row: dict = {"Persona": persona.replace("_", " ").title(), "Parameter": param}
            for r in rounds_found:
                cfg_node = persona_cfg.get(f"round_{r}", {})
                val = cfg_node.get(persona, {}).get(param, "—")
                row[f"R{r}"] = str(val)
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ── Main render ──────────────────────────────────────────────────────────────

def render_metrics_tab(run_dir: Path | None) -> None:
    if run_dir is None:
        st.warning("No run selected.")
        return

    with st.spinner("Loading metrics…"):
        df = load_cumulative_metrics(str(run_dir))
        persona_cfg = load_persona_config()
        llm_log = load_llm_reasoning(str(run_dir))

    if df.empty:
        st.warning(
            "No cumulative metrics found. "
            "Run the full pipeline to generate `round_metrics.parquet`."
        )
        return

    rounds   = sorted(df["round"].unique())
    personas = sorted(df["persona"].unique())

    # ── Summary KPI row ──────────────────────────────────────────────────
    latest_round = df[df["round"] == max(rounds)]
    avg_f1    = latest_round["f1"].mean()
    avg_prec  = latest_round["precision"].mean()
    avg_rec   = latest_round["recall"].mean()
    avg_fpr   = latest_round["fpr"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📊 Rounds Completed", f"{len(rounds)}")
    c2.metric("✅ Avg F1 (Latest)",   f"{avg_f1:.3f}")
    c3.metric("🎯 Avg Precision",     f"{avg_prec:.3f}")
    c4.metric("🔍 Avg Recall",        f"{avg_rec:.3f}")
    c5.metric("⚠️ Avg FPR",           f"{avg_fpr:.3f}", delta_color="inverse")

    st.markdown("---")

    # ── Section 1: F1 Heatmap + Radar ────────────────────────────────────
    st.markdown("<div class='section-title'>🗺️ Performance Overview</div>", unsafe_allow_html=True)
    col_heat, col_radar = st.columns([1.4, 1])

    with col_heat:
        fig_heat = _build_f1_heatmap(df)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False}, key="perf_fig_heat")

    with col_radar:
        fig_radar = _build_radar_chart(df)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False}, key="perf_fig_radar")

    st.markdown("---")

    # ── Section 2: Metric evolution ──────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Metric Evolution Across Rounds</div>", unsafe_allow_html=True)
    st.caption(
        "Each line represents one manipulation persona. "
        "Adversarial mutations from the LLM strategist make it progressively harder to detect."
    )

    metric_tab_f1, metric_tab_pr, metric_tab_rec, metric_tab_fpr = st.tabs(
        ["F1 Score", "Precision", "Recall", "FPR"]
    )
    with metric_tab_f1:
        st.plotly_chart(_build_metric_evolution(df, "f1", "F1 Score per Persona"),
                        use_container_width=True, config={"displayModeBar": False}, key="perf_ev_f1")
    with metric_tab_pr:
        st.plotly_chart(_build_metric_evolution(df, "precision", "Precision per Persona"),
                        use_container_width=True, config={"displayModeBar": False}, key="perf_ev_pr")
    with metric_tab_rec:
        st.plotly_chart(_build_metric_evolution(df, "recall", "Recall per Persona"),
                        use_container_width=True, config={"displayModeBar": False}, key="perf_ev_rec")
    with metric_tab_fpr:
        st.plotly_chart(_build_fpr_chart(df),
                        use_container_width=True, config={"displayModeBar": False}, key="perf_ev_fpr")

    st.markdown("---")

    # ── Section 3: P–R scatter ───────────────────────────────────────────
    st.markdown("<div class='section-title'>🎯 Precision–Recall Trade-off</div>", unsafe_allow_html=True)
    st.plotly_chart(_build_pr_scatter(df), use_container_width=True, config={"displayModeBar": False}, key="perf_pr_scatter")

    st.markdown("---")

    # ── Section 4: Raw metrics table ─────────────────────────────────────
    st.markdown("<div class='section-title'>📋 Full Metrics Table</div>", unsafe_allow_html=True)
    display_df = df.copy()
    display_df["persona"] = display_df["persona"].str.replace("_", " ").str.title()
    for col in ["precision", "recall", "f1", "fpr"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 5: Parameter evolution ───────────────────────────────────
    st.markdown("<div class='section-title'>🧬 Adversarial Parameter Evolution</div>", unsafe_allow_html=True)
    st.caption(
        "The LLM Adversarial Strategist mutates persona parameters each round "
        "to evade the detector. This table tracks how each parameter changed."
    )

    param_df = _build_parameter_table(persona_cfg)
    if param_df is not None and not param_df.empty:
        # Colour-code by persona
        persona_filter_param = st.multiselect(
            "Filter by Persona",
            options=sorted(param_df["Persona"].unique()),
            default=sorted(param_df["Persona"].unique()),
            key="metrics_persona_filter",
        )
        filtered_param = param_df[param_df["Persona"].isin(persona_filter_param)]
        st.dataframe(filtered_param, use_container_width=True, hide_index=True)
    else:
        st.info("No parameter evolution data found. Run multiple adversarial rounds to populate this table.")

    st.markdown("---")

    # ── Section 6: LLM Reasoning Log ─────────────────────────────────────
    st.markdown("<div class='section-title'>🤖 LLM Adversarial Strategist Reasoning</div>", unsafe_allow_html=True)
    st.caption(
        "The LLM strategist reads the previous round's metrics and reasons "
        "about which parameters to mutate. Full reasoning log:"
    )

    if llm_log:
        # Split by round sections
        sections = llm_log.split("=" * 60)
        sections = [s.strip() for s in sections if s.strip()]

        if sections:
            for i, section in enumerate(sections):
                if not section:
                    continue
                # Extract round header if present
                lines = section.split("\n")
                header = lines[0] if lines else f"Section {i+1}"
                body   = "\n".join(lines[1:]).strip()

                with st.expander(f"📝 {header}", expanded=(i == len(sections) - 1)):
                    # Parse persona blocks within a section
                    persona_blocks = {}
                    current_persona = "General"
                    current_lines   = []

                    for line in body.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith("[") and stripped.endswith("]") and stripped[1:-1].upper() in ["SPOOFING", "WASH_TRADING", "PUMP_AND_DUMP", "LAYERING"]:
                            if current_lines:
                                persona_blocks[current_persona] = "\n".join(current_lines)
                            current_persona = stripped[1:-1]
                            current_lines = []
                        else:
                            current_lines.append(line)

                    if current_lines:
                        persona_blocks[current_persona] = "\n".join(current_lines)

                    for p_name, p_text in persona_blocks.items():
                        if not p_text.strip():
                            continue
                        p_color = PERSONA_LINE_COLORS.get(p_name.lower(), "#94a3b8")
                        st.markdown(
                            f"""
                            <div style="background:rgba(15,23,42,0.85); border-left:3px solid {p_color};
                                        border-radius:0 10px 10px 0; padding:0.85rem 1rem;
                                        margin-bottom:0.6rem;">
                                <div style="font-size:0.7rem; color:{p_color}; font-weight:700;
                                            text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.4rem;">
                                    {p_name}
                                </div>
                                <div style="font-size:0.84rem; color:#94a3b8; white-space:pre-wrap; line-height:1.55;">
                                    {p_text.strip()}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
    else:
        st.info(
            "No LLM reasoning log found. The log is created when the Adversarial Strategist "
            "runs for Round 1+ (`llm_reasoning.log`)."
        )
