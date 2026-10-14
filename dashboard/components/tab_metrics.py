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

from dashboard.data_loader import load_cumulative_metrics, load_params_history, load_strategist_history


def _rgba(hex_color: str, alpha: float) -> str:
    """'#3b82f6' + alpha -> 'rgba(59,130,246,a)'. 8-digit hex is rejected by some Plotly versions."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"

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
            fillcolor=_rgba(color, 0x14 / 255),
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
            fillcolor=_rgba(c, 0x22 / 255),
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
            line=dict(color=_rgba(color, 0x55 / 255), width=1.5, dash="dot"),
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


def _build_pre_post_chart(both: pd.DataFrame, metric: str) -> go.Figure:
    """Pre-retrain (frozen detector, dashed) vs post-retrain (solid) per persona across rounds."""
    fig = go.Figure()
    palette = ["#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]
    for i, persona in enumerate(sorted(both["persona"].unique())):
        color = palette[i % len(palette)]
        for phase, dash, label in (("pre", "dash", "pre-retrain"), ("post", "solid", "post-retrain")):
            d = both[(both["persona"] == persona) & (both["phase"] == phase)].sort_values("round")
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["round"], y=d[metric], mode="lines+markers",
                name=f"{persona.replace('_', ' ')} · {label}",
                line=dict(color=color, dash=dash, width=2),
                marker=dict(symbol="circle-open" if phase == "pre" else "circle", size=8),
            ))
    fig.update_layout(
        paper_bgcolor="rgba(10,14,26,0)", plot_bgcolor="rgba(10,14,26,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=36, b=10), height=360,
        title_text=f"{metric.upper()} — dashed: frozen detector vs. mutated personas · solid: after retraining",
        xaxis=dict(title="Round", dtick=1, gridcolor="rgba(99,130,200,0.1)"),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(99,130,200,0.1)"),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _pre_post_summary(both: pd.DataFrame) -> str:
    """One-sentence, number-derived summary of the largest evasion and its recovery (no LLM)."""
    pre = both[both["phase"] == "pre"].set_index(["round", "persona"])["recall"]
    post = both[both["phase"] == "post"].set_index(["round", "persona"])["recall"]
    base = post.xs(0, level="round") if 0 in post.index.get_level_values("round") else None
    if pre.empty or base is None:
        return "Only the baseline round has been evaluated so far — pre/post-retrain comparison starts at round 1."
    drops = []
    for (rnd, persona), v in pre.items():
        b = base.get(persona)
        if b and b == b and v == v:
            drops.append(((b - v) / b, rnd, persona, v, post.get((rnd, persona))))
    if not drops:
        return "Not enough data for a pre/post comparison yet."
    drop, rnd, persona, v, rec = max(drops)
    name = persona.replace("_", " ")
    if drop <= 0:
        return f"No mutation reduced recall below the round-0 baseline; the closest was {name} in round {rnd}."
    tail = f" and recovered to {rec:.0%} after retraining" if rec == rec and rec is not None else ""
    return f"Recall for {name} dropped {drop:.0%} below baseline under round {rnd}'s mutation (to {v:.0%}){tail}."


def render_metrics_tab(run_dir: Path | None) -> None:
    if run_dir is None:
        st.warning("No run selected.")
        return

    with st.spinner("Loading metrics…"):
        df = load_cumulative_metrics(str(run_dir))

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

    # ── Section 0: Adversarial degrade / recover (pre vs post retrain) ────
    both = load_cumulative_metrics(str(run_dir), phase=None)
    st.caption("Metrics on this tab combine all test markets of each round (they are not split by the sidebar's test-market selector).")
    st.markdown("<div class='section-title'>🥊 Evasion vs. Recovery</div>", unsafe_allow_html=True)
    st.markdown(f"**{_pre_post_summary(both)}**")
    st.caption(
        "Pre-retrain = this round's mutated personas scored by the detector frozen at the end of the previous "
        "round (how well the mutation evaded). Post-retrain = after retraining on all rounds so far (recovery)."
    )
    if (both["phase"] == "pre").any():
        pr_tab, f1_tab = st.tabs(["Recall", "F1"])
        with pr_tab:
            st.plotly_chart(_build_pre_post_chart(both, "recall"), use_container_width=True,
                            config={"displayModeBar": False}, key="perf_prepost_recall")
        with f1_tab:
            st.plotly_chart(_build_pre_post_chart(both, "f1"), use_container_width=True,
                            config={"displayModeBar": False}, key="perf_prepost_f1")
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
    display_df = both.copy() if not both.empty else df.copy()
    display_df["persona"] = display_df["persona"].str.replace("_", " ").str.title()
    for col in ["precision", "recall", "f1", "fpr"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 5: Parameter evolution ───────────────────────────────────
    st.markdown("<div class='section-title'>🧬 Adversarial Parameter Evolution</div>", unsafe_allow_html=True)
    st.caption("Parameters actually used by each persona in each round of this run (from the round's metrics.json).")
    param_df = load_params_history(str(run_dir))
    if not param_df.empty:
        persona_filter_param = st.multiselect(
            "Filter by Persona",
            options=sorted(param_df["Persona"].unique()),
            default=sorted(param_df["Persona"].unique()),
            key="metrics_persona_filter",
        )
        view = param_df[param_df["Persona"].isin(persona_filter_param)].copy()
        view["Persona"] = view["Persona"].str.replace("_", " ").str.title()
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.info("This run has no recorded parameters (it predates parameter tracking). Re-run the pipeline to populate this table.")

    st.markdown("---")

    # ── Section 6: Strategist decisions ──────────────────────────────────
    st.markdown("<div class='section-title'>🤖 Adversarial Strategist Decisions</div>", unsafe_allow_html=True)
    st.caption(
        "Before each round the strategist (LLM, or the hand-written library as fallback) changes each persona's "
        "parameters. The outcome row shows how that mutation did in the same round: recall of the frozen detector "
        "(lower = better evasion), episodes caught and detection delay, recall after retraining, the manipulation's "
        "effect vs. round 0, and the score (evasion × retained effect, capped at 1×; 0 if the effect fell below half)."
    )
    history = load_strategist_history(str(run_dir))
    if not history:
        st.info("No strategist rounds yet — mutations start from round 1.")
        return
    for entry in reversed(history):
        rnd = entry["round"]
        with st.expander(f"Round {rnd} — mutations tested in this round", expanded=(rnd == history[-1]["round"])):
            personas = entry["personas"]
            cols = st.columns(max(1, len(personas)))
            for col, (persona, d) in zip(cols, personas.items()):
                with col, st.container(border=True):
                    _render_strategist_card(persona, d)


SOURCE_BADGES = {
    "agentic": "🤖 LLM strategist",
    "hand_authored": "✍️ Hand-written fallback",
    "baseline": "Baseline",
    "unchanged": "— Unchanged",
}


def _fmt_num(v, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.0%}" if pct else f"{v:.2f}"


def _format_reasoning(text: str) -> str:
    """LLM / library reasoning → clean Markdown: one numbered item per line, knobs as code."""
    import re
    if not text:
        return "_No reasoning recorded._"
    t = " ".join(text.replace("\r", " ").split())          # collapse stray newlines / indentation
    t = t.replace("->", "→").replace("$", "\\$")
    parts = re.split(r"(?:(?<=\s)|^)(\d{1,2})\.\s+(?=\S)", t)
    lines = [parts[0].strip()] if parts[0].strip() else []
    for i in range(1, len(parts) - 1, 2):
        lines.append(f"{parts[i]}. {parts[i + 1].strip()}")
    md = "\n".join(lines)
    return re.sub(r"(?<![`\w])([a-z]+(?:_[a-z0-9]+)+)(?![`\w])", r"`\1`", md)


def _render_strategist_card(persona: str, d: dict) -> None:
    color = PERSONA_LINE_COLORS.get(persona, "#94a3b8")
    st.markdown(
        f"<div style='font-size:0.78rem; font-weight:700; color:{color}; text-transform:uppercase; "
        f"letter-spacing:0.07em;'>{persona.replace('_', ' ')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(SOURCE_BADGES.get(d.get("source"), d.get("source") or "unknown"))

    m1, m2 = st.columns(2)
    cov = d.get("frozen_coverage")
    m1.metric("Frozen coverage" if cov is not None else "Frozen recall",
              _fmt_num(cov if cov is not None else d.get("frozen_recall"), pct=True),
              help="Share of each episode's account-windows that raised ANY alert from the detector trained BEFORE it "
                   "saw this mutation (lower = better evasion). Being flagged under another persona's label still "
                   "counts as caught. Older runs show the persona's own classifier or row recall.")
    m2.metric("Score", _fmt_num(d.get("score")),
              help="(0.5×(1−coverage) + 0.3×episodes missed + 0.2×delay) × min(1, effect ratio); "
                   "0 when the effect fell below half of round 0.")
    own = d.get("own_label_coverage")
    if cov is not None and own is not None and abs(own - cov) >= 0.05:
        st.caption(f"Own-label coverage: {_fmt_num(own, pct=True)} (the rest was flagged under another persona's label)")
    if d.get("effect_recovery_from_round") is not None:
        st.caption(f"⚠ Effect had fallen below half of round 0 — parameters were reset to round "
                   f"{d['effect_recovery_from_round']} before this mutation.")
    if d.get("effect_guard_reverts"):
        reverted = ", ".join(f"`{k}` {_fmt_val(a)} → {_fmt_val(b)}" for k, (a, b) in d["effect_guard_reverts"].items())
        st.caption(f"⚠ Effect was at risk — the guard reverted {reverted} after the proposal.")
    if d.get("episodes_caught"):
        st.caption(f"Episodes caught by frozen detector: {d['episodes_caught']} · "
                   f"mean detection delay: {_fmt_num(d.get('frozen_delay'))} ticks")
    m3, m4 = st.columns(2)
    m3.metric("Retrained recall", _fmt_num(d.get("retrained_recall"), pct=True),
              help="Recall after retraining on this round's data (recovery).")
    m4.metric("Effect vs R0", f"{_fmt_num(d.get('effect_ratio'))}×" if d.get("effect_ratio") is not None else "–",
              help="Manipulation effectiveness relative to round 0 (spoof price impact, wash volume share, pump pressure).")

    changes = d.get("changes") or {}
    st.markdown("**Changes**")
    if changes:
        st.markdown("\n".join(f"- `{k}`: {_fmt_val(old)} → **{_fmt_val(new)}**" for k, (old, new) in changes.items()))
    else:
        st.markdown("_No parameter changes._")
    with st.expander("Reasoning", expanded=False):
        st.markdown(_format_reasoning(d.get("reasoning", "")))


def _fmt_val(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
