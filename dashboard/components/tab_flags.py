"""
dashboard/components/tab_flags.py
==================================
Tab 2 — Flagged Trades

For each detected manipulation window, shows:
  • Confidence score + gauge
  • Manipulation type badge
  • Key signal features (cancel_rate, OTR, volume_spike_ratio, etc.)
  • SHAP feature importance waterfall/bar chart (computed live using SHAP)
  • LLM narrative explanation (calls NarrativeGenerator)
  • Detailed trade log in the flagged window
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from dashboard.data_loader import (
    load_manipulation_events,
    load_trade_log,
    load_order_log,
    load_ohlcv,
    build_flagged_windows,
)

logger = logging.getLogger(__name__)

# ── Design tokens ────────────────────────────────────────────────────────────
CHART_BG    = "rgba(10,14,26,0)"
PAPER_BG    = "rgba(10,14,26,0)"
GRID_COLOR  = "rgba(99,130,200,0.1)"
FONT_COLOR  = "#94a3b8"

PERSONA_COLORS = {
    "spoofing":      ("#ef4444", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.35)"),
    "wash_trading":  ("#f59e0b", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.35)"),
    "pump_and_dump": ("#8b5cf6", "rgba(139,92,246,0.12)", "rgba(139,92,246,0.35)"),
    "layering":      ("#06b6d4", "rgba(6,182,212,0.12)",  "rgba(6,182,212,0.35)"),
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=11),
    margin=dict(l=10, r=10, t=36, b=10),
)


def _canon_persona(raw: str) -> str:
    if "spoof" in raw: return "spoofing"
    if "wash"  in raw: return "wash_trading"
    if "pump"  in raw: return "pump_and_dump"
    if "layer" in raw: return "layering"
    return raw


def _confidence_gauge(confidence: float, persona: str) -> str:
    """Returns HTML for a confidence gauge bar."""
    color, _, _ = PERSONA_COLORS.get(persona, ("#94a3b8", "", ""))
    pct = int(confidence * 100)
    return (
        f'<div style="display:flex; align-items:center; gap:0.75rem; margin:0.4rem 0;">'
        f'<div style="flex:1; background:rgba(255,255,255,0.06); border-radius:99px; height:8px;">'
        f'<div style="width:{pct}%; height:8px; border-radius:99px; background:linear-gradient(90deg, {color}88, {color}); box-shadow:0 0 8px {color}66;"></div>'
        f'</div>'
        f'<span style="font-size:0.85rem; font-weight:700; color:{color}; font-family:monospace; min-width:3rem;">{confidence:.1%}</span>'
        f'</div>'
    )


def _feature_signal_pills(row: pd.Series, persona: str) -> str:
    """Returns HTML feature signal pills for the flagged window."""
    color, bg, border = PERSONA_COLORS.get(persona, ("#94a3b8", "rgba(148,163,184,0.1)", "rgba(148,163,184,0.3)"))
    pills_html = []

    feature_map = {
        "cancel_rate":                   ("Cancel Rate",       "%",    True),
        "otr":                            ("OTR",               "×",    True),
        "volume_spike_ratio":            ("Vol Spike",         "×",    True),
        "stateful_circular_volume_ratio":("Circular Vol",      "%",    True),
        "price_momentum":                ("Price Mom.",        "%",    True),
        "abnormal_return_z":             ("Abnormal Ret (Z)",  "",     True),
        "trade_size_cv":                 ("Trade Size CV",     "",     False),
        "avg_cancel_distance_to_mid":    ("Cancel Dist.",      "%",    False),
        "topk_buy_imbalance":            ("TopK Buy Imbal.",   "%",    False),
    }

    for feat, (label, unit, highlight) in feature_map.items():
        val = row.get(feat, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if unit == "%":
            display = f"{val:.1%}"
        elif unit == "×":
            display = f"{val:.2f}×"
        else:
            display = f"{val:.3f}"

        pill_color = color if highlight else "#94a3b8"
        pill_bg    = bg    if highlight else "rgba(148,163,184,0.08)"
        pill_border= border if highlight else "rgba(148,163,184,0.25)"

        pills_html.append(
            f"<span style='background:{pill_bg}; color:{pill_color}; border:1px solid {pill_border}; "
            f"padding:3px 10px; border-radius:6px; font-size:0.72rem; font-weight:600; "
            f"margin:2px; display:inline-block;'>"
            f"{label}: {display}</span>"
        )

    return " ".join(pills_html) if pills_html else "<span style='color:#64748b; font-size:0.75rem;'>No feature data available</span>"


def _build_shap_chart(shap_values: dict[str, float], persona: str) -> go.Figure:
    """Horizontal waterfall/bar chart of SHAP values."""
    color, _, _ = PERSONA_COLORS.get(persona, ("#94a3b8", "", ""))
    if not shap_values:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No SHAP data")
        return fig

    # Sort by absolute value
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    bar_colors = []
    for v in values:
        if v > 0:
            bar_colors.append("rgba(16,185,129,0.8)")  # positive → green (raises risk)
        else:
            bar_colors.append("rgba(239,68,68,0.7)")   # negative → reduces risk

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=FONT_COLOR),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(labels) * 28 + 60),
        title_text=f"SHAP Feature Attribution — {persona.replace('_', ' ').title()}",
        xaxis_title="SHAP Value (impact on model output)",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.2)", zerolinewidth=1.5),
        yaxis=dict(gridcolor=GRID_COLOR),
        showlegend=False,
    )
    return fig


@st.cache_resource(show_spinner=False)
def _get_fitted_pipeline_and_features(run_dir_str: str, round_num: int):
    """
    Fits the detector pipeline and generates test features once per (run_dir, round).
    Cached as a resource so all flagged window cards can compute SHAP instantaneously.
    """
    try:
        run_dir = Path(run_dir_str)
        cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "detector_config.yaml"
        with open(cfg_path) as f:
            det_cfg = yaml.safe_load(f)

        from src.detector.features import FeatureEngine
        from src.detector.pipeline import DetectorPipeline

        engine = FeatureEngine(det_cfg)

        train_dir = run_dir / f"round_{round_num}" / "train"
        test_dir  = run_dir / f"round_{round_num}" / "test"

        if not train_dir.exists() or not test_dir.exists():
            return None, None

        # Build feature matrix from test data
        trade_df = pd.read_parquet(test_dir / "trade_log.parquet")
        order_df = pd.read_parquet(test_dir / "order_log.parquet")

        base_time = pd.Timestamp("2026-01-01")
        trade_df["timestamp"] = base_time + pd.to_timedelta(trade_df["tick"], unit="s")
        order_df["timestamp"] = base_time + pd.to_timedelta(order_df["tick"], unit="s")

        X_test = engine.generate_features(trade_df, order_df)
        if X_test.empty:
            return None, None

        # Build train features and fit pipeline
        train_trade = pd.read_parquet(train_dir / "trade_log.parquet")
        train_order = pd.read_parquet(train_dir / "order_log.parquet")
        train_manip = pd.read_parquet(train_dir / "manipulation_events.parquet")

        train_trade["timestamp"] = base_time + pd.to_timedelta(train_trade["tick"], unit="s")
        train_order["timestamp"] = base_time + pd.to_timedelta(train_order["tick"], unit="s")

        X_train = engine.generate_features(train_trade, train_order)
        if X_train.empty:
            return None, None

        pipeline = DetectorPipeline(det_cfg)
        y_train = pipeline._generate_labels(X_train, train_manip)
        pipeline.supervised_detector.fit(X_train, y_train)

        return pipeline, X_test

    except Exception as e:
        logger.warning(f"Failed to fit pipeline and generate features: {e}")
        return None, None


def _compute_shap_for_window(
    run_dir: Path,
    round_num: int,
    event_row: pd.Series,
) -> dict[str, float]:
    """
    Computes SHAP values for the flagged window using the cached pipeline and features.
    Returns a dict of {feature: shap_value}.
    """
    try:
        import shap

        pipeline, X_test = _get_fitted_pipeline_and_features(str(run_dir), round_num)
        if pipeline is None or X_test is None or X_test.empty:
            return {}

        persona = _canon_persona(str(event_row.get("persona", "")))
        commodity = event_row.get("commodity", "")
        start_t = event_row.get("start_time", event_row.get("window_start", None))
        end_t   = event_row.get("end_time", event_row.get("window_end", None))

        try:
            X_comm = X_test.xs(commodity, level="commodity") if "commodity" in X_test.index.names else X_test
        except KeyError:
            X_comm = X_test

        if start_t is not None and end_t is not None:
            window_mask = (X_comm.index >= start_t) & (X_comm.index <= end_t)
            X_window = X_comm[window_mask]
        else:
            X_window = X_comm.head(5)

        if X_window.empty:
            X_window = X_comm.head(3)

        model = pipeline.supervised_detector.models.get(persona)
        if model is None:
            return {}

        # Clean X to match what the model was trained on
        absolute_cols = ["total_volume", "buy_volume", "sell_volume", "trade_count",
                         "buy_trade_count", "sell_trade_count", "total_vol"]
        X_window_clean = X_window.drop(columns=[c for c in absolute_cols if c in X_window.columns])

        # SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X_window_clean)

        # Average SHAP across window rows
        signed_shap = shap_vals.values.mean(axis=0)
        feature_names = X_window_clean.columns.tolist()
        result = {feat: float(signed_shap[i]) for i, feat in enumerate(feature_names)}
        # Keep top-12 by absolute value
        result = dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)[:12])
        return result

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}


def _get_llm_narrative(event_row: pd.Series, shap_summary: dict, run_dir: Path) -> str | None:
    """
    Calls NarrativeGenerator to produce an LLM-written case note.
    Returns None if LLM is unavailable.
    """
    try:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "llm_config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path) as f:
            llm_cfg = yaml.safe_load(f)

        narrator_cfg = llm_cfg.get("narrative_generator", {})
        if not narrator_cfg:
            return None

        from src.agentic.narrative_generator import NarrativeGenerator
        gen = NarrativeGenerator(config=narrator_cfg, run_dir=run_dir)

        start_tick = int(event_row.get("start_tick", 0))
        end_tick = int(event_row.get("end_tick", 0))

        flag_row = {
            "persona": event_row.get("persona", "unknown"),
            "commodity": event_row.get("commodity", "unknown"),
            "confidence": round(float(event_row.get("confidence", 0.5)), 3),
            "start_tick": start_tick,
            "end_tick":   end_tick,
        }

        # If shap_summary is empty, provide feature indicators from event_row so prompt is grounded
        payload_shap = dict(shap_summary)
        if not payload_shap:
            for k in ["cancel_rate", "otr", "volume_spike_ratio", "stateful_circular_volume_ratio",
                      "price_momentum", "abnormal_return_z", "trade_size_cv"]:
                v = event_row.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    payload_shap[k] = round(float(v), 3)

        result = gen.generate(flag_row=flag_row, shap_summary=payload_shap)
        return result.get("narrative_text", "")

    except Exception as e:
        logger.warning(f"LLM narrative failed: {e}")
        return None


def _window_trade_chart(trades: pd.DataFrame, start_t, end_t, commodity: str) -> go.Figure | None:
    """Mini price scatter chart for the flagged window."""
    if trades.empty or "timestamp" not in trades.columns:
        return None

    mask = (trades["timestamp"] >= start_t) & (trades["timestamp"] <= end_t)
    if "commodity" in trades.columns:
        mask &= (trades["commodity"] == commodity)
    window_trades = trades[mask].sort_values("timestamp")

    if window_trades.empty or "price" not in window_trades.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window_trades["timestamp"],
        y=window_trades["price"],
        mode="lines+markers",
        name="Trade Price",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=5, color="#3b82f6"),
    ))
    if "quantity" in window_trades.columns and len(window_trades) > 0:
        max_q = window_trades["quantity"].max()
        if max_q > 0:
            sizes = (window_trades["quantity"] / max_q * 14 + 4).clip(3, 18)
            fig.data[0].marker.size = sizes.tolist()

    fig.update_layout(
        paper_bgcolor="rgba(10,14,26,0)",
        plot_bgcolor="rgba(10,14,26,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=10),
        margin=dict(l=5, r=5, t=30, b=5),
        height=180,
        showlegend=False,
        title_text="Trade Price in Flagged Window",
        title_font=dict(size=11),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig



# ── Main render ──────────────────────────────────────────────────────────────

def render_flags_tab(run_dir: Path | None, selected_round: int) -> None:
    if run_dir is None:
        st.warning("No run selected.")
        return

    with st.spinner("Loading flagged windows…"):
        events = load_manipulation_events(str(run_dir), selected_round)
        trades = load_trade_log(str(run_dir), selected_round)

    if events.empty:
        st.info("✅ No manipulation events found for this round. Market appears clean.")
        return

    # ── Build flagged windows ────────────────────────────────────────────
    flagged = build_flagged_windows(run_dir, selected_round)
    if flagged.empty:
        st.warning("Could not build flagged windows. Check data integrity.")
        return

    # ── Summary KPI row ──────────────────────────────────────────────────
    total_flags = len(flagged)
    avg_conf    = flagged["confidence"].mean()
    personas    = flagged["persona"].unique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 Total Flags",       f"{total_flags}")
    c2.metric("🎯 Avg Confidence",    f"{avg_conf:.1%}")
    c3.metric("🎭 Persona Types",     f"{len(personas)}")
    c4.metric("🏆 Highest Conf.",     f"{flagged['confidence'].max():.1%}")

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2])
    with col_f1:
        persona_filter = st.multiselect(
            "Filter by Persona",
            options=sorted(set(flagged["persona"].unique())),
            default=sorted(set(flagged["persona"].unique())),
            key="flags_persona_filter",
        )
    with col_f2:
        commodity_filter = st.multiselect(
            "Filter by Commodity",
            options=sorted(set(flagged["commodity"].unique())),
            default=sorted(set(flagged["commodity"].unique())),
            key="flags_commodity_filter",
        )
    with col_f3:
        min_conf = st.slider(
            "Min Confidence Threshold",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key="flags_conf_slider",
        )

    filtered = flagged[
        flagged["persona"].isin(persona_filter) &
        flagged["commodity"].isin(commodity_filter) &
        (flagged["confidence"] >= min_conf)
    ].reset_index(drop=True)

    if filtered.empty:
        st.info("No flagged windows match the current filters.")
        return

    PAGE_SIZE = 6
    total_flags = len(filtered)
    total_pages = max(1, (total_flags + PAGE_SIZE - 1) // PAGE_SIZE)

    col_meta, col_page = st.columns([3, 1])
    with col_meta:
        curr_page = st.session_state.get("flags_page_sel", 1)
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.82rem; margin-top:0.4rem;'>"
            f"Showing <b style='color:#e2e8f0;'>{total_flags}</b> flagged windows "
            f"(Page <b style='color:#38bdf8;'>{curr_page}</b> of {total_pages})</div>",
            unsafe_allow_html=True,
        )
    with col_page:
        if total_pages > 1:
            page = st.selectbox(
                "Page Selection",
                options=list(range(1, total_pages + 1)),
                index=0,
                key="flags_page_sel",
                label_visibility="collapsed",
            )
        else:
            page = 1

    start_idx = (page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_flags)
    page_items = filtered.iloc[start_idx:end_idx]

    # ── Per-flag expandable cards ────────────────────────────────────────
    for idx, row in page_items.iterrows():
        persona = row["persona"]
        color, bg_color, border_color = PERSONA_COLORS.get(persona, ("#94a3b8", "rgba(148,163,184,0.1)", "rgba(148,163,184,0.3)"))

        label = persona.replace("_", " ").title()
        start_str = str(row["window_start"])[:19] if pd.notna(row.get("window_start")) else "N/A"
        end_str   = str(row["window_end"])[:19]   if pd.notna(row.get("window_end"))   else "N/A"
        conf_pct  = f"{row['confidence']:.1%}"
        commodity = row.get("commodity", "?")

        expander_label = f"#{idx+1}  ·  {label}  ·  {commodity}  ·  Confidence: {conf_pct}  ·  {start_str}"

        with st.expander(expander_label, expanded=(idx == 0)):

            # ── Header row ───────────────────────────────────────────────
            h1, h2 = st.columns([1, 2])

            with h1:
                gauge_html = _confidence_gauge(row['confidence'], persona)
                st.markdown(
                    f'<div style="background:{bg_color}; border:1px solid {border_color}; border-radius:12px; padding:1rem 1.2rem;">'
                    f'<div style="font-size:0.7rem; color:{color}; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:0.4rem;">Manipulation Type</div>'
                    f'<div style="font-size:1.3rem; font-weight:700; color:#e2e8f0;">{label}</div>'
                    f'<div style="margin-top:0.7rem; font-size:0.72rem; color:#94a3b8;">Commodity: <b style="color:#e2e8f0;">{commodity}</b></div>'
                    f'<div style="font-size:0.72rem; color:#94a3b8;">Window: <b style="color:#e2e8f0;">{start_str}</b></div>'
                    f'<div style="font-size:0.72rem; color:#94a3b8; margin-bottom:0.7rem;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <b style="color:#e2e8f0;">{end_str}</b></div>'
                    f'<div style="font-size:0.72rem; color:#94a3b8; margin-bottom:0.25rem;">Confidence</div>'
                    f'{gauge_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Feature price/volume info
                if row.get("price_at_flag") is not None:
                    st.markdown(
                        f"<div style='margin-top:0.5rem; font-size:0.75rem; color:#94a3b8;'>"
                        f"Avg Price: <b style='color:#e2e8f0;'>${row['price_at_flag']:,.4f}</b>&nbsp;&nbsp;"
                        f"Volume: <b style='color:#e2e8f0;'>{int(row['volume_at_flag'] or 0):,}</b></div>",
                        unsafe_allow_html=True,
                    )

            with h2:
                # Feature signal pills
                st.markdown(
                    "<div style='font-size:0.72rem; color:#94a3b8; text-transform:uppercase; "
                    "letter-spacing:0.06em; font-weight:700; margin-bottom:0.5rem;'>Signal Features</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(_feature_signal_pills(row, persona), unsafe_allow_html=True)

                # Window trade chart
                fig_win = _window_trade_chart(trades, row.get("window_start"), row.get("window_end"), commodity)
                if fig_win:
                    st.plotly_chart(fig_win, use_container_width=True, config={"displayModeBar": False}, key=f"fig_win_{idx}_{commodity}")

            st.markdown("---")

            # ── SHAP Analysis ────────────────────────────────────────────
            shap_col, narr_col = st.columns([1.2, 1])

            with shap_col:
                st.markdown(
                    "<div style='font-size:0.75rem; font-weight:700; color:#94a3b8; "
                    "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
                    "🔬 SHAP Feature Attribution</div>",
                    unsafe_allow_html=True,
                )

                # Button to trigger SHAP computation (expensive, on-demand)
                shap_key = f"shap_{idx}_{persona}"
                compute_shap = st.button(
                    "Compute SHAP",
                    key=f"btn_{shap_key}",
                    help="Runs SHAP TreeExplainer on the detector model for this window. May take 10-30s.",
                )

                shap_vals_key = f"shap_vals_{idx}"
                if compute_shap:
                    with st.spinner("Computing SHAP values with TreeExplainer…"):
                        try:
                            shap_result = _compute_shap_for_window(run_dir, selected_round, row)
                            st.session_state[shap_vals_key] = shap_result
                        except Exception as e:
                            st.error(f"SHAP failed: {e}")
                            st.session_state[shap_vals_key] = {}

                shap_vals = st.session_state.get(shap_vals_key, {})

                if shap_vals:
                    fig_shap = _build_shap_chart(shap_vals, persona)
                    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False}, key=f"fig_shap_{idx}_{persona}")

                    # Feature explanation table
                    feat_descriptions = {
                        "cancel_rate":                   "Fraction of orders cancelled (spoofing hallmark)",
                        "otr":                           "Order-to-trade ratio — high = phantom liquidity",
                        "volume_spike_ratio":            "Current volume vs historical rolling mean",
                        "stateful_circular_volume_ratio":"Graph-based circular trading ratio",
                        "price_momentum":                "(VWAP - Min_VWAP) / Min_VWAP",
                        "abnormal_return_z":             "Z-score of volume × absolute price change",
                        "trade_size_cv":                 "Coefficient of variation in trade sizes",
                        "topk_buy_imbalance":            "Top-K trader buy fraction (pump signal)",
                        "avg_cancel_distance_to_mid":    "Avg distance of cancelled orders to mid-price",
                        "net_position_ratio":            "Proximity of trader net position to zero",
                    }
                    rows_table = [
                        {
                            "Feature": k.replace("_", " ").title(),
                            "SHAP": f"{v:+.4f}",
                            "Direction": "↑ Raises Risk" if v > 0 else "↓ Lowers Risk",
                            "Description": feat_descriptions.get(k, "—"),
                        }
                        for k, v in shap_vals.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(rows_table),
                        use_container_width=True,
                        hide_index=True,
                        key=f"df_shap_table_{idx}",
                    )
                else:
                    # Show synthetic feature importance (from feature engineering knowledge)
                    st.info("Click **Compute SHAP** to run the XGBoost SHAP explainer for this window.")
                    # Show placeholder bar chart from synthetic weights
                    placeholder_weights = {
                        "spoofing":      {"cancel_rate": 0.42, "otr": 0.31, "avg_cancel_distance_to_mid": 0.15, "order_rate_ratio": 0.09, "volume_concentration": 0.03},
                        "wash_trading":  {"stateful_circular_volume_ratio": 0.38, "net_position_ratio": 0.26, "unique_counterparty_ratio": 0.19, "trade_size_cv": 0.12, "volume_concentration": 0.05},
                        "pump_and_dump": {"volume_spike_ratio": 0.35, "abnormal_return_z": 0.28, "topk_buy_imbalance": 0.22, "price_momentum": 0.10, "price_range": 0.05},
                        "layering":      {"cancel_rate": 0.36, "order_book_imbalance": 0.29, "max_cancel_ratio_topk": 0.20, "avg_cancel_size": 0.10, "otr": 0.05},
                    }
                    weights = placeholder_weights.get(persona, {})
                    if weights:
                        fig_ph = _build_shap_chart(weights, persona)
                        fig_ph.update_layout(title_text="📊 Typical Feature Importance (Placeholder)")
                        st.plotly_chart(fig_ph, use_container_width=True, config={"displayModeBar": False}, key=f"fig_ph_{idx}_{persona}")

            with narr_col:
                st.markdown(
                    "<div style='font-size:0.75rem; font-weight:700; color:#94a3b8; "
                    "text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
                    "🤖 LLM Case Narrative</div>",
                    unsafe_allow_html=True,
                )

                narr_key = f"narrative_{idx}"
                gen_narr = st.button(
                    "Generate LLM Explanation",
                    key=f"btn_narr_{idx}",
                    help="Calls the NarrativeGenerator LLM to write a compliance analyst case note.",
                )

                if gen_narr:
                    with st.spinner("Generating AI compliance case note…"):
                        shap_for_narr = st.session_state.get(shap_vals_key, {})
                        if not shap_for_narr:
                            # Auto-compute SHAP if not yet computed so LLM narrative has exact numbers
                            try:
                                shap_for_narr = _compute_shap_for_window(run_dir, selected_round, row)
                                if shap_for_narr:
                                    st.session_state[shap_vals_key] = shap_for_narr
                            except Exception:
                                pass
                        narrative = _get_llm_narrative(row, shap_for_narr, run_dir)
                        st.session_state[narr_key] = narrative

                narrative = st.session_state.get(narr_key)

                if narrative:
                    if "REJECTED" in narrative:
                        st.error(narrative)
                    else:
                        st.markdown(
                            f"""
                            <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(99,130,200,0.2);
                                        border-radius:10px; padding:1.1rem 1.25rem; line-height:1.65;">
                                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;
                                            letter-spacing:0.07em; font-weight:700; margin-bottom:0.6rem;">
                                    Compliance Analyst Case Note · AI Generated
                                </div>
                                <div style="font-size:0.88rem; color:#cbd5e1;">{narrative}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    # Context info while narrative not yet generated
                    persona_explainers = {
                        "spoofing": (
                            "**Spoofing** involves placing large orders to create the illusion of demand/supply, "
                            "then cancelling them before execution. Key signals: very high cancel rate, "
                            "abnormally large order sizes placed far from mid-price."
                        ),
                        "wash_trading": (
                            "**Wash Trading** involves a trader (or colluding pair) trading with themselves "
                            "to inflate volume metrics artificially. Key signals: circular volume patterns, "
                            "near-zero net position, uniform trade sizes."
                        ),
                        "pump_and_dump": (
                            "**Pump & Dump** involves coordinated buying to inflate price, followed by "
                            "a rapid sell-off. Key signals: abnormal volume spike, strong price momentum, "
                            "high top-K buy imbalance during accumulation phase."
                        ),
                        "layering": (
                            "**Layering** places multiple orders at different price levels to create "
                            "a false impression of a deep order book, then cancels them rapidly. "
                            "Key signals: high OTR, elevated order book imbalance, high cancel rate."
                        ),
                    }
                    expl = persona_explainers.get(persona, "")
                    if expl:
                        st.markdown(
                            f"""
                            <div style="background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.2);
                                        border-radius:10px; padding:1rem 1.2rem; line-height:1.6;">
                                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;
                                            letter-spacing:0.07em; font-weight:700; margin-bottom:0.5rem;">
                                    📚 Pattern Context
                                </div>
                                <div style="font-size:0.84rem; color:#94a3b8;">{expl}</div>
                                <div style="margin-top:0.75rem; font-size:0.75rem; color:#475569;">
                                    Click <b>Generate LLM Explanation</b> for a case-specific narrative.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("")  # spacing between cards
