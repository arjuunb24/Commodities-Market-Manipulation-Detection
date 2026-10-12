"""
dashboard/components/tab_market.py
===================================
Tab 1 — Market Overview

Sections:
  • KPI strip    : Total trades, volume, unique traders, manipulation events
  • Candlestick  : OHLCV with manipulation event overlays (shaded regions)
  • Volume bars  : Trade volume with buy/sell split + anomaly highlights
  • Microstructure metrics: cancel_rate, OTR, order book imbalance over time
  • Price return distribution: histogram + KDE overlay
  • Trader activity heatmap: top traders by volume × hour
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.data_loader import (
    load_ohlcv,
    load_trade_log,
    load_microstructure_resampled,
    load_manipulation_events,
    load_round_metrics_json,
)

ALL_COMMODITIES_KEY = "ALL_COMMODITIES"

COMMODITY_LABELS = {
    ALL_COMMODITIES_KEY: "🌐 All Commodities (Cross-Asset Overview)",
    "brent_crude":   "🛢️ Brent Crude",
    "crude_oil_wti": "⛽ WTI Crude",
    "natural_gas":   "⚡ Natural Gas",
    "gold":          "🥇 Gold",
    "silver":        "🥈 Silver",
    "copper":        "🥉 Copper",
    "corn":          "🌽 Corn",
    "wheat":         "🌾 Wheat",
}

COMMODITY_COLORS = {
    "brent_crude":   "#0284c7",  # Sky / Navy Blue
    "crude_oil_wti": "#06b6d4",  # Cyan
    "natural_gas":   "#38bdf8",  # Ice Blue
    "gold":          "#eab308",  # Gold / Amber
    "silver":        "#cbd5e1",  # Silver / Slate
    "copper":        "#f97316",  # Copper / Orange
    "corn":          "#84cc16",  # Lime Green
    "wheat":         "#d97706",  # Wheat / Dark Gold
}

# ── Design tokens ────────────────────────────────────────────────────────────
CHART_BG     = "rgba(10,14,26,0)"
PAPER_BG     = "rgba(10,14,26,0)"
GRID_COLOR   = "rgba(99,130,200,0.1)"
FONT_COLOR   = "#94a3b8"
ACCENT_BLUE  = "#3b82f6"
ACCENT_CYAN  = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_RED   = "#ef4444"
ACCENT_AMBER = "#f59e0b"
ACCENT_PURPLE= "#8b5cf6"

PERSONA_COLORS = {
    "spoofing":      ACCENT_RED,
    "wash_trading":  ACCENT_AMBER,
    "pump_and_dump": ACCENT_PURPLE,
    "layering":      ACCENT_CYAN,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(99,130,200,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
)


def _canon_persona(raw: str) -> str:
    if "spoof" in raw: return "spoofing"
    if "wash"  in raw: return "wash_trading"
    if "pump"  in raw: return "pump_and_dump"
    if "layer" in raw: return "layering"
    return raw


def _persona_badge(persona: str) -> str:
    colors = {
        "spoofing":      ("#f87171", "rgba(239,68,68,0.18)",  "rgba(239,68,68,0.35)"),
        "wash_trading":  ("#fbbf24", "rgba(245,158,11,0.18)", "rgba(245,158,11,0.35)"),
        "pump_and_dump": ("#a78bfa", "rgba(139,92,246,0.18)", "rgba(139,92,246,0.35)"),
        "layering":      ("#22d3ee", "rgba(6,182,212,0.18)",  "rgba(6,182,212,0.35)"),
    }
    c, bg, border = colors.get(persona, ("#94a3b8", "rgba(148,163,184,0.15)", "rgba(148,163,184,0.3)"))
    label = persona.replace("_", " ").title()
    return (
        f"<span style='background:{bg}; color:{c}; border:1px solid {border}; "
        f"padding:2px 9px; border-radius:99px; font-size:0.7rem; font-weight:700; "
        f"letter-spacing:0.05em; text-transform:uppercase;'>{label}</span>"
    )


# ── KPI helpers ──────────────────────────────────────────────────────────────

def _render_kpi_strip(trades: pd.DataFrame, events: pd.DataFrame, round_metrics: dict):
    """Top row of KPI metric cards."""
    personas = round_metrics.get("personas", {})

    total_trades = len(trades) if not trades.empty else 0
    total_volume = int(trades["quantity"].sum()) if not trades.empty and "quantity" in trades.columns else 0
    unique_traders = 0
    if not trades.empty:
        buyers = set(trades.get("buyer_trader_id", pd.Series()).dropna().unique())
        sellers = set(trades.get("seller_trader_id", pd.Series()).dropna().unique())
        unique_traders = len(buyers | sellers)
    manip_count = len(events) if not events.empty else 0

    # Build F1 labels for each persona
    f1_labels = []
    for persona, m in personas.items():
        f1_labels.append(f"{persona.replace('_',' ').title()}: {m.get('f1', 0.0):.3f}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💹 Total Trades",    f"{total_trades:,}")
    c2.metric("📦 Total Volume",    f"{total_volume:,}")
    c3.metric("👥 Unique Traders",  f"{unique_traders:,}")
    c4.metric("🚨 Manip Events",    f"{manip_count:,}")

    # Best and worst F1 from personas
    if personas:
        sorted_personas = sorted(personas.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        best_p, best_m = sorted_personas[0]
        worst_p, worst_m = sorted_personas[-1]
        c5.metric(
            f"✅ Best F1 ({best_p.replace('_',' ').title()[:8]})",
            f"{best_m.get('f1', 0.0):.3f}",
            delta=f"P={best_m.get('precision',0):.2f} R={best_m.get('recall',0):.2f}",
        )
        c6.metric(
            f"⚠️ Worst F1 ({worst_p.replace('_',' ').title()[:8]})",
            f"{worst_m.get('f1', 0.0):.3f}",
            delta=f"FPR={worst_m.get('fpr',0):.2f}",
            delta_color="inverse",
        )


# ── Chart builders ───────────────────────────────────────────────────────────

def _build_candlestick_chart(ohlcv: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """
    Professional OHLCV candlestick with:
    - Manipulation event shaded regions
    - VWAP line
    - Volume subplot
    """
    if ohlcv.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No OHLCV data available for this round")
        return fig

    # Use timestamp column (added by data_loader from bar_start)
    ts_col   = "timestamp" if "timestamp" in ohlcv.columns else "bar_start"
    open_col  = next((c for c in ["open",  "Open"]  if c in ohlcv.columns), None)
    high_col  = next((c for c in ["high",  "High"]  if c in ohlcv.columns), None)
    low_col   = next((c for c in ["low",   "Low"]   if c in ohlcv.columns), None)
    close_col = next((c for c in ["close", "Close", "vwap"] if c in ohlcv.columns), None)
    volume_col= next((c for c in ["volume","Volume","total_volume","total_vol"] if c in ohlcv.columns), None)

    if not all([open_col, high_col, low_col, close_col]):
        # Fallback: build OHLCV from close-only data
        if close_col:
            ohlcv = ohlcv.copy()
            ohlcv["open"]  = ohlcv[close_col].shift(1).fillna(ohlcv[close_col])
            ohlcv["high"]  = ohlcv[close_col]
            ohlcv["low"]   = ohlcv[close_col]
            open_col, high_col, low_col = "open", "high", "low"
        else:
            fig = go.Figure()
            fig.update_layout(**PLOTLY_LAYOUT, title="No OHLCV price data found")
            return fig

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=["Price (OHLCV + Manipulation Events)", "Volume"],
    )

    # ── Candlesticks ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=ohlcv[ts_col],
            open=ohlcv[open_col],
            high=ohlcv[high_col],
            low=ohlcv[low_col],
            close=ohlcv[close_col],
            name="OHLCV",
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
            increasing_fillcolor=f"rgba(16,185,129,0.7)",
            decreasing_fillcolor=f"rgba(239,68,68,0.7)",
            line_width=1,
        ),
        row=1, col=1,
    )

    # ── VWAP line (if available) ─────────────────────────────────────────
    if "vwap" in ohlcv.columns and close_col != "vwap":
        fig.add_trace(
            go.Scatter(
                x=ohlcv[ts_col], y=ohlcv["vwap"],
                name="VWAP",
                line=dict(color=ACCENT_CYAN, width=1.5, dash="dot"),
                opacity=0.85,
            ),
            row=1, col=1,
        )

    # ── Manipulation event shaded regions ────────────────────────────────
    if not events.empty and "start_time" in events.columns:
        for _, ev in events.iterrows():
            persona = _canon_persona(str(ev.get("persona", "unknown")))
            color = PERSONA_COLORS.get(persona, "#94a3b8")
            fig.add_vrect(
                x0=ev["start_time"], x1=ev["end_time"],
                fillcolor=color, opacity=0.08,
                layer="below", line_width=0,
                row=1, col=1,
            )
            # Add annotation marker at top
            fig.add_vline(
                x=ev["start_time"],
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.5,
                row=1, col=1,
            )

    # ── Volume bars ──────────────────────────────────────────────────────
    if volume_col:
        fig.add_trace(
            go.Bar(
                x=ohlcv[ts_col],
                y=ohlcv[volume_col],
                name="Volume",
                marker_color=ACCENT_BLUE,
                marker_opacity=0.6,
            ),
            row=2, col=1,
        )

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1,
                     title_font=dict(size=11), gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     title_font=dict(size=11), gridcolor=GRID_COLOR, zeroline=False)
    return fig


def _build_microstructure_chart(trades: pd.DataFrame, micro_df: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """
    Rolling microstructure metrics chart:
    - Cancel rate (spoofing signal)
    - Order-to-Trade Ratio (OTR)
    - Buy/sell imbalance
    """
    if trades.empty or "timestamp" not in trades.columns:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No trade data for microstructure analysis")
        return fig

    trades_r = trades.set_index("timestamp").sort_index()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.3, 0.32],
        subplot_titles=["Cancel Rate (Spoofing Signal)", "Order-to-Trade Ratio (OTR)", "Buy/Sell Volume Imbalance"],
    )

    freq = "5min"

    # ── Cancel Rate & OTR (from pre-aggregated micro_df) ───────────────────
    if not micro_df.empty and "cancel_rate" in micro_df.columns:
        fig.add_trace(
            go.Scatter(
                x=micro_df.index, y=micro_df["cancel_rate"].values,
                name="Cancel Rate",
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.12)",
                line=dict(color=ACCENT_RED, width=2),
            ),
            row=1, col=1,
        )
        # Reference line at 0.5
        fig.add_hline(y=0.5, line=dict(color=ACCENT_RED, dash="dash", width=1), opacity=0.5, row=1, col=1)

        # OTR
        trade_count = trades_r.resample(freq).size()
        submits = micro_df["submits"] if "submits" in micro_df.columns else pd.Series(0, index=micro_df.index)
        otr = (submits / (trade_count + 1e-9)).reindex(micro_df.index).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=otr.index, y=otr.values,
                name="OTR",
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.12)",
                line=dict(color=ACCENT_AMBER, width=2),
            ),
            row=2, col=1,
        )
    else:
        # Fallback empty traces
        fig.add_trace(go.Scatter(x=[], y=[], name="Cancel Rate"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], name="OTR"), row=2, col=1)

    # ── Buy/Sell Imbalance ───────────────────────────────────────────────
    if "quantity" in trades_r.columns:
        buy_col  = "buy_vol"  if "buy_vol"  in trades_r.columns else "quantity"
        sell_col = "sell_vol" if "sell_vol" in trades_r.columns else "quantity"

        if "buyer_trader_id" in trades_r.columns:
            is_buy = ~trades_r["buyer_trader_id"].str.startswith("mm", na=False)
            buy_vol  = trades_r[is_buy]["quantity"].resample(freq).sum()
            sell_vol = trades_r[~is_buy]["quantity"].resample(freq).sum()
        else:
            buy_vol  = trades_r["quantity"].resample(freq).sum()
            sell_vol = pd.Series(0, index=buy_vol.index)

        total = buy_vol + sell_vol + 1e-9
        imbalance = (buy_vol - sell_vol) / total

        fig.add_trace(
            go.Bar(
                x=imbalance.index,
                y=imbalance.clip(-1, 1).values,
                name="Buy/Sell Imbalance",
                marker_color=[ACCENT_GREEN if v >= 0 else ACCENT_RED for v in imbalance.values],
                marker_opacity=0.75,
            ),
            row=3, col=1,
        )
        fig.add_hline(y=0, line=dict(color=FONT_COLOR, width=1), opacity=0.4, row=3, col=1)

    # ── Manipulation event markers ────────────────────────────────────────
    if not events.empty and "start_time" in events.columns:
        for _, ev in events.iterrows():
            persona = _canon_persona(str(ev.get("persona", "")))
            color = PERSONA_COLORS.get(persona, "#94a3b8")
            for r in [1, 2, 3]:
                fig.add_vrect(
                    x0=ev["start_time"], x1=ev["end_time"],
                    fillcolor=color, opacity=0.06, layer="below", line_width=0,
                    row=r, col=1,
                )

    fig.update_layout(**PLOTLY_LAYOUT, height=480, showlegend=True)
    return fig


def _build_volume_profile(trades: pd.DataFrame) -> go.Figure:
    """
    Horizontal volume profile (price-level histogram) + time-volume bar.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        shared_yaxes=True,
        subplot_titles=["Volume Over Time (Buy vs Sell)", "Volume Profile by Price"],
    )

    if trades.empty or "timestamp" not in trades.columns:
        fig.update_layout(**PLOTLY_LAYOUT, title="No trade data")
        return fig

    trades_r = trades.set_index("timestamp").sort_index()
    freq = "5min"

    if "buyer_trader_id" in trades_r.columns and "quantity" in trades_r.columns:
        is_buy  = ~trades_r["buyer_trader_id"].str.startswith("mm", na=False)
        buy_vol  = trades_r[is_buy]["quantity"].resample(freq).sum()
        sell_vol = trades_r[~is_buy]["quantity"].resample(freq).sum()

        fig.add_trace(
            go.Bar(x=buy_vol.index, y=buy_vol.values, name="Buy Volume",
                   marker_color=f"rgba(16,185,129,0.7)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=sell_vol.index, y=-sell_vol.values, name="Sell Volume",
                   marker_color=f"rgba(239,68,68,0.7)"),
            row=1, col=1,
        )

    # ── Price volume profile ──────────────────────────────────────────────
    if "price" in trades_r.columns and "quantity" in trades_r.columns:
        price_bins = pd.cut(trades_r["price"], bins=40)
        vol_by_price = trades_r.groupby(price_bins, observed=False)["quantity"].sum()
        mid_prices   = [iv.mid for iv in vol_by_price.index]
        fig.add_trace(
            go.Bar(
                x=vol_by_price.values,
                y=mid_prices,
                orientation="h",
                name="Price Profile",
                marker_color=ACCENT_CYAN,
                marker_opacity=0.65,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        barmode="overlay",
        showlegend=True,
    )
    return fig


def _build_price_return_dist(trades: pd.DataFrame) -> go.Figure:
    """Returns distribution histogram + normal overlay."""
    if trades.empty or "price" not in trades.columns:
        return None

    prices   = trades["price"].dropna().sort_values()
    returns  = prices.pct_change().dropna()

    if len(returns) < 10:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=80,
            name="Returns",
            marker_color=ACCENT_BLUE,
            marker_opacity=0.7,
            histnorm="probability density",
        )
    )

    # Normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x_norm = np.linspace(returns.quantile(0.01), returns.quantile(0.99), 200)
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
    fig.add_trace(
        go.Scatter(x=x_norm, y=y_norm, name="Normal Fit",
                   line=dict(color=ACCENT_AMBER, width=2, dash="dash"))
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title_text="Price Return Distribution",
        xaxis_title="Return",
        yaxis_title="Density",
        showlegend=True,
    )
    return fig


def _build_trader_heatmap(trades: pd.DataFrame) -> go.Figure | None:
    """Top-20 traders activity heatmap: trader × time_bucket."""
    if trades.empty or "timestamp" not in trades.columns:
        return None
    if "buyer_trader_id" not in trades.columns:
        return None

    trades = trades.copy()
    trades["hour_bucket"] = trades["timestamp"].dt.floor("30min")

    # Aggregate buyer + seller activity per trader
    buy_agg  = trades.groupby(["buyer_trader_id",  "hour_bucket"])["quantity"].sum().reset_index()
    sell_agg = trades.groupby(["seller_trader_id", "hour_bucket"])["quantity"].sum().reset_index()
    buy_agg.columns  = ["trader_id", "hour_bucket", "volume"]
    sell_agg.columns = ["trader_id", "hour_bucket", "volume"]
    combined = pd.concat([buy_agg, sell_agg]).groupby(["trader_id", "hour_bucket"])["volume"].sum().reset_index()

    # Top 20 traders by total volume
    top_traders = combined.groupby("trader_id")["volume"].sum().nlargest(20).index
    combined = combined[combined["trader_id"].isin(top_traders)]

    pivot = combined.pivot(index="trader_id", columns="hour_bucket", values="volume").fillna(0)
    if pivot.empty:
        return None

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(c)[-8:] for c in pivot.columns],
            y=pivot.index,
            colorscale=[
                [0.0, "rgba(10,14,26,1)"],
                [0.3, "rgba(30,58,138,0.8)"],
                [0.6, "rgba(59,130,246,0.8)"],
                [1.0, "rgba(6,182,212,1)"],
            ],
            showscale=True,
            colorbar=dict(title="Volume", tickfont=dict(size=10)),
            hovertemplate="Trader: %{y}<br>Time: %{x}<br>Volume: %{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title_text="Top-20 Trader Activity Heatmap (30-min buckets)",
        xaxis_title="Time",
        yaxis_title="Trader ID",
    )
    fig.update_yaxes(tickfont=dict(size=9), gridcolor=GRID_COLOR)
    return fig


# ── Multi-Commodity (Cross-Asset) Chart Builders ──────────────────────────

def _build_all_commodities_normalized_chart(ohlcv: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    """Indexed relative performance chart (Base = 100) for all commodities on a single timeline."""
    ohlcv_valid = ohlcv.dropna(subset=["close", "timestamp"])
    if ohlcv_valid.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No price data available")
        return fig
    pivot = ohlcv_valid.pivot_table(index="timestamp", columns="commodity", values="close", aggfunc="last").ffill().bfill()
    if pivot.empty or len(pivot) < 2:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="Insufficient price history for multi-asset comparison")
        return fig

    first_row = pivot.iloc[0].replace(0, np.nan)
    norm_df = (pivot / first_row) * 100.0

    fig = go.Figure()
    for c in norm_df.columns:
        color = COMMODITY_COLORS.get(c, ACCENT_BLUE)
        label = COMMODITY_LABELS.get(c, c.replace("_", " ").title())
        fig.add_trace(
            go.Scatter(
                x=norm_df.index,
                y=norm_df[c],
                mode="lines",
                name=label,
                line=dict(color=color, width=2.2),
                hovertemplate="<b>" + label + "</b><br>Time: %{x}<br>Norm Index: %{y:.2f} (<b>%{customdata:+.2f}%</b>)<extra></extra>",
                customdata=norm_df[c] - 100.0,
            )
        )

    # 100 baseline
    fig.add_hline(
        y=100, line_dash="dash", line_color="rgba(255,255,255,0.25)",
        annotation_text="Baseline (100)", annotation_position="bottom right",
        annotation_font_size=10, annotation_font_color="#94a3b8"
    )

    # Highlight manipulation windows
    if not events.empty and "start_tick" in events.columns and "end_tick" in events.columns:
        ts_map = ohlcv.dropna(subset=["bar_start", "timestamp"]).drop_duplicates(subset=["bar_start"]).set_index("bar_start")["timestamp"]
        for _, ev in events.iterrows():
            st_tick = ev.get("start_tick")
            en_tick = ev.get("end_tick")
            p_color = PERSONA_COLORS.get(_canon_persona(ev.get("persona", "")), ACCENT_RED)
            if st_tick in ts_map.index and en_tick in ts_map.index:
                fig.add_vrect(
                    x0=ts_map.loc[st_tick],
                    x1=ts_map.loc[en_tick],
                    fillcolor=p_color,
                    opacity=0.10,
                    line_width=1,
                    line_dash="dot",
                    line_color=p_color,
                )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=420,
        title_text="Normalized Price Trajectories (Base = 100, Cross-Asset Comparison)",
        xaxis_title="Time",
        yaxis_title="Indexed Performance (100 = Initial Price)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15,23,42,0.8)",
            font=dict(size=10),
        ),
    )
    return fig


def _build_all_commodities_grid_charts(ohlcv: pd.DataFrame, events: pd.DataFrame, commodities: list[str]) -> go.Figure:
    """2x4 subplot grid displaying mini-candlestick / price charts for all 8 commodities simultaneously."""
    display_comms = [c for c in commodities if c != ALL_COMMODITIES_KEY][:8]
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[COMMODITY_LABELS.get(c, c.replace("_", " ").title()) for c in display_comms],
        vertical_spacing=0.18,
        horizontal_spacing=0.06,
    )

    for i, c in enumerate(display_comms):
        row = (i // 4) + 1
        col = (i % 4) + 1
        c_ohlcv = ohlcv[ohlcv["commodity"] == c].dropna(subset=["close", "timestamp"]).sort_values("timestamp")
        c_events = events[events["commodity"] == c] if not events.empty and "commodity" in events.columns else pd.DataFrame()
        c_color = COMMODITY_COLORS.get(c, ACCENT_CYAN)

        if not c_ohlcv.empty:
            fig.add_trace(
                go.Scatter(
                    x=c_ohlcv["timestamp"],
                    y=c_ohlcv["close"],
                    mode="lines",
                    name=COMMODITY_LABELS.get(c, c),
                    line=dict(color=c_color, width=2),
                    showlegend=False,
                    hovertemplate=f"<b>{c.replace('_',' ').title()}</b>: $%{{y:,.2f}}<extra></extra>",
                ),
                row=row, col=col,
            )
            if not c_events.empty:
                ts_map = c_ohlcv.drop_duplicates(subset=["bar_start"]).set_index("bar_start")["timestamp"]
                for _, ev in c_events.iterrows():
                    st_tick = ev.get("start_tick")
                    en_tick = ev.get("end_tick")
                    if st_tick in ts_map.index and en_tick in ts_map.index:
                        fig.add_vrect(
                            x0=ts_map.loc[st_tick],
                            x1=ts_map.loc[en_tick],
                            fillcolor="rgba(239,68,68,0.25)",
                            line_width=1,
                            line_color=ACCENT_RED,
                            row=row, col=col,
                        )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=480,
        title_text="Simultaneous 8-Commodity Price Profiles with Manipulation Shading",
    )
    fig.update_annotations(font_size=11, font_color="#e2e8f0")
    return fig


def _build_cross_asset_correlation_matrix(ohlcv: pd.DataFrame, commodities: list[str]) -> go.Figure | None:
    """8x8 Pearson correlation heatmap of returns across all commodities."""
    display_comms = [c for c in commodities if c != ALL_COMMODITIES_KEY][:8]
    ohlcv_valid = ohlcv[ohlcv["commodity"].isin(display_comms)].dropna(subset=["close", "timestamp"])
    if ohlcv_valid.empty:
        return None
    pivot = ohlcv_valid.pivot_table(index="timestamp", columns="commodity", values="close", aggfunc="last").ffill().bfill()
    if len(pivot) < 3:
        return None
    returns = pivot.pct_change().dropna()
    corr = returns.corr().fillna(0)

    short_labels = [COMMODITY_LABELS.get(c, c).split(" ")[-1] for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=short_labels,
            y=short_labels,
            colorscale=[
                [0.0, "rgba(239,68,68,0.9)"],
                [0.5, "rgba(15,23,42,0.95)"],
                [1.0, "rgba(6,182,212,0.95)"],
            ],
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Corr", tickfont=dict(size=10)),
            text=corr.values.round(2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=10, color="#ffffff"),
            hovertemplate="Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=360,
        title_text="Cross-Asset Return Correlation Matrix (Inter-Market Dynamics)",
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def _build_cross_asset_volume_chart(trades: pd.DataFrame, commodities: list[str]) -> go.Figure:
    """Total volume distribution across all commodities."""
    display_comms = [c for c in commodities if c != ALL_COMMODITIES_KEY][:8]
    vol_series = trades.groupby("commodity")["quantity"].sum().reindex(display_comms).fillna(0) if not trades.empty and "commodity" in trades.columns else pd.Series(0, index=display_comms)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[COMMODITY_LABELS.get(c, c) for c in vol_series.index],
            y=vol_series.values,
            marker_color=[COMMODITY_COLORS.get(c, ACCENT_BLUE) for c in vol_series.index],
            text=[f"{v:,.0f}" for v in vol_series.values],
            textposition="auto",
            hovertemplate="Commodity: %{x}<br>Volume: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=360,
        title_text="Trading Volume Distribution Across Assets",
        xaxis_title="",
        yaxis_title="Volume",
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def _build_cross_asset_manipulation_chart(events: pd.DataFrame, commodities: list[str]) -> go.Figure:
    """Stacked bar chart of manipulation events by commodity and persona."""
    display_comms = [c for c in commodities if c != ALL_COMMODITIES_KEY][:8]
    fig = go.Figure()
    if not events.empty and "commodity" in events.columns and "persona" in events.columns:
        events_agg = events.groupby(["commodity", "persona"]).size().unstack(fill_value=0).reindex(display_comms).fillna(0)
        for p_col in events_agg.columns:
            p_canon = _canon_persona(p_col)
            fig.add_trace(
                go.Bar(
                    name=p_col.replace("_", " ").title(),
                    x=[COMMODITY_LABELS.get(c, c) for c in events_agg.index],
                    y=events_agg[p_col].values,
                    marker_color=PERSONA_COLORS.get(p_canon, ACCENT_RED),
                    hovertemplate="Commodity: %{x}<br>Persona: " + p_col.title() + "<br>Events: %{y}<extra></extra>",
                )
            )
    else:
        fig.add_trace(go.Bar(x=[COMMODITY_LABELS.get(c, c) for c in display_comms], y=[0]*len(display_comms)))

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=360,
        barmode="stack",
        title_text="Manipulation Events by Commodity & Adversarial Persona",
        xaxis_title="",
        yaxis_title="Flagged Windows",
    )
    fig.update_xaxes(tickangle=-25)
    return fig


def _render_all_commodities_view(
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    round_metrics: dict,
    commodities: list[str],
) -> None:
    """Renders the comprehensive cross-asset simultaneous surveillance overview."""
    display_comms = [c for c in commodities if c != ALL_COMMODITIES_KEY][:8]

    # Global KPI strip across the entire commodities market
    _render_kpi_strip(trades, events, round_metrics)

    # ── Snapshot Deck ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🌐 Cross-Asset Market Snapshot</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, c in enumerate(display_comms):
        col = cols[i % 4]
        c_ohlcv = ohlcv[ohlcv["commodity"] == c] if not ohlcv.empty and "commodity" in ohlcv.columns else pd.DataFrame()
        c_trades = trades[trades["commodity"] == c] if not trades.empty and "commodity" in trades.columns else pd.DataFrame()
        c_events = events[events["commodity"] == c] if not events.empty and "commodity" in events.columns else pd.DataFrame()

        valid_p = c_ohlcv["close"].dropna() if not c_ohlcv.empty and "close" in c_ohlcv.columns else pd.Series(dtype=float)
        if valid_p.empty and not c_trades.empty and "price" in c_trades.columns:
            valid_p = c_trades["price"].dropna()
        latest_p = float(valid_p.iloc[-1]) if not valid_p.empty else 0.0
        first_p = float(valid_p.iloc[0]) if not valid_p.empty else 0.0
        pct = ((latest_p - first_p) / first_p * 100.0) if first_p > 0 else 0.0
        c_vol = int(c_trades["quantity"].sum()) if not c_trades.empty and "quantity" in c_trades.columns else 0
        ev_cnt = len(c_events)

        status_bg = "rgba(239,68,68,0.2)" if ev_cnt > 0 else "rgba(16,185,129,0.15)"
        status_color = "#f87171" if ev_cnt > 0 else "#34d399"
        status_label = f"🚨 {ev_cnt} Flagged" if ev_cnt > 0 else "🟢 Clean"
        pct_color = "#10b981" if pct >= 0 else "#ef4444"
        pct_sign = "+" if pct >= 0 else ""

        with col:
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.7); border:1px solid rgba(99,130,200,0.22);
                            border-radius:10px; padding:0.75rem 0.9rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#ffffff; font-size:0.88rem;">{COMMODITY_LABELS.get(c, c)}</span>
                        <span style="font-size:0.7rem; padding:0.12rem 0.45rem; border-radius:4px; font-weight:600; background:{status_bg}; color:{status_color};">
                            {status_label}
                        </span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:0.4rem;">
                        <span style="font-size:1.15rem; font-weight:700; color:#38bdf8;">${latest_p:,.2f}</span>
                        <span style="font-size:0.82rem; font-weight:600; color:{pct_color};">{pct_sign}{pct:.2f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-top:0.35rem; border-top:1px solid rgba(99,130,200,0.1); padding-top:0.3rem;">
                        <span>Vol: <b style="color:#e2e8f0;">{c_vol:,}</b></span>
                        <span>Trades: <b style="color:#e2e8f0;">{len(c_trades):,}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Section 1: Normalized Relative Performance ──────────────────────────
    st.markdown("<div class='section-title'>📈 Normalized Relative Price Performance (Base = 100)</div>", unsafe_allow_html=True)
    st.caption("All 8 commodities indexed to 100 at round open. Enables direct cross-asset divergence, contagion, and lead-lag analysis.")
    fig_norm = _build_all_commodities_normalized_chart(ohlcv, events)
    st.plotly_chart(fig_norm, use_container_width=True, config={"displayModeBar": False}, key="fig_all_norm")

    # ── Section 2: Small Multiples 8-Commodity Simultaneous Grid ─────────────
    st.markdown("<div class='section-title'>⚡ Simultaneous 8-Commodity Price Profiles</div>", unsafe_allow_html=True)
    st.caption("Side-by-side synchronized price trajectories for all 8 commodities. Red shaded spans indicate ground-truth manipulation intervals.")
    fig_grid = _build_all_commodities_grid_charts(ohlcv, events, commodities)
    st.plotly_chart(fig_grid, use_container_width=True, config={"displayModeBar": False}, key="fig_all_grid")

    # ── Section 3: Correlation & Volume Share Side-by-Side ───────────────────
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("<div class='section-title'>🔗 Cross-Asset Return Correlation Matrix</div>", unsafe_allow_html=True)
        fig_corr = _build_cross_asset_correlation_matrix(ohlcv, commodities)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False}, key="fig_all_corr")
        else:
            st.info("Insufficient data for correlation matrix.")

    with col_r:
        st.markdown("<div class='section-title'>📦 Volume & Trade Distribution</div>", unsafe_allow_html=True)
        fig_vol_dist = _build_cross_asset_volume_chart(trades, commodities)
        st.plotly_chart(fig_vol_dist, use_container_width=True, config={"displayModeBar": False}, key="fig_all_vol_dist")

    # ── Section 4: Manipulation Attack Exposure ──────────────────────────────
    st.markdown("<div class='section-title'>🚨 Manipulation Attack Distribution Across Commodities</div>", unsafe_allow_html=True)
    st.caption("Distribution of ground-truth manipulation events per commodity broken down by adversarial persona.")
    fig_manip_dist = _build_cross_asset_manipulation_chart(events, commodities)
    st.plotly_chart(fig_manip_dist, use_container_width=True, config={"displayModeBar": False}, key="fig_all_manip_dist")


# ── Main render ──────────────────────────────────────────────────────────────

def render_market_tab(run_dir: Path | None, selected_round: int) -> None:
    if run_dir is None:
        st.warning("No run selected. Use the sidebar to pick a run.")
        return

    # ── Fast Data loading (without loading massive raw order log) ─────────────
    with st.spinner("Loading market data…"):
        ohlcv   = load_ohlcv(str(run_dir), selected_round)
        trades  = load_trade_log(str(run_dir), selected_round)
        events  = load_manipulation_events(str(run_dir), selected_round)
        r_metrics = load_round_metrics_json(str(run_dir), selected_round)

    # ── Commodity selector ───────────────────────────────────────────────
    raw_commodities = []
    for df in [ohlcv, trades, events]:
        if not df.empty and "commodity" in df.columns:
            raw_commodities = sorted(df["commodity"].unique().tolist())
            break

    if not raw_commodities:
        st.info("No commodities found in this round's dataset.")
        return

    options = [ALL_COMMODITIES_KEY] + raw_commodities
    default_comm_idx = 0
    saved_comm = st.session_state.get("market_commodity_sel")
    if saved_comm in options:
        default_comm_idx = options.index(saved_comm)

    col_c1, col_c2 = st.columns([1.7, 3.3])
    with col_c1:
        sel_commodity = st.selectbox(
            "Select Commodity Asset / Scope",
            options=options,
            index=default_comm_idx,
            format_func=lambda c: COMMODITY_LABELS.get(c, c.replace("_", " ").title()),
            key="market_commodity_sel",
        )

    # ── Branch: All Commodities Simultaneous Overview ────────────────────────
    if sel_commodity == ALL_COMMODITIES_KEY:
        total_market_vol = int(trades["quantity"].sum()) if not trades.empty and "quantity" in trades.columns else 0
        total_events = len(events) if not events.empty else 0
        with col_c2:
            st.markdown(
                f"""
                <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(99,130,200,0.22);
                            border-radius:10px; padding:0.6rem 1.2rem; margin-top:1.55rem;
                            display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">View Scope:</span>
                        <b style="color:#38bdf8; font-size:0.95rem; margin-left:0.4rem;">🌐 All 8 Commodities Simultaneously</b>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:#94a3b8;">Total Market Vol:</span>
                        <b style="color:#e2e8f0; font-size:0.95rem; margin-left:0.4rem;">{total_market_vol:,}</b>
                    </div>
                    <div>
                        <span style="font-size:0.75rem; color:#94a3b8;">Total Flagged Windows:</span>
                        <b style="color:{'#ef4444' if total_events > 0 else '#10b981'}; font-size:0.95rem; margin-left:0.4rem;">{total_events}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        _render_all_commodities_view(ohlcv, trades, events, r_metrics, raw_commodities)
        return

    # Filter data for selected commodity
    comm_trades = trades[trades["commodity"] == sel_commodity] if not trades.empty and "commodity" in trades.columns else trades
    comm_ohlcv  = ohlcv[ohlcv["commodity"] == sel_commodity] if not ohlcv.empty and "commodity" in ohlcv.columns else ohlcv
    comm_events = events[events["commodity"] == sel_commodity] if not events.empty and "commodity" in events.columns else events

    valid_prices = comm_ohlcv["close"].dropna() if not comm_ohlcv.empty and "close" in comm_ohlcv.columns else pd.Series(dtype=float)
    if valid_prices.empty and not comm_trades.empty and "price" in comm_trades.columns:
        valid_prices = comm_trades["price"].dropna()
    last_price = float(valid_prices.iloc[-1]) if not valid_prices.empty else 0.0
    comm_vol = int(comm_trades["quantity"].sum()) if not comm_trades.empty and "quantity" in comm_trades.columns else 0
    event_cnt = len(comm_events)

    with col_c2:
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.65); border:1px solid rgba(99,130,200,0.22);
                        border-radius:10px; padding:0.6rem 1.2rem; margin-top:1.55rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Asset Focus:</span>
                    <b style="color:#ffffff; font-size:0.95rem; margin-left:0.4rem;">{COMMODITY_LABELS.get(sel_commodity, sel_commodity)}</b>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94a3b8;">Latest Price:</span>
                    <b style="color:#38bdf8; font-size:1.05rem; margin-left:0.4rem;">${last_price:,.2f}</b>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94a3b8;">Volume:</span>
                    <b style="color:#e2e8f0; font-size:0.95rem; margin-left:0.4rem;">{comm_vol:,}</b>
                </div>
                <div>
                    <span style="font-size:0.75rem; color:#94a3b8;">Flagged Windows:</span>
                    <b style="color:{'#ef4444' if event_cnt > 0 else '#10b981'}; font-size:0.95rem; margin-left:0.4rem;">{event_cnt}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── KPI strip ────────────────────────────────────────────────────────
    _render_kpi_strip(comm_trades, comm_events, r_metrics)

    # ── Manipulation event legend ────────────────────────────────────────
    if not comm_events.empty and "persona" in comm_events.columns:
        personas_in_round = comm_events["persona"].apply(_canon_persona).unique()
        badges = " &nbsp; ".join(_persona_badge(p) for p in sorted(set(personas_in_round)))
        st.markdown(
            f"<div style='margin: 0.5rem 0 0.75rem; font-size:0.78rem; color:#94a3b8;'>"
            f"Active manipulation personas in window: &nbsp; {badges}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section 1: Candlestick ────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 OHLCV Price Chart</div>", unsafe_allow_html=True)
    fig_candle = _build_candlestick_chart(comm_ohlcv, comm_events)
    st.plotly_chart(fig_candle, use_container_width=True, config={"displayModeBar": False}, key=f"fig_candle_{sel_commodity}")

    # ── Section 2: Volume profile ────────────────────────────────────────
    st.markdown("<div class='section-title'>📦 Volume Analysis</div>", unsafe_allow_html=True)
    fig_vol = _build_volume_profile(comm_trades)
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False}, key=f"fig_vol_{sel_commodity}")

    # ── Section 3: Microstructure (Cached Pre-aggregated) ────────────────
    st.markdown("<div class='section-title'>⚡ Microstructure Signals</div>", unsafe_allow_html=True)
    st.caption(
        "Rolling 5-min window metrics. "
        "Elevated cancel rate (>50%) and high OTR are hallmarks of spoofing. "
        "Shaded regions indicate ground-truth manipulation windows."
    )
    micro_df = load_microstructure_resampled(str(run_dir), selected_round, sel_commodity)
    fig_micro = _build_microstructure_chart(comm_trades, micro_df, comm_events)
    st.plotly_chart(fig_micro, use_container_width=True, config={"displayModeBar": False}, key=f"fig_micro_{sel_commodity}")

    # ── Section 4: Returns + Heatmap side-by-side ────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-title'>📉 Return Distribution</div>", unsafe_allow_html=True)
        fig_ret = _build_price_return_dist(comm_trades)
        if fig_ret:
            st.plotly_chart(fig_ret, use_container_width=True, config={"displayModeBar": False}, key=f"fig_ret_{sel_commodity}")
        else:
            st.info("Insufficient trade data for return distribution.")

    with col_right:
        st.markdown("<div class='section-title'>🔥 Trader Activity Heatmap</div>", unsafe_allow_html=True)
        fig_heat = _build_trader_heatmap(comm_trades)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False}, key=f"fig_heat_{sel_commodity}")
        else:
            st.info("Insufficient trader data for heatmap.")
