"""
dashboard/components/tab_flags.py
==================================
Tab 2 — Flagged Trades (SRS FR-5.3, UC-2)

Shows what the DETECTOR flagged (round_N/flags.parquet, post-retrain), not the
simulator's ground truth. Consecutive flagged windows of the same account and
persona are merged into one case. Each case shows:
  • score, which layer fired (classifier / IsolationForest)
  • the stored SHAP attributions of the model that fired
  • whether the simulator's ground truth confirms it (kept visually distinct —
    ground truth, detector belief and LLM narrative are three different levels
    of certainty, SRS §8.1)
  • the account's trades over the case window, and its raw orders/trades
  • an optional LLM case note, fact-checked before display (FR-4.1)
Ground-truth episodes the detector missed are listed separately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from dashboard.data_loader import (
    load_flags,
    load_manipulation_events,
    load_order_log,
    load_trade_log,
    market_tick_range,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRID_COLOR = "rgba(99,130,200,0.1)"
FONT_COLOR = "#94a3b8"
PERSONA_COLORS = {
    "spoofing":      ("#ef4444", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.35)"),
    "wash_trading":  ("#f59e0b", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.35)"),
    "pump_and_dump": ("#8b5cf6", "rgba(139,92,246,0.12)", "rgba(139,92,246,0.35)"),
    "anomaly":       ("#06b6d4", "rgba(6,182,212,0.12)", "rgba(6,182,212,0.35)"),
}
PERSONA_ICONS = {"spoofing": "🎭", "wash_trading": "🔁", "pump_and_dump": "🚀", "anomaly": "❓"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(10,14,26,0)",
    plot_bgcolor="rgba(10,14,26,0)",
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=11),
    margin=dict(l=10, r=10, t=36, b=10),
)
FEATURE_DESCRIPTIONS = {
    "spoof_signature": "Share of the account's large cancelled orders during whose life it was filled on the opposite side",
    "large_cancel_ratio": "Share of the account's large orders that were cancelled",
    "fast_cancel_ratio": "Share of order volume cancelled within 10 s of placement",
    "max_order_size_rel": "Largest order vs. the market's median order size",
    "n_large_cancels": "Number of large orders cancelled in the window",
    "cancel_ratio": "Cancelled volume / placed volume",
    "top_cpty_share": "Share of the account's volume done with a single counterparty",
    "reciprocal_ratio": "Volume traded back and forth with the same counterparties",
    "net_flatness": "1 − |net position / gross volume| (1 = no net position change)",
    "price_dev_bp": "Average distance of the account's trades from VWAP (bp)",
    "coordinated_peers": "Other accounts bursting in the same direction in this window",
    "long_net_ratio": "Net position built over ~12 minutes / gross volume",
    "reversal_score": "Net buyer in prior windows, net seller now (or vice versa)",
    "signed_flow_return": "Own net direction × market return (bp)",
    "activity_vs_history": "Activity now vs. the account's own history (log ratio)",
    "aggressor_ratio": "Share of the account's volume where it took liquidity",
    "mkt_anomaly_score": "Market-regime anomaly percentile for the window",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_cases(flags: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive flagged windows of the same account + persona into cases."""
    f = flags[flags["final_flag"]].copy()
    if f.empty:
        return pd.DataFrame()
    f = f.sort_values(["commodity", "trader_id", "predicted_persona", "window_start"])
    cases = []
    for (comm, trader, persona), g in f.groupby(["commodity", "trader_id", "predicted_persona"], sort=False):
        current = None
        for _, row in g.iterrows():
            if current is not None and row["window_start"] <= current["end_tick"] + 1:
                current["end_tick"] = max(current["end_tick"], int(row["window_end"]))
                current["n_windows"] += 1
                current["triggered_by"] |= set(row["triggered_by"])
                if row["final_score"] > current["score"]:
                    current["score"] = float(row["final_score"])
                    current["top_shap_features"] = row["top_shap_features"]
                continue
            if current is not None:
                cases.append(current)
            current = {
                "commodity": comm, "trader_id": trader, "persona": persona,
                "start_tick": int(row["window_start"]), "end_tick": int(row["window_end"]),
                "score": float(row["final_score"]), "n_windows": 1,
                "triggered_by": set(row["triggered_by"]),
                "top_shap_features": row["top_shap_features"],
            }
        if current is not None:
            cases.append(current)
    df = pd.DataFrame(cases)
    df["triggered_by"] = df["triggered_by"].apply(sorted)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _match_ground_truth(cases: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the matching ground-truth episode (if any) to each case; return missed episodes."""
    cases = cases.copy()
    cases["gt_event"] = None
    cases["gt_persona"] = None
    matched: set[str] = set()
    if events.empty:
        return cases, events
    for i, c in cases.iterrows():
        for _, e in events.iterrows():
            if e["commodity"] != c["commodity"] or c["trader_id"] not in set(e["trader_ids"]):
                continue
            if c["end_tick"] < e["start_tick"] or c["start_tick"] > e["end_tick"]:
                continue
            cases.at[i, "gt_event"] = e["event_id"]
            cases.at[i, "gt_persona"] = e["persona"]
            if c["persona"] == e["persona"]:
                matched.add(e["event_id"])
            break
    missed = events[~events["event_id"].isin(matched)]
    return cases, missed


def _build_shap_chart(shap_values: dict[str, float], persona: str) -> go.Figure:
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k.replace("_", " ") for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=["rgba(239,68,68,0.8)" if v > 0 else "rgba(16,185,129,0.7)" for v in values],
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        textfont=dict(size=10, color=FONT_COLOR),
    ))
    source = "IsolationForest" if persona == "anomaly" else f"{persona.replace('_', ' ')} classifier"
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(labels) * 30 + 70),
        title_text=f"SHAP attribution — {source}",
        xaxis_title="Contribution (positive = pushes toward 'manipulation' / 'anomalous')",
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor="rgba(255,255,255,0.25)"),
        yaxis=dict(gridcolor=GRID_COLOR),
        showlegend=False,
    )
    return fig


def _case_chart(trades: pd.DataFrame, case: pd.Series, pad: int = 120) -> go.Figure | None:
    """Market price around the case window, with the flagged account's own trades marked."""
    if trades.empty:
        return None
    t = trades[(trades["commodity"] == case["commodity"])
               & trades["tick"].between(case["start_tick"] - pad, case["end_tick"] + pad)]
    if t.empty:
        return None
    px = t.groupby("tick")["price"].last()
    own = t[(t["buyer_trader_id"] == case["trader_id"]) | (t["seller_trader_id"] == case["trader_id"])]
    buys = own[own["buyer_trader_id"] == case["trader_id"]]
    sells = own[own["seller_trader_id"] == case["trader_id"]]
    fig = go.Figure()
    fig.add_vrect(x0=case["start_tick"], x1=case["end_tick"], fillcolor="rgba(239,68,68,0.08)",
                  line_width=0, annotation_text="flagged span", annotation_position="top left")
    fig.add_trace(go.Scatter(x=px.index, y=px.values, mode="lines", name="market price",
                             line=dict(color="#3b82f6", width=1.5)))
    fig.add_trace(go.Scatter(x=buys["tick"], y=buys["price"], mode="markers", name="account buys",
                             marker=dict(symbol="triangle-up", size=8, color="#10b981")))
    fig.add_trace(go.Scatter(x=sells["tick"], y=sells["price"], mode="markers", name="account sells",
                             marker=dict(symbol="triangle-down", size=8, color="#ef4444")))
    fig.update_layout(**PLOTLY_LAYOUT, height=240, title_text="Price around the case (tick axis)",
                      xaxis=dict(gridcolor=GRID_COLOR, title="tick"), yaxis=dict(gridcolor=GRID_COLOR),
                      legend=dict(orientation="h", y=-0.25))
    return fig


@st.cache_resource(show_spinner=False)
def _llm_config() -> dict:
    path = REPO_ROOT / "configs" / "llm_config.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _narrative(case: pd.Series, run_dir: Path) -> dict | None:
    cfg = _llm_config().get("narrative_generator", {})
    from src.agentic.llm_client import llm_unavailable_reason
    reason = llm_unavailable_reason(cfg)
    if reason:
        return {"error": reason}
    try:
        from src.agentic.narrative_generator import NarrativeGenerator
        gen = NarrativeGenerator(config=cfg, run_dir=run_dir)
        flag_row = {
            "trader_id": case["trader_id"],
            "commodity": case["commodity"],
            "flagged_persona": case["persona"],
            "window_start_tick": int(case["start_tick"]),
            "window_end_tick": int(case["end_tick"]),
            "detector_score": round(float(case["score"]), 3),
            "triggered_by": list(case["triggered_by"]),
        }
        shap = {k: round(float(v), 3) for k, v in case["top_shap_features"].items()}
        return gen.generate(flag_row=flag_row, shap_summary=shap)
    except Exception as exc:
        logger.warning("Narrative generation failed: %s", exc)
        return {"error": str(exc)}


# ── Main render ──────────────────────────────────────────────────────────────

def render_flags_tab(run_dir: Path | None, selected_round: int, market: str | None = None) -> None:
    if run_dir is None:
        st.info("No run selected. Run `scripts/run_full_pipeline.py` and pick the run in the sidebar.")
        return

    with st.spinner(f"Loading round {selected_round} detector output…"):
        flags = load_flags(str(run_dir), selected_round)
        events = load_manipulation_events(str(run_dir), selected_round, market)
        trades = load_trade_log(str(run_dir), selected_round, market)
        # flags.parquet covers all test markets of the round; keep the selected market's tick range.
        rng = market_tick_range(str(run_dir), selected_round, market) if market else None
        if rng is not None and not flags.empty:
            flags = flags[(flags["window_start"] >= rng[0]) & (flags["window_start"] <= rng[1])]

    if flags.empty:
        st.info(
            "No detector output (flags.parquet) for this round. Either the round did not finish or the "
            "run was produced before trader-level flags existed — re-run `scripts/run_full_pipeline.py`."
        )
        return

    cases = _build_cases(flags)
    cases, missed = _match_ground_truth(cases, events) if not cases.empty else (cases, events)

    # Persona classifiers name a manipulation type; the IsolationForest only says "unusual".
    # They are reported separately so the persona results are not buried under the much
    # noisier unsupervised layer.
    is_anom = cases["persona"].eq("anomaly") if not cases.empty else pd.Series(dtype=bool)
    cls = cases[~is_anom] if not cases.empty else cases
    n_conf = int((cls["gt_persona"] == cls["persona"]).sum()) if not cls.empty else 0
    n_fp = int(cls["gt_event"].isna().sum()) if not cls.empty else 0
    n_anom = int(is_anom.sum()) if not cases.empty else 0
    n_anom_hit = int((cases[is_anom]["gt_event"].notna()).sum()) if n_anom else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🚨 Persona cases", f"{len(cls)}", help="Cases raised by a persona classifier (spoofing / wash trading / pump-and-dump). Consecutive flagged windows of one account are merged.")
    c2.metric("✅ Confirmed by ground truth", f"{n_conf}", help="The simulator's own manipulation log contains this account, persona and time span.")
    c3.metric("⚠️ No matching episode", f"{n_fp}", help="Persona cases where the account was not manipulating at that time. Each costs an analyst investigation.")
    c4.metric("🕳️ Missed episodes", f"{len(missed)}", help="Ground-truth episodes no persona classifier flagged with the right persona.")
    c5.metric("❓ Unsupervised anomalies", f"{n_anom}",
              help=f"Accounts only the IsolationForest found unusual (no persona). {n_anom_hit} of them overlap a real episode; "
                   "the rest are mostly legitimate accounts with unusual but lawful behaviour (e.g. hedgers' large orders). "
                   "Shown only when 'anomaly' is selected in the Persona filter.")
    st.caption("Detector flags are model output on synthetic data — not findings about real traders. "
               "Metrics cover the test market selected in the sidebar.")

    if not missed.empty:
        with st.expander(f"Missed ground-truth episodes ({len(missed)})"):
            st.dataframe(missed[["event_id", "persona", "start_tick", "end_tick", "trader_ids"]],
                         use_container_width=True, hide_index=True)

    if cases.empty:
        st.success("The detector raised no flags in this round's test market. This is a valid result, not an error.")
        return

    st.markdown("---")
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        options = sorted(cases["persona"].unique())
        persona_filter = st.multiselect("Persona", options,
                                        default=[p for p in options if p != "anomaly"] or options,
                                        key="flags_persona_filter",
                                        help="Add 'anomaly' to review IsolationForest-only cases.")
    with f2:
        status_filter = st.multiselect("Ground truth", ["confirmed", "other persona", "no episode"],
                                       default=["confirmed", "other persona", "no episode"], key="flags_status_filter")
    with f3:
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.05, key="flags_score_slider")

    cases["status"] = np.where(cases["gt_event"].isna(), "no episode",
                               np.where(cases["gt_persona"] == cases["persona"], "confirmed", "other persona"))
    view = cases[cases["persona"].isin(persona_filter) & cases["status"].isin(status_filter)
                 & (cases["score"] >= min_score)].reset_index(drop=True)
    if view.empty:
        st.info("No cases match the current filters.")
        return

    PAGE = 6
    pages = max(1, (len(view) + PAGE - 1) // PAGE)
    page = st.selectbox("Page", list(range(1, pages + 1)), key="flags_page_sel") if pages > 1 else 1
    for idx, case in view.iloc[(page - 1) * PAGE: page * PAGE].iterrows():
        _render_case(idx, case, trades, run_dir, selected_round, market)


def _render_case(idx: int, case: pd.Series, trades: pd.DataFrame, run_dir: Path, selected_round: int,
                 market: str | None = None) -> None:
    persona = case["persona"]
    color, bg, border = PERSONA_COLORS.get(persona, PERSONA_COLORS["anomaly"])
    icon = PERSONA_ICONS.get(persona, "•")
    status_badge = {
        "confirmed": "✅ ground truth confirms",
        "other persona": f"🟨 ground truth: {case['gt_persona']}",
        "no episode": "⚠️ no ground-truth episode",
    }[case["status"]]
    label = (f"{icon} {persona.replace('_', ' ').title()} · {case['trader_id']} · ticks "
             f"{case['start_tick']}–{case['end_tick']} · score {case['score']:.2f} · {status_badge}")

    with st.expander(label, expanded=(idx == 0)):
        h1, h2 = st.columns([1, 2])
        with h1:
            st.markdown(
                f'<div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:1rem 1.2rem;">'
                f'<div style="font-size:0.7rem; color:{color}; text-transform:uppercase; font-weight:700;">Detector belief</div>'
                f'<div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">{icon} {persona.replace("_", " ").title()}</div>'
                f'<div style="font-size:0.75rem; color:#94a3b8; margin-top:0.5rem;">Account: <b style="color:#e2e8f0;">{case["trader_id"]}</b></div>'
                f'<div style="font-size:0.75rem; color:#94a3b8;">Commodity: <b style="color:#e2e8f0;">{case["commodity"]}</b></div>'
                f'<div style="font-size:0.75rem; color:#94a3b8;">Windows flagged: <b style="color:#e2e8f0;">{case["n_windows"]}</b></div>'
                f'<div style="font-size:0.75rem; color:#94a3b8;">Score: <b style="color:#e2e8f0;">{case["score"]:.3f}</b></div>'
                f'<div style="font-size:0.75rem; color:#94a3b8;">Fired: <b style="color:#e2e8f0;">{", ".join(case["triggered_by"])}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Simulator ground truth: {status_badge}")
        with h2:
            fig = _case_chart(trades, case)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"case_chart_{selected_round}_{market}_{idx}")

        s_col, n_col = st.columns([1.2, 1])
        shap = case["top_shap_features"] or {}
        with s_col:
            if shap:
                st.plotly_chart(_build_shap_chart(shap, persona), use_container_width=True,
                                config={"displayModeBar": False}, key=f"case_shap_{selected_round}_{market}_{idx}")
                st.dataframe(pd.DataFrame([
                    {"Feature": k, "SHAP": f"{v:+.4f}", "Meaning": FEATURE_DESCRIPTIONS.get(k, "—")}
                    for k, v in shap.items()
                ]), use_container_width=True, hide_index=True)
            else:
                st.info("No SHAP attribution stored for this case.")
        with n_col:
            st.markdown("**🤖 Case note** · _AI-generated summary of detector output — simulation only_")
            key = f"narr_{selected_round}_{market}_{idx}"
            if st.button("Generate case note", key=f"btn_{key}"):
                with st.spinner("Writing and fact-checking the case note…"):
                    st.session_state[key] = _narrative(case, run_dir)
            res = st.session_state.get(key)
            if res:
                if res.get("error"):
                    st.info(f"AI features are unavailable: {res['error']} The SHAP chart shows the evidence.")
                elif res.get("grounding_check_passed"):
                    st.success("verified: every number in the note appears in the detector output")
                    st.markdown(res["narrative_text"])
                else:
                    st.info("The generated note mentioned numbers that are not in the detector output, so it "
                            "was discarded. The SHAP chart on the left is the evidence for this case.")

        with st.expander("Show the account's raw orders and trades in this span"):
            orders = load_order_log(str(run_dir), selected_round, market)
            o = orders[(orders["trader_id"] == case["trader_id"])
                       & orders["tick"].between(case["start_tick"], case["end_tick"])]
            t = trades[((trades["buyer_trader_id"] == case["trader_id"]) | (trades["seller_trader_id"] == case["trader_id"]))
                       & trades["tick"].between(case["start_tick"], case["end_tick"])]
            st.markdown(f"Orders ({len(o)})")
            st.dataframe(o.drop(columns=["timestamp"], errors="ignore").head(500), use_container_width=True, hide_index=True)
            st.markdown(f"Trades ({len(t)})")
            st.dataframe(t.drop(columns=["timestamp"], errors="ignore").head(500), use_container_width=True, hide_index=True)
