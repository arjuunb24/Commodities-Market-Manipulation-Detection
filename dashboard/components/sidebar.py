"""
dashboard/components/sidebar.py
================================
Left sidebar: run selector, round selector, and quick stats.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.data_loader import list_run_dirs, list_rounds, load_round_metrics_json


PERSONA_COLORS = {
    "spoofing": "#ef4444",
    "wash_trading": "#f59e0b",
    "pump_and_dump": "#8b5cf6",
    "layering": "#06b6d4",
}


def render_sidebar() -> tuple[Path | None, int]:
    """
    Renders the sidebar and returns (run_dir, selected_round).
    Returns (None, 0) if no runs are available.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
                <div style="font-size:2.2rem;">🛡️</div>
                <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0;">Veridex</div>
                <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">
                    Market Surveillance
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:0.82rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin: 0.8rem 0 0.4rem;">🗂️ Run Selection</div>',
            unsafe_allow_html=True,
        )
        run_dirs = list_run_dirs()

        if not run_dirs:
            st.warning("No adversarial runs found in `data/runs/`.")
            st.info("Run the pipeline first:\n```\n.venv\\Scripts\\python.exe scripts/run_full_pipeline.py\n```")
            return None, 0

        run_names = [d.name for d in run_dirs]
        selected_run_name = st.selectbox(
            "Adversarial Run",
            options=run_names,
            index=0,
            key="sidebar_run_select",
        )
        run_dir = next(d for d in run_dirs if d.name == selected_run_name)

        # ── Round selector ───────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin: 1.2rem 0 0.4rem;">🔄 Round Selection</div>',
            unsafe_allow_html=True,
        )
        available_rounds = list_rounds(run_dir)

        if not available_rounds:
            st.warning("No rounds found for this run.")
            return run_dir, 0

        default_round = st.session_state.get("sidebar_round_slider", available_rounds[0])
        if default_round not in available_rounds:
            default_round = available_rounds[0]

        selected_round = st.select_slider(
            "Select Round",
            options=available_rounds,
            value=default_round,
            key="sidebar_round_slider",
        )

        # ── Quick stats for selected round ───────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.82rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin: 0.8rem 0 0.6rem;">📋 Round Summary</div>',
            unsafe_allow_html=True,
        )

        m = load_round_metrics_json(str(run_dir), selected_round)
        personas = m.get("personas", {})

        if personas:
            for persona, stats in personas.items():
                color = PERSONA_COLORS.get(persona, "#94a3b8")
                f1_val = stats.get("f1", 0.0)
                prec_val = stats.get("precision", 0.0)
                rec_val = stats.get("recall", 0.0)
                st.markdown(
                    f"""
                    <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(99,130,200,0.18);
                                border-radius:10px; padding:0.65rem 0.85rem; margin-bottom:0.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.72rem; font-weight:700; color:{color};
                                         text-transform:uppercase; letter-spacing:0.06em;">
                                {persona.replace("_", " ")}
                            </span>
                            <span style="font-size:0.8rem; font-weight:700; color:#e2e8f0;">
                                F1&nbsp;{f1_val:.3f}
                            </span>
                        </div>
                        <div style="display:flex; gap:1rem; margin-top:0.35rem;">
                            <span style="font-size:0.7rem; color:#94a3b8;">
                                P&nbsp;<b style="color:#e2e8f0;">{prec_val:.2f}</b>
                            </span>
                            <span style="font-size:0.7rem; color:#94a3b8;">
                                R&nbsp;<b style="color:#e2e8f0;">{rec_val:.2f}</b>
                            </span>
                            <span style="font-size:0.7rem; color:#94a3b8;">
                                FPR&nbsp;<b style="color:#e2e8f0;">{stats.get('fpr', 0.0):.2f}</b>
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No metrics available for this round.")

        # ── Info footer ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:0.68rem; color:#475569; text-align:center;">
                <div>Run: <code style="color:#94a3b8;">{selected_run_name[:22]}…</code></div>
                <div style="margin-top:4px;">Rounds available: {len(available_rounds)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return run_dir, selected_round
