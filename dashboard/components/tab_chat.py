"""
dashboard/components/tab_chat.py
================================
Tab 4 — Analyst Chat (SRS FR-5.4, UC-3)

Free-text questions about the selected run, answered by the Module 4C
AnalystAssistant using function calls over the run's logged outputs only.
Every answer shows a "Sources" expander listing the exact tool calls used.
If no LLM key is configured the panel is replaced by an explanatory message
(FR-4.4) — the rest of the dashboard keeps working.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUGGESTIONS = [
    "Which accounts were flagged for spoofing in the latest round, and why?",
    "Which persona is hardest to catch before retraining?",
    "Summarise the trading activity of the top flagged account.",
]


def _assistant_config() -> dict:
    path = REPO_ROOT / "configs" / "llm_config.yaml"
    return (yaml.safe_load(path.read_text()) or {}).get("analyst_assistant", {}) if path.exists() else {}


def render_chat_tab(run_dir: Path | None) -> None:
    if run_dir is None:
        st.info("No run selected. Pick a run in the sidebar to ask questions about it.")
        return

    from src.agentic.llm_client import llm_unavailable_reason
    cfg = _assistant_config()
    reason = llm_unavailable_reason(cfg)
    if reason:
        st.info(
            f"AI features are temporarily unavailable ({reason}). Everything else in the dashboard works; "
            "the **Flagged Trades** tab shows the SHAP evidence for every case."
        )
        return

    st.caption("Answers are generated only from this run's logged detector output. Simulation data — not real traders.")

    state_key = f"chat::{run_dir}"
    if state_key not in st.session_state:
        from src.agentic.analyst_assistant import AnalystAssistant
        st.session_state[state_key] = {"assistant": AnalystAssistant(cfg, run_dir=run_dir), "log": []}
    session = st.session_state[state_key]

    for turn in session["log"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources") is not None:
                with st.expander(f"Sources ({len(turn['sources'])})"):
                    if not turn["sources"]:
                        st.markdown("_No tool calls — the assistant answered without reading data._")
                    for s in turn["sources"]:
                        flag = " — returned an error" if s["error"] else ""
                        st.code(f"{s['tool']}({s['args']}){flag}", language="text")

    prompt = None
    if not session["log"]:
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            if col.button(text, key=f"suggest::{text}"):
                prompt = text
    typed = st.chat_input("Ask about flags, accounts or detector performance…")
    prompt = typed or prompt
    if not prompt:
        return

    session["log"].append({"role": "user", "content": prompt})
    with st.spinner("Querying the run's logs…"):
        try:
            result = session["assistant"].ask(prompt)
        except Exception as exc:
            result = {"answer": f"AI features are temporarily unavailable: {exc}", "sources": []}
    session["log"].append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
    st.rerun()
