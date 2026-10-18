"""
src/live/predict.py
===================
Score a saved real-market dataset with a round's bar model, group flagged minutes
into windows ("cases"), run the mechanism checker (and optionally the LLM
reviewer) on them, and write a report.

    data/live/<dataset>/predictions/<run>_r<N>_<timestamp>/
        minute_scores.parquet   every minute: per-persona probability, flag, margin
        cases.json              flagged windows with evidence, rule verdicts, LLM opinion
        report.md               readable summary
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.live.bars import bar_features, minute_to_utc
from src.live.checker import LLMReviewer, MechanismChecker
from src.live.datasets import load_dataset
from src.live.registry import REPO_ROOT, load_bar_model, load_manifest

MERGE_GAP_MIN = 2         # flagged minutes this close together form one window


def build_cases(scores: pd.DataFrame, gap: int = MERGE_GAP_MIN) -> list[dict]:
    """Merge consecutive flagged minutes (same commodity and segment) into windows."""
    flagged = scores[scores["flag"]].sort_values(["commodity", "segment", "minute"])
    personas = [c[5:] for c in scores.columns if c.startswith("prob_")]
    cases = []
    for (c, seg), g in flagged.groupby(["commodity", "segment"], sort=False):
        block = (g["minute"].diff().fillna(gap + 1) > gap).cumsum()
        for _, w in g.groupby(block):
            # The persona that clears its threshold by the most, summed over the window
            margin_sum = {p: float((w[f"prob_{p}"]).sum()) for p in personas}
            votes = w["predicted_persona"].value_counts()
            persona = votes.index[0] if len(votes) else max(margin_sum, key=margin_sum.get)
            cases.append({
                "commodity": c,
                "segment": int(seg),
                "start_minute": int(w["minute"].min()),
                "end_minute": int(w["minute"].max()),
                "minutes": int(w["minute"].max() - w["minute"].min() + 1),
                "flagged_minutes": int(len(w)),
                "model_persona": persona,
                "model_probability": round(float(w[f"prob_{persona}"].max()), 4),
                "model_margin": round(float(w["margin"].max()), 4),
                "persona_votes": {k: int(v) for k, v in votes.items()},
            })
    return sorted(cases, key=lambda x: -x["model_margin"])


def run_prediction(dataset_dir: Path, models_dir: Path, *, top: int = 20, use_llm: bool = False,
                   llm_max: int = 5, out_root: Path | None = None, personas: list[str] | None = None) -> dict[str, Any]:
    bars, data_manifest = load_dataset(dataset_dir)
    if bars.empty:
        raise ValueError(f"{dataset_dir} has no bars.")
    model = load_bar_model(models_dir)
    model_manifest = load_manifest(models_dir)
    if personas:
        # Score only these personas (e.g. the ones bars can reveal: spoofing, pump_and_dump)
        unknown = [p for p in personas if p not in model.models]
        if unknown:
            raise ValueError(f"Model has no classifier for {unknown}; it has {list(model.models)}.")
        model.models = {p: model.models[p] for p in personas}
        for key in ("heldout", "train_oof"):
            if key in model.metrics:
                model.metrics[key] = {p: v for p, v in model.metrics[key].items() if p in personas}
    X = bar_features(bars)
    scores = model.score(bars, X)
    scores["ts_utc"] = minute_to_utc(scores["minute"])

    cases = build_cases(scores)
    checker = MechanismChecker(bars, real_time=True)
    reviewer, llm_error = None, None
    if use_llm:
        try:
            reviewer = LLMReviewer(run_dir=None)
        except Exception as exc:  # no key, budget used up, provider disabled
            llm_error = str(exc)
    for i, case in enumerate(cases[:top]):
        case["start_utc"] = minute_to_utc([case["start_minute"]])[0].strftime("%Y-%m-%d %H:%M")
        case["end_utc"] = (minute_to_utc([case["end_minute"]])[0] + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
        check = checker.check(case["commodity"], case["start_minute"], case["end_minute"])
        case["check"] = check
        mp = check["personas"].get(case["model_persona"], {})
        case["checker_verdict_for_model_persona"] = mp.get("verdict")
        case["checker_best_match"] = check.get("best_match")
        if reviewer is not None and i < llm_max:
            case["llm_review"] = reviewer.review(case, check)

    n_min = int(len(scores))
    by_persona = {p: int((scores["predicted_persona"] == p).sum()) for p in model.models}
    skill = model.skill_summary()
    sim_flag_rate = {p: (m or {}).get("flag_rate") for p, m in (model.metrics.get("heldout") or {}).items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name, rnd = model_manifest.get("run"), model_manifest.get("round")
    out = Path(out_root or Path(dataset_dir) / "predictions") / f"{run_name}_r{rnd}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "minute_scores.parquet", index=False)
    summary = {
        "dataset": Path(dataset_dir).name,
        "window_utc": [str(data_manifest.get("window_start_utc")), str(data_manifest.get("window_end_utc"))],
        "model": {"run": run_name, "round": rnd, "models_dir": str(Path(models_dir).relative_to(REPO_ROOT))
                  if Path(models_dir).is_relative_to(REPO_ROOT) else str(models_dir)},
        "model_skill_on_simulator": skill,
        "simulator_flag_rate": sim_flag_rate,
        "minutes_scored": n_min,
        "minutes_flagged": int(scores["flag"].sum()),
        "flag_rate": round(float(scores["flag"].mean()), 4) if n_min else 0.0,
        "flagged_minutes_by_persona": by_persona,
        "n_cases": len(cases),
        "n_checked": min(len(cases), top),
        "llm": {"used": reviewer is not None, "reviewed": sum(1 for c in cases if "llm_review" in c), "error": llm_error},
        "verdict_counts": _verdict_counts(cases[:top]),
    }
    (out / "cases.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_report(summary, cases[:top], model_manifest), encoding="utf-8")
    summary["out_dir"] = str(out)
    return {"summary": summary, "cases": cases, "out_dir": out}


def _after_text(ev: dict) -> str:
    """What the price did in the 30 minutes after the window, in words."""
    if ev.get("post_bars", 0) == 0 or ev.get("post_30min_move_bp") is None:
        return "no bars after the window"
    gb = ev.get("giveback_pct")
    if gb is None:
        return f"then moved {ev['post_30min_move_bp']} bp in the next 30 min (move too small to call a reversal)"
    if gb >= 0:
        return f"then gave back {int(round(gb))}% of it in the next 30 min"
    return f"then kept going the same way ({ev['post_30min_move_bp']} bp more in the next 30 min)"


def _verdict_counts(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cases:
        v = c.get("checker_verdict_for_model_persona") or "not checked"
        out[v] = out.get(v, 0) + 1
    return out


def _skill_line(skill: dict) -> str:
    aucs = ", ".join(f"{p.replace('_', ' ')} {a:.2f}" for p, a in skill["auc"].items() if a is not None)
    return (f"Held-out simulator AUC (0.5 = coin flip, 1.0 = perfect): {aucs}. "
            f"Mean {skill['mean_auc']} ({skill['level']}).")


def render_report(summary: dict, cases: list[dict], model_manifest: dict) -> str:
    s = summary
    L = [
        f"# Live market check: {s['dataset']}",
        "",
        f"- Data window (UTC): {s['window_utc'][0]} to {s['window_utc'][1]}",
        f"- Model: bar model from `{s['model']['run']}`, round {s['model']['round']}",
        f"- Minutes scored: {s['minutes_scored']:,}; flagged: {s['minutes_flagged']:,} ({s['flag_rate']:.1%})",
        f"- Flagged windows: {s['n_cases']} (top {s['n_checked']} checked)",
        "",
        "## How much to trust this",
        "",
        _skill_line(s["model_skill_on_simulator"]),
        "",
        "This model sees only anonymous 1-minute bars (no orders, cancels or account ids), and it was trained on the "
        "simulator, not on real confirmed cases. Treat every flag as a window worth a look, not as evidence of "
        "manipulation. The mechanism checker below tests each window against how the personas actually move a market, "
        "and looks for ordinary explanations first.",
        "",
        "## Flagged windows",
        "",
    ]
    if not cases:
        L.append("No windows were flagged.")
    else:
        L.append("| # | Commodity | Window (UTC) | Model says | Checker on that persona | Best mechanism match | Ordinary explanation |")
        L.append("|---|---|---|---|---|---|---|")
        for i, c in enumerate(cases, 1):
            ch = c.get("check", {})
            best = ch.get("best_match")
            bv = ch.get("personas", {}).get(best, {}).get("verdict") if best else None
            benign = "; ".join(ch.get("benign_explanations", [])) or "none found"
            L.append(f"| {i} | {c['commodity'].replace('_', ' ')} | {c.get('start_utc', '')} to {c.get('end_utc', '')[-5:]} "
                     f"({c['minutes']} min) | {c['model_persona'].replace('_', ' ')} (p={c['model_probability']:.2f}) | "
                     f"{c.get('checker_verdict_for_model_persona') or '-'} | {(best or '-').replace('_', ' ')}: {bv or '-'} | {benign} |")
        L.append("")
        for i, c in enumerate(cases, 1):
            ch = c.get("check", {})
            ev = ch.get("evidence", {})
            L += [f"### {i}. {c['commodity'].replace('_', ' ')}, {c.get('start_utc', '')} to {c.get('end_utc', '')} UTC", ""]
            if not ev.get("available"):
                L += ["No bars available for this window.", ""]
                continue
            L.append(f"Move {ev['move_bp']} bp ({ev['move_z']} sigma); peak {ev['peak_excursion_bp']} bp; "
                     f"volume {ev['volume_ratio']}x normal; range {ev['range_ratio']}x normal; wick share {ev['wick_share']}; "
                     f"{_after_text(ev)}.")
            if ev.get("scheduled_events"):
                L.append("Scheduled events: " + "; ".join(f"{e['event']} ({e['at_local']})" for e in ev["scheduled_events"]) + ".")
            if ev.get("comoving_commodities"):
                L.append("Moved with: " + ", ".join(f"{p['commodity']} ({p['move_z']} sigma)" for p in ev["comoving_commodities"]) + ".")
            L.append("")
            mp = c["model_persona"]
            pr = ch["personas"].get(mp, {})
            L.append(f"Checker on {mp.replace('_', ' ')}: **{pr.get('verdict')}** ({pr.get('passed')} tests passed)")
            for t in pr.get("tests", []):
                L.append(f"- {'pass' if t['passed'] else 'fail'}: {t['test']}")
            if pr.get("data_needed_to_confirm"):
                L.append(f"- To confirm: {pr['data_needed_to_confirm']}")
            if c.get("llm_review"):
                r = c["llm_review"]
                if "error" in r:
                    L.append(f"\nLLM review unavailable: {r['error']}")
                elif "reasoning" in r:
                    L.append(f"\nLLM review (AI opinion, not a finding): **{r.get('verdict')}**, {r.get('confidence')} confidence, "
                             f"best fit {r.get('persona_assessed')}. {r.get('reasoning')}")
                else:
                    L.append(f"\nLLM review (unparsed): {r.get('raw', '')}")
            L.append("")
    L += ["---", "Generated by scripts/predict_live.py. Simulated-data model applied to real bars; not a regulatory finding."]
    return "\n".join(L)
