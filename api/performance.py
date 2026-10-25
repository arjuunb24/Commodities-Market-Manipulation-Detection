"""
api/performance.py
==================
Model Performance views: per-round metrics (frozen vs retrained), the
evasion/recovery headline, parameter history and the strategist's decisions.
Logic ported from dashboard/data_loader.py and dashboard/components/tab_metrics.py.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from api import data
from api.data import clean

METRIC_COLS = ["precision", "recall", "f1", "fpr", "support", "events_total", "events_detected",
               "detection_latency_ticks", "episode_coverage", "caught_coverage", "caught_weak_share", "family_coverage",
               "family_precision", "family_recall", "family_f1"]


def _round_dirs(run: str):
    d = data.run_dir(run)
    return sorted((p for p in d.glob("round_*") if p.is_dir() and p.name.split("_", 1)[1].isdigit()),
                  key=lambda p: int(p.name.split("_", 1)[1]))


def _all_metrics(run: str) -> dict[int, dict]:
    return {int(p.name.split("_")[1]): data.metrics(run, int(p.name.split("_")[1]))
            for p in _round_dirs(run) if (p / "metrics.json").exists()}


SCOPED = ("personas", "pre_retrain", "effectiveness", "effectiveness_ratio", "mutation_score")


def _scoped(run: str, market: str | None) -> dict[int, dict]:
    """Every round's metrics, with the detection/effect blocks swapped for one market's when `market` is set."""
    allm = _all_metrics(run)
    if not market:
        return allm
    from api.bymarket import market_metrics

    def one(rnd: int) -> dict:
        names = data.list_markets(run, rnd)
        return market_metrics(run, rnd, market if market in names else names[0]) if names else {}

    # Sequential on purpose: the background warm-up (api.main) fills the same cache from the
    # latest round down, and extra threads only fight it and the page's other requests for the GIL.
    per_round = {rnd: one(rnd) for rnd in allm}
    out = {}
    for rnd, m in allm.items():
        mm = per_round[rnd]
        out[rnd] = {**{k: v for k, v in m.items() if k not in SCOPED},
                    **{k: mm.get(k) for k in SCOPED},
                    "_source": mm.get("source"), "_has_frozen": mm.get("has_frozen")}
    return out


@data.ttl_cache(600, maxsize=64)
def _family_post(run: str, rnd: int, market: str | None) -> dict:
    """Retrained-detector family P/R/F1 from flags.parquet + labels, for runs recorded before metrics.py had it."""
    import pandas as pd
    from src.adversarial.metrics import PERSONA_FAMILY, family_prf
    try:
        flags = data.flags_for_market(run, rnd, market)
        dirs = [d for d in data.test_dirs(run, rnd) if market is None or d.name == market]
        lab = pd.concat([pd.read_parquet(d / "labels.parquet") for d in dirs if (d / "labels.parquet").exists()])
    except Exception:   # noqa: BLE001 - a missing file only means no fallback
        return {}
    if flags.empty or lab.empty:
        return {}
    lab = lab[~lab.index.duplicated()]
    f = flags.set_index(["commodity", "window_start", "trader_id"])[["final_flag", "predicted_persona"]]
    # the API's flags table holds only alerted windows: every other labelled window is "no alert"
    f = f[~f.index.duplicated()].reindex(lab.index)
    f["final_flag"] = f["final_flag"].fillna(False).astype(bool)
    f["predicted_persona"] = f["predicted_persona"].fillna("none")
    return family_prf(f, lab, PERSONA_FAMILY["spoofing"])


def cumulative(run: str, market: str | None = None) -> list[dict]:
    from src.adversarial.metrics import PERSONA_FAMILY
    rows = []
    for rnd, m in _scoped(run, market).items():
        for phase, key in (("post", "personas"), ("pre", "pre_retrain")):
            for persona, v in (m.get(key) or {}).items():
                row = {"round": rnd, "persona": persona, "phase": phase,
                       **{c: v.get(c) for c in METRIC_COLS if c in v}}
                if phase == "post" and persona in PERSONA_FAMILY and row.get("family_f1") is None:
                    row.update(_family_post(run, rnd, market))
                rows.append(row)
    return clean(rows)


def headline(rows: list[dict]) -> str:
    """Number-derived sentence on the largest evasion and its recovery (no LLM)."""
    df = pd.DataFrame(rows)
    if df.empty or not (df["phase"] == "pre").any():
        return "Only the baseline round has been evaluated so far; the frozen-vs-retrained comparison starts at round 1."
    pre = df[df["phase"] == "pre"].set_index(["round", "persona"])["recall"]
    post = df[df["phase"] == "post"].set_index(["round", "persona"])["recall"]
    if 0 not in post.index.get_level_values("round"):
        return "Not enough data for a frozen-vs-retrained comparison yet."
    base = post.xs(0, level="round")
    drops = []
    for (rnd, persona), v in pre.items():
        b = base.get(persona)
        if b and b == b and v == v and v is not None:
            drops.append(((b - v) / b, rnd, persona, v, post.get((rnd, persona))))
    if not drops:
        return "Not enough data for a frozen-vs-retrained comparison yet."
    drop, rnd, persona, v, rec = max(drops)
    name = persona.replace("_", " ")
    if drop <= 0:
        return f"No mutation pushed recall below the round-0 baseline; the closest was {name} in round {rnd}."
    tail = f", then recovered to {rec:.0%} after retraining" if rec is not None and rec == rec else ""
    return f"Recall for {name} fell {drop:.0%} below baseline under round {rnd}'s mutation (to {v:.0%}){tail}."


def params_history(run: str) -> dict:
    rounds, table = [], {}
    for rnd, m in _all_metrics(run).items():
        rounds.append(rnd)
        for persona, params in (m.get("params_used") or {}).items():
            for k, v in params.items():
                table.setdefault((persona, k), {})[rnd] = v
    rows = [{"persona": p, "parameter": k, "values": {str(r): vals.get(r) for r in rounds}}
            for (p, k), vals in sorted(table.items())]
    for r in rows:   # mark the rounds where the value changed
        vals = [r["values"][str(x)] for x in rounds]
        r["changed"] = [str(rounds[i]) for i in range(1, len(vals)) if vals[i] != vals[i - 1]]
    return clean({"rounds": rounds, "rows": rows})


def _split_reasoning(text: str) -> dict:
    """Reasoning -> numbered points; probe / guard / recovery notes split out."""
    if not text:
        return {"points": [], "notes": []}
    notes, body = [], []
    for line in text.replace("\r", "").split("\n"):
        (notes if line.strip().startswith("[") else body).append(line.strip())
    t = " ".join(" ".join(body).split()).replace("->", "→")
    parts = re.split(r"(?:(?<=\s)|^)(\d{1,2})\.\s+(?=\S)", t)
    points = [parts[0].strip()] if parts[0].strip() else []
    points += [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]
    return {"points": [p for p in points if p], "notes": [n for n in notes if n]}


def strategist(run: str, market: str | None = None) -> list[dict]:
    allm = _scoped(run, market)
    out = []
    for rnd in sorted(r for r in allm if r >= 1):
        m, prev = allm[rnd], allm.get(rnd - 1, {})
        rec = data.strategist_record(run, rnd).get("personas", {})
        personas = {}
        for p in sorted(set(m.get("params_used", {})) | set(rec)):
            r = rec.get(p, {})
            new = m.get("params_used", {}).get(p, r.get("params", {}))
            old = prev.get("params_used", {}).get(p, {})
            changes = r.get("changes") or {k: [old.get(k), v] for k, v in new.items() if old.get(k) != v}
            pre = (m.get("pre_retrain") or {}).get(p, {})
            post = (m.get("personas") or {}).get(p, {})
            probe = r.get("probe") or {}
            personas[p] = {
                "source": r.get("source") or (m.get("mutation_source") or {}).get(p, "unknown"),
                "changes": [{"knob": k, "old": v[0], "new": v[1]} for k, v in changes.items()],
                "reasoning": _split_reasoning(r.get("reasoning", "")),
                "frozen_coverage": pre.get("caught_coverage", pre.get("episode_coverage")),
                "own_label_coverage": pre.get("episode_coverage"),
                "family_coverage": pre.get("family_coverage"),
                "weak_share": pre.get("caught_weak_share"),
                "episodes_caught": pre.get("caught_events_detected", pre.get("events_detected")),
                "episodes_total": pre.get("events_total"),
                "frozen_delay": pre.get("caught_latency_ticks", pre.get("detection_latency_ticks")),
                "frozen_recall": pre.get("recall"),
                "retrained_recall": post.get("recall"),
                "effect_ratio": (m.get("effectiveness_ratio") or {}).get(p),
                "score": (m.get("mutation_score") or {}).get(p),
                "recovered_from_round": r.get("effect_recovery_from_round"),
                "guard_reverts": [{"knob": k, "old": v[0], "new": v[1]} for k, v in (r.get("effect_guard_reverts") or {}).items()],
                "probe": {"chosen": probe.get("chosen"),
                          "candidates": [{"changes": c.get("changes"), "coverage": c.get("coverage"),
                                          "effect_ratio": c.get("effect_ratio"), "score": c.get("score"),
                                          "origin": c.get("origin"),
                                          # older runs only marked exploration candidates with "origin"
                                          "source": c.get("candidate_source")
                                          or ("code_exploration" if c.get("origin") else "llm")}
                                         for c in probe.get("candidates", [])]} if probe else None,
            }
        out.append({"round": rnd, "personas": personas})
    return clean(out)


def effectiveness(run: str, market: str | None = None) -> list[dict]:
    rows = []
    for rnd, m in _scoped(run, market).items():
        for p in data.PERSONAS:
            e = (m.get("effectiveness") or {}).get(p) or {}
            rows.append({"round": rnd, "persona": p, "ratio": (m.get("effectiveness_ratio") or {}).get(p),
                         "score": (m.get("mutation_score") or {}).get(p), "primary_metric": e.get("primary_metric"),
                         "primary": e.get(e.get("primary_metric")) if e.get("primary_metric") else None})
    return clean(rows)


def performance(run: str, market: str | None = None) -> dict:
    rows = cumulative(run, market)
    allm = _all_metrics(run)
    scoped = _scoped(run, market) if market else {}
    frozen_missing = bool(market) and any(r > 0 and not m.get("_has_frozen") for r, m in scoped.items())
    thr = {str(r): m.get("thresholds") for r, m in allm.items()}
    unsup = {str(r): m.get("unsupervised") for r, m in allm.items()}
    return clean({"rows": rows, "headline": headline(rows) if not frozen_missing else
                  "This run predates per-market frozen-detector results, so only retrained metrics and effect are split by market. "
                  "Runs made from now on record both.",
                  "rounds": sorted(allm), "scope": market or "pooled", "frozen_by_market": not frozen_missing,
                  "thresholds": thr, "unsupervised": unsup, "effectiveness": effectiveness(run, market)})


@data.ttl_cache(120, maxsize=256)
def run_summary(run: str) -> dict:
    allm = _all_metrics(run)
    rounds = sorted(allm)
    if not rounds:
        return {"run": run, "rounds": []}
    last = allm[rounds[-1]]
    post = last.get("personas") or {}
    f1s = [v.get("f1") for v in post.values() if v.get("f1") is not None]
    trend = []
    for r in rounds:
        vals = [v.get("f1") for v in (allm[r].get("personas") or {}).values() if v.get("f1") is not None]
        trend.append(sum(vals) / len(vals) if vals else None)
    llm = data.run_dir(run) / "llm_calls.jsonl"
    cost = 0.0
    n_calls = 0
    if llm.exists():
        for line in llm.read_text(encoding="utf-8").splitlines():
            try:
                cost += float(json.loads(line).get("cost_usd") or 0); n_calls += 1
            except (ValueError, TypeError):
                continue
    sources = {}
    for r in rounds[1:]:
        for p, s in (allm[r].get("mutation_source") or {}).items():
            sources[s] = sources.get(s, 0) + 1
    return clean({"run": run, "rounds": rounds, "latest_round": rounds[-1],
                  "avg_f1": sum(f1s) / len(f1s) if f1s else None, "f1_trend": trend,
                  "n_test_markets": last.get("n_test_markets"), "llm_calls": n_calls, "llm_cost_usd": round(cost, 4),
                  "mutation_sources": sources})
