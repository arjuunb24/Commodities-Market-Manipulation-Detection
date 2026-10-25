"""
src/adversarial/metrics.py
==========================
Computing, comparing and persisting per-round detection metrics.

Two levels, both per persona (SRS FR-2.6 — never one lumped number):

  Row level    precision / recall / F1 / FPR over (account, window) rows, using
               each persona classifier's tuned-threshold flag.
  Event level  an episode counts as detected when any of its accounts is flagged
               for that persona in a window overlapping the episode. Detection
               latency = window_end of the first such flag − episode start
               (FR-2.7); missed episodes have NaN latency and are reported as
               misses, never averaged in as zero.
               Episode coverage = share of each episode's own labelled
               (account, window) rows that were flagged, averaged over episodes.
               Unlike row recall it does not change when a mutation simply adds
               more accounts (more rows) to an episode.
  Caught       the same three event-level numbers (caught_coverage,
               caught_events_detected, caught_latency_ticks) counted on ANY alert
               (final_flag: any persona classifier or the anomaly model). An account
               flagged under another persona's label is still caught; the
               strategist's evasion score uses these, so relabelling (e.g. split
               spoofing flagged as layering) is not rewarded as evasion.
               caught_weak_share = share of episodes with under WEAK_COVERAGE of
               their windows caught: an episode flagged in one window of twenty is
               practically missed, and "any window" detection was 16/16 every round.
  Family       family_coverage (spoofing, layering): episode coverage counting
               flags under the persona's own label OR its fake-depth sibling's, so
               "spoofing caught as layering" is visible as a labelling issue rather
               than read as a miss by the persona's own classifier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import io as io_utils

logger = logging.getLogger(__name__)


def _persona_pred(results: pd.DataFrame, persona: str, threshold: float) -> np.ndarray:
    if f"flag_{persona}" in results.columns:
        return results[f"flag_{persona}"].astype(int).to_numpy()
    if f"prob_{persona}" in results.columns:
        return (results[f"prob_{persona}"] > threshold).astype(int).to_numpy()
    return results["final_flag" if "final_flag" in results else "ensemble_flag"].astype(int).to_numpy()


def event_detection(
    results: pd.DataFrame,
    events: pd.DataFrame,
    persona: str,
    flag_col: str | None = None,
) -> tuple[int, int, list[float]]:
    """Returns (n_events, n_detected, latencies_of_detected_events)."""
    ev = events[events["persona"] == persona] if events is not None and not events.empty else pd.DataFrame()
    if ev.empty:
        return 0, 0, []
    flag_col = flag_col or f"flag_{persona}"
    if flag_col not in results.columns:
        return len(ev), 0, []
    hits = results[results[flag_col].astype(bool)].reset_index()
    latencies: list[float] = []
    for _, e in ev.iterrows():
        accts = set(e["trader_ids"])
        h = hits[(hits["commodity"] == e["commodity"]) & hits["trader_id"].isin(accts)
                 & (hits["window_end"] >= e["start_tick"]) & (hits["window_start"] <= e["end_tick"])]
        if not h.empty:
            latencies.append(float(max(0, int(h["window_end"].min()) - int(e["start_tick"]))))
    return len(ev), len(latencies), latencies


WEAK_COVERAGE = 0.2   # an episode with under 20% of its windows flagged counts as (practically) missed

# Schemes the law treats as one offence and the detector cannot always tell apart: layering is
# a form of spoofing (CEA 4c(a)(5)(C)), and spoofing split over consecutive levels is a ladder.
PERSONA_FAMILY = {"spoofing": ("spoofing", "layering"), "layering": ("spoofing", "layering")}


def family_prf(results: pd.DataFrame, y: pd.DataFrame, family: tuple[str, ...]) -> dict[str, float]:
    """
    Window-level precision / recall / F1 of a persona family as ONE class: truth = labelled with any
    member, prediction = an alert labelled with any member. Spoofing and layering are one offence
    (CEA 4c(a)(5)(C)) and mutations push them into each other's look, so their per-label scores
    fall through label swaps that this metric does not count as errors.
    """
    if "final_flag" not in results.columns or "predicted_persona" not in results.columns:
        return {}
    members = [p for p in family if p in y.columns]
    if not members:
        return {}
    truth = (y[members].to_numpy() > 0).any(axis=1)
    pred = results["final_flag"].astype(bool).to_numpy() & results["predicted_persona"].astype(str).isin(members).to_numpy()
    tp = int((truth & pred).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / truth.sum() if truth.sum() else float("nan")
    f1 = 2 * p * r / (p + r) if truth.sum() and (p + r) else (float("nan") if not truth.sum() else 0.0)
    return {"family_precision": round(float(p), 4), "family_recall": round(float(r), 4), "family_f1": round(float(f1), 4)}


def episode_coverage(results: pd.DataFrame, y: pd.DataFrame, events: pd.DataFrame, persona: str,
                     flag_col: str | None = None) -> float:
    """Mean over episodes of the flagged share of that episode's labelled rows (NaN if none)."""
    shares = episode_shares(results, y, events, persona, flag_col)
    return round(float(np.mean(shares)), 4) if shares else float("nan")


def episode_shares(results: pd.DataFrame, y: pd.DataFrame, events: pd.DataFrame, persona: str,
                   flag_col: str | None = None) -> list[float]:
    """Flagged share of each episode's labelled rows (episodes without labelled rows are skipped)."""
    ev = events[events["persona"] == persona] if events is not None and not events.empty else pd.DataFrame()
    flag_col = flag_col or f"flag_{persona}"
    if ev.empty or flag_col not in results.columns or persona not in y.columns:
        return []
    frame = results[[flag_col, "window_end"]].join(y[[persona]].rename(columns={persona: "_label"})).reset_index()
    frame = frame[frame["_label"] == 1]
    shares = []
    for _, e in ev.iterrows():
        rows = frame[(frame["commodity"] == e["commodity"]) & frame["trader_id"].isin(set(e["trader_ids"]))
                     & (frame["window_end"] >= e["start_tick"]) & (frame["window_start"] <= e["end_tick"])]
        if len(rows):
            shares.append(float(rows[flag_col].astype(bool).mean()))
    return shares


def compute_round_metrics(
    results: pd.DataFrame,
    y: pd.DataFrame,
    personas: list[str],
    threshold: float = 0.5,
    events: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-persona metrics for one scored test set.

    Returns {persona: {precision, recall, f1, fpr, support, events_total,
                       events_detected, detection_latency_ticks}}.
    A persona with no positive rows gets NaN precision/recall/F1 (undefined,
    not zero) and support 0.
    """
    metrics: dict[str, dict[str, float]] = {}
    common_idx = results.index.intersection(y.index)
    if common_idx.empty:
        logger.warning("No common index between results and labels. Returning empty metrics.")
        return {}
    res = results.loc[common_idx]
    lab = y.loc[common_idx]

    for persona in personas:
        if persona not in lab.columns:
            continue
        y_true = lab[persona].astype(int).to_numpy()
        y_pred = _persona_pred(res, persona, threshold)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        n_ev, n_det, lats = event_detection(res, events, persona) if events is not None else (0, 0, [])
        entry: dict[str, Any] = {
            "fpr": round(fpr, 4),
            "support": int(y_true.sum()),
            "events_total": n_ev,
            "events_detected": n_det,
            "detection_latency_ticks": round(float(np.mean(lats)), 1) if lats else float("nan"),
            "episode_coverage": episode_coverage(res, lab, events, persona) if events is not None else float("nan"),
        }
        if events is not None and "final_flag" in res.columns:
            family = [p for p in PERSONA_FAMILY.get(persona, ()) if f"flag_{p}" in res.columns]
            if len(family) > 1:
                # flagged under its own label OR its family sibling's (spoofing <-> layering)
                fam = res.assign(_family_flag=res[[f"flag_{p}" for p in family]].any(axis=1))
                entry["family_coverage"] = episode_coverage(fam, lab, events, persona, flag_col="_family_flag")
                entry.update(family_prf(res, lab, PERSONA_FAMILY[persona]))
            c_ev, c_det, c_lats = event_detection(res, events, persona, flag_col="final_flag")
            c_shares = episode_shares(res, lab, events, persona, flag_col="final_flag")
            entry.update({
                "caught_coverage": round(float(np.mean(c_shares)), 4) if c_shares else float("nan"),
                # share of episodes barely caught (< WEAK_COVERAGE of their windows on any alert)
                "caught_weak_share": round(float(np.mean([s < WEAK_COVERAGE for s in c_shares])), 4) if c_shares else float("nan"),
                "caught_events_detected": c_det,
                "caught_latency_ticks": round(float(np.mean(c_lats)), 1) if c_lats else float("nan"),
            })
        if y_true.sum() == 0:
            logger.warning("No positive labels for persona '%s' in this test set.", persona)
            entry.update({"precision": float("nan"), "recall": float("nan"), "f1": float("nan")})
        else:
            entry.update({
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            })
            logger.info("[Metrics] %s: P=%.3f R=%.3f F1=%.3f FPR=%.3f events %d/%d",
                        persona, entry["precision"], entry["recall"], entry["f1"], fpr, n_det, n_ev)
        metrics[persona] = entry
    return metrics


def compute_unsupervised_metrics(results: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
    """IsolationForest flags vs. 'any manipulation' (it is persona-agnostic)."""
    idx = results.index.intersection(y.index)
    if idx.empty or "iso_flag" not in results:
        return {}
    y_any = (y.loc[idx].sum(axis=1) > 0).astype(int).to_numpy()
    pred = results.loc[idx, "iso_flag"].astype(int).to_numpy()
    fp = int(((y_any == 0) & (pred == 1)).sum())
    tn = int(((y_any == 0) & (pred == 0)).sum())
    return {
        "precision": round(float(precision_score(y_any, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_any, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_any, pred, zero_division=0)), 4),
        "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "support": int(y_any.sum()),
    }


def compute_shap_drift(shap_a: dict[str, float], shap_b: dict[str, float]) -> dict[str, float]:
    """Change in feature importance between two rounds, sorted by |delta|."""
    drift = {f: round(shap_b.get(f, 0.0) - shap_a.get(f, 0.0), 6) for f in set(shap_a) | set(shap_b)}
    return dict(sorted(drift.items(), key=lambda kv: abs(kv[1]), reverse=True))


def _clean(obj):
    """JSON-safe copy (NaN → None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_round_metrics(
    metrics: dict[str, dict[str, float]],
    round_num: int,
    round_dir: Path,
    *,
    pre_retrain: dict[str, dict[str, float]] | None = None,
    unsupervised: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist round metrics to round_dir/metrics.json ("personas" = post-retrain)."""
    round_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"round": round_num, "personas": metrics}
    if pre_retrain is not None:
        payload["pre_retrain"] = pre_retrain
    if unsupervised is not None:
        payload["unsupervised"] = unsupervised
    if extra:
        payload.update(extra)
    with open(round_dir / "metrics.json", "w") as f:
        json.dump(_clean(payload), f, indent=2)
    logger.info("Round %d metrics saved to %s", round_num, round_dir / "metrics.json")


def load_round_metrics(round_dir: Path, phase: str = "post") -> dict[str, dict[str, float]]:
    """Load metrics.json; phase 'post' (default) or 'pre'."""
    path = Path(round_dir) / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("personas" if phase == "post" else "pre_retrain", {}) or {}


PHASE_KEYS = (("post", "personas"), ("pre", "pre_retrain"), ("r0", "round0_detector"), ("bench", "benchmark"))


def build_summary_table(all_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Tidy table: one row per (round, persona, phase).

    all_metrics entries: {"round": r, "personas": post_metrics, "pre_retrain": pre_metrics?,
                          "round0_detector": metrics of the round-0 detector on round r's markets?}
    Phases: post (retrained), pre (frozen previous detector: the evasion measure), r0
    (round-0 detector: drift from the baseline) and bench (retrained detector on the fixed
    benchmark of unseen random variants: transfer). caught_coverage counts any alert.
    """
    rows = []
    for entry in all_metrics:
        for phase, key in PHASE_KEYS:
            for persona, m in (entry.get(key) or {}).items():
                rows.append({
                    "round": entry["round"],
                    "persona": persona,
                    "phase": phase,
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "fpr": m.get("fpr"),
                    "support": m.get("support", 0),
                    "events_total": m.get("events_total", 0),
                    "events_detected": m.get("events_detected", 0),
                    "detection_latency_ticks": m.get("detection_latency_ticks"),
                    "episode_coverage": m.get("episode_coverage"),
                    "caught_coverage": m.get("caught_coverage"),
                    "family_coverage": m.get("family_coverage"),
                    "effectiveness_ratio": (entry.get("effectiveness_ratio") or {}).get(persona),
                    "mutation_score": (entry.get("mutation_score") or {}).get(persona),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        for c in ("precision", "recall", "f1", "fpr", "detection_latency_ticks"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_round_metrics_parquet(summary: pd.DataFrame, commodity: str, path: Path) -> None:
    """Schema-validated round_metrics.parquet (SRS §6.1 round_metrics contract)."""
    summary = summary[summary["phase"].isin(["pre", "post"])] if not summary.empty else summary
    if summary.empty:
        return
    df = pd.DataFrame({
        "round_num": summary["round"].astype("int64"),
        "commodity": commodity,
        "persona": summary["persona"].astype(str),
        "precision": summary["precision"].astype(float),
        "recall": summary["recall"].astype(float),
        "f1": summary["f1"].astype(float),
        "fpr": summary["fpr"].astype(float),
        "detection_latency_ticks": summary["detection_latency_ticks"].astype(float),
        "is_post_retrain": (summary["phase"] == "post").astype(bool),
    })
    io_utils.write_parquet(df, path, "round_metrics", overwrite=True)
