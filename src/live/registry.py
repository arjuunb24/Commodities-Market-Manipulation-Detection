"""
src/live/registry.py
====================
Saved models per adversarial round, for prediction outside the training loop.

    data/runs/<run>/round_N/models/
        detector.joblib    trader-level DetectorPipeline (needs order logs with account ids)
        bar_model.joblib   bar-level BarModel (works on anonymous 1-minute OHLCV bars)
        manifest.json      what was saved, when, from which rounds, and the skill metrics

The orchestrator calls save_round_models() after every round; scripts/save_round_models.py
backfills runs made before this existed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "data" / "runs"
MODELS_DIR = "models"
DETECTOR_FILE = "detector.joblib"
BAR_FILE = "bar_model.joblib"
MANIFEST = "manifest.json"


def save_round_models(round_dir: Path, *, pipeline=None, bar_model=None, extra: dict | None = None) -> Path:
    """Write the round's models and a manifest; returns the models directory."""
    out = Path(round_dir) / MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    # Saving one model keeps the other's earlier entry (e.g. a bar-model-only backfill).
    previous = json.loads((out / MANIFEST).read_text(encoding="utf-8")) if (out / MANIFEST).exists() else {}
    manifest: dict[str, Any] = {
        **previous,
        "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "round": int(str(Path(round_dir).name).split("_")[-1]) if Path(round_dir).name.startswith("round_") else None,
        "run": Path(round_dir).parent.name,
        "files": dict(previous.get("files", {})),
    }
    if pipeline is not None:
        # SHAP explainer caches are large and rebuilt on demand; they are not saved.
        sup = pipeline.supervised_detector
        cache, sup._explainers = sup._explainers, {}
        try:
            joblib.dump(pipeline, out / DETECTOR_FILE, compress=3)
        finally:
            sup._explainers = cache
        manifest["files"]["detector"] = DETECTOR_FILE
        manifest["detector"] = {
            "unit": "(commodity, 3-minute window, account)",
            "input": "simulator-format order_log + trade_log with account ids",
            "personas": list(sup.models),
            "thresholds": {p: round(float(t), 4) for p, t in sup.thresholds.items()},
            "n_features": len(sup.feature_columns),
            "config": pipeline.config,   # the exact detector settings, for reproducing this model
        }
    if bar_model is not None:
        joblib.dump(bar_model, out / BAR_FILE, compress=3)
        manifest["files"]["bar_model"] = BAR_FILE
        manifest["bar_model"] = {
            "unit": "(commodity, 1-minute bar)",
            "input": "anonymous 1-minute OHLCV bars (simulator or exchange)",
            "personas": list(bar_model.models),
            "thresholds": bar_model.thresholds,
            "features": bar_model.features,
            "skill": bar_model.skill_summary(),
            "metrics": bar_model.metrics,
        }
    if extra:
        manifest.update(extra)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Saved round models to %s", out)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_run(run: str | None) -> Path:
    """A run name, a path, or None/'latest' for the newest run that has saved models."""
    if run and run != "latest":
        p = Path(run)
        p = p if p.is_dir() else RUNS_DIR / run
        if not p.is_dir():
            raise FileNotFoundError(f"Run not found: {run}")
        return p
    runs = sorted((d for d in RUNS_DIR.glob("adversarial_*") if any(d.glob(f"round_*/{MODELS_DIR}/{MANIFEST}"))),
                  key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError("No run has saved models yet. Run the pipeline, or backfill with "
                                "scripts/save_round_models.py --run <run>.")
    return runs[-1]


def saved_rounds(run_dir: Path) -> list[int]:
    return sorted(int(p.parent.parent.name.split("_")[1]) for p in Path(run_dir).glob(f"round_*/{MODELS_DIR}/{MANIFEST}"))


def models_dir(run: str | None, round_num: int | str | None) -> Path:
    run_dir = resolve_run(run)
    rounds = saved_rounds(run_dir)
    if not rounds:
        raise FileNotFoundError(f"No saved models in {run_dir.name}. Backfill with scripts/save_round_models.py --run {run_dir.name}")
    r = rounds[-1] if round_num in (None, "latest") else int(round_num)
    if r not in rounds:
        raise FileNotFoundError(f"Round {r} of {run_dir.name} has no saved models (saved: {rounds}).")
    return run_dir / f"round_{r}" / MODELS_DIR


def load_manifest(mdir: Path) -> dict:
    return json.loads((Path(mdir) / MANIFEST).read_text(encoding="utf-8"))


def load_bar_model(mdir: Path):
    path = Path(mdir) / BAR_FILE
    if not path.exists():
        raise FileNotFoundError(f"No bar model in {mdir}")
    return joblib.load(path)


def load_detector(mdir: Path):
    path = Path(mdir) / DETECTOR_FILE
    if not path.exists():
        raise FileNotFoundError(f"No trader-level detector in {mdir}")
    return joblib.load(path)
