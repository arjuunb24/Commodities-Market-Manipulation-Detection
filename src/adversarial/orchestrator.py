"""
src/adversarial/orchestrator.py
================================
RoundOrchestrator: orchestrates one complete round of the adversarial loop.

One round = Simulate → Detect → Score → Log Metrics.
The full multi-round loop is driven by scripts/run_full_pipeline.py.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.adversarial.metrics import (
    compute_round_metrics,
    save_round_metrics,
    load_round_metrics,
)
from src.detector.pipeline import DetectorPipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class RoundOrchestrator:
    """
    Runs one complete simulation + detection round.

    Directory convention:
        data/runs/<master_run_id>/
            round_0/          ← simulation parquet outputs
            round_1/
            ...
            round_metrics.parquet  ← cumulative metrics across all rounds
    """

    def __init__(
        self,
        master_run_dir: Path,
        detector_config: dict[str, Any],
        sim_ticks: int = 5000,
        sim_seed: int = 42,
        commodity: str = "crude_oil_wti",
    ) -> None:
        self.master_run_dir = Path(master_run_dir)
        self.master_run_dir.mkdir(parents=True, exist_ok=True)

        self.detector_config = detector_config
        self.sim_ticks = sim_ticks
        self.sim_seed = sim_seed
        self.commodity = commodity

        self.pipeline = DetectorPipeline(detector_config)
        self.personas = detector_config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])

        # Cumulative metrics list — grows each round
        self._all_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_round(self, round_num: int) -> dict[str, dict[str, float]]:
        """
        Executes one full round:
          1. Run the simulation twice (train and test) with different seeds
          2. Train & score the detector on separate datasets
          3. Compute metrics
          4. Save everything to disk

        Returns the per-persona metrics dict for this round.
        """
        round_dir = self.master_run_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        train_dir = round_dir / "train"
        test_dir = round_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"ROUND {round_num} — Simulating market (Training data)...")
        self._run_simulation(train_dir, seed=self.sim_seed + round_num * 10)
        
        logger.info(f"ROUND {round_num} — Simulating market (Testing data)...")
        self._run_simulation(test_dir, seed=self.sim_seed + round_num * 10 + 1)

        # Step 2: Load, train & score
        logger.info(f"Round {round_num} — Training & scoring detector...")
        X_train, y_train = self.pipeline.load_and_preprocess(train_dir)
        X_test, y_test = self.pipeline.load_and_preprocess(test_dir)

        if X_train.empty or X_test.empty:
            logger.warning(f"Round {round_num}: empty feature matrix. Skipping.")
            return {}

        self.pipeline.train(X_train, y_train)
        results = self.pipeline.score(X_test)

        # Step 3: Compute metrics on the test set
        metrics = compute_round_metrics(results, y_test, self.personas)

        # Step 4: Save
        save_round_metrics(metrics, round_num, round_dir)
        self._all_metrics.append({"round": round_num, "personas": metrics})
        self._save_cumulative_metrics()

        return metrics

    def get_metrics_for_round(self, round_num: int) -> dict[str, dict[str, float]]:
        """Reads metrics for a specific round from disk (used by tools.py)."""
        round_dir = self.master_run_dir / f"round_{round_num}"
        return load_round_metrics(round_dir)

    def get_shap_drift_for_persona(
        self,
        persona: str,
        round_a: int,
        round_b: int,
    ) -> dict[str, float]:
        """
        Computes a simple feature importance drift between two rounds.
        Currently uses precision/recall delta as a proxy for SHAP drift.
        (Full SHAP integration requires storing shap_values.parquet per round — future work.)
        """
        metrics_a = self.get_metrics_for_round(round_a).get(persona, {})
        metrics_b = self.get_metrics_for_round(round_b).get(persona, {})

        if not metrics_a or not metrics_b:
            return {"note": "Insufficient round data for SHAP drift."}

        return {
            "precision_delta": round(metrics_b.get("precision", 0) - metrics_a.get("precision", 0), 4),
            "recall_delta":    round(metrics_b.get("recall", 0) - metrics_a.get("recall", 0), 4),
            "f1_delta":        round(metrics_b.get("f1", 0) - metrics_a.get("f1", 0), 4),
            "interpretation":  (
                "Negative delta means the detector got WORSE at catching this persona (evasion worked). "
                "Positive delta means the detector RECOVERED after retraining."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_simulation(self, out_dir: Path, seed: int) -> None:
        """
        Calls run_simulation.py as a subprocess, writing outputs directly
        into out_dir so each run has isolated Parquet files.
        """
        python_exe = sys.executable
        script = REPO_ROOT / "scripts" / "run_simulation.py"

        cmd = [
            python_exe, str(script),
            "--ticks",     str(self.sim_ticks),
            "--seed",      str(seed),
            "--outdir",    str(out_dir),
            "--commodity", self.commodity,
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Simulation subprocess failed with code {result.returncode}"
            )

        # run_simulation.py writes to outdir/<timestamp>_<seed>/
        # We need to move those files up one level into out_dir
        self._flatten_sim_output(out_dir)

    def _flatten_sim_output(self, round_dir: Path) -> None:
        """
        run_simulation.py writes to: <outdir>/runs/<timestamp>_<seed>/
        We pass round_dir as --outdir, so simulation output lands in:
            round_dir/runs/<timestamp>_<seed>/
        This method moves Parquet files up into round_dir directly.
        """
        # Check for a 'runs/' subdirectory first
        runs_subdir = round_dir / "runs"
        search_root = runs_subdir if runs_subdir.is_dir() else round_dir

        # Find the most recently created timestamped folder inside
        sub_dirs = [d for d in search_root.iterdir() if d.is_dir()]
        if not sub_dirs:
            logger.warning("No simulation output subdirectory found in %s", search_root)
            return

        sim_output_dir = sorted(sub_dirs)[-1]
        logger.debug(f"Flattening {sim_output_dir} → {round_dir}")

        for parquet_file in sim_output_dir.glob("*.parquet"):
            dest = round_dir / parquet_file.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(parquet_file), str(dest))

        for yaml_file in sim_output_dir.glob("*.yaml"):
            dest = round_dir / yaml_file.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(yaml_file), str(dest))

        # Clean up empty dirs
        try:
            sim_output_dir.rmdir()
        except OSError:
            pass
        try:
            runs_subdir.rmdir()
        except OSError:
            pass


    def _save_cumulative_metrics(self) -> None:
        """Saves all rounds' metrics to a single Parquet file for easy loading."""
        from src.adversarial.metrics import build_summary_table
        df = build_summary_table(self._all_metrics)
        out = self.master_run_dir / "round_metrics.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"Cumulative metrics saved to {out}")
