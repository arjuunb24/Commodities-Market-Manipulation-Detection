"""
src/adversarial/orchestrator.py
================================
RoundOrchestrator: orchestrates one complete round of the adversarial loop.

One round = Simulate → Detect → Score → Log Metrics.
The full multi-round loop is driven by scripts/run_full_pipeline.py.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.adversarial.metrics import (
    build_summary_table,
    compute_round_metrics,
    compute_shap_drift,
    compute_unsupervised_metrics,
    load_round_metrics,
    save_round_metrics,
    write_round_metrics_parquet,
)
from src.adversarial.effectiveness import (
    effectiveness_ratio,
    episode_effectiveness,
    mutation_score,
    summarize,
)
from src.core.persona_params import effective_params, resolve_persona_config
from src.detector.pipeline import DetectorPipeline
from src.utils import io as io_utils

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
        n_test_markets: int = 2,
        persona_config_path: Path | None = None,
    ) -> None:
        self.master_run_dir = Path(master_run_dir)
        self.master_run_dir.mkdir(parents=True, exist_ok=True)

        self.detector_config = detector_config
        self.sim_ticks = sim_ticks
        self.sim_seed = sim_seed
        self.commodity = commodity
        self.n_test_markets = max(1, int(n_test_markets))
        self.persona_config_path = Path(persona_config_path) if persona_config_path else REPO_ROOT / "configs" / "persona_config.yaml"

        self.pipeline = DetectorPipeline(detector_config)
        self.personas = detector_config.get("personas", ["spoofing", "wash_trading", "pump_and_dump"])

        # Cumulative metrics list — grows each round
        self._all_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _test_dirs(self, round_dir: Path) -> list[Path]:
        return [round_dir / ("test" if m == 0 else f"test_{m}") for m in range(self.n_test_markets)]

    def _params_in_force(self) -> dict[str, dict[str, Any]]:
        import yaml
        cfg = yaml.safe_load(self.persona_config_path.read_text()) if self.persona_config_path.exists() else {}
        active = resolve_persona_config(cfg or {})
        return {p: effective_params(active, p) for p in self.personas}

    def _baseline_effectiveness(self) -> dict:
        path = self.master_run_dir / "round_0" / "metrics.json"
        return json.loads(path.read_text()).get("effectiveness", {}) if path.exists() else {}

    def run_round(self, round_num: int, mutation_source: dict[str, str] | None = None) -> dict[str, dict[str, float]]:
        """
        Executes one full round (SRS FR-3.1 / FR-3.2):
          1. Simulate one training market and `n_test_markets` held-out test
             markets (different seeds) under the current persona_config.
          2. PRE-retrain: score the test markets with the detector frozen at the
             end of the previous round — how well this round's mutation evaded.
          3. Retrain on all training rounds 0..N and score again (POST-retrain).
          4. Measure manipulation effectiveness on the test markets and the
             strategist objective (evasion x retained effectiveness).
          5. Persist metrics, flags (post-retrain) and SHAP importances.

        Returns the post-retrain per-persona metrics.
        """
        round_dir = self.master_run_dir / f"round_{round_num}"
        train_dir = round_dir / "train"
        test_dirs = self._test_dirs(round_dir)
        for d in [train_dir, *test_dirs]:
            d.mkdir(parents=True, exist_ok=True)
        params_used = self._params_in_force()

        logger.info("=" * 60)
        logger.info(f"ROUND {round_num} — simulating 1 train + {len(test_dirs)} test markets...")
        T = self.sim_ticks
        jobs = [(train_dir, self.sim_seed + round_num * 10, round_num * T)]
        for m, d in enumerate(test_dirs):
            jobs.append((d, self.sim_seed + round_num * 10 + 1 + m, (round_num * self.n_test_markets + m) * T))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for f in [executor.submit(self._run_simulation, *job) for job in jobs]:
                f.result()

        X_parts, y_parts, ev_parts, eff_parts = [], [], [], []
        for d in test_dirs:
            X_m, y_m = self.pipeline.load_and_preprocess(d)
            ev_m = pd.read_parquet(d / "manipulation_events.parquet")
            if not X_m.empty:
                X_parts.append(X_m)
                y_parts.append(y_m)
            ev_parts.append(ev_m)
            eff_parts.append(episode_effectiveness(
                ev_m, pd.read_parquet(d / "trade_log.parquet"), pd.read_parquet(d / "order_log.parquet")))
        if not X_parts:
            logger.warning(f"Round {round_num}: empty test feature matrix. Skipping.")
            return {}
        X_test, y_test = pd.concat(X_parts), pd.concat(y_parts)
        events_test = pd.concat(ev_parts, ignore_index=True)
        eff = summarize(pd.concat(eff_parts, ignore_index=True))

        # --- Pre-retrain (frozen detector from previous round) ---
        pre_metrics = None
        if self.pipeline.is_trained:
            logger.info(f"Round {round_num} — PRE-retrain scoring with frozen detector...")
            pre_results = self.pipeline.score(X_test)
            pre_metrics = compute_round_metrics(pre_results, y_test, self.personas, events=events_test)

        # --- Retrain on cumulative training data ---
        logger.info(f"Round {round_num} — retraining on rounds 0..{round_num}...")
        X_tr, y_tr = [], []
        for r in range(round_num + 1):
            X_r, y_r = self.pipeline.load_and_preprocess(self.master_run_dir / f"round_{r}" / "train")
            if not X_r.empty:
                X_tr.append(X_r)
                y_tr.append(y_r)
        if not X_tr:
            logger.warning(f"Round {round_num}: empty training data. Skipping.")
            return {}
        self.pipeline.train(pd.concat(X_tr), pd.concat(y_tr), save_dir=round_dir)

        # --- Post-retrain ---
        results = self.pipeline.score(X_test)
        post_metrics = compute_round_metrics(results, y_test, self.personas, events=events_test)
        unsup = compute_unsupervised_metrics(results, y_test)
        flags = self.pipeline.build_flags(X_test, results)
        self.pipeline.write_flags(flags, round_dir / "flags.parquet")

        # --- Effectiveness and strategist objective ---
        baseline = eff if round_num == 0 else self._baseline_effectiveness()
        ratios, scores = {}, {}
        for p in self.personas:
            ratios[p] = effectiveness_ratio(eff.get(p), baseline.get(p), p) if p in eff else None
            scores[p] = mutation_score((pre_metrics or {}).get(p), ratios[p])

        save_round_metrics(
            post_metrics, round_num, round_dir,
            pre_retrain=pre_metrics,
            unsupervised=unsup,
            extra={
                "thresholds": self.pipeline.supervised_detector.thresholds,
                "n_flags": int(flags["final_flag"].sum()),
                "n_test_markets": len(test_dirs),
                "params_used": params_used,
                "mutation_source": mutation_source or {p: ("baseline" if round_num == 0 else "unknown") for p in self.personas},
                "effectiveness": eff,
                "effectiveness_ratio": ratios,
                "mutation_score": scores,
            },
        )
        entry = {"round": round_num, "personas": post_metrics, "effectiveness_ratio": ratios, "mutation_score": scores}
        if pre_metrics is not None:
            entry["pre_retrain"] = pre_metrics
        self._all_metrics.append(entry)
        self._save_cumulative_metrics()
        return post_metrics

    def get_metrics_for_round(self, round_num: int, phase: str = "post") -> dict[str, dict[str, float]]:
        """Reads metrics for a specific round from disk (used by tools.py)."""
        return load_round_metrics(self.master_run_dir / f"round_{round_num}", phase)

    def get_shap_drift_for_persona(self, persona: str, round_a: int, round_b: int) -> dict[str, float]:
        """Mean-|SHAP| importance change for a persona's classifier between two rounds (FR-3.4)."""
        def _load(r: int) -> dict:
            path = self.master_run_dir / f"round_{r}" / "shap_importance.json"
            return json.loads(path.read_text()).get(persona, {}) if path.exists() else {}
        a, b = _load(round_a), _load(round_b)
        if not a or not b:
            return {"note": "SHAP importance not available for both rounds."}
        return compute_shap_drift(a, b)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_simulation(self, out_dir: Path, seed: int, tick_offset: int = 0) -> None:
        """
        Calls run_simulation.py as a subprocess, writing outputs directly
        into out_dir so each run has isolated Parquet files.
        """
        python_exe = sys.executable
        script = REPO_ROOT / "scripts" / "run_simulation.py"

        cmd = [
            python_exe, str(script),
            "--ticks",       str(self.sim_ticks),
            "--seed",        str(seed),
            "--outdir",      str(out_dir),
            "--commodity",   self.commodity,
            "--tick-offset", str(tick_offset),
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
        df = build_summary_table(self._all_metrics)
        df.to_parquet(self.master_run_dir / "round_summary.parquet", index=False)
        write_round_metrics_parquet(df, self.commodity, self.master_run_dir / "round_metrics.parquet")
        logger.info("Cumulative metrics saved to %s", self.master_run_dir)
