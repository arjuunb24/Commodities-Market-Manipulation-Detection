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
    PRIMARY_METRIC,
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

PROBE_COMMODITIES = ("crude_oil_wti", "gold", "natural_gas", "corn")
PROBE_SEEDS = (9101, 9102)
PROBE_TICKS = 6000


def _probe_market(args) -> str:
    """Process-pool worker: simulate one probe market with the given persona config and write it."""
    import logging as _lg, warnings as _w
    _w.filterwarnings("ignore"); _lg.disable(_lg.WARNING)
    from src.core.calibration import CalibrationEngine
    from src.core.persona_params import apply_persona_config
    from src.core.simulation import SimConfig, run_simulation
    active, commodity, seed, ticks, out_dir = args
    ou, _ = CalibrationEngine().get_fallback(commodity)
    cfg = SimConfig(commodity=commodity, fundamental_value_drift=ou.drift, fundamental_value_vol=ou.volatility,
                    n_ticks=ticks, data_dir=Path(out_dir), overwrite=True, write_to_disk=True)
    apply_persona_config(cfg, active)
    return str(run_simulation(cfg, seed).run_dir)


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

    def probe(self, configs: list[dict[str, dict]], round_num: int | None = None) -> list[dict[str, dict]]:
        """
        Score candidate persona parameter sets against the CURRENT detector (the one
        that will be frozen for the next round) before committing to one.

        Each config maps persona -> full parameter dict. Every config is simulated on the
        same short markets (PROBE_COMMODITIES x PROBE_SEEDS, PROBE_TICKS ticks; common
        random numbers, so candidates differ only by their parameters) and returns, per
        persona, any-alert coverage, effect ratio vs. round 0 and the mutation score.
        """
        import yaml
        if not self.pipeline.is_trained:
            raise RuntimeError("probe() needs a trained detector")
        cfg = yaml.safe_load(self.persona_config_path.read_text()) if self.persona_config_path.exists() else {}
        base_active = resolve_persona_config(cfg or {})
        tag = f"round_{round_num}" if round_num is not None else "adhoc"
        root = self.master_run_dir / "probes" / tag
        # Effect ratios divide by the round-0 effect measured on the SAME probe markets:
        # the full test markets differ in commodities and length, and e.g. wash volume
        # share depends on how much everyone else trades (probe vs. full-market baseline
        # overstated the wash effect by 0.25-0.75 in every round of two runs).
        baseline = self._probe_baseline(base_active, root) or self._baseline_effectiveness()
        # Anchor on the last REAL round: simulate its parameters on the same probe markets and
        # scale its measured effect ratio by candidate / reference. Same seeds, so the probe
        # markets' bias and most of their noise cancel (the 8-episode round-0 probe baseline
        # alone was off by 0.2-0.36 for pump-and-dump and 0.09 for layering).
        real = json.loads((self.master_run_dir / f"round_{round_num}" / "metrics.json").read_text()) \
            if round_num is not None and (self.master_run_dir / f"round_{round_num}" / "metrics.json").exists() else {}
        real_ratio, ref_params = real.get("effectiveness_ratio") or {}, real.get("params_used") or {}
        ref_eff: dict = {}
        if real_ratio and ref_params:
            ref_active = {p: dict(v) for p, v in base_active.items()}
            for p in set().union(*configs) if configs else ():
                if p in ref_params:
                    ref_active[p] = {**ref_active.get(p, {}), **ref_params[p]}
            _, _, _, ref_eff = self._simulate_probe_set(ref_active, root / "reference", features=False)
        out: list[dict[str, dict]] = []
        for k, overrides in enumerate(configs):
            active = {p: dict(v) for p, v in base_active.items()}
            for p, params in overrides.items():
                active[p] = {**active.get(p, {}), **params}
            X, y, ev, eff = self._simulate_probe_set(active, root / f"cand{k}")
            met = compute_round_metrics(self.pipeline.score(X), y, self.personas, events=ev)
            res = {}
            for p in overrides:
                m = met.get(p, {})
                rel = effectiveness_ratio(eff.get(p), ref_eff.get(p), p) if p in eff and p in ref_eff else None
                anchored = round(real_ratio[p] * rel, 3) if rel is not None and real_ratio.get(p) is not None else None
                direct = effectiveness_ratio(eff.get(p), baseline.get(p), p) if p in eff else None
                # Two estimates with different errors: `anchored` inherits the noise of the last real
                # round's 2 test markets (wash off by 0.17-0.24 in run 124309Z), `direct` the bias of
                # the 8-episode round-0 probe baseline (pump-and-dump off by 0.2-0.36 in 092237Z).
                # Their mean is used; both are kept for checking against the realised ratio later.
                parts = [x for x in (anchored, direct) if x is not None]
                ratio = round(sum(parts) / len(parts), 3) if parts else None
                key = PRIMARY_METRIC[p]
                res[p] = {"coverage": m.get("caught_coverage"), "effect_ratio": ratio, "score": mutation_score(m, ratio),
                          "effect_detail": {"anchored": anchored, "direct": direct, "probe_primary": (eff.get(p) or {}).get(key),
                                            "reference_primary": (ref_eff.get(p) or {}).get(key),
                                            "baseline_primary": (baseline.get(p) or {}).get(key),
                                            "real_ratio_last_round": real_ratio.get(p)}}
            logger.info(f"[Probe] candidate set {k}: " + ", ".join(f"{p}: cov {v['coverage']} eff {v['effect_ratio']} score {v['score']}" for p, v in res.items()))
            out.append(res)
        shutil.rmtree(root, ignore_errors=True)   # probe markets are disposable
        return out

    def _simulate_probe_set(self, active: dict, root: Path, features: bool = True):
        """Simulate the probe markets under `active`; returns (X, y, events, summarised effect)."""
        from concurrent.futures import ProcessPoolExecutor
        jobs = [(active, c, s, PROBE_TICKS, str(root / f"{c}_{s}")) for c in PROBE_COMMODITIES for s in PROBE_SEEDS]
        with ProcessPoolExecutor(min(len(jobs), 8)) as ex:
            dirs = list(ex.map(_probe_market, jobs))
        Xs, Ys, Es, effs = [], [], [], []
        for d in dirs:
            if features:
                X, y = self.pipeline.load_and_preprocess(Path(d))
                Xs.append(X); Ys.append(y)
            ev = pd.read_parquet(Path(d) / "manipulation_events.parquet"); Es.append(ev)
            effs.append(episode_effectiveness(ev, pd.read_parquet(Path(d) / "trade_log.parquet"),
                                              pd.read_parquet(Path(d) / "order_log.parquet")))
        X = pd.concat(Xs) if Xs else None
        y = pd.concat(Ys) if Ys else None
        return X, y, pd.concat(Es, ignore_index=True), summarize(pd.concat(effs, ignore_index=True))

    def _probe_baseline(self, base_active: dict, root: Path) -> dict:
        """
        Round-0 effect on the probe markets (simulated once with the round-0 parameters,
        then cached in probe_baseline.json). Empty when round 0 has no params_used.
        """
        cache = self.master_run_dir / "probe_baseline.json"
        if cache.exists():
            return json.loads(cache.read_text())
        r0 = self.master_run_dir / "round_0" / "metrics.json"
        params0 = json.loads(r0.read_text()).get("params_used", {}) if r0.exists() else {}
        if not params0:
            return {}
        active = {p: dict(v) for p, v in base_active.items()}
        for p, params in params0.items():
            active[p] = {**active.get(p, {}), **params}
        _, _, _, eff = self._simulate_probe_set(active, root / "baseline", features=False)
        cache.write_text(json.dumps(eff, indent=2))
        logger.info(f"[Probe] round-0 effect on probe markets: " + ", ".join(
            f"{p}: {d.get(d.get('primary_metric'))}" for p, d in eff.items()))
        return eff

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
