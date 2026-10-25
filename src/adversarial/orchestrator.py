"""
src/adversarial/orchestrator.py
================================
RoundOrchestrator: orchestrates one complete round of the adversarial loop.

One round = Simulate → Detect → Score → Log Metrics.
The full multi-round loop is driven by scripts/run_full_pipeline.py.
"""

from __future__ import annotations

import copy
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
    account_pnl,
    effectiveness_ratio,
    episode_effectiveness,
    mutation_score,
    summarize,
)
from src.core.persona_params import effective_params, resolve_persona_config
from src.detector.pipeline import DetectorPipeline
from src.utils import io as io_utils

logger = logging.getLogger(__name__)


def _merge_pnl(parts: list[dict]) -> dict:
    """Episode-weighted combination of per-market account_pnl summaries."""
    out: dict[str, dict] = {}
    for part in parts:
        for p, v in part.items():
            o = out.setdefault(p, {"_bp": [], "_prof": [], "episodes": 0})
            o["_bp"] += [v["pnl_bp_median"]] * v["episodes"]
            o["_prof"] += [v["profitable_share"]] * v["episodes"]
            o["episodes"] += v["episodes"]
    return {p: {"pnl_bp_median": round(float(np.median(o["_bp"])), 2),
                "profitable_share": round(float(np.mean(o["_prof"])), 3),
                "episodes": o["episodes"]} for p, o in out.items()}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PROBE_COMMODITIES = ("crude_oil_wti", "gold", "natural_gas", "corn")
PROBE_SEEDS = (9101, 9102)
PROBE_TICKS = 6000


def _probe_market(args) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Process-pool worker: simulate one probe market with the given persona config, write it,
    cache its features (when `detector_config` is given) and measure its episodes' effect.
    Doing the feature and effect work here runs it in parallel across markets.
    """
    import logging as _lg, warnings as _w
    _w.filterwarnings("ignore"); _lg.disable(_lg.WARNING)
    from src.core.calibration import CalibrationEngine
    from src.core.persona_params import apply_persona_config
    from src.core.simulation import SimConfig, run_simulation
    active, commodity, seed, ticks, out_dir, detector_config = args
    ou, _ = CalibrationEngine().get_fallback(commodity)
    cfg = SimConfig(commodity=commodity, fundamental_value_drift=ou.drift, fundamental_value_vol=ou.volatility,
                    n_ticks=ticks, data_dir=Path(out_dir), overwrite=True, write_to_disk=True)
    apply_persona_config(cfg, active)
    run_dir = Path(run_simulation(cfg, seed).run_dir)
    if detector_config is not None:
        DetectorPipeline(detector_config).load_and_preprocess(run_dir)   # writes the features cache
    ev = pd.read_parquet(run_dir / "manipulation_events.parquet")
    eff = episode_effectiveness(ev, pd.read_parquet(run_dir / "trade_log.parquet"),
                                pd.read_parquet(run_dir / "order_log.parquet"))
    return str(run_dir), ev, eff


def _cache_features(args) -> str:
    """Process-pool worker: compute and cache one market's features/labels (read back from the cache)."""
    import logging as _lg, warnings as _w
    _w.filterwarnings("ignore"); _lg.disable(_lg.WARNING)
    detector_config, run_dir = args
    DetectorPipeline(detector_config).load_and_preprocess(Path(run_dir))
    return run_dir


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
        commodity: str = "all",
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
        loop = detector_config.get("loop", {}) or {}
        self.feature_workers = max(1, int(loop.get("feature_workers", 3)))   # parallel feature extraction (round markets)
        self.probe_workers = max(1, int(loop.get("probe_workers", 8)))       # parallel probe markets
        self.replay_check_every = max(1, int(loop.get("replay_check_every", 1)))
        # Randomised persona variants (src/adversarial/variants.py): extra training markets per
        # round, and a fixed benchmark of unseen variants scored every round. Both are
        # simulated alongside the round's own markets and are half length by default.
        self.train_variants = max(0, int(loop.get("train_variants", 1)))
        self.benchmark_markets = max(0, int(loop.get("benchmark_markets", 2)))
        self.variant_ticks = max(1000, int(self.sim_ticks * float(loop.get("variant_ticks_fraction", 0.5))))
        self.last_round: int | None = None   # set by the driver so the final round always runs the replay check

        # Cumulative metrics list — grows each round
        self._all_metrics: list[dict] = []
        self._round0 = None   # copy of the round-0 detector (scores later rounds' markets too)
        self._concurrent_sims = 1   # simulations running at once (they share the CPUs)

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
        `round_num` is the last COMPLETED round (its metrics.json anchors the effect estimate).
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
        # The reference and every candidate are simulated in ONE process pool (one batch of
        # markets instead of a pool per set, with features computed in the workers).
        specs = []
        if real_ratio and ref_params:
            ref_active = {p: dict(v) for p, v in base_active.items()}
            for p in set().union(*configs) if configs else ():
                if p in ref_params:
                    ref_active[p] = {**ref_active.get(p, {}), **ref_params[p]}
            specs.append((ref_active, root / "reference", False))
        for k, overrides in enumerate(configs):
            active = {p: dict(v) for p, v in base_active.items()}
            for p, params in overrides.items():
                active[p] = {**active.get(p, {}), **params}
            specs.append((active, root / f"cand{k}", True))
        sets = self._simulate_probe_sets(specs)
        ref_eff: dict = sets.pop(0)[3] if len(sets) > len(configs) else {}
        out: list[dict[str, dict]] = []
        for k, (overrides, (X, y, ev, eff)) in enumerate(zip(configs, sets)):
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
        return self._simulate_probe_sets([(active, root, features)])[0]

    def _simulate_probe_sets(self, specs: list[tuple[dict, Path, bool]]) -> list[tuple]:
        """
        Simulate the probe markets for several (active, root, features) sets in one process
        pool; returns one (X, y, events, summarised effect) per set, in order.
        """
        from concurrent.futures import ProcessPoolExecutor
        jobs, owner = [], []
        for i, (active, root, features) in enumerate(specs):
            for c in PROBE_COMMODITIES:
                for s in PROBE_SEEDS:
                    jobs.append((active, c, s, PROBE_TICKS, str(Path(root) / f"{c}_{s}"),
                                 self.detector_config if features else None))
                    owner.append(i)
        if not jobs:
            return []
        with ProcessPoolExecutor(min(len(jobs), self.probe_workers)) as ex:
            done = list(ex.map(_probe_market, jobs))
        out = []
        for i, (_, _, features) in enumerate(specs):
            Xs, Ys, Es, effs = [], [], [], []
            for (d, ev, eff), o in zip(done, owner):
                if o != i:
                    continue
                if features:
                    X, y = self.pipeline.load_and_preprocess(Path(d))   # cache written by the worker
                    Xs.append(X); Ys.append(y)
                Es.append(ev); effs.append(eff)
            out.append((pd.concat(Xs) if Xs else None, pd.concat(Ys) if Ys else None,
                        pd.concat(Es, ignore_index=True), summarize(pd.concat(effs, ignore_index=True))))
        return out

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

    def _by_market(self, per_market: list, results: pd.DataFrame, pre_results: pd.DataFrame | None, baseline: dict) -> dict:
        """
        The round's metrics split by test market (each is an independent simulation).
        The pooled numbers above stay the official ones; these let a viewer compare markets.
        Effect ratios use the same pooled round-0 baseline as the pooled ratios.
        """
        out = {}
        for name, idx, y_m, ev_m, eff_m in per_market:
            entry: dict[str, Any] = {"personas": compute_round_metrics(results.loc[idx], y_m, self.personas, events=ev_m)}
            if pre_results is not None:
                entry["pre_retrain"] = compute_round_metrics(pre_results.loc[idx], y_m, self.personas, events=ev_m)
            e = summarize(eff_m) if not eff_m.empty else {}
            ratio = {p: effectiveness_ratio(e.get(p), baseline.get(p), p) if p in e else None for p in self.personas}
            entry["effectiveness"] = e
            entry["effectiveness_ratio"] = ratio
            entry["mutation_score"] = {p: mutation_score((entry.get("pre_retrain") or {}).get(p), ratio[p]) for p in self.personas}
            entry["n_episodes"] = int(len(ev_m))
            out[name] = entry
        return out

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
        variant_dirs = [round_dir / ("train_variant" if v == 0 else f"train_variant_{v}") for v in range(self.train_variants)]
        bench_dirs = [] if self._benchmark_ready() else self._benchmark_dirs()
        logger.info(f"ROUND {round_num} — simulating 1 train + {len(variant_dirs)} variant + {len(test_dirs)} test"
                    + (f" + {len(bench_dirs)} benchmark" if bench_dirs else "") + " markets...")
        T = self.sim_ticks
        jobs = [(train_dir, self.sim_seed + round_num * 10, round_num * T)]
        for m, d in enumerate(test_dirs):
            jobs.append((d, self.sim_seed + round_num * 10 + 1 + m, (round_num * self.n_test_markets + m) * T))
        # Variant and benchmark seeds live in their own ranges, clear of every round's train/test seeds.
        for v, d in enumerate(variant_dirs):
            cfg = self._write_variant_config(d, self._params_active(), seed=(self.sim_seed, round_num, v))
            jobs.append((d, self.sim_seed + 50_000 + round_num * 10 + v, 0, cfg, self.variant_ticks))
        for b, d in enumerate(bench_dirs):
            cfg = self._write_variant_config(d, self._baseline_active(), seed=(self.sim_seed, 99_999, b))
            jobs.append((d, self.sim_seed + 90_000 + b, 0, cfg, self.variant_ticks))
        import concurrent.futures
        self._concurrent_sims = len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for f in [executor.submit(self._run_simulation, *job) for job in jobs]:
                f.result()
        self._concurrent_sims = 1
        self._cache_features([*test_dirs, train_dir, *variant_dirs, *bench_dirs])

        X_parts, y_parts, ev_parts, eff_parts, tr_parts = [], [], [], [], []
        per_market = []   # (market name, row index, labels, events, per-episode effect) for by-market metrics
        for d in test_dirs:
            X_m, y_m = self.pipeline.load_and_preprocess(d)
            ev_m = pd.read_parquet(d / "manipulation_events.parquet")
            if not X_m.empty:
                X_parts.append(X_m)
                y_parts.append(y_m)
            ev_parts.append(ev_m)
            tr_m = pd.read_parquet(d / "trade_log.parquet")
            tr_parts.append(account_pnl(ev_m, tr_m))
            eff_parts.append(episode_effectiveness(
                ev_m, tr_m, pd.read_parquet(d / "order_log.parquet")))
            if not X_m.empty:
                per_market.append((d.name, X_m.index, y_m, ev_m, eff_parts[-1]))
        if not X_parts:
            logger.warning(f"Round {round_num}: empty test feature matrix. Skipping.")
            return {}
        X_test, y_test = pd.concat(X_parts), pd.concat(y_parts)
        events_test = pd.concat(ev_parts, ignore_index=True)
        eff = summarize(pd.concat(eff_parts, ignore_index=True))
        pnl = _merge_pnl(tr_parts)
        for p, v in pnl.items():
            logger.info(f"Round {round_num} — {p}: P&L median {v['pnl_bp_median']:+.1f} bp, "
                        f"{v['profitable_share']:.0%} of {v['episodes']} episodes profitable (logged, not optimised)")

        # --- Pre-retrain (frozen detector from previous round) ---
        pre_metrics = None
        pre_results = None
        if self.pipeline.is_trained:
            logger.info(f"Round {round_num} — PRE-retrain scoring with frozen detector...")
            pre_results = self.pipeline.score(X_test)
            pre_metrics = compute_round_metrics(pre_results, y_test, self.personas, events=events_test)
        # The round-0 detector on this round's markets: how far the personas have moved from
        # the baseline overall (round 1's pre-retrain detector already is the round-0 one).
        round0_metrics = None
        if round_num >= 2 and (r0 := self._round0_detector()) is not None:
            logger.info(f"Round {round_num} — scoring with the round-0 detector...")
            round0_metrics = compute_round_metrics(r0.score(X_test), y_test, self.personas, events=events_test)

        # --- Retrain on cumulative training data ---
        logger.info(f"Round {round_num} — retraining on rounds 0..{round_num}...")
        X_tr, y_tr, w_tr = [], [], []
        anchor = float(self.detector_config.get("continual", {}).get("anchor_round_weight", 1.0))
        for r in range(round_num + 1):
            rd = self.master_run_dir / f"round_{r}"
            markets = [rd / "train"] + sorted(d for d in rd.glob("train_variant*") if d.is_dir())
            for market in markets:
                x_train_market, y_train_market = self.pipeline.load_and_preprocess(market)
                if not x_train_market.empty:
                    x_train_market = x_train_market.copy()
                    x_train_market["cv_group"] = f"round_{r}:{market.name}"
                    X_tr.append(x_train_market)
                    y_tr.append(y_train_market)
                    w_tr.append(np.full(len(x_train_market), anchor if r == 0 and market.name == "train" else 1.0))
        if not X_tr:
            logger.warning(f"Round {round_num}: empty training data. Skipping.")
            return {}
        for x_replay, y_replay in self._replay_training_markets():
            X_tr.append(x_replay)
            y_tr.append(y_replay)
            w_tr.append(np.ones(len(x_replay)))
        weights = np.concatenate(w_tr)
        self.pipeline.train(pd.concat(X_tr), pd.concat(y_tr), save_dir=round_dir,
                            sample_weight=None if np.all(weights == 1.0) else weights)

        # --- Post-retrain ---
        results = self.pipeline.score(X_test)
        post_metrics = compute_round_metrics(results, y_test, self.personas, events=events_test)
        unsup = compute_unsupervised_metrics(results, y_test)
        flags = self.pipeline.build_flags(X_test, results)
        self.pipeline.write_flags(flags, round_dir / "flags.parquet")
        bench_metrics = self._score_benchmark()

        # --- Effectiveness and strategist objective ---
        baseline = eff if round_num == 0 else self._baseline_effectiveness()
        ratios, scores = {}, {}
        for p in self.personas:
            ratios[p] = effectiveness_ratio(eff.get(p), baseline.get(p), p) if p in eff else None
            scores[p] = mutation_score((pre_metrics or {}).get(p), ratios[p])
        by_market = self._by_market(per_market, results, pre_results, baseline)

        save_round_metrics(
            post_metrics, round_num, round_dir,
            pre_retrain=pre_metrics,
            unsupervised=unsup,
            extra={
                "thresholds": self.pipeline.supervised_detector.thresholds,
                "n_flags": int(flags["final_flag"].sum()),
                "n_test_markets": len(test_dirs),
                "evaluation_protocol": {
                    "threshold_cv_grouping": "simulation_market",
                    "test_markets_in_model_training": False,
                    "window_rows_are_independent": False,
                    "primary_evaluation_unit": "manipulation_episode",
                    "train_seeds": [self.sim_seed + round_num * 10],
                    "test_seeds": [
                        self.sim_seed + round_num * 10 + 1 + m
                        for m in range(len(test_dirs))
                    ],
                    "probe_seeds": list(PROBE_SEEDS),
                    "probe_run_seed_overlap": bool(set(PROBE_SEEDS) & {
                        self.sim_seed + r * 10 + m
                        for r in range(round_num + 1)
                        for m in range(self.n_test_markets + 1)
                    }),
                    "probe_commodities": list(PROBE_COMMODITIES),
                    "held_out_case_replays": list(
                        (self.detector_config.get("replay_check", {}) or {}).get("cases", [])
                    ),
                    "note": (
                        "Post-retrain metrics use held-out test markets; frozen pre-retrain "
                        "metrics measure mutation evasion. Test markets are not used for "
                        "model fitting or threshold tuning. Test markets share the simulator "
                        "persona implementations with training markets."
                    ),
                },
                "params_used": params_used,
                "mutation_source": mutation_source or {p: ("baseline" if round_num == 0 else "unknown") for p in self.personas},
                "effectiveness": eff,
                "manipulator_pnl": pnl,
                "effectiveness_ratio": ratios,
                "mutation_score": scores,
                "by_market": by_market,
                **({"round0_detector": round0_metrics} if round0_metrics is not None else {}),
                **({"benchmark": bench_metrics} if bench_metrics else {}),
            },
        )
        self._save_models(round_num, round_dir)
        if round_num == 0:
            self._round0 = copy.deepcopy(self.pipeline)
        replay_check = self._replay_check(round_num) if self._replay_check_due(round_num) else {}
        if replay_check:
            with open(round_dir / "replay_check.json", "w") as f:
                json.dump(replay_check, f, indent=2)
        entry = {"round": round_num, "personas": post_metrics, "effectiveness_ratio": ratios, "mutation_score": scores}
        if pre_metrics is not None:
            entry["pre_retrain"] = pre_metrics
        if round0_metrics is not None:
            entry["round0_detector"] = round0_metrics
        if bench_metrics:
            entry["benchmark"] = bench_metrics
        self._all_metrics.append(entry)
        self._save_cumulative_metrics()
        return post_metrics

    # ---- randomised variants and the benchmark (src/adversarial/variants.py) ----

    def _persona_doc(self) -> dict:
        import yaml
        return (yaml.safe_load(self.persona_config_path.read_text()) if self.persona_config_path.exists() else {}) or {}

    def _params_active(self) -> dict:
        return resolve_persona_config(self._persona_doc())

    def _baseline_active(self) -> dict:
        return resolve_persona_config({"round_0": self._persona_doc().get("round_0") or {}})

    def _write_variant_config(self, market_dir: Path, base: dict, seed: tuple) -> str:
        """Write a random variant of `base` next to the market and return its path."""
        import yaml
        from src.adversarial.variants import variant_config
        market_dir.mkdir(parents=True, exist_ok=True)
        doc = variant_config(base, np.random.default_rng(list(seed)), self.personas)
        path = market_dir / "variant_persona_config.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=True))
        return str(path)

    def _benchmark_dirs(self) -> list[Path]:
        return [self.master_run_dir / "benchmark" / f"market_{b}" for b in range(self.benchmark_markets)]

    def _benchmark_ready(self) -> bool:
        return all((d / "order_log.parquet").exists() for d in self._benchmark_dirs())

    def _score_benchmark(self) -> dict:
        """Metrics of the just-retrained detector on the fixed benchmark of unseen variants."""
        dirs = [d for d in self._benchmark_dirs() if (d / "order_log.parquet").exists()]
        Xs, ys, evs = [], [], []
        for d in dirs:
            X_b, y_b = self.pipeline.load_and_preprocess(d)
            if not X_b.empty:
                Xs.append(X_b)
                ys.append(y_b)
                evs.append(pd.read_parquet(d / "manipulation_events.parquet"))
        if not Xs:
            return {}
        return compute_round_metrics(self.pipeline.score(pd.concat(Xs)), pd.concat(ys), self.personas,
                                     events=pd.concat(evs, ignore_index=True))

    def _round0_detector(self):
        """The detector trained in round 0 (kept in memory; loaded from round_0/models after a restart)."""
        if self._round0 is None:
            path = self.master_run_dir / "round_0" / "models" / "detector.joblib"
            if path.exists():
                try:
                    import joblib
                    self._round0 = joblib.load(path)
                except Exception as exc:  # pragma: no cover - logged, not fatal
                    logger.warning("Could not load the round-0 detector: %s", exc)
        return self._round0

    def _cache_features(self, dirs: list[Path]) -> None:
        """Compute the round markets' features in parallel worker processes (load_and_preprocess then hits the cache)."""
        if self.feature_workers <= 1 or len(dirs) <= 1:
            return
        from concurrent.futures import ProcessPoolExecutor
        logger.info("Computing features for %d markets (%d workers)...", len(dirs), self.feature_workers)
        with ProcessPoolExecutor(min(len(dirs), self.feature_workers)) as ex:
            list(ex.map(_cache_features, [(self.detector_config, str(d)) for d in dirs]))

    def _replay_markets(self, block: str) -> list[Path]:
        """Replay markets (simulated once, cached) for detector_config[block] = {cases, seeds, ticks}."""
        cfg = self.detector_config.get(block, {}) or {}
        if not cfg.get("cases"):
            return []
        from src.replay.evaluate import load_cases, simulate_market
        cases = load_cases()
        dirs = []
        for cid in cfg["cases"]:
            if cid not in cases:
                logger.warning("%s: unknown case %s (see configs/case_replays.yaml)", block, cid)
                continue
            for com in cases[cid].get("markets", []):
                for seed in cfg.get("seeds", [1]):
                    dirs.append(simulate_market(cid, cases[cid], com, int(seed), int(cfg.get("ticks", 6000))))
        return dirs

    def _replay_training_markets(self):
        """Labelled replay markets added to every retrain (replay_augmentation); features cached per market."""
        if not hasattr(self, "_aug_cache"):
            self._aug_cache = []
            for d in self._replay_markets("replay_augmentation"):
                X_a, y_a = self.pipeline.load_and_preprocess(d)
                if not X_a.empty:
                    X_a = X_a.copy()
                    X_a["cv_group"] = f"replay:{d.resolve()}"
                    self._aug_cache.append((X_a, y_a))
            if self._aug_cache:
                logger.info("Replay augmentation: %d labelled real-case markets added to training", len(self._aug_cache))
        return self._aug_cache

    def _replay_check_due(self, round_num: int) -> bool:
        """Round 0, every `replay_check_every`-th round and the final round (when the driver set it)."""
        return round_num == 0 or round_num % self.replay_check_every == 0 or round_num == self.last_round

    def _replay_check(self, round_num: int) -> dict:
        """
        Score held-out real-case replays with this round's retrained detector. Logged and
        saved; a case that stops being recognised (no window with its own label, family
        spoofing/layering) triggers a warning, so forgetting is visible while the loop runs.
        """
        try:
            dirs = self._replay_markets("replay_check")
            if not dirs:
                return {}
            from src.replay.evaluate import score_market
            rows = pd.DataFrame([score_market(d, self.pipeline, round_num) for d in dirs])
            out = {}
            for case, g in rows.groupby("case"):
                out[case] = {"markets": int(len(g)), "caught": round(float(g["caught"].mean()), 3),
                             "recognised_family": round(float(g["recognised_family"].mean()), 3),
                             "label_share": round(float(g["label_share"].mean()), 3),
                             "max_prob": round(float(g["max_prob"].median()), 3)}
            prev = getattr(self, "_last_replay_check", {})
            for case, v in out.items():
                before = prev.get(case, {}).get("recognised_family")
                if before is not None and v["recognised_family"] < before - 0.25:
                    logger.warning("Round %d: real-case replay %s recognised in %.0f%% of markets (was %.0f%%): "
                                   "the retrained detector is forgetting this tactic.", round_num, case,
                                   100 * v["recognised_family"], 100 * before)
            self._last_replay_check = out
            return out
        except Exception as exc:  # pragma: no cover - diagnostic only, never stops the loop
            logger.warning("Round %d: replay check failed: %s", round_num, exc)
            return {}

    def _save_models(self, round_num: int, round_dir: Path) -> None:
        """
        Persist this round's retrained detector and a bar-level model (anonymous 1-minute
        bars, for real market data) to round_N/models/. Saving must never stop the loop.
        """
        try:
            from src.live.bar_model import train_round_bar_model
            from src.live.registry import save_round_models
            bar_model = train_round_bar_model(self.master_run_dir, round_num, self.personas)
            save_round_models(round_dir, pipeline=self.pipeline, bar_model=bar_model)
        except Exception as exc:  # pragma: no cover - logged, not fatal
            logger.warning("Round %d: saving models failed: %s", round_num, exc)

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

    def _run_simulation(self, out_dir: Path, seed: int, tick_offset: int = 0,
                        persona_config: str | None = None, ticks: int | None = None) -> None:
        """
        Calls run_simulation.py as a subprocess, writing outputs directly
        into out_dir so each run has isolated Parquet files.
        """
        python_exe = sys.executable
        script = REPO_ROOT / "scripts" / "run_simulation.py"
        import os
        # Each concurrent simulation gets a share of the CPUs: 8 markets x one worker per CPU
        # (64 processes on 8 cores) broke a worker pool (seed222: "semaphore or lock released
        # too many times") and left a market with 1 of 8 commodities.
        workers = max(1, (os.cpu_count() or 4) * 2 // max(1, self._concurrent_sims))

        cmd = [
            python_exe, str(script),
            "--ticks",       str(ticks or self.sim_ticks),
            "--seed",        str(seed),
            "--outdir",      str(out_dir),
            "--commodity",   self.commodity,
            "--tick-offset", str(tick_offset),
            "--workers",     str(workers),
            *(["--persona-config", str(persona_config)] if persona_config else []),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        for attempt in (1, 2):
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                break
            # The script exits non-zero when any commodity failed; retry once from a clean folder.
            shutil.rmtree(Path(out_dir) / "runs", ignore_errors=True)
            if attempt == 2:
                raise RuntimeError(f"Simulation subprocess failed twice with code {result.returncode} ({out_dir})")
            logger.warning("Simulation for %s failed (code %d); retrying once.", out_dir, result.returncode)

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
