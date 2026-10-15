"""
scripts/calibrate_sim_volatility.py
===================================
Fit the simulator's per-commodity volatility correction factors.

The sourced daily volatilities (src/core/calibration.py) are targets for the
*realised* price volatility. The simulated price is driven by the fundamental
value but also by order-flow noise and the strength of mean-reversion anchoring,
so realised volatility differs from the fundamental input by a commodity-specific
factor. This script simulates each commodity (legitimate population only, several
seeds), measures realised 5-minute log-return volatility, compares it with
    target = daily_vol / sqrt(23400 / 300)
and updates   scale <- scale / ratio**damping   until every ratio is within tolerance.
The factors are written to configs/sim_calibration.yaml; the sourced volatilities
are never changed. 5-minute returns are used because 1-minute returns of large-tick
contracts (corn, wheat) are dominated by bid-ask bounce, which is real and should
not be calibrated away.

Usage:
    .venv\\Scripts\\python.exe scripts/calibrate_sim_volatility.py [--iterations 4] [--seeds 3]
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT = REPO_ROOT / "configs" / "sim_calibration.yaml"
COMMODITIES = ["crude_oil_wti", "brent_crude", "gold", "silver", "natural_gas", "copper", "wheat", "corn"]
HORIZONS = (60, 300, 1800)


def _simulate(args):
    import logging
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    import numpy as np
    from src.core.calibration import CalibrationEngine
    from src.core.simulation import SimConfig, run_simulation
    commodity, scale, seed, ticks = args
    ou, _ = CalibrationEngine().get_fallback(commodity)
    cfg = SimConfig(commodity=commodity, fundamental_value_drift=ou.drift, fundamental_value_vol=ou.volatility * scale,
                    apply_vol_scale=False, n_ticks=ticks, write_to_disk=False, enable_spoofing=False,
                    enable_wash_trading=False, enable_pump_and_dump=False, enable_layering=False)
    t = run_simulation(cfg, seed).trade_log
    out = {}
    for h in HORIZONS:
        px = t.groupby(t["tick"] // h)["price"].last()
        out[h] = np.log(px).diff().dropna().to_numpy()
    return commodity, out


def measure(scales: dict, seeds: list[int], ticks: int) -> dict:
    import numpy as np
    from src.core.calibration import CalibrationEngine
    jobs = [(c, scales[c], s, ticks) for c in COMMODITIES for s in seeds]
    pooled: dict = {c: {h: [] for h in HORIZONS} for c in COMMODITIES}
    with ProcessPoolExecutor(8) as ex:
        for c, rets in ex.map(_simulate, jobs):
            for h in HORIZONS:
                pooled[c][h].append(rets[h])
    ratios = {}
    for c in COMMODITIES:
        ou, _ = CalibrationEngine().get_fallback(c)
        ratios[c] = {h: float(np.concatenate(pooled[c][h]).std() / (ou.volatility / math.sqrt(23400 / h))) for h in HORIZONS}
    return ratios


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--tolerance", type=float, default=0.08)
    ap.add_argument("--damping", type=float, default=0.8)
    a = ap.parse_args()
    seeds = [7001 + i for i in range(a.seeds)]
    scales = {c: 1.0 for c in COMMODITIES}
    history = []
    for it in range(a.iterations + 1):
        ratios = measure(scales, seeds, a.ticks)
        history.append({c: round(ratios[c][300], 3) for c in COMMODITIES})
        print(f"iteration {it}: " + ", ".join(f"{c} {scales[c]:.3f}->{ratios[c][300]:.2f}" for c in COMMODITIES), flush=True)
        if it == a.iterations or all(abs(ratios[c][300] - 1) <= a.tolerance for c in COMMODITIES):
            break
        for c in COMMODITIES:
            scales[c] = float(min(2.0, max(0.3, scales[c] / ratios[c][300] ** a.damping)))
    doc = {
        "vol_scale": {c: round(scales[c], 3) for c in COMMODITIES},
        "fit": {"horizon_ticks": 300, "seeds": seeds, "ticks": a.ticks, "iterations": len(history) - 1,
                "realised_over_target": {c: {f"{h // 60}min": round(ratios[c][h], 3) for h in HORIZONS} for c in COMMODITIES}},
    }
    header = ("# Simulator volatility correction factors (written by scripts/calibrate_sim_volatility.py).\n"
              "# vol_scale multiplies the sourced daily volatility INSIDE the simulator so that realised\n"
              "# 5-minute volatility matches the sourced target. The sourced values are unchanged.\n")
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
