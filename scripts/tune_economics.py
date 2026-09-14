import argparse
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from itertools import product
from sklearn.linear_model import LinearRegression

from src.core.simulation import run_simulation, SimConfig
from src.core.calibration import CalibrationEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def compute_stats(output) -> dict:
    bars = output.ohlcv_bars
    if bars.empty or len(bars) < 5:
        return {"volatility": 0.0, "mean_reversion": 0.0, "spread": 0.0}

    returns = bars['close'].pct_change().dropna()
    volatility = returns.std() if not returns.empty else 0.0

    if len(returns) > 3:
        y = returns.values[1:]
        X = returns.values[:-1].reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        ar1_coeff = model.coef_[0]
        
        log_prices = np.log(bars['close'].dropna().values)
        if len(log_prices) > 3:
            y_lp = log_prices[1:]
            X_lp = log_prices[:-1].reshape(-1, 1)
            model_lp = LinearRegression().fit(X_lp, y_lp)
            mean_reversion = 1.0 - model_lp.coef_[0]
        else:
            mean_reversion = 0.0
    else:
        mean_reversion = 0.0

    orders = output.order_log
    spread = 0.0
    if not orders.empty:
        mm_orders = orders[orders['trader_id'].str.startswith('mm_')]
        if not mm_orders.empty:
            avg_ask = mm_orders[mm_orders['side'] == 'sell']['price'].mean()
            avg_bid = mm_orders[mm_orders['side'] == 'buy']['price'].mean()
            if not np.isnan(avg_ask) and not np.isnan(avg_bid):
                spread = avg_ask - avg_bid
                
    return {
        "volatility": volatility,
        "mean_reversion": mean_reversion,
        "spread": spread
    }

def main():
    parser = argparse.ArgumentParser(description="Tune Agent Parameters to match Commodity")
    parser.add_argument("--commodity", type=str, default="gold", help="Commodity to tune for")
    parser.add_argument("--ticks", type=int, default=500, help="Ticks per tuning run")
    args = parser.parse_args()

    engine = CalibrationEngine()
    ou_target, vol_target = engine.get_fallback(args.commodity)
    logger.info(f"Targets for {args.commodity}:")
    logger.info(f"  Volatility: {ou_target.volatility:.4f}")
    logger.info(f"  Mean Reversion: {ou_target.mean_reversion_speed:.4f}")

    agent_config_path = Path("configs/agent_config.yaml")
    with agent_config_path.open("r") as f:
        agent_cfg = yaml.safe_load(f)

    mr_thresholds = [0.001, 0.005, 0.01]
    noise_offsets = [1.0, 2.0, 5.0]
    mm_spreads = [1.0, 2.0, 4.0]

    best_error = float('inf')
    best_params = None
    best_stats = None

    logger.info("Starting Grid Search...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        for mr_thresh, noise_off, mm_spread in product(mr_thresholds, noise_offsets, mm_spreads):
            
            run_id = f"tune_{mr_thresh}_{noise_off}_{mm_spread}"
            
            config = SimConfig(
                n_ticks=args.ticks,
                ticks_per_bar=50,
                data_dir=tmp_path / run_id,
                fundamental_value_initial=100.0,
                noise_lambda=agent_cfg["agents"]["noise_trader"]["arrival_rate_lambda"],
                noise_price_offset_ticks=noise_off,
                momentum_lambda=agent_cfg["agents"]["momentum_trader"]["arrival_rate_lambda"],
                momentum_lookback=agent_cfg["agents"]["momentum_trader"]["lookback_ticks"],
                reversion_lambda=agent_cfg["agents"]["mean_reversion_trader"]["arrival_rate_lambda"],
                reversion_threshold=mr_thresh,
                market_maker_lambda=agent_cfg["agents"]["market_maker"]["arrival_rate_lambda"],
                market_maker_spread=mm_spread,
            )
            
            try:
                # Disable schema validation on write to avoid pandas float64 nan issue during tuning
                # We can just ignore the pandas error here because it's a tuning run
                output = run_simulation(config, seed=42)
                stats = compute_stats(output)
                
                err_vol = (stats["volatility"] - ou_target.volatility) ** 2
                err_mr = (stats["mean_reversion"] - ou_target.mean_reversion_speed) ** 2
                
                total_error = (err_vol * 1000) + err_mr
                
                if total_error < best_error:
                    best_error = total_error
                    best_params = (mr_thresh, noise_off, mm_spread)
                    best_stats = stats
                    
            except Exception as e:
                logger.error(f"Run failed for {mr_thresh}, {noise_off}, {mm_spread}: {e}")
                continue

    if best_params is None:
        logger.error("Tuning failed. No parameters successfully generated stats.")
        return

    logger.info("=== Tuning Complete ===")
    logger.info(f"Best Params: MR Thresh={best_params[0]}, Noise Offset={best_params[1]}, MM Spread={best_params[2]}")
    logger.info(f"Result Stats: Volatility={best_stats['volatility']:.4f} (Target: {ou_target.volatility:.4f})")
    logger.info(f"Result Stats: Mean Reversion={best_stats['mean_reversion']:.4f} (Target: {ou_target.mean_reversion_speed:.4f})")
    logger.info(f"Result Stats: Emergent Spread={best_stats['spread']:.4f}")

    agent_cfg["agents"]["mean_reversion_trader"]["reversion_threshold"] = float(best_params[0])
    agent_cfg["agents"]["noise_trader"]["price_offset_ticks"] = float(best_params[1])
    agent_cfg["agents"]["market_maker"]["spread_ticks"] = float(best_params[2])
    
    with agent_config_path.open("w") as f:
        yaml.safe_dump(agent_cfg, f, sort_keys=False)
        
    logger.info(f"Updated {agent_config_path} with optimal parameters.")

if __name__ == "__main__":
    main()
