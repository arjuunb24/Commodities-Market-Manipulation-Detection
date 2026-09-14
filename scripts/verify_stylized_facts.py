import argparse
import tempfile
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import yaml
import concurrent.futures

from src.core.simulation import run_simulation, SimConfig
from src.core.calibration import CalibrationEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COMMODITIES = [
    "crude_oil_wti",
    "brent_crude",
    "gold",
    "silver",
    "natural_gas",
    "copper",
    "wheat",
    "corn"
]

def analyze_stylized_facts(output, commodity: str) -> dict:
    bars = output.ohlcv_bars
    trade_log = output.trade_log
    order_log = output.order_log
    manip_events = output.manipulation_events

    stats = {"Commodity": commodity}

    # 1. Macroeconomic Stylized Facts
    if len(bars) > 5:
        log_returns = np.log(bars['close'] / bars['close'].shift(1)).dropna()
        
        # Excess Kurtosis (Fat Tails)
        kurt = kurtosis(log_returns, fisher=True)
        stats["Excess Kurtosis"] = f"{kurt:.2f}"
        
        # Volatility Clustering (Autocorrelation of absolute returns)
        abs_returns = np.abs(log_returns)
        if len(abs_returns) > 5:
            acf_abs = abs_returns.autocorr(lag=1)
            stats["Abs Rtn Autocorr (Lag 1)"] = f"{acf_abs:.3f}"
        else:
            stats["Abs Rtn Autocorr (Lag 1)"] = "N/A"
            
        # Absence of Autocorrelation in Raw Returns
        if len(log_returns) > 5:
            acf_raw = log_returns.autocorr(lag=1)
            stats["Raw Rtn Autocorr (Lag 1)"] = f"{acf_raw:.3f}"
        else:
            stats["Raw Rtn Autocorr (Lag 1)"] = "N/A"
            
    else:
        stats["Excess Kurtosis"] = "N/A"
        stats["Abs Rtn Autocorr (Lag 1)"] = "N/A"
        stats["Raw Rtn Autocorr (Lag 1)"] = "N/A"

    # 2. Microstructure Stats
    if not order_log.empty and not trade_log.empty:
        otr = len(order_log) / len(trade_log)
        stats["Order-to-Trade Ratio"] = f"{otr:.2f}"
    else:
        stats["Order-to-Trade Ratio"] = "N/A"

    if not order_log.empty:
        mm_orders = order_log[order_log['trader_id'].str.startswith('mm_')]
        if not mm_orders.empty:
            avg_ask = mm_orders[mm_orders['side'] == 'sell']['price'].mean()
            avg_bid = mm_orders[mm_orders['side'] == 'buy']['price'].mean()
            if not np.isnan(avg_ask) and not np.isnan(avg_bid):
                stats["Avg Spread (Ticks)"] = f"{(avg_ask - avg_bid):.2f}"
            else:
                stats["Avg Spread (Ticks)"] = "N/A"
        else:
            stats["Avg Spread (Ticks)"] = "N/A"
            
    # 3. Manipulation Footprints
    if not order_log.empty and not manip_events.empty:
        is_manip = np.zeros(len(order_log), dtype=bool)
        ticks = order_log['tick'].values
        
        for _, row in manip_events.iterrows():
            start = row['start_tick']
            end = row['end_tick']
            is_manip |= (ticks >= start) & (ticks <= end)
            
        manip_log = order_log[is_manip]
        normal_log = order_log[~is_manip]
        
        def calc_cancel_ratio(df):
            if df.empty: return 0.0
            cancels = len(df[df['event_type'] == 'cancel'])
            return cancels / len(df)
            
        cr_norm = calc_cancel_ratio(normal_log)
        cr_manip = calc_cancel_ratio(manip_log)
        
        stats["Normal Cancel Ratio"] = f"{cr_norm:.2%}"
        stats["Manip Cancel Ratio"] = f"{cr_manip:.2%}"
    else:
        stats["Normal Cancel Ratio"] = "N/A"
        stats["Manip Cancel Ratio"] = "N/A"

    return stats


def run_single_commodity(commodity: str, n_ticks: int, tmp_path: str):
    logger.info(f"Running simulation for {commodity}...")
    engine = CalibrationEngine()
    agent_config_path = Path("configs/agent_config.yaml")
    with agent_config_path.open("r") as f:
        agent_cfg = yaml.safe_load(f)

    ou_target, _ = engine.get_fallback(commodity)
    
    run_id = f"stylized_{commodity}"
    
    config = SimConfig(
        commodity=commodity,
        n_ticks=n_ticks,
        ticks_per_bar=50,
        data_dir=Path(tmp_path) / run_id,
        fundamental_value_initial=100.0,
        fundamental_value_drift=ou_target.drift,
        fundamental_value_vol=ou_target.volatility,
        noise_lambda=agent_cfg["agents"]["noise_trader"]["arrival_rate_lambda"],
        noise_price_offset_ticks=agent_cfg["agents"]["noise_trader"]["price_offset_ticks"],
        momentum_lambda=agent_cfg["agents"]["momentum_trader"]["arrival_rate_lambda"],
        momentum_lookback=agent_cfg["agents"]["momentum_trader"]["lookback_ticks"],
        reversion_lambda=agent_cfg["agents"]["mean_reversion_trader"]["arrival_rate_lambda"],
        reversion_threshold=agent_cfg["agents"]["mean_reversion_trader"]["reversion_threshold"],
        market_maker_lambda=agent_cfg["agents"]["market_maker"]["arrival_rate_lambda"],
        market_maker_spread=agent_cfg["agents"]["market_maker"]["spread_ticks"],
    )
    
    seed = hash(commodity) % (2**32)
    output = run_simulation(config, seed=seed)
    stats = analyze_stylized_facts(output, commodity)
    logger.info(f"Completed {commodity}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify Large-Scale Stylized Facts")
    parser.add_argument("--ticks", type=int, default=10000, help="Ticks per simulation run")
    args = parser.parse_args()

    results = []
    logger.info(f"Starting stylized facts verification across {len(COMMODITIES)} commodities for {args.ticks} ticks each (Parallel)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(run_single_commodity, commodity, args.ticks, tmpdir): commodity
                for commodity in COMMODITIES
            }
            
            for future in concurrent.futures.as_completed(futures):
                com = futures[future]
                try:
                    stats = future.result()
                    results.append(stats)
                except Exception as e:
                    logger.error(f"Run failed for {com}: {e}")
                
    # Output markdown report
    df_results = pd.DataFrame(results)
    
    report_path = Path("stylized_facts_report.md")
    
    markdown_content = "# Stylized Facts & Market Microstructure Verification\n\n"
    markdown_content += f"**Simulation Length:** {args.ticks} ticks per commodity\n\n"
    markdown_content += "This report verifies that the simulated limit order book organically exhibits universal statistical properties of real financial markets.\n\n"
    
    markdown_content += "## 1. Macroeconomic Stylized Facts\n"
    markdown_content += "- **Excess Kurtosis (> 0):** Proves 'fat tails' (extreme events occur more frequently than a normal distribution).\n"
    markdown_content += "- **Abs Rtn Autocorr (> 0):** Proves 'volatility clustering' (large moves follow large moves).\n"
    markdown_content += "- **Raw Rtn Autocorr (~ 0):** Proves 'local efficiency' (returns themselves are not trivially predictable).\n\n"
    
    markdown_content += "## 2. Microstructure & Manipulation Footprints\n"
    markdown_content += "- **Order-to-Trade Ratio:** Usually high in algorithmic markets (3.0 - 15.0).\n"
    markdown_content += "- **Avg Spread:** Organically maintained by the MM inventory model.\n"
    markdown_content += "- **Cancel Ratios:** Should spike significantly during manipulation episodes (Spoofing/Layering) compared to normal ledgers.\n\n"
    
    markdown_content += "## Results Table\n\n"
    markdown_content += df_results.to_markdown(index=False)
    
    with report_path.open("w") as f:
        f.write(markdown_content)
        
    logger.info(f"Wrote report to {report_path}")

if __name__ == "__main__":
    main()
