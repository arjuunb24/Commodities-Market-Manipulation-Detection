"""
src/core/simulation.py
=======================
Simulation loop for the Veridex ABM.

This is the orchestrator for Modules 1 & 3:
  - Creates and seeds all agents (legitimate + manipulation personas)
  - Runs the tick-by-tick simulation loop
  - Collects all trade and order log rows
  - Validates and persists outputs via src/utils/io.py (with Pandera schemas)
  - Produces a unique run_id = {timestamp}_{seed} for every run

SRS FR-0.3: "Every simulation run must produce a unique, timestamp-seeded run_id."
SRS FR-0.4: "Two runs with the same master seed and config must produce byte-identical
             outputs." (enforced via numpy.random.default_rng + rng.spawn)
NFR-R2: Atomic write-then-rename for all output files (implemented in io.py).

Architecture:
  - run_simulation() is the single public entry point.
  - All state lives in local variables within run_simulation() — no module-level
    mutable state (Implementation Plan constraint: "no global mutable state").
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.core.agents.base import Agent
from src.core.agents.manipulation.layering import LayeringAgent
from src.core.agents.manipulation.pump_and_dump import PumpDumpCoalition
from src.core.agents.manipulation.spoofing import SpoofingAgent
from src.core.agents.manipulation.wash_trading import WashTradingPair
from src.core.agents.market_maker import MarketMaker
from src.core.agents.mean_reversion_trader import MeanReversionTrader
from src.core.agents.momentum_trader import MomentumTrader
from src.core.agents.noise_trader import NoiseTrader
from src.core.calibration import CalibrationEngine
from src.core.order_book import BookState, Order, OrderBook, Trade
from src.utils import io as io_utils
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class SimConfig:
    """
    Full configuration for one simulation run.

    All parameters can be specified via the CLI or config files.
    Defaults are loaded from configs/agent_config.yaml + configs/persona_config.yaml.
    """
    # Simulation parameters
    commodity: str = "gold"
    n_ticks: int = 10_000
    ticks_per_bar: int = 100
    fundamental_value_initial: float = 100.0
    fundamental_value_drift: float = 0.0001
    fundamental_value_vol: float = 0.002

    # Agent population (counts)
    n_noise_traders: int = 20
    n_momentum_traders: int = 8
    n_mean_reversion_traders: int = 6
    n_market_makers: int = 1

    # Agent parameters (passed through from agent_config.yaml)
    noise_lambda: float = 0.6
    noise_price_offset_ticks: float = 2.0
    momentum_lambda: float = 0.3
    momentum_lookback: int = 20
    momentum_threshold: float = 0.002
    reversion_lambda: float = 0.25
    reversion_threshold: float = 0.005
    market_maker_lambda: float = 0.95
    market_maker_spread: float = 2.0

    # Persona flags
    enable_spoofing: bool = True
    enable_wash_trading: bool = True
    enable_pump_and_dump: bool = True
    enable_layering: bool = False

    # Spoofing parameters
    spoof_size_multiplier: float = 8.0
    spoof_cancel_delay: int = 5
    spoof_frequency: float = 0.4
    spoof_aggressiveness: float = 0.003

    # Wash trading parameters
    wash_frequency: float = 0.5
    wash_price_deviation: float = 0.001
    wash_n_pairs: int = 1

    # Pump-and-dump parameters
    pnd_burst_duration: int = 200
    pnd_n_accounts: int = 3
    pnd_dump_delay: int = 300
    pnd_accumulation_size: int = 150

    # Layering parameters
    layer_n_layers: int = 4
    layer_spacing: float = 0.002
    layer_cancel_delay: int = 3

    # Output
    data_dir: Path = Path("data")
    overwrite: bool = False
    write_to_disk: bool = True


# ===========================================================================
# Simulation output
# ===========================================================================


@dataclass
class SimulationOutput:
    """
    All outputs from a completed simulation run.

    DataFrames are in the exact schema expected by Pandera validators.
    Files are also written to disk by run_simulation() before returning.
    """
    run_id: str
    run_dir: Path
    trade_log: pd.DataFrame
    order_log: pd.DataFrame
    manipulation_events: pd.DataFrame
    ohlcv_bars: pd.DataFrame
    config: SimConfig


# ===========================================================================
# Fundamental value process
# ===========================================================================


class FundamentalValueProcess:
    """
    Slow random walk modelling the 'true' underlying commodity value.

    The fundamental value is provided as an exogenous signal to
    MeanReversionTrader agents each tick. It is NOT the market price.

    Driven by: V_{t+1} = V_t * exp(drift + vol * Z_t)  where Z_t ~ N(0,1)
    This is a simple GBM with very low volatility — intentionally slow-moving.
    """

    def __init__(
        self,
        initial_value: float,
        drift: float,
        volatility: float,
        rng: np.random.Generator,
    ) -> None:
        self.value = initial_value
        self.drift = drift
        self.volatility = volatility
        self.rng = rng

    def step(self) -> float:
        """Advance by one tick and return the new fundamental value."""
        shock = self.rng.standard_normal()
        self.value *= np.exp(self.drift + self.volatility * shock)
        return self.value


# ===========================================================================
# Agent factory
# ===========================================================================


def _build_agents(
    config: SimConfig,
    master_rng: np.random.Generator,
) -> tuple[list[Agent], list[Any], list[dict]]:
    """
    Instantiate all agents and manipulation controllers.

    Returns:
        agents: All individual Agent instances (legitimate + manipulation)
        controllers: Manipulation controller objects (PumpDumpCoalition, WashTradingPair, etc.)
                     that the simulation loop must call .update_phase() or .get_cancels() on.
        manipulation_event_rows: Pre-populated manipulation_events rows from controllers.

    Each agent and controller gets its own independent sub-generator via rng.spawn(1)[0].
    """
    # Spawn a pool of sub-generators — one per entity needing randomness
    # Conservative count: 50 agents + 10 controllers
    sub_rngs = iter(master_rng.spawn(100))

    agents: list[Agent] = []
    controllers: list[Any] = []
    manipulation_event_rows: list[dict] = []

    # --- Legitimate agents ---
    for i in range(config.n_noise_traders):
        agents.append(NoiseTrader(
            f"noise_{i}", next(sub_rngs),
            arrival_rate_lambda=config.noise_lambda,
            price_offset_ticks=config.noise_price_offset_ticks,
        ))

    for i in range(config.n_momentum_traders):
        agents.append(MomentumTrader(
            f"momentum_{i}", next(sub_rngs),
            lookback_ticks=config.momentum_lookback,
            momentum_threshold=config.momentum_threshold,
            arrival_rate_lambda=config.momentum_lambda,
        ))

    for i in range(config.n_mean_reversion_traders):
        agents.append(MeanReversionTrader(
            f"meanrev_{i}", next(sub_rngs),
            reversion_threshold=config.reversion_threshold,
            arrival_rate_lambda=config.reversion_lambda,
        ))

    for i in range(config.n_market_makers):
        agents.append(MarketMaker(
            f"mm_{i}", next(sub_rngs),
            spread_ticks=config.market_maker_spread,
            arrival_rate_lambda=config.market_maker_lambda,
        ))

    # --- Manipulation personas (Dynamic Spawning) ---
    # We dynamically calculate how many episodes to run based on total n_ticks.
    # We aim for roughly 1 episode of each type per 5000 ticks.
    n_episodes = max(1, config.n_ticks // 5000)
    
    def _get_random_window(window_size: int = 2000) -> tuple[int, int]:
        # Ensure we have enough ticks to place the window
        max_start = max(0, config.n_ticks - window_size - 100)
        start = master_rng.integers(0, max_start + 1)
        return int(start), int(start + window_size)

    if config.enable_spoofing:
        for i in range(n_episodes):
            start, end = _get_random_window(2000)
            spoofer = SpoofingAgent(
                f"spoofer_{i}", next(sub_rngs),
                order_size_multiplier=config.spoof_size_multiplier,
                cancellation_delay_ticks=config.spoof_cancel_delay,
                frequency=config.spoof_frequency,
                price_aggressiveness=config.spoof_aggressiveness,
                start_tick=start,
                end_tick=end,
            )
            agents.append(spoofer)
            controllers.append(spoofer)
            manipulation_event_rows.append(spoofer.log_activity(start))

    if config.enable_wash_trading:
        for i in range(n_episodes):
            start, end = _get_random_window(2000)
            for j in range(config.wash_n_pairs):
                pair = WashTradingPair.create(
                    next(sub_rngs), i * 100 + j,
                    trade_frequency=config.wash_frequency,
                    price_deviation_from_mid=config.wash_price_deviation,
                    start_tick=start,
                    end_tick=end,
                )
                agents.extend(pair.agents)
                controllers.append(pair)
                manipulation_event_rows.append(pair.log_activity())

    if config.enable_pump_and_dump:
        for i in range(n_episodes):
            start, end = _get_random_window(2000)
            coalition = PumpDumpCoalition.create(
                next(sub_rngs),
                n_coordinated_accounts=config.pnd_n_accounts,
                burst_duration_ticks=config.pnd_burst_duration,
                dump_delay_ticks=config.pnd_dump_delay,
                accumulation_size=config.pnd_accumulation_size,
                start_tick=start,
                end_tick=end,
            )
            agents.extend(coalition.agents)
            controllers.append(coalition)
            manipulation_event_rows.append(coalition.log_activity())

    if config.enable_layering:
        for i in range(n_episodes):
            start, end = _get_random_window(2000)
            layerer = LayeringAgent(
                f"layerer_{i}", next(sub_rngs),
                n_layers=config.layer_n_layers,
                layer_spacing=config.layer_spacing,
                cancellation_delay_ticks=config.layer_cancel_delay,
                start_tick=start,
                end_tick=end,
            )
            agents.append(layerer)
            controllers.append(layerer)
            manipulation_event_rows.append(layerer.log_activity())

    return agents, controllers, manipulation_event_rows


# ===========================================================================
# OHLCV bar aggregation
# ===========================================================================


def _resample_to_bars(trade_log_df: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """
    Resample trade_log to OHLCV bars.

    SRS FR-1.7: Zero-trade bars have NaN OHLC and volume=0.
    Bars are NOT forward-filled here — only optionally in the dashboard layer.
    """
    if trade_log_df.empty:
        df = pd.DataFrame(columns=["bar_start", "commodity", "open", "high", "low", "close", "volume"])
        return df.astype({
            "bar_start": "int64",
            "commodity": str,
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        })

    max_tick = int(trade_log_df["tick"].max())
    bar_starts = range(0, max_tick + ticks_per_bar, ticks_per_bar)

    # Vectorized groupby instead of O(N^2) boolean masking
    # Create an independent series to group by, so we don't mutate trade_log_df
    bar_start_series = (trade_log_df["tick"] // ticks_per_bar) * ticks_per_bar
    bar_start_series.name = "bar_start"
    
    aggs = trade_log_df.groupby(bar_start_series).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum")
    ).reset_index()

    # Reindex to ensure we have empty bars for periods with no trades (FR-1.7)
    aggs = aggs.set_index("bar_start").reindex(bar_starts).reset_index()
    
    aggs["volume"] = aggs["volume"].fillna(0).astype("int64")
    # NaN for OHLC is already handled by pandas reindex

    return aggs


# ===========================================================================
# Main simulation loop
# ===========================================================================


def run_simulation(
    config: SimConfig,
    seed: int,
) -> SimulationOutput:
    """
    Run the full simulation and produce all output files.

    Args:
        config: Complete simulation configuration.
        seed: Master random seed. Determines ALL randomness in the simulation
              deterministically (SRS FR-0.4).

    Returns:
        SimulationOutput with all 4 DataFrames + run metadata.

    Raises:
        FileExistsError: If run_id already exists and config.overwrite=False.
        pandera.errors.SchemaErrors: If any output violates its schema.
    """
    # --- Unique run_id ---
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{seed}"
    logger.info("Starting simulation: run_id=%s, seed=%d, n_ticks=%d", run_id, seed, config.n_ticks)

    # --- Create run directory (fail fast on duplicate) ---
    if config.write_to_disk:
        run_dir = io_utils.ensure_run_dir(config.data_dir, run_id, overwrite=config.overwrite)
    else:
        run_dir = config.data_dir / run_id

    # --- Seeded RNG ---
    master_rng = np.random.default_rng(seed)

    # --- Fundamental value process ---
    fundamental_rng, *_ = master_rng.spawn(1)
    fundamental = FundamentalValueProcess(
        initial_value=config.fundamental_value_initial,
        drift=config.fundamental_value_drift,
        volatility=config.fundamental_value_vol,
        rng=fundamental_rng,
    )

    # --- Build agents ---
    agents, controllers, manipulation_event_rows = _build_agents(config, master_rng)

    # Index agents for fast lookup
    agent_map: dict[str, Agent] = {a.trader_id: a for a in agents}
    market_makers = [a for a in agents if isinstance(a, MarketMaker)]
    momentum_traders = [a for a in agents if isinstance(a, MomentumTrader)]
    mean_rev_traders = [a for a in agents if isinstance(a, MeanReversionTrader)]
    spoofing_agents = [c for c in controllers if isinstance(c, SpoofingAgent)]
    layering_agents = [c for c in controllers if isinstance(c, LayeringAgent)]
    pnd_coalitions = [c for c in controllers if isinstance(c, PumpDumpCoalition)]

    # Pre-filter regular agents to avoid isinstance in the hot tick loop
    regular_agents = [a for a in agents if not isinstance(a, (MarketMaker, LayeringAgent))]

    # --- Order book ---
    book = OrderBook(initial_price=config.fundamental_value_initial)

    # --- Accumulators ---
    all_trades: list[dict] = []
    all_order_events: list[dict] = []

    # --- Main tick loop ---
    for tick in range(config.n_ticks):
        # --- Update fundamental value ---
        fv = fundamental.step()

        # --- Broadcast fundamental to relevant agents ---
        for agent in mean_rev_traders:
            agent.set_fundamental_value(fv)

        # --- Update pump-dump coalition phases ---
        for coalition in pnd_coalitions:
            coalition.update_phase(tick)

        # --- Collect cancel instructions from spoofing/layering agents ---
        cancel_orders: list[Order] = []
        for spoofer in spoofing_agents:
            cancel_orders.extend(spoofer.get_cancels(tick))
        for layerer in layering_agents:
            cancel_orders.extend(layerer.get_cancels(tick))

        # --- Process cancels FIRST (before new orders this tick) ---
        for cancel_order in cancel_orders:
            _, events = book.submit(cancel_order, simulation_mode=True)
            all_order_events.extend(e.to_dict() for e in events)

        # --- Market maker requote (atomic cancel-and-replace) ---
        for mm in market_makers:
            cancel_ids, new_orders = mm.decide_pair(book.book_state(depth=5), tick)
            for cid in cancel_ids:
                events = book.cancel_order(cid, mm.trader_id, tick)
                all_order_events.extend(e.to_dict() for e in events)
            for order in new_orders:
                trades, events = book.submit(order, simulation_mode=True)
                _collect(trades, events, all_trades, all_order_events, mm, momentum_traders)

        # --- Get current book state for all other agents ---
        state = book.book_state(depth=5)
        state = BookState(
            best_bid=state.best_bid,
            best_ask=state.best_ask,
            mid_price=state.mid_price,
            bid_depth=state.bid_depth,
            ask_depth=state.ask_depth,
            last_trade_price=state.last_trade_price,
            tick=tick,
        )

        # --- Layering agent (batch submit) ---
        for layerer in layering_agents:
            layer_orders = layerer.decide_batch(state, tick)
            for order in layer_orders:
                trades, events = book.submit(order, simulation_mode=True)
                _collect(trades, events, all_trades, all_order_events, layerer, momentum_traders)

        # --- All other agents (one order per tick) ---
        for agent in regular_agents:
            order = agent.decide(state, tick)
            if order is None:
                continue

            trades, events = book.submit(order, simulation_mode=True)
            _collect(trades, events, all_trades, all_order_events, agent, momentum_traders)

    # --- Assemble DataFrames ---
    trade_df = pd.DataFrame(all_trades) if all_trades else _empty_trade_df()
    order_df = pd.DataFrame(all_order_events) if all_order_events else _empty_order_df()
    if not order_df.empty and "quantity" in order_df.columns:
        order_df["quantity"] = order_df["quantity"].astype("Int64")
    manip_df = _build_manipulation_df(manipulation_event_rows)
    bars_df = _resample_to_bars(trade_df, config.ticks_per_bar)

    # --- Inject Commodity column before writing ---
    trade_df["commodity"] = config.commodity
    order_df["commodity"] = config.commodity
    manip_df["commodity"] = config.commodity
    bars_df["commodity"] = config.commodity

    # --- Validate & write (Pandera validates at write time — FR-0.2) ---
    logger.info(
        "Run complete: %d trades, %d order events, %d manipulation episodes",
        len(trade_df), len(order_df), len(manip_df),
    )

    # Write run metadata
    if config.write_to_disk:
        io_utils.write_parquet(trade_df, run_dir / "trade_log.parquet", "trade_log", overwrite=True)
        io_utils.write_parquet(order_df, run_dir / "order_log.parquet", "order_log", overwrite=True)
        io_utils.write_parquet(manip_df, run_dir / "manipulation_events.parquet", "manipulation_events", overwrite=True)
        io_utils.write_parquet(bars_df, run_dir / "ohlcv_bars.parquet", "ohlcv_bars", overwrite=True)
        _write_run_metadata(run_dir, run_id, seed, config)

    return SimulationOutput(
        run_id=run_id,
        run_dir=run_dir,
        trade_log=trade_df,
        order_log=order_df,
        manipulation_events=manip_df,
        ohlcv_bars=bars_df,
        config=config,
    )


# ===========================================================================
# Internal helpers
# ===========================================================================


from src.core.order_book import Side

def _collect(
    trades: list[Trade],
    events: list,
    all_trades: list[dict],
    all_order_events: list[dict],
    agent: Agent,
    momentum_traders: list[MomentumTrader],
) -> None:
    """Append trades and events to accumulators; update momentum price feeds."""
    for trade in trades:
        all_trades.append(trade.to_dict())
        # Feed last trade price to all momentum traders
        for mt in momentum_traders:
            mt.update_price(trade.price)
        # Update market maker inventory if it was involved
        if isinstance(agent, MarketMaker):
            if trade.buyer_trader_id == agent.trader_id:
                agent.update_inventory(Side.BUY, trade.quantity)
            elif trade.seller_trader_id == agent.trader_id:
                agent.update_inventory(Side.SELL, trade.quantity)

    all_order_events.extend(e.to_dict() for e in events)


def _empty_trade_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=[
        "trade_id", "commodity", "tick", "price", "quantity", "buyer_trader_id", "seller_trader_id"
    ])
    return df.astype({
        "trade_id": str,
        "commodity": str,
        "tick": "int64",
        "price": "float64",
        "quantity": "int64",
        "buyer_trader_id": str,
        "seller_trader_id": str,
    })


def _empty_order_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=[
        "order_id", "commodity", "trader_id", "side", "price", "quantity", "tick", "event_type"
    ])
    return df.astype({
        "order_id": str,
        "commodity": str,
        "trader_id": str,
        "side": str,
        "price": "float64",
        "quantity": "Int64",  # nullable int
        "tick": "int64",
        "event_type": str,
    })


def _build_manipulation_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build the manipulation_events DataFrame.
    Zero-persona runs produce a valid EMPTY DataFrame (not a missing file).
    """
    if not rows:
        df = pd.DataFrame(columns=[
            "event_id", "commodity", "persona", "start_tick", "end_tick", "trader_ids", "parameters"
        ])
        return df.astype({
            "event_id": str,
            "commodity": str,
            "persona": str,
            "start_tick": "int64",
            "end_tick": "int64",
            "trader_ids": object,
            "parameters": object,
        })
    return pd.DataFrame(rows)


def _write_run_metadata(
    run_dir: Path,
    run_id: str,
    seed: int,
    config: SimConfig,
) -> None:
    """Write a metadata YAML summarising the run parameters."""
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "n_ticks": config.n_ticks,
        "ticks_per_bar": config.ticks_per_bar,
        "personas_enabled": {
            "spoofing": config.enable_spoofing,
            "wash_trading": config.enable_wash_trading,
            "pump_and_dump": config.enable_pump_and_dump,
            "layering": config.enable_layering,
        },
    }
    io_utils.write_yaml(metadata, run_dir / "run_metadata.yaml")
