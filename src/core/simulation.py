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
             outputs." (numpy default_rng + spawn; deterministic order/trade/event ids)
NFR-R2: Atomic write-then-rename for all output files (implemented in io.py).

Market-realism notes (see the persona modules for manipulation realism):
  - Time: 1 tick = 1 second of a ~6.5h session (`ticks_per_day`). Calibrated
    parameters are DAILY; they are converted to per-tick scale here
    (vol / sqrt(ticks_per_day), drift / ticks_per_day). Previously a daily vol
    was applied every tick, i.e. the fundamental moved ~150x too fast.
  - Prices live on each contract's real tick grid (COMMODITY_TICK_SIZES).
  - Arrival order inside a tick is shuffled each tick (seeded), so no trader
    systematically gets first look at the book.
  - Every participant's unfilled orders expire/cancel; the book no longer
    accumulates stale orders forever.
  - Agents see a fresh book snapshot after any other agent changes it.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.agents.base import Agent
from src.core.agents.manipulation.common import ManipulationController
from src.core.agents.manipulation.layering import LayeringAgent
from src.core.agents.manipulation.pump_and_dump import PumpDumpCoalition
from src.core.agents.manipulation.spoofing import SpoofingAgent
from src.core.agents.manipulation.wash_trading import WashTradingPair
from src.core.agents.execution_algo import ExecutionAlgo
from src.core.agents.hft_liquidity_provider import HFTLiquidityProvider
from src.core.agents.market_maker import MarketMaker
from src.core.agents.mean_reversion_trader import MeanReversionTrader
from src.core.agents.momentum_trader import MomentumTrader
from src.core.agents.noise_trader import NoiseTrader
from src.core.calibration import (
    COMMODITY_INITIAL_PRICES,
    COMMODITY_PERSONA_PROFILES,
    COMMODITY_TICK_SIZES,
)
from src.core.order_book import BookState, Order, OrderBook, Side, Trade
from src.utils import io as io_utils
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ACTIVE_PERSONAS = ("spoofing", "wash_trading", "pump_and_dump", "layering")


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class SimConfig:
    """
    Full configuration for one simulation run.

    fundamental_value_drift / fundamental_value_vol are DAILY figures (as produced
    by calibration); the simulation rescales them to one tick.
    """
    # Simulation parameters
    commodity: str = "gold"
    n_ticks: int = 10_000
    ticks_per_bar: int = 100
    ticks_per_day: int = 23_400          # 6.5h session at 1 tick / second
    fundamental_value_initial: float | None = None
    fundamental_value_drift: float = 0.0001
    fundamental_value_vol: float = 0.0094
    # Multiply the fundamental volatility by the commodity's simulator correction
    # factor (configs/sim_calibration.yaml) so that REALISED 5-minute volatility
    # matches the sourced daily volatility. The sourced figure itself is untouched.
    apply_vol_scale: bool = True
    jumps_per_day: float = 4.0           # macro / inventory news shocks
    jump_size_daily_vol: float = 0.25    # jump std as a fraction of daily vol
    tick_offset: int = 0

    # Optional explicit window overrides (used by tests and targeted scenarios)
    spoof_start_tick: int | None = None
    spoof_end_tick: int | None = None
    wash_start_tick: int | None = None
    wash_end_tick: int | None = None
    pnd_start_tick: int | None = None
    pnd_end_tick: int | None = None
    layer_start_tick: int | None = None
    layer_end_tick: int | None = None

    # Agent population (counts)
    n_noise_traders: int = 20
    n_momentum_traders: int = 8
    n_mean_reversion_traders: int = 6
    n_market_makers: int = 1
    # Legitimate look-alikes of manipulators (hard negatives): high-cancel quoting
    # firms and institutional algos that rest and pull large orders.
    n_hft_liquidity_providers: int = 3
    n_execution_algos: int = 3

    # Agent parameters (relative to price unless stated)
    noise_lambda: float = 0.6
    noise_price_offset_ticks: float = 2.0   # legacy; offset is now noise_price_offset_pct
    noise_price_offset_pct: float = 0.0005
    momentum_lambda: float = 0.05
    momentum_lookback: int = 60             # ticks (seconds), sampled once per tick
    momentum_threshold: float = 0.001
    reversion_lambda: float = 0.25
    reversion_threshold: float = 0.001
    market_maker_lambda: float = 0.95
    market_maker_spread: float = 2.0        # legacy; half-spread is market_maker_half_spread_pct
    market_maker_half_spread_pct: float = 0.0003
    manipulator_background_rate: float = 0.05

    # Persona flags
    enable_spoofing: bool = True
    enable_wash_trading: bool = True
    enable_pump_and_dump: bool = True
    enable_layering: bool = False

    # Spoofing parameters
    spoof_size_multiplier: float = 8.0
    spoof_cancel_delay: int = 5
    spoof_frequency: float = 0.4
    spoof_aggressiveness: float = 0.0005
    spoof_size_jitter: float = 0.0
    spoof_cancel_jitter: float = 0.0
    spoof_n_split: int = 1
    spoof_n_accounts: int = 1
    spoof_reprice: int = 0
    spoof_flip: float = 0.0
    spoof_busy_bias: float = 0.0
    spoof_cover_rate: float = 0.0
    spoof_cover_size: float = 0.0

    # Wash trading parameters
    wash_frequency: float = 0.5
    wash_price_deviation: float = 0.001
    wash_n_pairs: int = 1                   # number of independent colluding groups
    wash_ring_size: int = 2
    wash_size_jitter: float = 0.0
    wash_camouflage_rate: float = 0.0
    wash_busy_bias: float = 0.0

    # Pump-and-dump parameters
    pnd_burst_duration: int = 200
    pnd_n_accounts: int = 3
    pnd_dump_delay: int = 100
    pnd_accumulation_size: int = 150
    pnd_stagger: float = 0.0
    pnd_ignition_aggression: float = 0.002
    pnd_decoy_rate: float = 0.0
    pnd_busy_bias: float = 0.0

    # Layering parameters (Tower Research pattern)
    layer_n_layers: int = 4
    layer_spacing: float = 0.0002
    layer_size_multiplier: float = 2.0
    layer_size_taper: float = 0.0
    layer_cancel_delay: int = 5
    layer_cancel_jitter: float = 0.0
    layer_frequency: float = 0.3
    layer_busy_bias: float = 0.0
    layer_placement_stagger: int = 0
    layer_cover_rate: float = 0.0
    layer_cover_size: float = 0.0

    # Output
    data_dir: Path = Path("data")
    overwrite: bool = False
    write_to_disk: bool = True


# ===========================================================================
# Simulation output
# ===========================================================================


@dataclass
class SimulationOutput:
    """All outputs from a completed simulation run (schema-shaped DataFrames)."""
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
    Jump-diffusion for the 'true' commodity value (an exogenous signal to
    MeanReversionTrader agents — NOT the market price).

        V_{t+1} = V_t * exp(mu + sigma * Z_t + J_t)

    mu and sigma are per-tick; J_t is a compound-Poisson news shock.
    """

    def __init__(
        self,
        initial_value: float,
        drift: float,
        volatility: float,
        rng: np.random.Generator,
        *,
        jump_prob: float = 0.0,
        jump_std: float = 0.0,
    ) -> None:
        self.value = initial_value
        self.drift = drift
        self.volatility = volatility
        self.rng = rng
        self.jump_prob = jump_prob
        self.jump_std = jump_std

    def step(self) -> float:
        shock = self.rng.standard_normal()
        jump = 0.0
        if self.rng.random() < self.jump_prob:
            jump = self.rng.standard_normal() * self.jump_std
        self.value *= np.exp(self.drift + self.volatility * shock + jump)
        return self.value


def _initial_price(config: SimConfig) -> float:
    init_price = config.fundamental_value_initial
    if init_price is None or init_price == 100.0:
        init_price = COMMODITY_INITIAL_PRICES.get(config.commodity, 100.0)
    return float(init_price)


# ===========================================================================
# Agent factory
# ===========================================================================


def _make_persona(
    persona: str,
    ep_idx: int,
    start: int,
    end: int,
    config: SimConfig,
    sub_rngs,
    jitter_rng: np.random.Generator | None,
) -> list[Any]:
    """Instantiate one manipulation episode. Returns a list of controllers."""
    controllers = _build_persona(persona, ep_idx, start, end, config, sub_rngs, jitter_rng)
    cover = {"spoofing": (config.spoof_cover_rate, config.spoof_cover_size),
             "layering": (config.layer_cover_rate, config.layer_cover_size)}.get(persona, (0.0, 0.0))
    for c in controllers:
        c.cover_rate, c.cover_size = float(cover[0]), float(cover[1])
    return controllers


def _build_persona(persona, ep_idx, start, end, config, sub_rngs, jitter_rng) -> list[Any]:

    def jit(x: float) -> float:
        return float(x * jitter_rng.uniform(0.95, 1.05)) if jitter_rng is not None else float(x)

    bg = config.manipulator_background_rate
    if persona == "spoofing":
        return [SpoofingAgent(
            f"spoofer_{ep_idx}", next(sub_rngs),
            order_size_multiplier=jit(config.spoof_size_multiplier),
            cancellation_delay_ticks=int(config.spoof_cancel_delay),
            frequency=min(1.0, jit(config.spoof_frequency)),
            price_aggressiveness=jit(config.spoof_aggressiveness),
            size_jitter=config.spoof_size_jitter,
            cancel_jitter=config.spoof_cancel_jitter,
            n_split_orders=int(config.spoof_n_split),
            n_accounts=int(config.spoof_n_accounts),
            reprice_to_touch=int(config.spoof_reprice),
            flip_after_cancel=float(config.spoof_flip),
            busy_period_bias=float(config.spoof_busy_bias),
            start_tick=start, end_tick=end, background_rate=bg,
        )]
    if persona == "wash_trading":
        n_groups = max(1, int(config.wash_n_pairs))
        busy = float(config.wash_busy_bias)
        # The persona's total cross rate is split across groups: more groups dilute
        # per-account activity instead of multiplying wash volume.
        freq = min(1.0, jit(config.wash_frequency)) / n_groups
        dev = jit(config.wash_price_deviation)
        return [
            WashTradingPair.create(
                next(sub_rngs), ep_idx * 10 + g_idx,
                trade_frequency=freq, price_deviation_from_mid=dev,
                start_tick=start, end_tick=end, background_rate=bg,
                ring_size=int(config.wash_ring_size),
                size_jitter=config.wash_size_jitter,
                camouflage_rate=config.wash_camouflage_rate,
                n_groups=n_groups,
                busy_period_bias=busy,
            )
            for g_idx in range(n_groups)
        ]
    if persona == "pump_and_dump":
        return [PumpDumpCoalition.create(
            next(sub_rngs),
            n_coordinated_accounts=int(config.pnd_n_accounts),
            burst_duration_ticks=int(config.pnd_burst_duration),
            dump_delay_ticks=int(config.pnd_dump_delay),
            accumulation_size=int(config.pnd_accumulation_size),
            start_tick=start, end_tick=end, episode_index=ep_idx, background_rate=bg,
            stagger=config.pnd_stagger,
            ignition_aggression=config.pnd_ignition_aggression,
            decoy_rate=config.pnd_decoy_rate,
            busy_period_bias=config.pnd_busy_bias,
        )]
    if persona == "layering":
        return [LayeringAgent(
            f"layerer_{ep_idx}", next(sub_rngs),
            n_layers=int(config.layer_n_layers),
            layer_spacing=float(config.layer_spacing),
            layer_size_multiplier=float(config.layer_size_multiplier),
            size_taper=float(config.layer_size_taper),
            cancellation_delay_ticks=int(config.layer_cancel_delay),
            cancel_jitter=float(config.layer_cancel_jitter),
            frequency=float(config.layer_frequency),
            busy_period_bias=float(config.layer_busy_bias),
            placement_stagger=int(config.layer_placement_stagger),
            start_tick=start, end_tick=end, background_rate=bg,
        )]
    raise ValueError(f"Unknown persona {persona!r}")


def _persona_duration(persona: str, config: SimConfig, rng: np.random.Generator) -> int:
    if persona == "spoofing":
        return int(rng.integers(140, 420))
    if persona == "layering":
        return int(rng.integers(120, 350))
    if persona == "wash_trading":
        return int(rng.integers(220, 600))
    if persona == "pump_and_dump":
        stagger = int(config.pnd_stagger * config.pnd_burst_duration)
        return int(config.pnd_burst_duration) + int(config.pnd_dump_delay) + 160 + stagger
    return int(rng.integers(150, 400))


def _build_agents(
    config: SimConfig,
    master_rng: np.random.Generator,
) -> tuple[list[Agent], list[Any]]:
    """
    Instantiate all legitimate agents and manipulation controllers.

    Persona mix, window timing and intensity are tailored to the commodity
    (COMMODITY_PERSONA_PROFILES) unless explicit window overrides are given.
    """
    init_price = _initial_price(config)
    tick_size = COMMODITY_TICK_SIZES.get(config.commodity, 0.01)
    mm_half_spread = max(2 * tick_size, init_price * config.market_maker_half_spread_pct)
    noise_offset = max(2 * tick_size, init_price * config.noise_price_offset_pct)

    sub_rngs = iter(master_rng.spawn(200))
    agents: list[Agent] = []
    controllers: list[Any] = []

    for i in range(config.n_noise_traders):
        agents.append(NoiseTrader(
            f"noise_{i}", next(sub_rngs),
            arrival_rate_lambda=config.noise_lambda,
            price_offset_ticks=noise_offset,
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
            spread_ticks=mm_half_spread,
            arrival_rate_lambda=config.market_maker_lambda,
        ))

    for i in range(config.n_hft_liquidity_providers):
        agents.append(HFTLiquidityProvider(f"hft_{i}", next(sub_rngs)))
    for i in range(config.n_execution_algos):
        agents.append(ExecutionAlgo(f"algo_{i}", next(sub_rngs)))

    flag_map = {
        "spoofing": config.enable_spoofing,
        "wash_trading": config.enable_wash_trading,
        "pump_and_dump": config.enable_pump_and_dump,
        "layering": config.enable_layering,
    }

    has_explicit_override = any(
        v is not None for v in [
            config.spoof_start_tick, config.wash_start_tick,
            config.pnd_start_tick, config.layer_start_tick,
        ]
    )

    if has_explicit_override:
        defaults = {
            "spoofing": (config.spoof_start_tick, config.spoof_end_tick, 100, 200),
            "wash_trading": (config.wash_start_tick, config.wash_end_tick, 50, 200),
            "pump_and_dump": (config.pnd_start_tick, config.pnd_end_tick, 150, 300),
            "layering": (config.layer_start_tick, config.layer_end_tick, 200, 200),
        }
        for ep_idx, persona in enumerate(ACTIVE_PERSONAS):
            if not flag_map[persona]:
                continue
            s, e, s_def, length = defaults[persona]
            start = s if s is not None else s_def
            end = e if e is not None else start + length
            controllers.extend(_make_persona(persona, ep_idx, start, end, config, sub_rngs, None))
        return agents, controllers

    if not any(flag_map.values()):
        return agents, controllers

    comm_affinities = COMMODITY_PERSONA_PROFILES.get(
        config.commodity,
        {"spoofing": 0.35, "wash_trading": 0.35, "pump_and_dump": 0.15, "layering": 0.15},
    )
    valid_personas = [p for p, on in flag_map.items() if on and p in comm_affinities] or \
                     [p for p, on in flag_map.items() if on]
    weights = np.array([comm_affinities.get(p, 0.25) for p in valid_personas], dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(valid_personas)) / len(valid_personas)

    min_episodes = max(1, len(valid_personas)) if config.n_ticks >= 2000 else 1
    max_episodes = max(min_episodes, min(4, config.n_ticks // 800)) if config.n_ticks >= 800 else 1
    n_episodes = int(master_rng.integers(min_episodes, max_episodes + 1))

    if n_episodes >= len(valid_personas):
        chosen = list(valid_personas)
        if n_episodes > len(valid_personas):
            chosen.extend(master_rng.choice(valid_personas, size=n_episodes - len(valid_personas), p=weights).tolist())
        master_rng.shuffle(chosen)
    else:
        chosen = master_rng.choice(valid_personas, size=n_episodes, p=weights, replace=True).tolist()

    # Stagger episodes across the timeline (one per segment) so ground-truth
    # windows do not overlap unless the run is very short.
    for ep_idx, persona in enumerate(chosen):
        duration = _persona_duration(persona, config, master_rng)
        max_start = max(10, config.n_ticks - duration - 30)
        segment_len = max_start // max(1, n_episodes)
        seg_start = ep_idx * segment_len + 15
        seg_end = min(max_start, (ep_idx + 1) * segment_len)
        if seg_end > seg_start:
            start = int(master_rng.integers(seg_start, seg_end + 1))
        else:
            start = int(master_rng.integers(15, max_start + 1))
        controllers.extend(_make_persona(persona, ep_idx, start, start + duration, config, sub_rngs, master_rng))

    return agents, controllers


# ===========================================================================
# OHLCV bar aggregation
# ===========================================================================


def _resample_to_bars(trade_log_df: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    """
    Resample trade_log to OHLCV bars.

    SRS FR-1.7: Zero-trade bars have NaN OHLC and volume=0 (never forward-filled).
    """
    if trade_log_df.empty:
        df = pd.DataFrame(columns=["bar_start", "commodity", "open", "high", "low", "close", "volume"])
        return df.astype({
            "bar_start": "int64", "commodity": str, "open": "float64", "high": "float64",
            "low": "float64", "close": "float64", "volume": "int64",
        })

    max_tick = int(trade_log_df["tick"].max())
    bar_starts = range(0, max_tick + ticks_per_bar, ticks_per_bar)
    bar_start_series = (trade_log_df["tick"] // ticks_per_bar) * ticks_per_bar
    bar_start_series.name = "bar_start"
    aggs = trade_log_df.groupby(bar_start_series).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
    ).reset_index()
    aggs = aggs.set_index("bar_start").reindex(bar_starts).reset_index()
    aggs["volume"] = aggs["volume"].fillna(0).astype("int64")
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
        seed: Master random seed — determines ALL randomness (SRS FR-0.4).

    Raises:
        FileExistsError: If run_id already exists and config.overwrite=False.
        pandera.errors.SchemaErrors: If any output violates its schema.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{seed}"
    logger.info("Starting simulation: run_id=%s, seed=%d, n_ticks=%d", run_id, seed, config.n_ticks)

    if config.write_to_disk:
        run_dir = io_utils.ensure_run_dir(config.data_dir, run_id, overwrite=config.overwrite)
    else:
        run_dir = config.data_dir / run_id

    master_rng = np.random.default_rng(seed)
    init_price = _initial_price(config)
    tick_size = COMMODITY_TICK_SIZES.get(config.commodity, 0.01)

    # Daily calibration → per-tick scale.
    tpd = max(1, int(config.ticks_per_day))
    fundamental_rng, order_rng = master_rng.spawn(2)
    fundamental = FundamentalValueProcess(
        initial_value=init_price,
        drift=config.fundamental_value_drift / tpd,
        volatility=config.fundamental_value_vol * _vol_scale(config) / np.sqrt(tpd),
        rng=fundamental_rng,
        jump_prob=config.jumps_per_day / tpd,
        jump_std=config.jump_size_daily_vol * config.fundamental_value_vol * _vol_scale(config),
    )

    agents, controllers = _build_agents(config, master_rng)

    # Run-scoped id prefix keeps ids unique when commodities/rounds are merged.
    id_prefix = f"{config.commodity[:3]}{seed}-"
    all_accounts: list[Agent] = list(agents)
    for c in controllers:
        all_accounts.extend(c.agents if hasattr(c, "agents") else [c])
    for a in all_accounts:
        a.id_prefix = id_prefix
        a.tick_size = tick_size

    agent_map: dict[str, Agent] = {a.trader_id: a for a in all_accounts}
    market_makers = [a for a in agents if isinstance(a, MarketMaker)]
    momentum_traders = [a for a in agents if isinstance(a, MomentumTrader)]
    mean_rev_traders = [a for a in agents if isinstance(a, MeanReversionTrader)]
    persona_controllers = [c for c in controllers if isinstance(c, ManipulationController)]
    expiring = [a for a in agents if not isinstance(a, MarketMaker)]

    # Actors that decide each tick (market makers requote separately, first).
    actors: list[Any] = [a for a in agents if not isinstance(a, MarketMaker)] + persona_controllers

    book = OrderBook(initial_price=init_price, id_prefix=id_prefix)
    for a in agents:
        if hasattr(a, "bind_book"):
            a.bind_book(book.is_resting)
    # Market-activity signal (advisory #5 "busy period" tactic): volume over the
    # last minute and a slow moving baseline of it.
    from collections import deque
    tick_volumes: deque[int] = deque(maxlen=60)
    activity = {"recent": 0, "baseline": 0.0}

    def _book_snapshot(book: OrderBook, tick: int) -> BookState:
        state = _book_snapshot(book, tick)
        state.recent_volume = activity["recent"]
        state.baseline_volume = activity["baseline"]
        return state

    all_trades: list[dict] = []
    all_order_events: list[dict] = []

    def submit(order: Order) -> bool:
        trades, events = book.submit(order, simulation_mode=True)
        _collect(trades, events, all_trades, all_order_events, agent_map, momentum_traders)
        return True

    n_trades_before = 0
    for tick in range(config.n_ticks):
        fv = fundamental.step()
        for agent in mean_rev_traders:
            agent.set_fundamental_value(fv)

        # --- Expiries / scheduled cancels first ---
        cancels: list[Order] = []
        for agent in expiring:
            cancels.extend(agent.expire_orders(tick, book.is_resting))
        for ctrl in persona_controllers:
            cancels.extend(ctrl.expire_orders(tick, book.is_resting))
        for c in cancels:
            _, events = book.submit(c, simulation_mode=True)
            all_order_events.extend(e.to_dict() for e in events)

        # --- Market maker requote (cancel-and-replace) ---
        for mm in market_makers:
            cancel_ids, new_orders = mm.decide_pair(_snapshot(book, tick), tick)
            for cid in cancel_ids:
                all_order_events.extend(e.to_dict() for e in book.cancel_order(cid, mm.trader_id, tick))
            for order in new_orders:
                submit(order)

        # --- Everyone else, in a seeded random arrival order ---
        state = _snapshot(book, tick)
        for idx in order_rng.permutation(len(actors)):
            actor = actors[idx]
            if hasattr(actor, "decide_batch"):
                orders = actor.decide_batch(state, tick)
            else:
                o = actor.decide(state, tick)
                orders = [o] if o is not None else []
            if orders:
                for order in orders:
                    submit(order)
                state = _snapshot(book, tick)

        # Update the activity signal with this tick's traded volume.
        traded = sum(t["quantity"] for t in all_trades[n_trades_before:])
        n_trades_before = len(all_trades)
        tick_volumes.append(traded)
        activity["recent"] = sum(tick_volumes)
        if len(tick_volumes) == tick_volumes.maxlen:
            b = activity["baseline"]
            activity["baseline"] = float(activity["recent"]) if b == 0 else 0.995 * b + 0.005 * activity["recent"]

        # Trend followers sample the last traded price once per second.
        if state.last_trade_price is not None:
            for mt in momentum_traders:
                mt.update_price(state.last_trade_price)

    # --- Assemble DataFrames ---
    trade_df = pd.DataFrame(all_trades) if all_trades else _empty_trade_df()
    order_df = pd.DataFrame(all_order_events) if all_order_events else _empty_order_df()
    if not order_df.empty and "quantity" in order_df.columns:
        order_df["quantity"] = order_df["quantity"].astype("Int64")
    manip_rows = _merge_episode_rows([c.log_activity() for c in controllers])
    for i, row in enumerate(manip_rows):
        row["event_id"] = f"{id_prefix}E{i}"
    manip_df = _build_manipulation_df(manip_rows)
    bars_df = _resample_to_bars(trade_df, config.ticks_per_bar)

    for df in (trade_df, order_df, manip_df, bars_df):
        df["commodity"] = config.commodity

    if config.tick_offset > 0:
        if not trade_df.empty:
            trade_df["tick"] = trade_df["tick"] + config.tick_offset
        if not order_df.empty:
            order_df["tick"] = order_df["tick"] + config.tick_offset
        if not manip_df.empty:
            manip_df["start_tick"] = manip_df["start_tick"] + config.tick_offset
            manip_df["end_tick"] = manip_df["end_tick"] + config.tick_offset
        if not bars_df.empty:
            bars_df["bar_start"] = bars_df["bar_start"] + config.tick_offset

    logger.info(
        "Run complete: %d trades, %d order events, %d manipulation episodes",
        len(trade_df), len(order_df), len(manip_df),
    )

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


def _merge_episode_rows(rows: list[dict]) -> list[dict]:
    """One ground-truth event per episode: groups of the same wash episode share one row."""
    merged: dict[tuple, dict] = {}
    for r in rows:
        key = (r["persona"], r["start_tick"], r["end_tick"])
        if key in merged:
            merged[key]["trader_ids"] = merged[key]["trader_ids"] + list(r["trader_ids"])
        else:
            merged[key] = {**r, "trader_ids": list(r["trader_ids"])}
    return list(merged.values())


def _snapshot(book: OrderBook, tick: int) -> BookState:
    state = book.book_state(depth=5)
    state.tick = tick
    return state


def _collect(
    trades: list[Trade],
    events: list,
    all_trades: list[dict],
    all_order_events: list[dict],
    agent_map: dict[str, Agent],
    momentum_traders: list[MomentumTrader],
) -> None:
    """Append trades/events and update participant inventories."""
    for trade in trades:
        all_trades.append(trade.to_dict())
        buyer = agent_map.get(trade.buyer_trader_id)
        if buyer is not None and hasattr(buyer, "update_inventory"):
            buyer.update_inventory(Side.BUY, trade.quantity)
        seller = agent_map.get(trade.seller_trader_id)
        if seller is not None and hasattr(seller, "update_inventory"):
            seller.update_inventory(Side.SELL, trade.quantity)
    all_order_events.extend(e.to_dict() for e in events)


_SIM_CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "configs" / "sim_calibration.yaml"
_VOL_SCALE_CACHE: dict | None = None


def _vol_scale(config: "SimConfig") -> float:
    """Simulator correction factor for the commodity's fundamental volatility (1.0 if none)."""
    global _VOL_SCALE_CACHE
    if not config.apply_vol_scale:
        return 1.0
    if _VOL_SCALE_CACHE is None:
        try:
            import yaml
            _VOL_SCALE_CACHE = (yaml.safe_load(_SIM_CALIBRATION_PATH.read_text()) or {}).get("vol_scale", {}) \
                if _SIM_CALIBRATION_PATH.exists() else {}
        except Exception:   # a broken calibration file must never stop a simulation
            _VOL_SCALE_CACHE = {}
    return float(_VOL_SCALE_CACHE.get(config.commodity, 1.0))


def multi_commodity_seed(seed: int, commodity: str) -> int:
    """Stable per-commodity seed (Python's hash() is salted per process)."""
    return (seed * 10007 + zlib.crc32(commodity.encode("utf-8"))) % (2**31 - 1)


def _empty_trade_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=[
        "trade_id", "commodity", "tick", "price", "quantity",
        "buyer_trader_id", "seller_trader_id", "aggressor_side",
    ])
    return df.astype({
        "trade_id": str, "commodity": str, "tick": "int64", "price": "float64",
        "quantity": "int64", "buyer_trader_id": str, "seller_trader_id": str,
        "aggressor_side": str,
    })


def _empty_order_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=[
        "order_id", "commodity", "trader_id", "side", "price", "quantity", "tick", "event_type"
    ])
    return df.astype({
        "order_id": str, "commodity": str, "trader_id": str, "side": str,
        "price": "float64", "quantity": "Int64", "tick": "int64", "event_type": str,
    })


def _build_manipulation_df(rows: list[dict]) -> pd.DataFrame:
    """Zero-persona runs produce a valid EMPTY DataFrame (not a missing file)."""
    if not rows:
        df = pd.DataFrame(columns=[
            "event_id", "commodity", "persona", "start_tick", "end_tick", "trader_ids", "parameters"
        ])
        return df.astype({
            "event_id": str, "commodity": str, "persona": str, "start_tick": "int64",
            "end_tick": "int64", "trader_ids": object, "parameters": object,
        })
    return pd.DataFrame(rows)


def _write_run_metadata(run_dir: Path, run_id: str, seed: int, config: SimConfig) -> None:
    metadata = {
        "run_id": run_id,
        "seed": seed,
        "commodity": config.commodity,
        "n_ticks": config.n_ticks,
        "ticks_per_bar": config.ticks_per_bar,
        "ticks_per_day": config.ticks_per_day,
        "personas_enabled": {
            "spoofing": config.enable_spoofing,
            "wash_trading": config.enable_wash_trading,
            "pump_and_dump": config.enable_pump_and_dump,
            "layering": config.enable_layering,
        },
    }
    io_utils.write_yaml(metadata, run_dir / "run_metadata.yaml")
