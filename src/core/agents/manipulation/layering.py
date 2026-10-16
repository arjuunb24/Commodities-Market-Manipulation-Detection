"""
src/core/agents/manipulation/layering.py
=========================================
Layering persona — several non-bona-fide orders stacked on consecutive price
levels of one side, cancelled together once the opposite-side order fills.

Grounded in CFTC Order No. 19-09, Tower Research Capital (2019, $67.4M), see
docs/enforcement_case_notes.md: algorithms placed 3–8 orders at consecutive
levels to create a false impression of depth, then cancelled the group after
the genuine order on the other side executed.

Difference from spoofing: spoofing shows ONE large order (or a few splits)
just behind the touch; layering spreads MODERATE orders over several levels,
so no single order is extreme — the deception is the depth profile itself.

Mechanism per cycle:
  1. A small genuine order rests at the best price on one side.
  2. `n_layers` orders are placed on the OTHER side at consecutive levels,
     starting `layer_spacing` behind the best price, each `layer_spacing` apart.
  3. After `cancellation_delay_ticks` (optionally jittered) all layers are cancelled.

Evasion knobs (Tower round 1/2 evasions, CFTC advisory):
  size_taper       0 = equal layer sizes; higher = smaller layers deeper in the
                   book ("decreasing volume deeper in the book to appear natural")
  cancel_jitter    randomised cancel timing (advisory #2)
  busy_period_bias act mainly when the market is busy (advisory #5)
  placement_stagger ticks between successive layers: 0 places the whole ladder in
                   one tick (the Tower pattern); k > 0 builds it one layer every k
                   ticks, so no single tick shows a multi-level batch ("orders
                   entered in rapid succession" rather than simultaneously). The
                   ladder is complete for less of the cycle, so fake depth falls.

Order ids carry audit tags ("l" layer leg, "g" genuine leg) used only for
effectiveness accounting, never by the detector.
"""

from __future__ import annotations

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount, fake_depth_share
from src.core.order_book import BookState, Order, Side


class LayeringAgent(ManipulationController):
    """Single-account layering persona (Tower Research pattern)."""

    PERSONA = "layering"

    def __init__(
        self,
        trader_id: str,
        rng: np.random.Generator,
        *,
        n_layers: int = 4,
        layer_spacing: float = 0.0002,
        layer_size_multiplier: float = 2.0,
        size_taper: float = 0.0,
        cancellation_delay_ticks: int = 5,
        cancel_jitter: float = 0.0,
        frequency: float = 0.3,
        busy_period_bias: float = 0.0,
        layer_side: str = "both",
        placement_stagger: int = 0,
        start_tick: int = 0,
        end_tick: int = 10000,
        background_rate: float = 0.05,
    ) -> None:
        account = ManipulatorAccount(trader_id, rng.spawn(1)[0], background_rate=background_rate)
        super().__init__([account], start_tick, end_tick, busy_period_bias=busy_period_bias)
        self.trader_id = trader_id
        self.rng = rng
        self.n_layers = max(2, int(n_layers))
        self.layer_spacing = float(layer_spacing)
        self.layer_size_multiplier = float(layer_size_multiplier)
        self.size_taper = float(np.clip(size_taper, 0.0, 0.9))
        self.cancellation_delay_ticks = max(1, int(cancellation_delay_ticks))
        self.cancel_jitter = float(np.clip(cancel_jitter, 0.0, 0.9))
        self.frequency = float(frequency)
        self.layer_side = layer_side
        self.placement_stagger = max(0, int(placement_stagger))
        self._cycle_busy_until = -1
        self._depth_shares: list[float] = []
        # staggered cycle in progress: (side, next layer index, next tick, cancel tick, base size, step)
        self._building: tuple | None = None

    @property
    def account(self) -> ManipulatorAccount:
        return self.accounts[0]

    def parameters_snapshot(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "layer_spacing": self.layer_spacing,
            "layer_size_multiplier": self.layer_size_multiplier,
            "size_taper": self.size_taper,
            "cancellation_delay_ticks": self.cancellation_delay_ticks,
            "cancel_jitter": self.cancel_jitter,
            "frequency": self.frequency,
            "busy_period_bias": self.busy_period_bias,
            "layer_side": self.layer_side,
            "placement_stagger": self.placement_stagger,
            "realised_fake_depth_share": round(float(np.mean(self._depth_shares)), 4) if self._depth_shares else 0.0,
        }

    def _pick_side(self) -> Side:
        if self.layer_side == "buy":
            return Side.BUY
        if self.layer_side == "sell":
            return Side.SELL
        return Side.BUY if self.rng.random() < 0.5 else Side.SELL

    def _layer(self, state: BookState, side: Side, i: int, base_size: float, step: float, tick: int, ttl: int) -> Order:
        tick_size = self.account.tick_size
        if side == Side.BUY:
            price = min(state.best_bid - step * (i + 1), state.best_ask - tick_size)
        else:
            price = max(state.best_ask + step * (i + 1), state.best_bid + tick_size)
        taper = 1.0 - self.size_taper * (i / max(1, self.n_layers - 1))
        qty = max(5, int(round(base_size * taper * self.rng.uniform(0.9, 1.1))))
        return self.account.limit(side, price, qty, tick, exact_ttl=max(1, ttl), tag="l")

    def _continue_build(self, state: BookState, tick: int) -> list[Order]:
        side, i, due, cancel_at, base_size, step = self._building
        if tick < due:
            return []
        if state.best_bid is None or state.best_ask is None:
            return []
        # Every layer expires at the cycle's cancel tick, so the group is still pulled together.
        o = self._layer(state, side, i, base_size, step, tick, cancel_at - tick)
        self._building = (side, i + 1, tick + self.placement_stagger, cancel_at, base_size, step) \
            if i + 1 < self.n_layers and tick + self.placement_stagger < cancel_at else None
        return [o]

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        if self._building is not None:
            return self._continue_build(state, tick)
        if tick < self._cycle_busy_until or state.best_bid is None or state.best_ask is None:
            return []
        if self.rng.random() > self.frequency:
            return []
        acct = self.account
        tick_size = acct.tick_size
        side = self._pick_side()

        levels = state.bid_depth + state.ask_depth
        avg_level = sum(q for _, q in levels) / max(1, len(levels))
        base_size = max(10.0, avg_level * self.layer_size_multiplier)
        delay = max(1, int(round(self.cancellation_delay_ticks
                                 * self.rng.uniform(1 - self.cancel_jitter, 1 + self.cancel_jitter))))

        genuine_side = Side.SELL if side == Side.BUY else Side.BUY
        genuine_price = state.best_ask if genuine_side == Side.SELL else state.best_bid
        orders = [acct.limit(genuine_side, genuine_price, int(self.rng.integers(5, 30)), tick,
                             exact_ttl=delay + 2, tag="g")]
        # At least one tick between layers so every layer sits on its own price level.
        step = max(tick_size, state.mid_price * self.layer_spacing if state.mid_price else tick_size)
        if self.placement_stagger > 0:
            # Build the ladder over time: the full ladder only exists from the last layer on,
            # so a staggered ladder also needs a cancel delay long enough to finish it.
            delay = max(delay, self.placement_stagger * (self.n_layers - 1) + 1)
            orders[0] = acct.limit(genuine_side, genuine_price, orders[0].quantity, tick, exact_ttl=delay + 2, tag="g")
            layers = [self._layer(state, side, i, base_size, step, tick, delay) for i in range(self.n_layers)]
            # Fake depth counts only while it rests: layer i is up for (delay - i x stagger) of the cycle.
            alive = np.mean([max(0, delay - i * self.placement_stagger) / delay for i in range(self.n_layers)])
            self._depth_shares.append(alive * fake_depth_share(state, side, [(o.price, o.quantity) for o in layers]))
            orders.append(layers[0])
            self._building = (side, 1, tick + self.placement_stagger, tick + delay, base_size, step)
        else:
            orders += [self._layer(state, side, i, base_size, step, tick, delay) for i in range(self.n_layers)]
            self._depth_shares.append(fake_depth_share(state, side, [(o.price, o.quantity) for o in orders[1:]]))
        self._cycle_busy_until = tick + delay
        return orders
