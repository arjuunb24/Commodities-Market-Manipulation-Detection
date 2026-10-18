"""
src/core/agents/manipulation/replay.py
======================================
Scripted replay of a documented real manipulation case (configs/case_replays.yaml).

The four personas are this project's own designs, and the detector was trained
against them and against the LLM strategist's mutations of them. A replay is the
held-out test: a script distilled from a public enforcement record, executed by
code that shares nothing with the persona agents, so a catch means the detector
recognises the documented tactic itself, not the simulator's own idiom.

One cycle:
  1. pick the fake side (fixed, or kept from the last cycle with
     `direction_persistence`, else reversed);
  2. optionally place a genuine order at the touch on the other side;
  3. place `n_orders` non-bona-fide orders on the fake side, one price level
     apart, starting `levels_from_touch` ticks behind the same-side best price,
     sized per order (x level depth) or to a target share of visible depth;
  4. while they rest: optionally reprice them to keep their distance (Sarao), and
     optionally add genuine orders at `genuine.rate` per tick;
  5. pull everything when the lifetime ends, or as soon as the genuine order
     fills (`cancel_on_genuine_fill`); optionally "flip" with an aggressive order
     the genuine way; wait `gap_ticks`, repeat.

Program mode (`mode: program`) runs timed phases instead of spoof cycles, for
cases that are not fake depth:
  trade   one account trades `side` at `rate` per tick, `marketable` (crosses the
          spread, like Optiver's closing-window buying) or `passive` (rests at its
          own touch), sized `size_x_level_depth` x average level depth;
  cross   two accounts of one owner trade with each other: the first rests an
          order inside the spread, the second takes it `cross_delay_ticks` later
          (Shinhan: offsetting orders within seconds). Roles alternate so the owner
          stays flat; with a one-tick spread it waits (the cross would hit others).

Ground truth carries the script's label and its case id.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.core.agents.manipulation.common import ManipulationController, ManipulatorAccount, fake_depth_share
from src.core.order_book import BookState, Order, Side


def _cancel(oid: str, acct: ManipulatorAccount, tick: int) -> Order:
    return Order(order_id=oid, trader_id=acct.trader_id, side=Side.BUY, price=1.0, quantity=1,
                 tick=tick, order_type="cancel")


class CaseReplayAgent(ManipulationController):
    """Executes one case script inside [start_tick, end_tick]."""

    def __init__(self, case_id: str, script: dict[str, Any], rng: np.random.Generator, *,
                 start_tick: int, end_tick: int, background_rate: float = 0.05, episode_index: int = 0) -> None:
        n_acc = max(1, int(script.get("n_accounts", 1)))
        rngs = rng.spawn(n_acc)
        stem = f"replay_{case_id}_{episode_index}"
        names = [stem] + [f"{stem}_{k}" for k in range(1, n_acc)]
        accounts = [ManipulatorAccount(n, r, background_rate=background_rate) for n, r in zip(names, rngs)]
        super().__init__(accounts, start_tick, end_tick)
        self.case_id = case_id
        self.script = script
        self.rng = rng
        self.PERSONA = script.get("label", "spoofing")          # ground-truth label for this episode
        f, g = script.get("fake", {}), script.get("genuine", {})
        self.fake_side = str(f.get("side", "both"))
        self.persistence = float(f.get("direction_persistence", 1.0))
        self.n_orders = max(1, int(f.get("n_orders", 1)))
        self.levels_from_touch = max(0, int(f.get("levels_from_touch", 0)))
        self.size_x = f.get("size_x_level_depth")
        self.target_share = f.get("target_share")
        lt = f.get("lifetime_ticks", [5, 5])
        self.lifetime = (max(1, int(lt[0])), max(1, int(lt[-1])))
        self.reprice = bool(f.get("reprice", False))
        self.cancel_on_fill = bool(f.get("cancel_on_genuine_fill", False))
        self.genuine_place = str(g.get("place", "per_cycle"))
        self.genuine_size_x = float(g.get("size_x_level_depth", 0.3))
        self.genuine_rate = float(g.get("rate", 0.0))
        self.flip = float(script.get("flip_after_cancel", 0.0))
        gp = script.get("gap_ticks", [1, 5])
        self.gap = (max(0, int(gp[0])), max(0, int(gp[-1])))

        self._is_resting: Callable[[str], bool] = lambda _oid: True
        self._legs: list[list] = []          # [order_id, account, side, qty, price, level]
        self._genuine: tuple[str, ManipulatorAccount] | None = None
        self._side: Side | None = None
        self._cancel_at = -1
        self._next_cycle = start_tick
        self._cycles = 0
        self._depth_shares: list[float] = []
        self.mode = str(script.get("mode", "spoof"))
        self.phases = list(script.get("phases", []))
        self._phase_ends = list(np.cumsum([int(ph.get("ticks", 0)) for ph in self.phases]))
        self._pending_cross: tuple | None = None   # (due_tick, taker, side, price, qty, maker_order_id)
        self._crosses = 0
        self._program_trades = 0

    # ------------------------------------------------------------------
    def expire_orders(self, tick: int, is_resting: Callable[[str], bool]) -> list[Order]:
        self._is_resting = is_resting        # lets manipulate() see whether the genuine order filled
        return super().expire_orders(tick, is_resting)

    def parameters_snapshot(self) -> dict:
        if self.mode == "program":
            return {"case": self.case_id, "title": self.script.get("title", ""), "mode": "program",
                    "phases": "; ".join(f"{ph.get('name', ph.get('kind'))}:{ph.get('kind')}:{ph.get('ticks')}" for ph in self.phases),
                    "crosses": self._crosses, "program_orders": self._program_trades}
        return {
            "case": self.case_id,
            "title": self.script.get("title", ""),
            "fake_side": self.fake_side, "n_orders": self.n_orders, "levels_from_touch": self.levels_from_touch,
            "size_x_level_depth": float(self.size_x or 0.0), "target_share": float(self.target_share or 0.0),
            "lifetime_ticks": f"{self.lifetime[0]}-{self.lifetime[1]}", "reprice": self.reprice,
            "cancel_on_genuine_fill": self.cancel_on_fill, "flip_after_cancel": self.flip,
            "cycles": self._cycles,
            "realised_fake_depth_share": round(float(np.mean(self._depth_shares)), 4) if self._depth_shares else 0.0,
        }

    # ------------------------------------------------------------------
    def _pick_side(self) -> Side:
        if self.fake_side == "buy":
            return Side.BUY
        if self.fake_side == "sell":
            return Side.SELL
        if self._side is None:
            return Side.BUY if self.rng.random() < 0.5 else Side.SELL
        if self.rng.random() < self.persistence:
            return self._side
        return Side.SELL if self._side == Side.BUY else Side.BUY

    @staticmethod
    def _avg_level(state: BookState) -> float:
        levels = state.bid_depth + state.ask_depth
        return sum(q for _, q in levels) / max(1, len(levels)) if levels else 40.0

    def _price(self, state: BookState, side: Side, level: int, tick_size: float) -> float:
        k = self.levels_from_touch + level
        if side == Side.BUY:
            return min(state.best_bid - k * tick_size, state.best_ask - tick_size)
        return max(state.best_ask + k * tick_size, state.best_bid + tick_size)

    def _sizes(self, state: BookState, side: Side) -> list[int]:
        if self.target_share:
            share = float(np.clip(self.target_share, 0.01, 0.9))
            band = state.bid_band_volume if side == Side.BUY else state.ask_band_volume
            band = band or sum(q for _, q in (state.bid_depth if side == Side.BUY else state.ask_depth))
            total = max(self.n_orders, int(share / (1 - share) * max(band, 1)))
            return [max(1, total // self.n_orders)] * self.n_orders
        per = max(1, int(float(self.size_x or 1.0) * self._avg_level(state)))
        return [per] * self.n_orders

    def _pull(self, tick: int) -> list[Order]:
        orders = [_cancel(leg[0], leg[1], tick) for leg in self._legs if self._is_resting(leg[0])]
        if self._genuine is not None and self._is_resting(self._genuine[0]):
            orders.append(_cancel(self._genuine[0], self._genuine[1], tick))
        self._legs, self._genuine = [], None
        return orders

    # ------------------------------------------------------------------
    def _phase(self, tick: int) -> dict | None:
        elapsed = tick - self.start_tick
        for ph, end in zip(self.phases, self._phase_ends):
            if elapsed < end:
                return ph
        return None

    def _program(self, state: BookState, tick: int) -> list[Order]:
        orders: list[Order] = []
        if self._pending_cross is not None and tick >= self._pending_cross[0]:
            _, taker, side, price, qty, maker_oid = self._pending_cross
            self._pending_cross = None
            if self._is_resting(maker_oid):
                orders.append(taker.limit(side, price, qty, tick, exact_ttl=2, tag="w"))
                self._crosses += 1
        ph = self._phase(tick)
        if ph is None or self.rng.random() >= float(ph.get("rate", 0.0)):
            return orders
        avg = self._avg_level(state)
        qty = max(1, int(float(ph.get("size_x_level_depth", 0.5)) * avg))
        ts = self.accounts[0].tick_size
        if ph.get("kind") == "trade":
            acct = self.accounts[min(int(ph.get("account", 0)), len(self.accounts) - 1)]
            side = Side.BUY if ph.get("side", "buy") == "buy" else Side.SELL
            if ph.get("how", "marketable") == "marketable":
                price = state.best_ask if side == Side.BUY else state.best_bid
                orders.append(acct.limit(side, price, qty, tick, exact_ttl=2, tag="p"))
            else:
                price = state.best_bid if side == Side.BUY else state.best_ask
                orders.append(acct.limit(side, price, qty, tick, exact_ttl=20, tag="p"))
            self._program_trades += 1
        elif ph.get("kind") == "cross" and self._pending_cross is None and len(self.accounts) >= 2:
            if state.best_ask - state.best_bid < 2 * ts - 1e-12:
                return orders                                   # one-tick spread: a cross would fill others
            maker, taker = (self.accounts[0], self.accounts[1]) if self._crosses % 2 == 0 else (self.accounts[1], self.accounts[0])
            maker_side = Side.SELL if self.rng.random() < 0.5 else Side.BUY
            price = maker.snap(state.best_ask - ts) if maker_side == Side.SELL else maker.snap(state.best_bid + ts)
            m = maker.limit(maker_side, price, qty, tick, exact_ttl=10, tag="w")
            orders.append(m)
            lo, hi = ph.get("cross_delay_ticks", [0, 2])
            delay = int(self.rng.integers(int(lo), int(hi) + 1))
            taker_side = Side.BUY if maker_side == Side.SELL else Side.SELL
            if delay == 0:
                orders.append(taker.limit(taker_side, price, qty, tick, exact_ttl=2, tag="w"))
                self._crosses += 1
            else:
                self._pending_cross = (tick + delay, taker, taker_side, price, qty, m.order_id)
            self._program_trades += 1
        return orders

    def manipulate(self, state: BookState, tick: int) -> list[Order]:
        if state.best_bid is None or state.best_ask is None:
            return []
        if self.mode == "program":
            return self._program(state, tick)
        genuine_acct = self.accounts[0]
        fake_acct = self.accounts[1] if len(self.accounts) > 1 else self.accounts[0]
        ts = fake_acct.tick_size

        if self._legs:
            filled = self._genuine is not None and not self._is_resting(self._genuine[0])
            if tick >= self._cancel_at or (self.cancel_on_fill and filled):
                orders = self._pull(tick)
                if self.flip > 0 and self.rng.random() < self.flip:
                    side = Side.SELL if self._side == Side.BUY else Side.BUY     # the genuine direction
                    price = state.best_bid if side == Side.SELL else state.best_ask
                    qty = max(1, int(self.genuine_size_x * 2 * self._avg_level(state)))
                    orders.append(genuine_acct.limit(side, price, qty, tick, exact_ttl=2, tag="f"))
                self._next_cycle = tick + int(self.rng.integers(self.gap[0], self.gap[1] + 1))
                return orders
            orders = []
            if self.reprice:
                for leg in self._legs:
                    oid, acct, side, qty, price, level = leg
                    target = acct.snap(self._price(state, side, level, ts))
                    if abs(target - price) >= ts / 2 and self._is_resting(oid):
                        orders.append(_cancel(oid, acct, tick))
                        new = acct.limit(side, target, qty, tick, exact_ttl=max(1, self._cancel_at - tick + 1), tag="s")
                        orders.append(new)
                        leg[0], leg[4] = new.order_id, new.price
            if self.genuine_rate > 0 and self.rng.random() < self.genuine_rate:
                g_side = Side.SELL if self._side == Side.BUY else Side.BUY
                price = state.best_ask if g_side == Side.SELL else state.best_bid
                orders.append(genuine_acct.limit(g_side, price, max(1, int(self.genuine_size_x * self._avg_level(state))),
                                                 tick, exact_ttl=20, tag="g"))
            return orders

        if tick < self._next_cycle:
            return []
        # --- new cycle ---
        self._side = self._pick_side()
        self._cycles += 1
        life = int(self.rng.integers(self.lifetime[0], self.lifetime[1] + 1))
        self._cancel_at = tick + life
        orders: list[Order] = []
        if self.genuine_place == "per_cycle":
            g_side = Side.SELL if self._side == Side.BUY else Side.BUY
            price = state.best_ask if g_side == Side.SELL else state.best_bid
            g = genuine_acct.limit(g_side, price, max(1, int(self.genuine_size_x * self._avg_level(state))),
                                   tick, exact_ttl=life + 2, tag="g")
            orders.append(g)
            self._genuine = (g.order_id, genuine_acct)
        for level, qty in enumerate(self._sizes(state, self._side)):
            o = fake_acct.limit(self._side, self._price(state, self._side, level, ts), qty, tick,
                                exact_ttl=life + 1, tag="s")
            orders.append(o)
            self._legs.append([o.order_id, fake_acct, self._side, qty, o.price, level])
        self._depth_shares.append(fake_depth_share(state, self._side, [(leg[4], leg[3]) for leg in self._legs]))
        return orders
