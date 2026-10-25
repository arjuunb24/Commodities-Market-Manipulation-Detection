"""
src/detector/features.py
========================
Feature engineering for the Veridex detector (Module 2, SRS FR-2.1).

Surveillance systems score *accounts*, not markets: a spoofer is identified by
what its own orders do, not by a market-wide cancel rate diluted by forty other
participants. The engine therefore produces two tables over the same rolling
windows (tick-based; 1 tick = 1 second):

  market features  — one row per (commodity, window): returns, volatility,
                     volume spikes, aggressor-side flow, order-book pressure.
  trader features  — one row per (commodity, window, trader_id) for every
                     account active in the window, with the market context of
                     that window joined on (prefix `mkt_`) and peer-relative
                     robust z-scores (prefix `peer_z_`) computed against all
                     other accounts in the same window.

Trader-level behavioural features, grouped by the persona they target:

  Layering     layered_order_share (volume placed as same-tick, same-side
               batches on >= 3 distinct price levels — the Tower pattern)
  Repricing    cancel_replace_ratio (cancels immediately replaced on the same
               side in the same tick — Sarao-style spoof repricing)
  Spoofing     spoof_signature (share of the account's large cancelled orders
               during whose life it was filled on the opposite side),
               linked_spoof_signature / linked_lift_log (the same bait-and-switch
               split over two accounts, see _linked_pair_steps),
               cancel_ratio, fast_cancel_ratio, mean_cancel_lifetime,
               large_order_share, large_cancel_ratio, max_order_size_rel,
               passive_distance_bp, cancel_exec_opposition (cancels on one
               side while executing on the other — the Coscia/Sarao signature)
  Wash trading reciprocal_ratio (volume traded back and forth with the same
               counterparty), top_cpty_share, net_flatness, price_dev_bp,
               trade_size_cv; and for rings that rotate wash trades over several
               accounts (where pairwise features fade): mutual_cpty_share,
               ring_size, ring_volume_share (see _add_ring_features)
  Pump & dump  exec_side_imbalance, aggressor_ratio, signed_flow_return
               (own buying while the price rises), reversal_score (net buyer in
               prior windows, net seller now), long_net_ratio (~12-minute
               position build-up), activity_vs_history, coordinated_peers
               (other accounts bursting in the same direction in the window)

Designated market makers (trader ids starting with an excluded prefix) are
dropped from the trader table: real surveillance monitors registered liquidity
providers under separate quoting obligations, and their requote cancels would
otherwise dominate every cancellation feature.

NaN policy (Implementation Plan §6.1): ratios with a zero denominator are
"no signal", encoded as an explicit neutral value per feature below and paired
with activity counts (n_orders, n_trades) so models can tell "inactive" from
"zero". Windows shorter than the full window length at the start of a run are
dropped rather than zero-padded.

Features are evaluated at completed-window time. Historical market-size
baselines use prior steps (with a first-step cold-start fallback), and
order-lifetime signatures never inspect fills after cancellation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAST_CANCEL_TICKS = 10        # cancelled within 10 s of placement
LARGE_ORDER_MULTIPLE = 4.0    # ≥ 4x the pre-step median of prior order-size step medians
PEER_Z_COLUMNS = [
    "cancel_ratio", "fast_cancel_ratio", "avg_order_size_rel", "max_order_size_rel",
    "gross_volume_share", "reciprocal_ratio", "aggressor_ratio", "large_order_share",
]
# Robust z-score of an account against ITS OWN earlier (non-overlapping) windows.
# A legitimate algorithm that always rests large orders looks normal against its
# own history; an account that only now starts spoofing does not.
SELF_Z_COLUMNS = [
    "avg_order_size_rel", "max_order_size_rel", "large_order_share", "cancel_ratio",
    "fast_cancel_ratio", "n_orders", "gross_volume_share", "layered_order_share",
]
# Bumped whenever feature definitions change, so cached features.parquet files are recomputed.
FEATURE_VERSION = 5
SELF_Z_MIN_HISTORY = 10         # past active windows (5 min) needed before a self z-score is reported
# Minimum scales: a sparse account's history barely varies, so tiny changes became huge z-scores
# (cancel_ratio drove most alerts on manipulator accounts outside their episodes).
SELF_Z_SCALE_FLOOR = {"n_orders": 2.0, "avg_order_size_rel": 0.5, "max_order_size_rel": 0.5,
                      "cancel_ratio": 0.10, "fast_cancel_ratio": 0.10, "large_order_share": 0.10,
                      "layered_order_share": 0.10, "gross_volume_share": 0.02}
# Past windows that were themselves this unusual (first pass, any self-z column) are left out of
# the account's baseline, so a burst of abnormal behaviour does not distort what "normal" means
# for the account afterwards. Uses no labels: only the account's own earlier deviations.
SELF_Z_TRIM = 4.0
SELF_Z_TRIM_PASSES = 4
# Evidence weighting: a self z-score is scaled by n / (n + K), n = orders the account placed in the
# window. Two orders say little about a cancel ratio; fifty say a lot. Without it, sparse but
# innocent accounts (occasional traders) collected alerts from noise. 0 turns it off. n_orders' own
# z-score is not scaled (it IS the count).
SELF_Z_EVIDENCE_K = 5.0
# activity_vs_history baseline: trimmed median of the account's earlier activity (same iterative
# cleaning as the self z-scores) instead of the plain mean, which a single long burst (a pump with a
# dozen accounts) inflated for the rest of the session, so normal trading afterwards looked like a
# collapse in activity and drew pump-and-dump alerts. False = the old mean.
ACTIVITY_ROBUST = True
ACTIVITY_TRIM_LOG = 1.5           # a window more than e^1.5 (~4.5x) above/below the baseline is unusual
# Wash rings: a pair is "mutual" when each side is at least MUTUAL_MIN_SHARE of the other's traded
# volume in the window and they traded at least RING_MIN_PAIR_SIZES x the pre-window size baseline.
# Ring partners are mutually important; a trader and a big market maker or HFT are not.
MUTUAL_MIN_SHARE = 0.10
RING_MIN_PAIR_SIZES = 2.0
RING_COLUMNS = ["mutual_cpty_share", "ring_size", "ring_volume_share"]
# Cross-account spoofing: a pair needs at least this many of A's large cancels in the window with
# B filled on the opposite side before its score counts (one coincidence says nothing).
LINK_MIN_EVENTS = 3
# Evidence weighting n / (n + K) of the pair score (as for the self z-scores): an execution algo
# cancels 2-6 large orders per window and two chance events gave it a score of 0.3-0.9; a spoof
# campaign produces 10-20 events per window.
LINK_EVIDENCE_K = 4.0
LINK_PLACE_TICKS = 1   # the partner's order is placed within this many ticks of the bait
LINK_COLUMNS = ["linked_spoof_signature", "linked_lift_log"]
MARKET_CONTEXT_COLUMNS = [
    "mkt_return_bp", "mkt_volatility_bp", "mkt_range_bp", "mkt_volume_spike",
    "mkt_trade_imbalance", "mkt_order_imbalance", "mkt_cancel_ratio", "mkt_abnormal_return_z",
]


def _ticks(value: Any) -> int:
    """'3min' / '30s' / 180 → ticks (1 tick = 1 second)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(pd.to_timedelta(value).total_seconds())


def _opposite_fill_flags(lc: pd.DataFrame, legs: pd.DataFrame, grace: int = 0) -> np.ndarray:
    """
    For each large cancelled order (trader_id, side, placed_tick, tick): 1.0 if the same
    account was filled on the OPPOSITE side at a tick in [placed_tick, tick + grace].
    The default uses no post-cancel grace period, so the feature is causal.

    Uses per-account sorted fill ticks and binary search, so memory grows with the number
    of orders, not orders x fills (a cross join exploded to hundreds of millions of rows
    for high-cancel HFT accounts).
    """
    out = np.zeros(len(lc))
    if lc.empty or legs.empty:
        return out
    sells = {t: np.sort(g["tick"].to_numpy()) for t, g in legs[legs["sell"] > 0].groupby("trader_id")}
    buys = {t: np.sort(g["tick"].to_numpy()) for t, g in legs[legs["buy"] > 0].groupby("trader_id")}
    lo_all = lc["placed_tick"].to_numpy(dtype=float)
    hi_all = lc["tick"].to_numpy(dtype=float) + grace
    side = lc["side"].to_numpy()
    for trader, idx in lc.groupby("trader_id").indices.items():
        for s, book in (("buy", sells), ("sell", buys)):      # a buy order needs a sell fill and vice versa
            arr = book.get(trader)
            sel = idx[side[idx] == s]
            if arr is None or not len(sel):
                continue
            n = np.searchsorted(arr, hi_all[sel], side="right") - np.searchsorted(arr, lo_all[sel], side="left")
            out[sel] = (n > 0).astype(float)
    return out


def _expand_ranges(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ranges [lo_i, hi_i): (owner index i, position) for every position in every range."""
    n = np.maximum(hi - lo, 0)
    owner = np.repeat(np.arange(len(lo)), n)
    pos = lo[owner] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    return owner, pos


def _linked_pair_steps(lc: pd.DataFrame, legs: pd.DataFrame, placed: pd.DataFrame, step_ticks: int,
                       exclude: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-account bait-and-switch. spoof_signature looks for the opposite-side fill in the SAME
    account; spoofing and layering spread over several accounts put the large cancelled bait in
    one account and the genuine order in another (run seed1001: 3-account spoofing reached a
    frozen spoofing-classifier probability of 0.001). The collusion shows in the timing: the
    partner places its opposite-side order together with the bait and is filled while the bait
    rests. An event is a large cancelled order of account A and another account B that

      * placed an opposite-side order within +-LINK_PLACE_TICKS of A's placement, and
      * was filled on the opposite side, against a third party, while A's order rested
      (up to its cancel; causal).

    Being filled against an algo that is chasing the price is common (in seed1001 every busy
    account was, often above its average rate); being filled on an order placed in the same
    tick as someone else's bait, time after time, is not. Returns, per cancel step:

      pairs  (step, trader_id=A, cpty=B, n_ev): A's large cancels that were events with B
      rates  (step, trader_id, lc_buy, lc_sell, life_buy, life_sell, pt_buy, pt_sell, f_buy, f_sell):
             large cancels and their resting ticks by side, ticks with a placement by side,
             fills by side (for the chance level in _add_linked_features)
    """
    rate_cols = ["lc_buy", "lc_sell", "life_buy", "life_sell", "pt_buy", "pt_sell", "f_buy", "f_sell"]
    empty_pairs = pd.DataFrame(columns=["step", "trader_id", "cpty", "n_ev"])
    keep = lambda df, col="trader_id": df[~df[col].astype(str).str.startswith(exclude)]  # noqa: E731
    lc, legs, placed = keep(lc), keep(legs), keep(placed)

    fills = pd.DataFrame({"tick": legs["tick"].to_numpy(dtype=np.int64), "trader_id": legs["trader_id"].to_numpy(),
                          "cpty": legs["cpty"].to_numpy() if "cpty" in legs else None,
                          "side": np.where(legs["buy"].to_numpy(dtype=float) > 0, "buy", "sell")})
    pl = placed[["trader_id", "side", "tick"]].drop_duplicates()
    life = (lc["tick"] - lc["placed_tick"] + 1).clip(lower=1).astype(float)
    tables = [
        lc.assign(lc_buy=(lc["side"] == "buy").astype(float), lc_sell=(lc["side"] == "sell").astype(float),
                  life_buy=np.where(lc["side"] == "buy", life, 0.0), life_sell=np.where(lc["side"] == "sell", life, 0.0))
          [["step", "trader_id", "lc_buy", "lc_sell", "life_buy", "life_sell"]],
        pl.assign(step=pl["tick"] // step_ticks, pt_buy=(pl["side"] == "buy").astype(float),
                  pt_sell=(pl["side"] == "sell").astype(float))[["step", "trader_id", "pt_buy", "pt_sell"]],
        fills.assign(step=fills["tick"] // step_ticks, f_buy=(fills["side"] == "buy").astype(float),
                     f_sell=(fills["side"] == "sell").astype(float))[["step", "trader_id", "f_buy", "f_sell"]],
    ]
    rates = pd.concat(tables, ignore_index=True).fillna(0.0).groupby(["step", "trader_id"])[rate_cols].sum().reset_index()
    if lc.empty or fills.empty or pl.empty:
        return empty_pairs, rates

    events = []
    for side, opp in (("buy", "sell"), ("sell", "buy")):
        L = lc[lc["side"] == side]
        P = pl[pl["side"] == opp].sort_values("tick", kind="stable")
        F = fills[fills["side"] == opp].sort_values("tick", kind="stable")
        if L.empty or P.empty or F.empty:
            continue
        pt = L["placed_tick"].to_numpy(dtype=float)
        # (bait, account placing on the opposite side within +-LINK_PLACE_TICKS)
        i, j = _expand_ranges(np.searchsorted(P["tick"].to_numpy(), pt - LINK_PLACE_TICKS, "left"),
                              np.searchsorted(P["tick"].to_numpy(), pt + LINK_PLACE_TICKS, "right"))
        co = pd.DataFrame({"cid": L["cid"].to_numpy()[i], "cpty": P["trader_id"].to_numpy()[j]}).drop_duplicates()
        # (bait, account filled on the opposite side while it rested)
        i, j = _expand_ranges(np.searchsorted(F["tick"].to_numpy(), pt, "left"),
                              np.searchsorted(F["tick"].to_numpy(), L["tick"].to_numpy(dtype=float), "right"))
        fl = pd.DataFrame({"cid": L["cid"].to_numpy()[i], "cpty": F["trader_id"].to_numpy()[j]})
        # ... against someone other than A: trading WITH the bait (an algo's resting order being
        # hit) is a counterparty, not a partner; a genuine leg is filled by third parties.
        fl = fl[F["cpty"].to_numpy()[j] != L["trader_id"].to_numpy()[i]].drop_duplicates()
        ev = co.merge(fl, on=["cid", "cpty"])
        events.append(ev.merge(L[["cid", "step", "trader_id"]], on="cid"))
    if not events:
        return empty_pairs, rates
    ev = pd.concat(events, ignore_index=True)
    ev = ev[ev["trader_id"] != ev["cpty"]]
    pairs = ev.groupby(["step", "trader_id", "cpty"]).size().rename("n_ev").reset_index()
    return pairs, rates


def _safe_div(a, b, fill: float = 0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, fill, dtype=float)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _to_windows(df: pd.DataFrame, keys: list[str], sums: list[str], maxes: list[str],
                n_steps: int, first_step: int, last_step: int) -> pd.DataFrame:
    """
    Aggregate per-step rows into rolling windows of `n_steps` steps.

    Each step row is replicated into the n_steps windows that contain it, then
    grouped. Window `w` covers steps [w - n_steps + 1, w]. Only complete windows
    inside the run are kept.
    """
    if df.empty:
        return pd.DataFrame(columns=["win", *keys, *sums, *maxes])
    parts = []
    for k in range(n_steps):
        part = df[["step", *keys, *sums, *maxes]].copy()
        part["win"] = part["step"] + k
        parts.append(part)
    rep = pd.concat(parts, ignore_index=True)
    rep = rep[(rep["win"] >= first_step + n_steps - 1) & (rep["win"] <= last_step)]
    agg = {c: "sum" for c in sums}
    agg.update({c: "max" for c in maxes})
    return rep.groupby(["win", *keys], sort=False).agg(agg).reset_index()


class FeatureEngine:
    """Computes market-level and trader-level rolling-window features."""

    def __init__(self, config: dict[str, Any]) -> None:
        fe = config.get("feature_engine", {})
        self.window = fe.get("window", "3min")
        self.step = fe.get("step", "30s")
        self.step_ticks = max(1, _ticks(self.step))
        self.window_ticks = max(self.step_ticks, _ticks(self.window))
        self.n_steps = max(1, self.window_ticks // self.step_ticks)
        self.top_k = fe.get("top_k_traders", 5)
        self.baseline_windows = fe.get("vol_spike_lookback", 10)
        self.exclude_prefixes = tuple(fe.get("exclude_trader_prefixes", ["mm_"]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_features(self, trade_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
        """Model input: trader-level features with market context joined."""
        _, trader = self.compute(trade_df, order_df)
        return trader

    def compute(self, trade_df: pd.DataFrame, order_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (market_features, trader_features).

        market_features index: (commodity, window_start)
        trader_features index: (commodity, window_start, trader_id)
        Both carry a `window_end` column (inclusive last tick of the window).
        """
        commodities = sorted(
            set(trade_df["commodity"].unique() if not trade_df.empty else [])
            | set(order_df["commodity"].unique() if not order_df.empty else [])
        )
        markets, traders = [], []
        for comm in commodities:
            t = trade_df[trade_df["commodity"] == comm] if not trade_df.empty else trade_df
            o = order_df[order_df["commodity"] == comm] if not order_df.empty else order_df
            m, tr = self._compute_commodity(t, o)
            if not m.empty:
                m["commodity"] = comm
                markets.append(m)
            if not tr.empty:
                tr["commodity"] = comm
                traders.append(tr)
        market = (pd.concat(markets, ignore_index=True).set_index(["commodity", "window_start"]).sort_index()
                  if markets else pd.DataFrame())
        trader = (pd.concat(traders, ignore_index=True).set_index(["commodity", "window_start", "trader_id"]).sort_index()
                  if traders else pd.DataFrame())
        return market, trader

    # ------------------------------------------------------------------
    # Per-commodity computation
    # ------------------------------------------------------------------

    def _compute_commodity(self, trades: pd.DataFrame, orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        S, W = self.step_ticks, self.n_steps
        ticks = []
        if not trades.empty:
            ticks += [trades["tick"].min(), trades["tick"].max()]
        if not orders.empty:
            ticks += [orders["tick"].min(), orders["tick"].max()]
        if not ticks:
            return pd.DataFrame(), pd.DataFrame()
        first_step, last_step = int(min(ticks)) // S, int(max(ticks)) // S
        if last_step - first_step + 1 < W:
            return pd.DataFrame(), pd.DataFrame()

        trades = trades.copy()
        orders = orders.copy()
        trades["step"] = trades["tick"] // S
        orders["step"] = orders["tick"] // S

        # ---------------- per-step market price path ----------------
        steps = pd.RangeIndex(first_step, last_step + 1, name="step")
        if not trades.empty:
            trades["notional"] = trades["price"] * trades["quantity"]
            g = trades.groupby("step")
            step_px = pd.DataFrame({
                "vol": g["quantity"].sum(),
                "n_trades": g.size(),
                "notional": g["notional"].sum(),
                "last": g["price"].last(),
                "hi": g["price"].max(),
                "lo": g["price"].min(),
                "agg_buy": trades[trades["aggressor_side"] == "buy"].groupby("step")["quantity"].sum(),
            }).reindex(steps)
        else:
            step_px = pd.DataFrame(index=steps, columns=["vol", "n_trades", "notional", "last", "hi", "lo", "agg_buy"], dtype=float)
        step_px[["vol", "n_trades", "notional", "agg_buy"]] = step_px[["vol", "n_trades", "notional", "agg_buy"]].fillna(0.0)
        step_px["last"] = step_px["last"].ffill()
        step_px["vwap"] = pd.Series(_safe_div(step_px["notional"], step_px["vol"], np.nan), index=steps).ffill()
        step_px["ret"] = np.log(step_px["last"]).diff().fillna(0.0)
        ref_price = step_px["last"].shift(1).ffill()

        # ---------------- order-level enrichment ----------------
        placed = orders[orders["event_type"] == "placed"][["order_id", "trader_id", "side", "price", "quantity", "tick", "step"]].copy()
        placed["quantity"] = placed["quantity"].astype(float)
        # Historical thresholds use earlier steps only. At the cold start,
        # initialize from the current step's observed median; this avoids
        # classifying every ordinary order in the first window as "large".
        step_size_median = placed.groupby("step")["quantity"].median().reindex(steps)
        prior_size_baseline = (
            step_size_median.shift(1)
            .expanding(min_periods=1)
            .median()
            .ffill()
        )
        current_size_baseline = step_size_median.expanding(min_periods=1).median()
        size_baseline = prior_size_baseline.fillna(current_size_baseline).ffill().fillna(1.0).clip(lower=1.0)
        placed["size_baseline"] = placed["step"].map(size_baseline)
        placed["is_large"] = placed["quantity"] >= LARGE_ORDER_MULTIPLE * placed["size_baseline"]
        placed["ref"] = placed["step"].map(ref_price)
        sign = np.where(placed["side"] == "buy", 1.0, -1.0)
        placed["dist_bp"] = sign * (placed["ref"] - placed["price"]) / placed["ref"] * 1e4
        placed["buy_q"] = np.where(placed["side"] == "buy", placed["quantity"], 0.0)
        placed["sell_q"] = placed["quantity"] - placed["buy_q"]
        placed["q2"] = placed["quantity"] ** 2
        placed["large_q"] = np.where(placed["is_large"], placed["quantity"], 0.0)
        placed["dist_q"] = placed["dist_bp"] * placed["quantity"]
        # Layering fingerprint: same account, same tick, same side, >= 3 distinct price levels.
        batch = placed.groupby(["trader_id", "tick", "side"])["price"]
        placed["layer_q"] = np.where(batch.transform("nunique") >= 3, placed["quantity"], 0.0)

        cancels = orders[orders["event_type"] == "cancelled"][["order_id", "trader_id", "quantity", "tick", "step"]].copy()
        cancels["quantity"] = cancels["quantity"].astype(float)
        cancels = cancels.merge(
            placed[["order_id", "side", "tick", "is_large"]].rename(columns={"tick": "placed_tick"}),
            on="order_id", how="left",
        )
        cancels["lifetime"] = (cancels["tick"] - cancels["placed_tick"]).fillna(self.window_ticks)
        cancels["is_large"] = cancels["is_large"].fillna(False).astype(bool)
        cancels["c_buy"] = np.where(cancels["side"] == "buy", cancels["quantity"], 0.0)
        cancels["c_sell"] = np.where(cancels["side"] == "sell", cancels["quantity"], 0.0)
        cancels["fast_q"] = np.where(cancels["lifetime"] <= FAST_CANCEL_TICKS, cancels["quantity"], 0.0)
        cancels["large_c"] = np.where(cancels["is_large"], cancels["quantity"], 0.0)
        # Cancel-and-replace: a new order on the same side placed by the same account in the
        # same tick as the cancel (Sarao-style repricing; market makers are excluded upstream).
        replaced_keys = placed[["trader_id", "tick", "side"]].drop_duplicates().assign(_replaced=1.0)
        cancels = cancels.merge(replaced_keys, on=["trader_id", "tick", "side"], how="left")
        cancels["replaced"] = cancels["_replaced"].fillna(0.0)

        # ---------------- trader per-step aggregates ----------------
        p_step = placed.groupby(["step", "trader_id"]).agg(
            n_placed=("quantity", "size"), placed_vol=("quantity", "sum"),
            placed_buy=("buy_q", "sum"), placed_sell=("sell_q", "sum"),
            placed_q2=("q2", "sum"), placed_max=("quantity", "max"),
            large_vol=("large_q", "sum"), dist_q=("dist_q", "sum"), layer_vol=("layer_q", "sum"),
        ).reset_index()
        c_step = cancels.groupby(["step", "trader_id"]).agg(
            n_cancel=("quantity", "size"), cancel_vol=("quantity", "sum"),
            cancel_buy=("c_buy", "sum"), cancel_sell=("c_sell", "sum"),
            fast_cancel=("fast_q", "sum"), large_cancel=("large_c", "sum"),
            lifetime_sum=("lifetime", "sum"), n_replaced=("replaced", "sum"),
        ).reset_index()

        if not trades.empty:
            trades["vwap_step"] = trades["step"].map(step_px["vwap"])
            trades["dev"] = (trades["price"] - trades["vwap_step"]).abs() / trades["vwap_step"] * 1e4 * trades["quantity"]
            buy_side = pd.DataFrame({
                "step": trades["step"], "trader_id": trades["buyer_trader_id"], "cpty": trades["seller_trader_id"],
                "buy": trades["quantity"].astype(float), "sell": 0.0,
                "aggr": np.where(trades["aggressor_side"] == "buy", trades["quantity"], 0).astype(float),
                "dev": trades["dev"], "q2": trades["quantity"].astype(float) ** 2,
            })
            sell_side = pd.DataFrame({
                "step": trades["step"], "trader_id": trades["seller_trader_id"], "cpty": trades["buyer_trader_id"],
                "buy": 0.0, "sell": trades["quantity"].astype(float),
                "aggr": np.where(trades["aggressor_side"] == "sell", trades["quantity"], 0).astype(float),
                "dev": trades["dev"], "q2": trades["quantity"].astype(float) ** 2,
            })
            legs = pd.concat([buy_side, sell_side], ignore_index=True)
            legs_with_tick = legs.assign(tick=np.concatenate([trades["tick"].to_numpy(), trades["tick"].to_numpy()]))
            t_step = legs.groupby(["step", "trader_id"]).agg(
                n_exec=("buy", "size"), exec_buy=("buy", "sum"), exec_sell=("sell", "sum"),
                aggr_vol=("aggr", "sum"), dev_sum=("dev", "sum"), exec_q2=("q2", "sum"),
            ).reset_index()
            pair_step = legs.groupby(["step", "trader_id", "cpty"]).agg(
                bought=("buy", "sum"), sold=("sell", "sum"),
            ).reset_index()
        else:
            t_step = pd.DataFrame(columns=["step", "trader_id", "n_exec", "exec_buy", "exec_sell", "aggr_vol", "dev_sum", "exec_q2"])
            pair_step = pd.DataFrame(columns=["step", "trader_id", "cpty", "bought", "sold"])
            legs_with_tick = pd.DataFrame(columns=["trader_id", "tick", "buy", "sell"])

        # ---------------- spoof signature (order level) ----------------
        # For each LARGE order that was cancelled: did the same account get filled
        # on the OPPOSITE side while that order was resting (or right after)?
        # That is the bait-and-switch the Coscia and Sarao cases turned on.
        lc = cancels[cancels["is_large"] & cancels["side"].notna()][["trader_id", "side", "placed_tick", "tick", "step"]]
        lc = lc.reset_index(drop=True).reset_index().rename(columns={"index": "cid"})
        lc["opp_fill"] = 0.0
        if not lc.empty and not trades.empty:
            lc["opp_fill"] = _opposite_fill_flags(lc, legs_with_tick)
        lc_step = lc.groupby(["step", "trader_id"]).agg(
            n_large_cancel=("cid", "size"), n_large_cancel_opp=("opp_fill", "sum"),
        ).reset_index()
        link_step, rate_step = _linked_pair_steps(lc, legs_with_tick, placed, S, self.exclude_prefixes)

        # ---------------- roll into windows ----------------
        p_win = _to_windows(p_step, ["trader_id"],
                            ["n_placed", "placed_vol", "placed_buy", "placed_sell", "placed_q2", "large_vol", "dist_q", "layer_vol"],
                            ["placed_max"], W, first_step, last_step)
        c_win = _to_windows(c_step, ["trader_id"],
                            ["n_cancel", "cancel_vol", "cancel_buy", "cancel_sell", "fast_cancel", "large_cancel", "lifetime_sum", "n_replaced"],
                            [], W, first_step, last_step)
        t_win = _to_windows(t_step, ["trader_id"],
                            ["n_exec", "exec_buy", "exec_sell", "aggr_vol", "dev_sum", "exec_q2"],
                            [], W, first_step, last_step)
        pair_win = _to_windows(pair_step, ["trader_id", "cpty"], ["bought", "sold"], [], W, first_step, last_step)
        lc_win = _to_windows(lc_step, ["trader_id"], ["n_large_cancel", "n_large_cancel_opp"], [], W, first_step, last_step)
        link_win = _to_windows(link_step, ["trader_id", "cpty"], ["n_ev"], [], W, first_step, last_step)
        rate_win = _to_windows(rate_step, ["trader_id"], ["lc_buy", "lc_sell", "life_buy", "life_sell",
                                                          "pt_buy", "pt_sell", "f_buy", "f_sell"], [], W, first_step, last_step)

        market = self._market_features(step_px, placed, cancels, first_step, last_step)

        tw = (p_win.merge(c_win, on=["win", "trader_id"], how="outer")
              .merge(t_win, on=["win", "trader_id"], how="outer")
              .merge(lc_win, on=["win", "trader_id"], how="outer"))
        tw = tw.fillna(0.0)
        tw = tw[~tw["trader_id"].astype(str).str.startswith(self.exclude_prefixes)]
        if tw.empty:
            return market.reset_index(), pd.DataFrame()

        if not pair_win.empty:
            pair_win["gross"] = pair_win["bought"] + pair_win["sold"]
            pair_win["recip"] = 2 * np.minimum(pair_win["bought"], pair_win["sold"])
            cp = pair_win.groupby(["win", "trader_id"]).agg(
                top_pair=("gross", "max"), recip=("recip", "sum"), n_cpty=("cpty", "nunique"),
            ).reset_index()
            tw = tw.merge(cp, on=["win", "trader_id"], how="left")
        for c in ("top_pair", "recip", "n_cpty"):
            if c not in tw:
                tw[c] = 0.0
        tw[["top_pair", "recip", "n_cpty"]] = tw[["top_pair", "recip", "n_cpty"]].fillna(0.0)

        window_first_step = first_step + W - 1
        window_starts = range(first_step, last_step - W + 2)
        size_baseline_by_win = pd.Series(
            [prior_size_baseline.get(step, np.nan) for step in window_starts],
            index=range(window_first_step, last_step + 1),
            dtype=float,
        )
        cold_start = size_baseline_by_win.isna()
        size_baseline_by_win.loc[cold_start] = [
            step_size_median.get(start, np.nan)
            for start in np.asarray(size_baseline_by_win.index)[cold_start.to_numpy()]
        ]
        size_baseline_by_win = size_baseline_by_win.ffill().fillna(1.0).clip(lower=1.0)
        tw["size_baseline"] = tw["win"].map(size_baseline_by_win).fillna(1.0)
        feats = self._trader_features(tw)
        feats = self._add_ring_features(feats, pair_win, size_baseline_by_win)
        feats = self._add_linked_features(feats, link_win, rate_win)
        feats = self._add_temporal(feats)
        feats = feats.merge(market[["win", *MARKET_CONTEXT_COLUMNS, "mkt_volume"]], on="win", how="left")
        feats["gross_volume_share"] = _safe_div(feats["exec_vol"], feats["mkt_volume"])
        feats["signed_flow_return"] = feats["exec_side_imbalance"] * feats["mkt_return_bp"]
        feats = self._add_peer_z(feats)
        feats = self._add_self_z(feats)

        feats["window_start"] = (feats["win"] - W + 1) * S
        feats["window_end"] = (feats["win"] + 1) * S - 1
        feats = feats.drop(columns=["win", "mkt_volume", "exec_vol_raw"], errors="ignore")
        market = market.copy()
        market["window_start"] = (market["win"] - W + 1) * S
        market["window_end"] = (market["win"] + 1) * S - 1
        market = market.drop(columns=["win"])
        return market, feats

    # ------------------------------------------------------------------
    # Market-level features
    # ------------------------------------------------------------------

    def _market_features(self, step_px: pd.DataFrame, placed: pd.DataFrame, cancels: pd.DataFrame,
                         first_step: int, last_step: int) -> pd.DataFrame:
        W = self.n_steps
        st = step_px.copy()
        st["placed_buy"] = placed.groupby("step")["buy_q"].sum().reindex(st.index).fillna(0.0)
        st["placed_sell"] = placed.groupby("step")["sell_q"].sum().reindex(st.index).fillna(0.0)
        st["n_placed"] = placed.groupby("step").size().reindex(st.index).fillna(0.0)
        st["n_cancel"] = cancels.groupby("step").size().reindex(st.index).fillna(0.0)

        roll = st[["vol", "n_trades", "agg_buy", "placed_buy", "placed_sell", "n_placed", "n_cancel"]].rolling(W).sum()
        m = pd.DataFrame(index=st.index)
        m["mkt_volume"] = roll["vol"]
        m["mkt_trades"] = roll["n_trades"]
        m["mkt_return_bp"] = (np.log(st["last"]) - np.log(st["last"].shift(W))).fillna(0.0) * 1e4
        m["mkt_volatility_bp"] = st["ret"].rolling(W).std().fillna(0.0) * 1e4
        hi = st["hi"].fillna(st["last"]).rolling(W).max()
        lo = st["lo"].fillna(st["last"]).rolling(W).min()
        m["mkt_range_bp"] = _safe_div(hi - lo, lo) * 1e4
        baseline = m["mkt_volume"].shift(W).rolling(self.baseline_windows * W, min_periods=W).mean()
        m["mkt_volume_spike"] = _safe_div(m["mkt_volume"], baseline.fillna(m["mkt_volume"]), 1.0)
        m["mkt_trade_imbalance"] = _safe_div(2 * roll["agg_buy"] - roll["vol"], roll["vol"])
        m["mkt_order_imbalance"] = _safe_div(roll["placed_buy"] - roll["placed_sell"], roll["placed_buy"] + roll["placed_sell"])
        m["mkt_cancel_ratio"] = _safe_div(roll["n_cancel"], roll["n_placed"])
        absret = m["mkt_return_bp"].abs()
        mu = absret.shift(W).rolling(self.baseline_windows * W, min_periods=W).mean()
        sd = absret.shift(W).rolling(self.baseline_windows * W, min_periods=W).std()
        m["mkt_abnormal_return_z"] = ((absret - mu) / sd.replace(0, np.nan)).fillna(0.0).clip(-10, 10)
        m = m.loc[first_step + W - 1: last_step].fillna(0.0)
        m.index.name = "win"
        return m.reset_index()

    # ------------------------------------------------------------------
    # Trader-level features
    # ------------------------------------------------------------------

    def _trader_features(self, tw: pd.DataFrame) -> pd.DataFrame:
        f = tw[["win", "trader_id"]].copy()
        exec_vol = tw["exec_buy"] + tw["exec_sell"]
        f["n_orders"] = tw["n_placed"]
        f["n_trades"] = tw["n_exec"]
        f["exec_vol"] = exec_vol
        avg_size = _safe_div(tw["placed_vol"], tw["n_placed"])
        f["avg_order_size_rel"] = avg_size / tw["size_baseline"]
        f["max_order_size_rel"] = tw["placed_max"] / tw["size_baseline"]
        f["order_size_cv"] = np.sqrt(np.clip(_safe_div(tw["placed_q2"], tw["n_placed"]) - avg_size ** 2, 0, None)) / np.where(avg_size > 0, avg_size, 1.0)
        f["large_order_share"] = _safe_div(tw["large_vol"], tw["placed_vol"])
        f["layered_order_share"] = _safe_div(tw["layer_vol"], tw["placed_vol"])
        f["cancel_replace_ratio"] = np.clip(_safe_div(tw["n_replaced"], tw["n_cancel"]), 0, 1)
        # Cancels of orders placed before the window with no new orders → ratio 1.
        no_new = np.where(tw["cancel_vol"] > 0, 1.0, 0.0)
        f["cancel_ratio"] = np.where(tw["placed_vol"] > 0,
                                     np.clip(_safe_div(tw["cancel_vol"], tw["placed_vol"]), 0, 1), no_new)
        f["fast_cancel_ratio"] = np.clip(_safe_div(tw["fast_cancel"], np.maximum(tw["placed_vol"], tw["cancel_vol"])), 0, 1)
        f["large_cancel_ratio"] = np.clip(_safe_div(tw["large_cancel"], tw["large_vol"]), 0, 1)
        f["mean_cancel_lifetime"] = _safe_div(tw["lifetime_sum"], tw["n_cancel"], float(self.window_ticks))
        f["passive_distance_bp"] = _safe_div(tw["dist_q"], tw["placed_vol"])
        f["fill_ratio"] = np.clip(_safe_div(exec_vol, tw["placed_vol"], 0.0), 0, 2)
        f["cancel_side_imbalance"] = _safe_div(tw["cancel_buy"] - tw["cancel_sell"], tw["cancel_buy"] + tw["cancel_sell"])
        f["exec_side_imbalance"] = _safe_div(tw["exec_buy"] - tw["exec_sell"], exec_vol)
        f["cancel_exec_opposition"] = np.clip(-f["cancel_side_imbalance"] * f["exec_side_imbalance"], 0, 1)
        f["n_large_cancels"] = tw["n_large_cancel"]
        f["spoof_signature"] = _safe_div(tw["n_large_cancel_opp"], tw["n_large_cancel"])
        f["net_flatness"] = np.where(exec_vol > 0, 1 - np.abs(f["exec_side_imbalance"]), 0.0)
        f["aggressor_ratio"] = _safe_div(tw["aggr_vol"], exec_vol)
        f["top_cpty_share"] = _safe_div(tw["top_pair"], exec_vol)
        f["reciprocal_ratio"] = np.clip(_safe_div(tw["recip"], exec_vol), 0, 1)
        f["n_counterparties"] = tw["n_cpty"]
        mean_t = _safe_div(exec_vol, tw["n_exec"])
        f["trade_size_cv"] = np.sqrt(np.clip(_safe_div(tw["exec_q2"], tw["n_exec"]) - mean_t ** 2, 0, None)) / np.where(mean_t > 0, mean_t, 1.0)
        f["price_dev_bp"] = _safe_div(tw["dev_sum"], exec_vol)
        f["net_qty"] = tw["exec_buy"] - tw["exec_sell"]
        return f

    def _add_linked_features(self, f: pd.DataFrame, link_win: pd.DataFrame, rate_win: pd.DataFrame) -> pd.DataFrame:
        """
        Per (window, account), from the account pairs of _linked_pair_steps (A = bait, B = partner):
          linked_spoof_signature  max over the account's pairs, as bait OR as partner, of
                                  share of A's large cancels that were events with B x (1 - 1/lift)
                                  x n / (n + LINK_EVIDENCE_K)
          linked_lift_log         log1p of the largest lift
        lift = min(chance lift, concentration lift). Concentration lift = (B's events with A / all
        B's events) / (A's large cancels / all large cancels in the window). Chance lift = events /
        chance, chance = per side, A's large cancels x P(B places on the opposite
        side within +-LINK_PLACE_TICKS of a random tick) x P(B filled on that side during one of
        A's resting periods), both from B's own rates in the window. A quoting firm that places
        and trades every tick has lift ~1 with everyone; a genuine-leg account has lift >> 1 with
        its bait account. Pairs need LINK_MIN_EVENTS events in the window.
        """
        for c in LINK_COLUMNS:
            f[c] = 0.0
        if link_win is None or link_win.empty:
            return f
        # Concentration: B's events with A as a share of all B's events, against A's share of all
        # large cancels in the window. Busy accounts' events spread over every account cancelling
        # large orders (and share the market regime with them); a partner's concentrate on A.
        ev_b = link_win.groupby(["win", "cpty"])["n_ev"].sum().rename("ev_b")
        lc_all = (rate_win["lc_buy"] + rate_win["lc_sell"]).groupby(rate_win["win"]).sum().rename("lc_all")
        lw = link_win[link_win["n_ev"] >= LINK_MIN_EVENTS]
        if lw.empty:
            return f
        lw = lw.join(ev_b, on=["win", "cpty"]).join(lc_all, on="win")
        a = rate_win[["win", "trader_id", "lc_buy", "lc_sell", "life_buy", "life_sell"]]
        b = rate_win[["win", "trader_id", "pt_buy", "pt_sell", "f_buy", "f_sell"]].rename(columns={"trader_id": "cpty"})
        lw = lw.merge(a, on=["win", "trader_id"], how="left").merge(b, on=["win", "cpty"], how="left").fillna(0.0)
        wt = float(max(1, self.window_ticks))
        span = 2 * LINK_PLACE_TICKS + 1
        expected = 0.0
        for side, opp in (("buy", "sell"), ("sell", "buy")):
            n_lc = lw[f"lc_{side}"]
            p_place = np.clip(span * lw[f"pt_{opp}"] / wt, 0, 1)
            mean_life = _safe_div(lw[f"life_{side}"], n_lc)
            p_fill = np.clip(mean_life * lw[f"f_{opp}"] / wt, 0, 1)
            expected = expected + n_lc * p_place * p_fill
        n_lc_a = lw["lc_buy"] + lw["lc_sell"]
        chance_lift = _safe_div(lw["n_ev"], np.maximum(expected, 1e-6))
        concentration_lift = _safe_div(_safe_div(lw["n_ev"], lw["ev_b"]), _safe_div(n_lc_a, lw["lc_all"]))
        lift = np.minimum(chance_lift, concentration_lift)   # both: above chance AND specific to A
        share = np.clip(_safe_div(lw["n_ev"], n_lc_a), 0, 1)
        evidence = lw["n_ev"] / (lw["n_ev"] + LINK_EVIDENCE_K)
        lw = lw.assign(_score=share * (1 - 1 / np.maximum(lift, 1.0)) * evidence, _lift=np.log1p(lift))
        both = pd.concat([lw[["win", "trader_id", "_score", "_lift"]],
                          lw[["win", "cpty", "_score", "_lift"]].rename(columns={"cpty": "trader_id"})])
        best = both.groupby(["win", "trader_id"])[["_score", "_lift"]].max()
        key = pd.MultiIndex.from_frame(f[["win", "trader_id"]])
        f["linked_spoof_signature"] = best["_score"].reindex(key).fillna(0.0).to_numpy()
        f["linked_lift_log"] = best["_lift"].reindex(key).fillna(0.0).to_numpy()
        return f

    def _add_ring_features(
        self, f: pd.DataFrame, pair_win: pd.DataFrame, size_baseline_by_win: pd.Series
    ) -> pd.DataFrame:
        """
        Wash-ring features per (window, account), from volume traded with each counterparty:
          mutual_cpty_share  share of the account's volume with MUTUAL partners (each side at least
                             MUTUAL_MIN_SHARE of the other's volume, pair volume >= RING_MIN_PAIR_SIZES
                             x the pre-window median order-size baseline)
          ring_size          accounts in the connected group of mutual partners (1 = none)
          ring_volume_share  share of the account's volume traded inside that group
        Pairwise features (reciprocal_ratio, top_cpty_share) fade when washing rotates around a ring
        of five or six accounts; the group stays visible.
        """
        for c in RING_COLUMNS:
            f[c] = 1.0 if c == "ring_size" else 0.0
        if pair_win is None or pair_win.empty:
            return f
        pw = pair_win[["win", "trader_id", "cpty", "bought", "sold"]].copy()
        excl = tuple(self.exclude_prefixes)
        pw = pw[~pw["trader_id"].astype(str).str.startswith(excl) & ~pw["cpty"].astype(str).str.startswith(excl)]
        if pw.empty:
            return f
        pw["gross"] = pw["bought"] + pw["sold"]
        vol = pw.groupby(["win", "trader_id"])["gross"].transform("sum")
        pw["share"] = _safe_div(pw["gross"], vol)
        rev = pw[["win", "trader_id", "cpty", "share"]].rename(columns={"trader_id": "cpty", "cpty": "trader_id", "share": "share_rev"})
        pw = pw.merge(rev, on=["win", "trader_id", "cpty"], how="left")
        baseline = pw["win"].map(size_baseline_by_win).fillna(1.0)
        pw["mutual"] = (
            (pw["share"] >= MUTUAL_MIN_SHARE)
            & (pw["share_rev"].fillna(0) >= MUTUAL_MIN_SHARE)
            & (pw["gross"] >= RING_MIN_PAIR_SIZES * baseline)
        )
        mutual_share = pw.loc[pw["mutual"]].groupby(["win", "trader_id"])["share"].sum()

        # Connected groups of mutual partners, per window (union-find over the few mutual edges)
        edges = pw.loc[pw["mutual"], ["win", "trader_id", "cpty"]]
        comp: dict[tuple, tuple] = {}
        for win, g in edges.groupby("win"):
            parent: dict[str, str] = {}

            def find(x):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in zip(g["trader_id"], g["cpty"]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            members: dict[str, list] = {}
            for x in list(parent):
                members.setdefault(find(x), []).append(x)
            for root, ms in members.items():
                for x in ms:
                    comp[(win, x)] = (root, len(ms))
        if comp:
            cdf = pd.DataFrame([(w, t, root, n) for (w, t), (root, n) in comp.items()],
                               columns=["win", "trader_id", "ring", "ring_size"])
            pw = pw.merge(cdf[["win", "trader_id", "ring"]], on=["win", "trader_id"], how="left")
            pw = pw.merge(cdf[["win", "trader_id", "ring"]].rename(columns={"trader_id": "cpty", "ring": "ring_cpty"}),
                          on=["win", "cpty"], how="left")
            inside = pw["ring"].notna() & (pw["ring"] == pw["ring_cpty"])
            ring_share = pw.loc[inside].groupby(["win", "trader_id"])["share"].sum()
            sizes = cdf.set_index(["win", "trader_id"])["ring_size"]
        else:
            ring_share = pd.Series(dtype=float)
            sizes = pd.Series(dtype=float)
        key = pd.MultiIndex.from_frame(f[["win", "trader_id"]])
        f["mutual_cpty_share"] = mutual_share.reindex(key).fillna(0.0).clip(0, 1).to_numpy()
        f["ring_volume_share"] = ring_share.reindex(key).fillna(0.0).clip(0, 1).to_numpy()
        f["ring_size"] = sizes.reindex(key).fillna(1.0).to_numpy()
        return f

    def _add_temporal(self, f: pd.DataFrame) -> pd.DataFrame:
        """Per-account history: position build-up/reversal and activity change."""
        W = self.n_steps
        grid_wins = np.arange(f["win"].min(), f["win"].max() + 1)
        full = (f.set_index(["trader_id", "win"])[["net_qty", "exec_vol", "n_orders"]]
                .unstack("trader_id").reindex(grid_wins).fillna(0.0))
        net = full["net_qty"]
        gross = full["exec_vol"]
        activity = gross + full["n_orders"]
        prev_net = net.shift(W).fillna(0) + net.shift(2 * W).fillna(0)
        prev_gross = gross.shift(W).fillna(0) + gross.shift(2 * W).fillna(0)
        prev_ratio = pd.DataFrame(_safe_div(prev_net, prev_gross), index=net.index, columns=net.columns)
        # Position built over the current + three previous non-overlapping windows (~12 min).
        long_net = net + net.shift(W).fillna(0) + net.shift(2 * W).fillna(0) + net.shift(3 * W).fillna(0)
        long_gross = gross + gross.shift(W).fillna(0) + gross.shift(2 * W).fillna(0) + gross.shift(3 * W).fillna(0)
        long_ratio = pd.DataFrame(_safe_div(long_net, long_gross), index=net.index, columns=net.columns)
        if ACTIVITY_ROBUST:
            # Trimmed median: windows far from the account's (cleaned) baseline leave it, iteratively.
            la = np.log1p(activity)
            keep = None
            for _ in range(SELF_Z_TRIM_PASSES):
                base = activity if keep is None else activity.where(keep)
                hist = base.shift(W).expanding().median()
                new_keep = ~((la - np.log1p(hist)).abs() > ACTIVITY_TRIM_LOG)
                if keep is not None and new_keep.equals(keep):
                    break
                keep = new_keep
            hist = hist.fillna(0.0)
        else:
            hist = activity.shift(W).expanding().mean().fillna(0.0)
        act_change = np.log1p(activity) - np.log1p(hist)

        stack = lambda df, name: df.stack().rename(name)  # noqa: E731
        temporal = pd.concat([stack(prev_ratio, "prev_net_ratio"), stack(long_ratio, "long_net_ratio"),
                              stack(act_change, "activity_vs_history")], axis=1)
        temporal.index.names = ["win", "trader_id"]
        f = f.merge(temporal.reset_index(), on=["win", "trader_id"], how="left")
        cols = ["prev_net_ratio", "long_net_ratio", "activity_vs_history"]
        f[cols] = f[cols].fillna(0.0)
        f["reversal_score"] = np.clip(-f["prev_net_ratio"] * f["exec_side_imbalance"], 0, 1)

        # Coordination: how many OTHER accounts are bursting (activity well above
        # their own history) and trading in the same direction in this window.
        # Coalitions (pump & dump) light up together; independent traders do not.
        bursting = (f["activity_vs_history"] > 0.7) & (f["exec_side_imbalance"].abs() > 0.5)
        direction = np.sign(f["exec_side_imbalance"]).where(bursting, 0.0)
        n_up = (direction > 0).groupby(f["win"]).transform("sum")
        n_down = (direction < 0).groupby(f["win"]).transform("sum")
        same = np.where(direction > 0, n_up - 1, np.where(direction < 0, n_down - 1, 0))
        f["coordinated_peers"] = np.clip(same, 0, None).astype(float)
        return f.drop(columns=["net_qty"])

    def _add_self_z(self, f: pd.DataFrame) -> pd.DataFrame:
        """
        Robust z-score of each account against its own history: median and IQR of its
        values in earlier windows that do not overlap the current one (shifted by one
        full window length). 0 until SELF_Z_MIN_HISTORY past active windows exist.

        Two passes: the first marks windows where the account deviated strongly from its
        own history on any column (|z| >= SELF_Z_TRIM); the second recomputes the baseline
        without those windows. Each window is still scored; it just stops counting as
        "normal" for later windows.
        """
        W = self.n_steps
        grid_wins = np.arange(f["win"].min(), f["win"].max() + 1)
        grids = {col: f.pivot_table(index="win", columns="trader_id", values=col, aggfunc="first").reindex(grid_wins)
                 for col in SELF_Z_COLUMNS}

        def zscores(col: str, keep: pd.DataFrame | None) -> pd.DataFrame:
            grid = grids[col]
            base = grid if keep is None else grid.where(keep.reindex_like(grid).fillna(True).astype(bool))
            exp = base.shift(W).expanding(min_periods=SELF_Z_MIN_HISTORY)
            med = exp.median()
            iqr = exp.quantile(0.75) - exp.quantile(0.25)
            scale = np.maximum(iqr / 1.349, SELF_Z_SCALE_FLOOR.get(col, 0.05))
            return ((grid - med) / scale).clip(-20, 20)

        # Iterate: each pass judges windows against the baseline cleaned by the previous one.
        # A single pass is not enough for a long burst: once its first windows enter the
        # history, later ones stop looking unusual and the burst absorbs itself.
        keep = None
        for _ in range(SELF_Z_TRIM_PASSES):
            zs = [zscores(col, keep).abs() for col in SELF_Z_COLUMNS]
            unusual = zs[0].ge(SELF_Z_TRIM)
            for z1 in zs[1:]:
                unusual = unusual | z1.reindex_like(unusual).ge(SELF_Z_TRIM)
            new_keep = ~unusual
            if keep is not None and new_keep.equals(keep):
                break
            keep = new_keep
        out = []
        n_orders = f.pivot_table(index="win", columns="trader_id", values="n_orders", aggfunc="first").reindex(grid_wins)
        weight = (n_orders / (n_orders + SELF_Z_EVIDENCE_K)) if SELF_Z_EVIDENCE_K > 0 else None
        for col in SELF_Z_COLUMNS:
            z = zscores(col, keep)
            if weight is not None and col != "n_orders":
                z = z * weight.reindex_like(z).fillna(0.0)
            s = z.stack(future_stack=True).rename(f"self_z_{col}")
            out.append(s)
        sz = pd.concat(out, axis=1)
        sz.index.names = ["win", "trader_id"]
        f = f.merge(sz.reset_index(), on=["win", "trader_id"], how="left")
        cols = [f"self_z_{c}" for c in SELF_Z_COLUMNS]
        f[cols] = f[cols].fillna(0.0)
        return f

    def _add_peer_z(self, f: pd.DataFrame) -> pd.DataFrame:
        """Robust z-score of each account against all accounts active in the same window."""
        g = f.groupby("win")
        for col in PEER_Z_COLUMNS:
            med = g[col].transform("median")
            mad = (f[col] - med).abs().groupby(f["win"]).transform("median")
            scale = 1.4826 * mad
            scale = scale.where(scale > 1e-9, f[col].groupby(f["win"]).transform("std").fillna(0) + 1e-6)
            f[f"peer_z_{col}"] = ((f[col] - med) / scale).clip(-20, 20).fillna(0.0)
        return f
