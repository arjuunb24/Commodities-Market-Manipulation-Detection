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

  Spoofing     spoof_signature (share of the account's large cancelled orders
               during whose life it was filled on the opposite side),
               cancel_ratio, fast_cancel_ratio, mean_cancel_lifetime,
               large_order_share, large_cancel_ratio, max_order_size_rel,
               passive_distance_bp, cancel_exec_opposition (cancels on one
               side while executing on the other — the Coscia/Sarao signature)
  Wash trading reciprocal_ratio (volume traded back and forth with the same
               counterparty), top_cpty_share, net_flatness, price_dev_bp,
               trade_size_cv
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
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAST_CANCEL_TICKS = 10        # cancelled within 10 s of placement
LARGE_ORDER_MULTIPLE = 4.0    # ≥ 4x the commodity's median order size
PEER_Z_COLUMNS = [
    "cancel_ratio", "fast_cancel_ratio", "avg_order_size_rel", "max_order_size_rel",
    "gross_volume_share", "reciprocal_ratio", "aggressor_ratio", "large_order_share",
]
MARKET_CONTEXT_COLUMNS = [
    "mkt_return_bp", "mkt_volatility_bp", "mkt_range_bp", "mkt_volume_spike",
    "mkt_trade_imbalance", "mkt_order_imbalance", "mkt_cancel_ratio", "mkt_abnormal_return_z",
]


def _ticks(value: Any) -> int:
    """'3min' / '30s' / 180 → ticks (1 tick = 1 second)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(pd.to_timedelta(value).total_seconds())


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
        step_px["last"] = step_px["last"].ffill().bfill()
        step_px["vwap"] = pd.Series(_safe_div(step_px["notional"], step_px["vol"], np.nan), index=steps).ffill().bfill()
        step_px["ret"] = np.log(step_px["last"]).diff().fillna(0.0)
        ref_price = step_px["last"].shift(1).fillna(step_px["last"])  # reference known before the step

        # ---------------- order-level enrichment ----------------
        placed = orders[orders["event_type"] == "placed"][["order_id", "trader_id", "side", "price", "quantity", "tick", "step"]].copy()
        placed["quantity"] = placed["quantity"].astype(float)
        median_size = float(placed["quantity"].median()) if not placed.empty else 1.0
        median_size = max(median_size, 1.0)
        placed["is_large"] = placed["quantity"] >= LARGE_ORDER_MULTIPLE * median_size
        placed["ref"] = placed["step"].map(ref_price)
        sign = np.where(placed["side"] == "buy", 1.0, -1.0)
        placed["dist_bp"] = sign * (placed["ref"] - placed["price"]) / placed["ref"] * 1e4
        placed["buy_q"] = np.where(placed["side"] == "buy", placed["quantity"], 0.0)
        placed["sell_q"] = placed["quantity"] - placed["buy_q"]
        placed["q2"] = placed["quantity"] ** 2
        placed["large_q"] = np.where(placed["is_large"], placed["quantity"], 0.0)
        placed["dist_q"] = placed["dist_bp"] * placed["quantity"]

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

        # ---------------- trader per-step aggregates ----------------
        p_step = placed.groupby(["step", "trader_id"]).agg(
            n_placed=("quantity", "size"), placed_vol=("quantity", "sum"),
            placed_buy=("buy_q", "sum"), placed_sell=("sell_q", "sum"),
            placed_q2=("q2", "sum"), placed_max=("quantity", "max"),
            large_vol=("large_q", "sum"), dist_q=("dist_q", "sum"),
        ).reset_index()
        c_step = cancels.groupby(["step", "trader_id"]).agg(
            n_cancel=("quantity", "size"), cancel_vol=("quantity", "sum"),
            cancel_buy=("c_buy", "sum"), cancel_sell=("c_sell", "sum"),
            fast_cancel=("fast_q", "sum"), large_cancel=("large_c", "sum"),
            lifetime_sum=("lifetime", "sum"),
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
            fills = legs_with_tick[["trader_id", "tick", "buy", "sell"]].rename(columns={"tick": "fill_tick"})
            j = lc.merge(fills, on="trader_id", how="inner")
            j = j[(j["fill_tick"] >= j["placed_tick"]) & (j["fill_tick"] <= j["tick"] + 2)]
            j = j[((j["side"] == "buy") & (j["sell"] > 0)) | ((j["side"] == "sell") & (j["buy"] > 0))]
            lc.loc[lc["cid"].isin(j["cid"].unique()), "opp_fill"] = 1.0
        lc_step = lc.groupby(["step", "trader_id"]).agg(
            n_large_cancel=("cid", "size"), n_large_cancel_opp=("opp_fill", "sum"),
        ).reset_index()

        # ---------------- roll into windows ----------------
        p_win = _to_windows(p_step, ["trader_id"],
                            ["n_placed", "placed_vol", "placed_buy", "placed_sell", "placed_q2", "large_vol", "dist_q"],
                            ["placed_max"], W, first_step, last_step)
        c_win = _to_windows(c_step, ["trader_id"],
                            ["n_cancel", "cancel_vol", "cancel_buy", "cancel_sell", "fast_cancel", "large_cancel", "lifetime_sum"],
                            [], W, first_step, last_step)
        t_win = _to_windows(t_step, ["trader_id"],
                            ["n_exec", "exec_buy", "exec_sell", "aggr_vol", "dev_sum", "exec_q2"],
                            [], W, first_step, last_step)
        pair_win = _to_windows(pair_step, ["trader_id", "cpty"], ["bought", "sold"], [], W, first_step, last_step)
        lc_win = _to_windows(lc_step, ["trader_id"], ["n_large_cancel", "n_large_cancel_opp"], [], W, first_step, last_step)

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

        feats = self._trader_features(tw, median_size)
        feats = self._add_temporal(feats)
        feats = feats.merge(market[["win", *MARKET_CONTEXT_COLUMNS, "mkt_volume"]], on="win", how="left")
        feats["gross_volume_share"] = _safe_div(feats["exec_vol"], feats["mkt_volume"])
        feats["signed_flow_return"] = feats["exec_side_imbalance"] * feats["mkt_return_bp"]
        feats = self._add_peer_z(feats)

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

    def _trader_features(self, tw: pd.DataFrame, median_size: float) -> pd.DataFrame:
        f = tw[["win", "trader_id"]].copy()
        exec_vol = tw["exec_buy"] + tw["exec_sell"]
        f["n_orders"] = tw["n_placed"]
        f["n_trades"] = tw["n_exec"]
        f["exec_vol"] = exec_vol
        avg_size = _safe_div(tw["placed_vol"], tw["n_placed"])
        f["avg_order_size_rel"] = avg_size / median_size
        f["max_order_size_rel"] = tw["placed_max"] / median_size
        f["order_size_cv"] = np.sqrt(np.clip(_safe_div(tw["placed_q2"], tw["n_placed"]) - avg_size ** 2, 0, None)) / np.where(avg_size > 0, avg_size, 1.0)
        f["large_order_share"] = _safe_div(tw["large_vol"], tw["placed_vol"])
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
