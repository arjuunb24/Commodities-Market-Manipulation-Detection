"""Cross-account bait-and-switch features (linked_spoof_signature / linked_lift_log)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detector.features import FeatureEngine

CFG = {"feature_engine": {"window": "3min", "step": "30s", "exclude_trader_prefixes": ["mm_"]}}


def _market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    600 ticks. Every 15 ticks: `bait` rests a 400-lot buy that it cancels 8 ticks later, while
    `partner` places a 10-lot sell in the same tick that a third party (`taker`) lifts. `algo`
    does the same with its own large buy, but `hitter` sells straight INTO it (a counterparty,
    not a partner). A crowd of small traders trades among itself throughout.
    """
    rng = np.random.default_rng(0)
    orders, trades, n = [], [], 0

    def order(trader, side, qty, tick, ev="placed", oid=None):
        nonlocal n
        n += 1
        oid = oid or f"o{n}"
        orders.append({"order_id": oid, "commodity": "gold", "trader_id": trader, "side": side,
                       "price": 100.0, "quantity": qty, "tick": tick, "event_type": ev})
        return oid

    def trade(tick, buyer, seller, qty=10):
        trades.append({"trade_id": f"t{len(trades)}", "commodity": "gold", "tick": tick, "price": 100.0,
                       "quantity": qty, "buyer_trader_id": buyer, "seller_trader_id": seller,
                       "aggressor_side": "buy"})

    for t in range(0, 600):
        a, b = rng.choice(40, 2, replace=False)
        order(f"crowd_{a}", "buy", 10, t)
        order(f"crowd_{b}", "sell", 10, t)
        trade(t, f"crowd_{a}", f"crowd_{b}")
    for t in range(5, 580, 15):
        bait = order("bait", "buy", 400, t)
        order("partner", "sell", 10, t)
        trade(t + 3, "taker", "partner")
        order("bait", "buy", 400, t + 8, "cancelled", oid=bait)
        big = order("algo", "buy", 400, t + 4)
        order("hitter", "sell", 10, t + 4)
        trade(t + 6, "algo", "hitter")
        order("algo", "buy", 400, t + 11, "cancelled", oid=big)
    o = pd.DataFrame(orders)
    o["quantity"] = o["quantity"].astype("Int64")
    return pd.DataFrame(trades), o


def test_linked_signature_marks_both_colluding_accounts_only():
    trades, orders = _market()
    X = FeatureEngine(CFG).generate_features(trades, orders)
    best = X.groupby(level="trader_id")["linked_spoof_signature"].max()
    assert best["bait"] > 0.2 and best["partner"] > 0.2
    # Filled against the large order itself = a counterparty, not a partner.
    assert best.get("hitter", 0.0) < 0.1 and best.get("algo", 0.0) < 0.1
    assert best[[t for t in best.index if t.startswith("crowd_")]].max() < 0.1
    assert X.groupby(level="trader_id")["linked_lift_log"].max()["bait"] > 1.0
