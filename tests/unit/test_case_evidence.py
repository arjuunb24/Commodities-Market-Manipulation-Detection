"""Case evidence: the cancelled-volume share only counts orders placed in the flagged span."""

import pandas as pd

from src.agentic.case_evidence import _cancelled_share_pct


def test_cancelled_share_ignores_orders_placed_before_the_span():
    # o0 was placed before the span and cancelled inside it; o1/o2 were placed in the span.
    acct = pd.DataFrame([
        {"order_id": "o0", "event_type": "placed", "tick": 5, "quantity": 500},
        {"order_id": "o0", "event_type": "cancelled", "tick": 12, "quantity": 500},
        {"order_id": "o1", "event_type": "placed", "tick": 11, "quantity": 100},
        {"order_id": "o1", "event_type": "cancelled", "tick": 14, "quantity": 100},
        {"order_id": "o2", "event_type": "placed", "tick": 13, "quantity": 100},
        {"order_id": "o2", "event_type": "partial_fill", "tick": 15, "quantity": 40},
        {"order_id": "o2", "event_type": "cancelled", "tick": 25, "quantity": 60},   # after the span: still counts
    ])
    placed = acct[(acct["event_type"] == "placed") & acct["tick"].between(10, 20)]
    # o0's 500 is not in the denominator, so it must not be in the numerator either (was 330%)
    assert _cancelled_share_pct(acct, placed) == 80.0


def test_cancelled_share_never_exceeds_100():
    acct = pd.DataFrame([
        {"order_id": "o1", "event_type": "placed", "tick": 1, "quantity": 10},
        {"order_id": "o1", "event_type": "cancelled", "tick": 2, "quantity": 10},
        {"order_id": "o1", "event_type": "cancelled", "tick": 3, "quantity": 10},   # duplicate cancel event
    ])
    placed = acct[acct["event_type"] == "placed"]
    assert _cancelled_share_pct(acct, placed) == 100.0
