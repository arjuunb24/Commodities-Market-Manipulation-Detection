"""
tests/unit/test_labels.py
=========================
Activity-based ground truth: a window is positive only where the account placed
its persona's manipulative orders; camouflage, decoys and quiet accumulation are not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.detector.pipeline import ACTIVITY_TAGS, DetectorPipeline

CFG = yaml.safe_load(open("configs/detector_config.yaml"))


def _windows(trader, starts, length=180, com="gold"):
    idx = pd.MultiIndex.from_tuples([(com, s, trader) for s in starts], names=["commodity", "window_start", "trader_id"])
    return pd.DataFrame({"window_end": [s + length - 1 for s in starts]}, index=idx)


def _orders(trader, ticks_tags, com="gold"):
    return pd.DataFrame([{"order_id": f"gol1-{trader}-{i}{tag}", "trader_id": trader, "side": "buy", "price": 1.0,
                          "quantity": 1, "tick": t, "event_type": "placed", "commodity": com}
                         for i, (t, tag) in enumerate(ticks_tags)])


def _events(persona, trader, start, end, com="gold"):
    return pd.DataFrame([{"commodity": com, "persona": persona, "start_tick": start, "end_tick": end, "trader_ids": [trader]}])


def test_only_windows_with_manipulative_orders_are_positive():
    X = _windows("pnd_0", [0, 200, 400, 600, 800])
    ev = _events("pump_and_dump", "pnd_0", 0, 1000)
    # decoys and accumulation early, ignition in the 3rd window, dump in the 5th
    orders = _orders("pnd_0", [(50, "d"), (90, "a"), (250, "a"), (450, "i"), (650, "d"), (850, "u")])
    act = DetectorPipeline({**CFG, "labels": {"mode": "activity"}}).generate_labels(X, ev, orders)
    span = DetectorPipeline({**CFG, "labels": {"mode": "span"}}).generate_labels(X, ev, orders)
    assert act["pump_and_dump"].tolist() == [0, 0, 1, 0, 1]
    assert span["pump_and_dump"].tolist() == [1, 1, 1, 1, 1]


def test_wash_camouflage_is_not_a_positive():
    X = _windows("wash_0a", [0, 200])
    ev = _events("wash_trading", "wash_0a", 0, 400)
    orders = _orders("wash_0a", [(30, "c"), (40, "c"), (250, "w")])
    y = DetectorPipeline({**CFG, "labels": {"mode": "activity"}}).generate_labels(X, ev, orders)
    assert y["wash_trading"].tolist() == [0, 1]


def test_orders_outside_the_episode_or_by_other_accounts_do_not_count():
    X = pd.concat([_windows("spoofer_0", [0, 500]), _windows("noise_1", [0])])
    ev = _events("spoofing", "spoofer_0", 0, 300)
    orders = pd.concat([_orders("spoofer_0", [(100, "s"), (600, "s")]), _orders("noise_1", [(100, "s")])])
    y = DetectorPipeline({**CFG, "labels": {"mode": "activity"}}).generate_labels(X, ev, orders)
    assert y["spoofing"].tolist() == [1, 0, 0]


def test_without_an_order_log_labels_fall_back_to_the_episode_span():
    X = _windows("layerer_0", [0, 200])
    ev = _events("layering", "layerer_0", 0, 400)
    y = DetectorPipeline({**CFG, "labels": {"mode": "activity"}}).generate_labels(X, ev)
    assert y["layering"].tolist() == [1, 1]


def test_every_persona_has_activity_tags():
    assert set(CFG["personas"]) <= set(ACTIVITY_TAGS)
    assert "d" not in ACTIVITY_TAGS["pump_and_dump"] and "a" not in ACTIVITY_TAGS["pump_and_dump"]
    assert "c" not in ACTIVITY_TAGS["wash_trading"]
