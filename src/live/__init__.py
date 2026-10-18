"""
src/live
========
Testing the detector against real market data.

Real public market data (exchange bars from Yahoo Finance) is anonymous: it has
prices and volumes per minute, but no orders, cancels or account ids. The
trader-level detector cannot run on it, so this package adds a second, bar-level
model trained on the simulator's 1-minute bars ("is a manipulation episode active
in this commodity in this minute, and which persona does it look like?"), plus a
mechanism checker that tests each flagged window against how the four personas
actually move a market.

  bars.py      1-minute bars from simulator logs, scale-free bar features, labels
  bar_model.py per-persona classifiers on bar features (trained each round)
  registry.py  save / load a round's models (round_N/models/)
  checker.py   rule-based mechanism checker and optional LLM reviewer
  calendar.py  scheduled market events that explain many real price moves
"""
