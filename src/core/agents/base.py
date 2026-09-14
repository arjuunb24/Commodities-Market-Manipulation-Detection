"""
src/core/agents/base.py
========================
Abstract base class for all agents in the Veridex ABM.

CONTRACT (Implementation Plan Section 5.3):
  - __init__(self, trader_id: str, rng: np.random.Generator)
    The rng must be a fully independent seeded generator obtained via
    master_rng.spawn(1)[0] in the simulation loop. Never use numpy's global
    random state.
  - decide(book_state: BookState, tick: int) -> Order | None
    Pure function: reads book_state, returns one Order (or None if the agent
    decides not to act this tick). Must never modify any shared mutable state.
    Must never raise exceptions for normal operating conditions (e.g., quiet
    market, empty book). Raises are reserved for programming errors only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.order_book import BookState, Order


class Agent(ABC):
    """
    Abstract base class for all Veridex simulation agents.

    Agents are pure decision-makers: they receive a read-only BookState snapshot
    and return an Order instruction (or None). They do NOT submit orders directly
    to the book — the simulation loop collects agent decisions and dispatches them.

    Args:
        trader_id: Unique string identifier for this agent instance.
        rng: An independent seeded numpy.random.Generator (never the global RNG).
             Each agent instance must receive its own distinct generator to ensure
             byte-level reproducibility across simulation runs (SRS FR-0.4).
    """

    def __init__(self, trader_id: str, rng: np.random.Generator) -> None:
        self.trader_id = trader_id
        self.rng = rng

    @abstractmethod
    def decide(self, book_state: BookState, tick: int) -> Order | None:
        """
        Decide what (if anything) to do this tick.

        Args:
            book_state: Read-only snapshot of the current order book state.
            tick: Current simulation tick.

        Returns:
            An Order to submit, or None if the agent decides not to act.

        Notes:
          - This method MUST be a pure function of (book_state, tick, self.rng).
            No other mutable state or global state should be read or written here.
          - Returning None is always valid (agent inactive this tick).
          - The returned Order must have price > 0 and quantity > 0.
            Implementations must enforce this before returning.
          - Empty order book (best_bid=None, best_ask=None): the agent must
            handle this gracefully without raising (return None if unsure).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(trader_id={self.trader_id!r})"
