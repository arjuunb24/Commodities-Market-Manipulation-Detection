Commodities Market Manipulation Detection — Implementation Plan

Agent-Based Simulation + Detector + Adversarial Loop + Agentic Intelligence Layer

Final-Year B.Tech Data Science Project — Build Document v2.0

Table of Contents

**Document purpose.** This is the engineering build companion to the Project Specification (_Commodities_ABM_Project_Plan.docx_). That document answers _why_ the architecture is what it is. This document answers _how to actually build it_ — repo layout, environment setup, module-by-module implementation detail down to class/function signatures, dataset sourcing steps, testing strategy, and a week-by-week execution checklist written so it can be handed directly to an agentic coding tool (e.g., Antigravity) as a work order.

**Companion document.** _Software Requirements Specification (SRS) v2.0_ — covers functional/non-functional requirements, UI/UX specification, and edge-case handling in formal requirements form. Read both together: this document tells you what to build and in what order; the SRS tells you exactly what "done" and "correct" mean for each piece.

# 0\. What's New in This Revision — LLM & Agentic AI Layer

The original specification is architecturally sound and deliberately keeps the price-generation mechanism free of any learned/generative component — that is the entire justification for choosing an Agent-Based Model over a GAN (Section 2 of the spec). **Nothing in this revision touches that guarantee.** No LLM ever generates a price, a trade, or an order. The additions sit strictly downstream of the deterministic simulation, in three places where language reasoning adds real product value without compromising the mechanism-first argument that is the project's core defensibility claim.

This is introduced as **Module 4 — LLM & Agentic Intelligence Layer**, with three components:

| Component                                   | What it does                                                                                                                                                                                                                                                                                                                                                                                                    | Why it's genuinely valuable, not decoration                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **4A — Explainability Narrative Generator** | Converts each flagged window's SHAP values + trade context into a plain-English compliance narrative ("this window was flagged primarily due to a 340% spike in order cancellation rate for trader T-118, consistent with spoofing")                                                                                                                                                                            | Real surveillance desks don't read SHAP bar charts — they read case narratives. This is the difference between a model output and a usable compliance artifact, and it's a strong, concrete "I made ML outputs usable for a non-technical stakeholder" talking point for interviews.                                                                                                        |
| **4B — Agentic Adversarial Strategist**     | Replaces/augments the manually-authored Section 5.2 mutation list with an LLM-driven agent that reads the current round's SHAP feature-importance drift and performance table, reasons about _why_ the detector is catching a persona, and proposes the next round's mutation parameters — then the loop actually runs the simulation, measures the result, and feeds that back to the agent for the next round | This is genuinely **agentic**: multi-step, tool-using, feedback-driven, not a single prompt-completion. It converts Section 5.4's "optional automated mutation search" from blind random/genetic search into a reasoned search, which is a stronger research claim and a better viva story ("the agent read what the detector was keying on and specifically tried to break that feature"). |
| **4C — Conversational Analyst Assistant**   | A chat panel on the dashboard where an analyst can ask questions in natural language ("why was trader T-118 flagged in round 3?", "which persona is hardest to catch after mutation?") and get answers grounded strictly in the logged data via function-calling (never free-generation of facts)                                                                                                               | Turns the dashboard from a set of charts into an actual analyst tool, and demonstrates function-calling / tool-use architecture, which is directly relevant to ML/AI engineering interview questions in 2026.                                                                                                                                                                               |

**Both 4A and 4C are load-bearing for correctness in one specific way: they must never be allowed to state a fact that isn't in the logs.** Section 7 of this document specifies the grounding/guardrail design that prevents hallucinated numbers from reaching an analyst. This is treated as a first-class requirement, not an afterthought — see SRS Section 5.4 (Security) and Section 5.8 (Compliance/Ethical).

**Scope discipline.** Module 4 is built _after_ Modules 1–3 are working end-to-end, exactly like the optional RL enhancement in Section 8 of the spec. If time runs out, the project is still complete and defensible without it — Modules 1–3 alone satisfy the original brief. Module 4 is what pushes the project from "very good final-year project" to "portfolio piece with a genuine AI engineering story," which matters directly for the ML/AI Engineer roles being targeted.

# 1\. System Architecture Overview

┌─────────────────────────────────────────────────────────────────────────────┐  
│ CALIBRATION LAYER (offline, run once per calibration cycle) │  
│ yfinance OHLCV ──► calibration notebook ──► agent_config.yaml (targets) │  
│ CFTC CoT / enforcement text ──► literature grounding notes │  
└───────────────────────────────┬───────────────────────────────────────────────┘  
▼  
┌─────────────────────────────────────────────────────────────────────────────┐  
│ MODULE 1 — Agent-Based Market Simulator (core.simulation) │  
│ OrderBook ◄──decide()── \[NoiseTrader|Momentum|MeanReversion|MarketMaker\] │  
│ ▲ \[SpoofingAgent|WashTradingPair| │  
│ │ PumpDumpCoalition|LayeringAgent\] │  
│ └──emits──► trade_log.csv · order_log.csv · manipulation_events.csv │  
│ · ohlcv_bars.csv │  
└───────────────────────────────┬───────────────────────────────────────────────┘  
▼  
┌─────────────────────────────────────────────────────────────────────────────┐  
│ MODULE 2 — Detector (detector.\*) │  
│ FeatureEngine (rolling window, per-trader) ──► feature_table.parquet │  
│ ├──► IsolationForestDetector (unsupervised) │  
│ └──► XGBoostDetector + SHAP (supervised, per manipulation type) │  
│ EnsembleScorer ──► flags.parquet (window, trader_id, score, flag, why) │  
└───────────────────────────────┬───────────────────────────────────────────────┘  
▼  
┌─────────────────────────────────────────────────────────────────────────────┐  
│ MODULE 3 — Adversarial Evaluation Loop (adversarial.\*) │  
│ RoundOrchestrator: for round in rounds: │  
│ mutate personas → simulate (Module 1) → score with frozen detector │  
│ (pre-retrain) → retrain (Module 2) → score again (post-retrain) │  
│ → log round_metrics.parquet, shap_drift.parquet │  
└───────────────────────────────┬───────────────────────────────────────────────┘  
▼  
┌─────────────────────────────────────────────────────────────────────────────┐  
│ MODULE 4 — LLM & Agentic Intelligence Layer (NEW) (agentic.\*) │  
│ 4A NarrativeGenerator(flags.parquet, shap) ──► case_narratives.jsonl │  
│ 4B AdversarialStrategist(round_metrics, shap_drift) ──\[tool calls\]──► │  
│ proposes mutation params ──► fed back into Module 3 orchestrator │  
│ 4C AnalystAssistant(chat) ──\[function calling over the parquet logs\]──► │  
│ grounded natural-language answers │  
└───────────────────────────────┬───────────────────────────────────────────────┘  
▼  
┌─────────────────────────────────────────────────────────────────────────────┐  
│ PRESENTATION — Streamlit Dashboard (dashboard/\*) │  
│ Live price+manipulation chart · Round performance curves · Case list │  
│ with narratives (4A) · Chat panel (4C) · Price-impact-avoided summary │  
└─────────────────────────────────────────────────────────────────────────────┘

**Design rule that governs every interface above:** every arrow crossing a module boundary is a file on disk (Parquet/CSV) with a frozen schema, never an in-memory object passed between modules. This is what lets three people build independently after Week 1–2 (spec Section 9), lets Module 4 be added without touching Modules 1–3, and lets the whole pipeline be replayed/debugged step by step — you can always re-run Module 2 alone against a saved Module 1 output while iterating.

# 2\. Repository Structure

commodities-manipulation-detection/  
├── .env.example # template for secrets (LLM API key etc.) — never commit .env  
├── .gitignore  
├── pyproject.toml # or requirements.txt + setup.cfg  
├── README.md  
├── docker-compose.yml  
├── Dockerfile  
├── configs/  
│ ├── agent_config.yaml # calibrated agent parameters (Module 1)  
│ ├── persona_config.yaml # manipulation persona parameters, per round  
│ ├── detector_config.yaml # model hyperparameters, feature window sizes  
│ └── llm_config.yaml # model name, temperature, tool schemas (Module 4)  
├── data/  
│ ├── raw/ # yfinance pulls, CFTC CSVs (gitignored, fetched by script)  
│ ├── calibration/ # calibration_report.ipynb outputs  
│ └── runs/  
│ └── &lt;run_id&gt;/ # every simulation run gets its own folder  
│ ├── trade_log.parquet  
│ ├── order_log.parquet  
│ ├── manipulation_events.parquet  
│ └── ohlcv_bars.parquet  
├── src/  
│ ├── core/  
│ │ ├── order_book.py # OrderBook, Order, matching engine  
│ │ ├── agents/  
│ │ │ ├── base.py # Agent ABC, decide(book_state) interface  
│ │ │ ├── noise_trader.py  
│ │ │ ├── momentum_trader.py  
│ │ │ ├── mean_reversion_trader.py  
│ │ │ ├── market_maker.py  
│ │ │ └── manipulation/  
│ │ │ ├── spoofing.py  
│ │ │ ├── wash_trading.py  
│ │ │ ├── pump_and_dump.py  
│ │ │ └── layering.py  
│ │ ├── simulation.py # SimulationLoop orchestration (Section 3.5 of spec)  
│ │ └── calibration.py # OU-process fit, yfinance pull, target estimation  
│ ├── detector/  
│ │ ├── features.py # FeatureEngine — all Section 4.1 features  
│ │ ├── unsupervised.py # IsolationForestDetector  
│ │ ├── supervised.py # XGBoostDetector, per-persona training  
│ │ ├── ensemble.py # EnsembleScorer, combination rule  
│ │ └── explain.py # SHAP computation wrapper  
│ ├── adversarial/  
│ │ ├── mutation_library.py # hand-authored mutation steps (Section 5.2)  
│ │ ├── orchestrator.py # RoundOrchestrator (Section 5.3)  
│ │ └── metrics.py # precision/recall/F1/FPR/latency/price-impact  
│ ├── agentic/ # MODULE 4 — new  
│ │ ├── llm_client.py # thin wrapper over Anthropic SDK, retries, cost logging  
│ │ ├── narrative_generator.py # 4A  
│ │ ├── adversarial_strategist.py # 4B  
│ │ ├── analyst_assistant.py # 4C  
│ │ ├── tools.py # function-calling tool definitions + implementations  
│ │ └── prompts/ # versioned prompt templates (.txt / .jinja)  
│ └── utils/  
│ ├── schemas.py # pydantic/pandera schemas for every log file  
│ ├── logging_config.py  
│ └── io.py # parquet read/write helpers with schema validation  
├── dashboard/  
│ ├── app.py # Streamlit entrypoint  
│ └── pages/  
│ ├── 1_live_market.py  
│ ├── 2_round_performance.py  
│ ├── 3_case_review.py # includes 4A narratives  
│ └── 4_analyst_chat.py # 4C  
├── notebooks/  
│ ├── 01_calibration.ipynb  
│ ├── 02_sanity_check_plots.ipynb  
│ └── 03_final_results.ipynb  
├── tests/  
│ ├── unit/ # one file per module above, mirrored structure  
│ ├── integration/ # end-to-end: run 500-tick sim → detector → check shapes  
│ └── property/ # invariant tests, see Section 8.3  
└── scripts/  
├── fetch_calibration_data.py  
├── run_simulation.py  
├── run_round.py  
├── run_full_pipeline.py # Weeks 7-8 full adversarial loop, resumable  
└── seed_llm_cache.py # pre-warms deterministic responses for offline demo

**Why this layout matters for an agentic coding tool.** Each leaf module (order_book.py, features.py, narrative_generator.py, …) is scoped to one class or one tight cluster of functions with a fully specified interface (Section 4 below gives signatures). This means Antigravity (or any agentic coder) can be pointed at one file at a time with its interface contract and the relevant SRS functional requirements as context, without needing the whole codebase in context — the Parquet-file boundaries between modules make this safe.

# 3\. Environment Setup

## 3.1 Prerequisites

- Python 3.11 (3.10–3.12 acceptable; avoid 3.13 until XGBoost/SHAP wheel support is confirmed)
- Git
- Docker Desktop (for the deployment step, Week 10)
- An Anthropic API key (for Module 4) — <https://console.anthropic.com> — free tier is sufficient for development; budget real spend only for the final adversarial-loop runs (see cost note, Section 7.5)

## 3.2 Virtual Environment

\# clone and enter  
git clone &lt;your-repo-url&gt; commodities-manipulation-detection  
cd commodities-manipulation-detection  
<br/>\# create and activate a venv (do this per-machine, never commit .venv)  
python3.11 -m venv .venv  
source .venv/bin/activate # Windows: .venv\\Scripts\\activate  
<br/>\# upgrade the basics before installing anything heavy  
pip install --upgrade pip setuptools wheel

## 3.3 requirements.txt (pin major versions; let patch versions float)

\# --- core simulation ---  
numpy>=1.26,<2.0  
pandas>=2.1  
scipy>=1.11  
<br/>\# --- calibration data ---  
yfinance>=0.2.40  
requests>=2.31  
<br/>\# --- detector ---  
scikit-learn>=1.4  
xgboost>=2.0  
shap>=0.45  
lightgbm>=4.3 # optional alternative to XGBoost, per spec Section 4.2  
<br/>\# --- adversarial orchestration / experiment tracking ---  
mlflow>=2.12  
<br/>\# --- LLM / agentic layer (Module 4) ---  
anthropic>=0.34  
pydantic>=2.6  
jinja2>=3.1  
tenacity>=8.2 # retry/backoff for LLM calls  
<br/>\# --- dashboard ---  
streamlit>=1.35  
plotly>=5.22  
altair>=5.3  
<br/>\# --- storage / schema validation ---  
pyarrow>=15.0  
pandera>=0.19  
<br/>\# --- testing ---  
pytest>=8.0  
pytest-cov>=5.0  
hypothesis>=6.100 # property-based testing, Section 8.3  
<br/>\# --- optional Phase 2 (RL stretch goal, spec Section 8) ---  
stable-baselines3>=2.3  
torch>=2.2  
gymnasium>=0.29  
<br/>\# --- dev quality ---  
black>=24.0  
ruff>=0.4  
mypy>=1.10  
python-dotenv>=1.0

pip install -r requirements.txt

If pip install --break-system-packages is needed inside a container that isn't the venv, use it there only — never on the host venv, where it's unnecessary and a sign something is wrong with the venv activation.

## 3.4 Secrets Management

cp .env.example .env

.env contents:

ANTHROPIC_API_KEY=sk-ant-...  
MLFLOW_TRACKING_URI=./mlruns  
LOG_LEVEL=INFO

.env is gitignored. src/agentic/llm_client.py loads it via python-dotenv at import time and raises a clear, actionable error (not a bare KeyError) if ANTHROPIC_API_KEY is missing — this matters because a missing key must fail fast and loudly, not silently degrade the narrative generator into blank text (see edge case table, Section 8.4).

## 3.5 Verifying the Environment

python -c "import numpy, pandas, sklearn, xgboost, shap, streamlit, anthropic; print('OK')"  
pytest tests/unit -q # should pass with zero tests initially, then grow with each module

# 4\. Dataset Sourcing — Step by Step

All sourcing described in spec Section 7. This section makes it concrete and executable.

## 4.1 yfinance Calibration Pull (scripts/fetch_calibration_data.py)

import yfinance as yf  
import pandas as pd  
from pathlib import Path  
<br/>TICKERS = {  
"gold": "GC=F",  
"crude_oil": "CL=F",  
"wheat": "ZW=F",  
}  
OUT_DIR = Path("data/raw/yfinance")  
OUT_DIR.mkdir(parents=True, exist_ok=True)  
<br/>def fetch_all(period="5y", interval="1d"):  
for name, ticker in TICKERS.items():  
df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)  
if df.empty:  
raise RuntimeError(f"yfinance returned no data for {ticker} — check ticker validity or rate limiting")  
df.to_parquet(OUT_DIR / f"{name}.parquet")  
print(f"{name} ({ticker}): {len(df)} rows, {df.index.min()} to {df.index.max()}")  
<br/>if \__name__== "\__main_\_":  
fetch_all()

**Edge cases to handle here (not optional — this script runs unattended in Week 1–2):** - yfinance occasionally returns an empty frame on rate limiting rather than raising — the script must check df.empty and raise, not silently proceed with zero rows. - Futures contracts roll (GC=F etc. are continuous front-month series from Yahoo, which already handles rollover, but verify there are no multi-day price gaps > 15% that would indicate a bad roll — flag these in the calibration notebook rather than silently including them in volatility estimates). - Missing trading days (holidays) — do not forward-fill before computing log returns; drop and note the gap.

## 4.2 CFTC Commitments of Traders (CoT)

- Source: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm> — "Legacy" report, futures-only, for the same three commodities.
- Format: fixed-width or CSV, downloadable directly, no API key required.
- Used only for: estimating the ratio of commercial (hedger) vs. non-commercial (speculative) positioning, which informs the ratio of MeanReversionTrader : MomentumTrader population sizes in agent_config.yaml — this is a modelling _input_, not data the detector ever sees.

## 4.3 CFTC Enforcement Action Summaries

- Source: <https://www.cftc.gov/LawRegulation/Enforcement/EnforcementActions.htm> — searchable by keyword ("spoofing", "wash trading").
- These are qualitative — read 4–6 case summaries per manipulation type, extract the described _tactics_ (order sizing, timing, account structuring) and record them as short notes in docs/enforcement_case_notes.md. This becomes the literature citation for Section 3.4/5.2 of the spec and is also fed as grounding context into Module 4B's mutation-proposal prompts (Section 7.3).

## 4.4 (Optional appendix credibility check) Kaggle Fraud Datasets

- Kaggle Credit Card Fraud (mlg-ulb/creditcardfraud) or IEEE-CIS Fraud Detection.
- Download via kaggle datasets download -d mlg-ulb/creditcardfraud (requires a free Kaggle API token in ~/.kaggle/kaggle.json).
- Used exactly once, in an appendix notebook, to show the feature-engineering _style_ (timing irregularity, concentration ratios) transfers to a real labelled fraud problem. This is never merged into the main training pipeline.

## 4.5 Data Provenance Rule

Every file under data/raw/ must be reproducible by re-running its fetch script — nothing under data/raw/ is ever hand-edited or committed to git. data/runs/&lt;run_id&gt;/ (Module 1 output) is the only data that is "real" for the project's own purposes; everything under data/raw/ exists solely to produce the numbers in configs/agent_config.yaml.

# 5\. Module 1 — Simulator Implementation Detail

## 5.1 Core Interfaces

\# src/core/order_book.py  
from dataclasses import dataclass  
from enum import Enum  
from typing import Literal  
<br/>class Side(str, Enum):  
BUY = "buy"  
SELL = "sell"  
<br/>@dataclass  
class Order:  
order_id: str  
trader_id: str  
side: Side  
price: float  
quantity: int  
timestamp: int  
order_type: Literal\["limit", "cancel"\]  
<br/>class OrderBook:  
def \__init_\_(self) -> None: ...  
def submit(self, order: Order) -> list\["Trade"\]:  
"""Processes one order/cancel. Returns list of Trade objects generated  
(empty if the order rested or the cancel found nothing to remove).  
Must be idempotent-safe against cancel of an already-filled order_id  
(no-op, logged as a rejected_cancel event, never an exception)."""  
def best_bid(self) -> float | None: ...  
def best_ask(self) -> float | None: ...  
def book_state(self, depth: int = 5) -> "BookState":  
"""Snapshot used as the input to every agent's decide() call."""

\# src/core/agents/base.py  
from abc import ABC, abstractmethod  
<br/>class Agent(ABC):  
def \__init_\_(self, trader_id: str, rng: "numpy.random.Generator") -> None: ...  
<br/>@abstractmethod  
def decide(self, book_state: "BookState", tick: int) -> "Order | None":  
"""Pure function of (own internal state, book_state, tick). Must not  
read any global mutable state. Returning None = agent does nothing  
this activation. Every concrete agent owns its RNG instance (passed  
in at construction) so simulation runs are reproducible given a seed."""

**Why the RNG-per-agent and pure-decide() rules are non-negotiable requirements, not style preferences:** the entire calibration story (spec Section 3.4) and the adversarial loop's "replay from any point" claim (spec Section 6.2, price-impact-avoided) depend on simulations being exactly reproducible from a seed. A shared global RNG or hidden mutable state breaks both.

## 5.2 Matching Algorithm — Edge Cases the Engine Must Handle

| Edge case                                                                                                              | Required behaviour                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Order crosses multiple resting orders (large incoming order eats several price levels)                                 | Walk the book level by level, filling at each resting order's price until the incoming quantity is exhausted or the book is empty on that side; log one Trade per resting order consumed, not one aggregated trade.                                                                               |
| Incoming order's price exactly equals the best resting price on the _same_ side (a buy at the current best bid)        | It rests behind existing orders at that price level (time priority), it does not match against same-side orders.                                                                                                                                                                                  |
| Cancel referencing an order_id that has already fully filled or was already cancelled                                  | No-op; log an event_type="rejected_cancel" row; never raise.                                                                                                                                                                                                                                      |
| Zero-quantity or negative-quantity order reaches the book (agent bug)                                                  | Reject at the book boundary with a ValueError in dev/test mode; in a full simulation run, log and drop rather than crash the whole run — a single agent bug should not lose hours of simulation.                                                                                                  |
| Two orders at the same price, same timestamp (tick collision)                                                          | Break ties by insertion order into that tick's activation queue — the queue itself must be deterministic given the RNG seed used to sample which agents activate.                                                                                                                                 |
| Market maker's own resting orders getting crossed by its _own_ new quote update                                        | Must cancel-and-replace atomically (both operations logged as one event_type="requote" pair) — never allow the book to briefly show two resting quotes from the same market maker on the same side.                                                                                               |
| Self-trade (an agent's buy crosses its own resting sell — should only be possible for the wash-trading pair by design) | For every agent _except_ the wash-trading persona, self-trade must be prevented at the agent-logic level (an agent should never route around its own resting order). For the wash-trading persona, self-crossing across its two colluding trader_ids is the entire mechanism and must be allowed. |
| Simulation run with zero manipulation personas configured                                                              | Must run and produce a clean manipulation_events.csv with zero rows (valid empty file, not a missing file) — this is the "clean baseline" run needed for the FPR-on-legitimate-only-data check in Module 2.                                                                                       |

## 5.3 Simulation Loop Skeleton

\# src/core/simulation.py  
def run_simulation(config: "SimConfig", seed: int) -> "SimulationOutput":  
rng = np.random.default_rng(seed)  
book = OrderBook()  
agents = instantiate_agents(config, rng) # noise/momentum/reversion/MM + personas  
fundamental_value = FundamentalValueProcess(config.fv_drift, rng)  
<br/>trade_log, order_log, manip_events = \[\], \[\], \[\]  
<br/>for tick in range(config.n_ticks):  
fundamental_value.step()  
active_agents = sample_activations(agents, rng, tick) # Poisson arrivals, spec 3.2  
for agent in active_agents:  
if isinstance(agent, ManipulationAgent) and not agent.is_active_window(tick):  
continue  
order = agent.decide(book.book_state(), tick)  
if order is None:  
continue  
trades = book.submit(order)  
trade_log.extend(trades)  
order_log.append(OrderEvent.from_order(order, tick))  
if isinstance(agent, ManipulationAgent):  
manip_events.append(agent.log_activity(tick))  
<br/>return SimulationOutput(  
trade_log=to_dataframe(trade_log),  
order_log=to_dataframe(order_log),  
manipulation_events=to_dataframe(manip_events),  
ohlcv_bars=resample_to_bars(trade_log, bar="1min"),  
)

run_simulation must be resumable/replayable in a specific sense required by Section 6.2's price-impact-avoided metric: given the same seed and config up to tick T, and a _modified_ agent population from tick T+1 onward (e.g., manipulation persona removed), the output up to T must be byte-identical. This is what makes the counterfactual replay valid — it is a hard functional requirement, not a nice-to-have (see SRS FR-6.2).

## 5.4 Persistence & Schema Enforcement

Every DataFrame written to data/runs/&lt;run_id&gt;/ is validated against a pandera schema in src/utils/schemas.py _before_ being written to Parquet. A schema violation (wrong dtype, an unexpected NaN in price, a trader_id referenced in trade_log that was never issued by instantiate_agents) fails the run loudly at write time — this is the single most important guardrail against Module 2 silently training on malformed data, which is a failure mode that is very hard to debug after the fact (see Risk Register, Section 11).

# 6\. Module 2 — Detector Implementation Detail

## 6.1 Feature Engine

\# src/detector/features.py  
class FeatureEngine:  
def \__init_\_(self, window: str = "5min", step: str = "1min") -> None: ...  
<br/>def compute(self, order_log: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:  
"""Returns one row per (window_start, trader_id) with columns:  
cancel_rate, volume_share_top_k, net_position_gross_ratio,  
price_volume_divergence, bid_ask_imbalance_spike, order_size_variance.  
Windows are rolling with \`step\` stride; the \*last\* window of a run  
that is shorter than \`window\` is dropped, not zero-padded (a  
truncated window has a genuinely different denominator and would  
bias the cancel_rate feature upward)."""

**Feature edge cases:**

| Feature                  | Edge case                                                                                | Handling rule                                                                                                                                                                                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cancel_rate              | trader placed 0 orders in the window                                                     | Undefined (0/0) — emit NaN, not 0. A trader with zero activity is not "zero cancellation rate," it's "no signal," and the detector must treat these differently (XGBoost handles NaN natively; IsolationForest requires an explicit is_active flag feature alongside it). |
| volume_share_top_k       | fewer than k active traders in the window                                                | Use min(k, n_active_traders); document k=5 as default in detector_config.yaml.                                                                                                                                                                                            |
| net_position_gross_ratio | trader's gross volume is 0                                                               | Same NaN rule as above.                                                                                                                                                                                                                                                   |
| price_volume_divergence  | window has zero trades (illiquid patch)                                                  | Emit NaN for the whole window-market-level row, and exclude from the unsupervised model's training set for that window (do not impute — an illiquid window is informative on its own and shouldn't be smoothed away).                                                     |
| bid_ask_imbalance_spike  | book was empty on one side for the whole window (extreme, near-zero-liquidity edge case) | Treat resting volume on the empty side as 0, not NaN; an empty side _is_ a real imbalance.                                                                                                                                                                                |

## 6.2 Unsupervised Layer

\# src/detector/unsupervised.py  
class IsolationForestDetector:  
def fit(self, features: pd.DataFrame) -> "IsolationForestDetector": ...  
def score(self, features: pd.DataFrame) -> pd.Series:  
"""Returns anomaly score in \[0,1\], higher = more anomalous."""  
def threshold_flags(self, scores: pd.Series, contamination: float) -> pd.Series: ...

Trained with contamination matched to the _expected_ manipulation rate in the calibration run (a free parameter to sweep — spec doesn't fix it, sweep 1–10% and report sensitivity in the results section, since this directly trades off precision against recall for the unsupervised layer and is worth a paragraph in the report).

## 6.3 Supervised Layer

Trained **separately per persona type** (spec Section 4.3's framing requirement) — three independent binary classifiers (spoofing_vs_rest, wash_trading_vs_rest, pump_dump_vs_rest), not one multiclass model. This is a deliberate implementation decision, not an oversight: a multiclass model forces mutually-exclusive labels per window, but a window can legitimately overlap two manipulation events near a boundary, and per-persona binary models handle that cleanly (a window can be positively flagged by more than one classifier).

\# src/detector/supervised.py  
class PersonaDetector:  
def \__init_\_(self, persona: str) -> None: ...  
def fit(self, features: pd.DataFrame, labels: pd.Series) -> "PersonaDetector": ...  
def predict_proba(self, features: pd.DataFrame) -> pd.Series: ...  
def shap_values(self, features: pd.DataFrame) -> "shap.Explanation": ...

**Class imbalance edge case (real and significant here):** manipulation windows are a small minority of all windows even with several personas active. Use scale_pos_weight in XGBoost (ratio of negative:positive in the training fold) rather than naive resampling, which would distort the feature distributions that SHAP later has to explain honestly.

## 6.4 Ensemble & Audit Trail

\# src/detector/ensemble.py  
class EnsembleScorer:  
def score(self, iso_scores, persona_scores: dict\[str, pd.Series\]) -> pd.DataFrame:  
"""Returns one row per (window, trader_id): final_flag (bool),  
final_score, triggered_by (list\[str\] — e.g. \["isolation_forest",  
"spoofing_classifier"\]), top_shap_features (per triggering  
classifier). This \`triggered_by\` + \`top_shap_features\` pair is the  
exact input contract for Module 4A's narrative generator — do not  
change this schema without updating agentic/narrative_generator.py."""

# 7\. Module 4 — LLM & Agentic Intelligence Layer (Detailed Design)

## 7.1 Shared Infrastructure

\# src/agentic/llm_client.py  
class LLMClient:  
def \__init_\_(self, model: str = "claude-sonnet-4-6", max_retries: int = 3) -> None: ...  
def complete(self, system: str, messages: list\[dict\], tools: list\[dict\] | None = None,  
max_tokens: int = 1024, temperature: float = 0.0) -> "LLMResponse": ...

- temperature=0.0 is the default everywhere in this project — narratives and strategist proposals need to be deterministic-ish and reproducible for grading/viva replay, not creative.
- Every call is wrapped in tenacity retry with exponential backoff on rate-limit/5xx errors, and a hard timeout — a hung LLM call must never block the simulation pipeline (Module 4 always runs _after_ Modules 1–3 have already written their Parquet outputs to disk, so a Module 4 failure never loses simulation results).
- Every call is logged to data/runs/&lt;run_id&gt;/llm_calls.jsonl (prompt, response, token counts, latency, cost estimate) — this becomes both a debugging trail and a cost-accounting artifact worth a line in the final report.

## 7.2 4A — Explainability Narrative Generator

**Input contract:** one row of the EnsembleScorer output (a single flagged window+trader) plus the numeric SHAP top-features already computed by Module 2. **The LLM never receives raw trade_log/order_log rows** — only the already-computed, already-correct numeric summary. This is the core anti-hallucination design decision: the model is asked to _phrase_ a finding, never to _derive_ one.

\# src/agentic/narrative_generator.py  
NARRATIVE_SYSTEM_PROMPT = """You write short compliance-analyst case notes from  
structured detector output. You are given: the persona classifier(s) that  
fired, the numeric SHAP feature attributions, and summary statistics for the  
flagged window. Write 2-4 sentences explaining WHY this window was flagged,  
referencing the specific feature values you were given. Do not invent any  
number, trader ID, or fact not present in the input. If the input is  
insufficient to explain the flag, say so explicitly rather than guessing."""  
<br/>def generate_narrative(flag_row: dict, shap_summary: dict) -> str:  
response = llm_client.complete(  
system=NARRATIVE_SYSTEM_PROMPT,  
messages=\[{"role": "user", "content": json.dumps({"flag": flag_row, "shap": shap_summary})}\],  
)  
return response.text

**Validation step (required, not optional):** after generation, a lightweight regex/entity check confirms every numeric value and trader_id mentioned in the narrative text actually appears in the input JSON. If a narrative fails this check, it is discarded and regenerated once with temperature=0.0 and an explicit re-ask; if it fails twice, the dashboard shows the raw SHAP bar chart instead of a narrative, with no silent fallback to an unverified narrative. This is the concrete implementation of the "never hallucinate a fact into a compliance artifact" requirement (SRS FR-4.1, NFR Security 5.4).

## 7.3 4B — Agentic Adversarial Strategist

This is the genuinely agentic component: a bounded loop, not a single call.

\# src/agentic/adversarial_strategist.py  
STRATEGIST_TOOLS = \[  
{"name": "get_round_metrics", "description": "Returns precision/recall/F1 per persona for a given round"},  
{"name": "get_shap_drift", "description": "Returns which features gained/lost importance between two rounds"},  
{"name": "get_enforcement_notes", "description": "Returns the CFTC case-note grounding text for a persona type"},  
{"name": "propose_mutation", "description": "Proposes a parameter dict for the next round's persona mutation — THIS IS THE ONLY TOOL THAT PRODUCES A SIDE EFFECT; it writes to persona_config.yaml, it does not touch the simulator or detector code"},  
\]  
<br/>def run_strategist_round(round_num: int, persona: str) -> dict:  
"""Agentic loop: the LLM is given the tool list and asked to decide which  
tools to call, in what order, before committing to a \`propose_mutation\`  
call. Bounded to at most 6 tool calls per round to control cost and  
prevent runaway loops. Every proposed mutation is validated against a  
hard-coded parameter-bounds schema (e.g., cancellation_delay_ticks must  
be a positive integer within the simulator's tick range) before being  
written to config — an out-of-bounds proposal is rejected and the agent  
is re-prompted once with the validation error, then falls back to the  
manually-authored mutation_library.py step for that round if it still  
fails."""

**This is where the "agentic" framing is earned, concretely, for the viva:** the strategist doesn't generate text describing an evasion — it calls tools to read the detector's actual weaknesses (SHAP drift), reasons over that structured data, and its output is a _validated, executable_ configuration change that the next simulation round genuinely runs. The fallback to mutation_library.py (the hand-authored Section 5.2 list) means the adversarial loop's core deliverable never depends on the LLM succeeding — Module 4B strictly improves on a baseline that already works without it.

## 7.4 4C — Conversational Analyst Assistant

Function-calling over the same Parquet logs, read-only, scoped per session to one run_id to bound context:

\# src/agentic/tools.py  
ANALYST_TOOLS = \[  
{"name": "query_flags", "description": "Filter flags.parquet by trader_id, round, persona, date range"},  
{"name": "query_round_metrics", "description": "Get precision/recall/F1/FPR/latency for a round"},  
{"name": "get_trader_history", "description": "All orders/trades for a specific trader_id"},  
{"name": "get_price_impact_avoided", "description": "Aggregated price-impact-avoided figure for a round"},  
\]

Every answer the assistant gives must cite which tool call(s) produced the numbers in its answer (surfaced in the UI as a small "sources" expander under the chat bubble — see SRS Section 8 UI spec). An analyst question that cannot be answered from the available tools ("what will the price do tomorrow") must be explicitly declined, not creatively answered — enforced via the system prompt and spot-checked in Module 4's test suite (Section 8.2).

## 7.5 Cost & Latency Budget

| Component                 | Calls per full pipeline run                                                               | Est. tokens/call            | Notes                                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------- |
| 4A Narrative Generator    | ~1 per flagged window (typically 20–150 per round)                                        | ~400 in / ~150 out          | Batchable; run once after each round's detector scoring completes, not live.                              |
| 4B Adversarial Strategist | ≤6 tool-augmented calls × number of mutation rounds (typically 4–5 personas × 3–5 rounds) | ~800 in / ~300 out per call | Bounded loop; this is the main cost driver — budget accordingly before the Week 7–8 full run.             |
| 4C Analyst Assistant      | On-demand, live during dashboard use/demo                                                 | ~1000 in / ~300 out         | Cache common questions (Section 8.5 offline-demo note) to keep the live demo resilient to network issues. |

Use a cheaper/faster model for 4A (high volume, low complexity — pure phrasing) and reserve the stronger model for 4B (genuine multi-step reasoning). Set a hard monthly spend cap on the API key used for development.

# 8\. Testing Strategy

## 8.1 Unit Tests (per module, tests/unit/)

- OrderBook: price-time priority, partial fills, cancel-of-filled no-op, self-trade prevention for non-wash-trading agents, empty-book queries return None not an exception.
- Each Agent subclass: decide() returns None under at least one plausible book state (agents must be capable of doing nothing), and never returns an order with non-positive quantity or a price ≤ 0.
- FeatureEngine: NaN-vs-zero rules from Section 6.1's table, each as an explicit test case with a hand-constructed tiny order/trade log.
- EnsembleScorer: schema of output matches the contract Module 4A depends on (a schema-level test, not just a values test — this is the seam most likely to silently break when someone "quickly" adds a column).

## 8.2 Integration Tests (tests/integration/)

- Run a short (500-tick) simulation end-to-end with a fixed seed → assert deterministic byte-identical output across two runs (this directly tests the reproducibility requirement in Section 5.3).
- Run a simulation with **zero** manipulation personas → assert manipulation_events.csv is a valid, correctly-schemad, zero-row file, and that Module 2's supervised layer training raises a clear, actionable error (not a cryptic sklearn stack trace) when asked to train on an all-negative label set — this is a realistic mistake a teammate will make once.
- Run one full baseline round (Round 0) → assert precision/recall/F1 land above a sanity floor for every persona (a hard-coded low bar, e.g. F1 > 0.3, just to catch "the pipeline is completely broken" regressions, not a tuning target).
- Module 4A: feed a hand-crafted flag_row + shap_summary with a known set of facts → assert the generated narrative passes the fact-grounding regex check, and assert a deliberately-broken input (missing SHAP data) triggers the documented fallback-to-chart behaviour rather than a hallucinated narrative.
- Module 4B: run one strategist round against a mocked LLMClient that returns a canned tool-call sequence → assert the proposed mutation is correctly validated and written to persona_config.yaml, and assert an out-of-bounds mocked proposal is correctly rejected and falls back to mutation_library.py.

## 8.3 Property-Based Tests (tests/property/, using hypothesis)

- For any sequence of randomly generated valid orders fed into a fresh OrderBook: total buy quantity filled always equals total sell quantity filled (conservation of matched volume — the single strongest invariant this whole project rests on).
- For any FeatureEngine window: net_position_gross_ratio is always in \[-1, 1\] or NaN, never out of range.

## 8.4 Edge-Case Regression Suite (explicit list, grows over the project)

| #   | Scenario                                                                                 | Expected behaviour                                                                                                                                                                                                                                          |
| --- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1  | Simulation with only 1 tick                                                              | Runs, produces mostly-empty logs, does not crash on empty-window feature computation downstream.                                                                                                                                                            |
| E2  | All agents are noise traders (no momentum/reversion/MM)                                  | Book still functions; calibration notebook should show this run is _not_ a good calibration match (used as a negative-control demonstration in the report, not a bug).                                                                                      |
| E3  | Manipulation window configured to start after n_ticks (persona never actually activates) | manipulation_events.csv has zero rows for that persona; no silent crash.                                                                                                                                                                                    |
| E4  | Two manipulation personas' active windows overlap                                        | Both are logged independently in manipulation_events.csv; feature engineering and labelling must handle a window belonging to two ground-truth events (Module 2 labels as positive for both relevant persona classifiers).                                  |
| E5  | ANTHROPIC_API_KEY missing or invalid at Module 4 runtime                                 | Module 4 fails fast with a clear message; Modules 1–3 outputs remain untouched and the dashboard falls back to non-narrative views (raw SHAP charts, no chat panel) rather than crashing the whole app.                                                     |
| E6  | LLM call succeeds but returns malformed/unparseable tool-call JSON                       | Caught, logged, one retry with an explicit "your last response was not valid JSON" correction message; second failure falls back per Section 7.3/7.2.                                                                                                       |
| E7  | Detector asked to score a window for a trader_id it never saw in training                | XGBoost handles unseen categoricals if encoded correctly (verify categorical encoding strategy explicitly — this is a common silent-bug source); IsolationForest is trader-agnostic by construction (features are the input, not trader identity directly). |
| E8  | Dashboard loaded with a run_id that has no completed rounds yet                          | Shows an explicit empty state ("no rounds run yet — start one from the sidebar"), never a blank page or stack trace (see SRS Section 8 empty-state requirements).                                                                                           |

## 8.5 Offline-Demo Resilience (Viva Day Requirement)

Conference-room wifi is not to be trusted on viva day. scripts/seed_llm_cache.py pre-generates and caches a full run's worth of 4A narratives and a canned 4C chat transcript for the demo run_id, keyed by input hash. LLMClient checks this cache before making a live call when DEMO_MODE=true in .env. This is a required deliverable, not a nice-to-have — a live API failure mid-viva must not be able to derail the demo.

# 9\. CI/CD & Deployment

## 9.1 Continuous Integration (GitHub Actions, .github/workflows/ci.yml)

name: CI  
on: \[push, pull_request\]  
jobs:  
test:  
runs-on: ubuntu-latest  
steps:  
\- uses: actions/checkout@v4  
\- uses: actions/setup-python@v5  
with: { python-version: "3.11" }  
\- run: pip install -r requirements.txt  
\- run: ruff check src/  
\- run: mypy src/ --ignore-missing-imports  
\- run: pytest tests/unit tests/property -q --cov=src  
\# integration tests run on a schedule, not every push (they're slower)

Module 4 tests always run against a **mocked** LLMClient in CI — CI never spends real API budget or depends on network access.

## 9.2 Dockerfile

FROM python:3.11-slim  
WORKDIR /app  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
COPY . .  
EXPOSE 8501  
CMD \["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"\]

## 9.3 docker-compose.yml

services:  
dashboard:  
build: .  
ports: \["8501:8501"\]  
env_file: .env  
volumes:  
\- ./data:/app/data # persist simulation runs across container restarts

docker compose up --build

# 10\. Week-by-Week Build Checklist (Task-Order Form)

This mirrors the spec's Section 10 timeline but is broken into concrete, independently-completable tasks — written so each bullet can be handed to an agentic coding tool as a self-contained work item, referencing the interface signatures in Sections 5–7 above and the corresponding FRs in the SRS.

**Weeks 1–2 — Foundations** - \[ \] Repo scaffold per Section 2; pyproject.toml/requirements.txt; pre-commit hooks (black, ruff). - \[ \] src/utils/schemas.py — pandera schemas for all four Module 1 output files (write these _before_ the simulator — they are the contract). - \[ \] scripts/fetch_calibration_data.py (Section 4.1) — run and commit the resulting calibration targets to configs/agent_config.yaml (values only, not raw data). - \[ \] Freeze Agent.decide() interface (Section 5.1) — Person A and Person B sign off explicitly before writing any agent logic.

**Weeks 3–4 — Core Simulator** - \[ \] OrderBook + edge cases from Section 5.2, with unit tests written alongside, not after. - \[ \] Four legitimate agent archetypes. - \[ \] First uncalibrated end-to-end run; sanity-check plot (spec Section 3.5 callout).

**Weeks 5–6 — Calibration + Detector v1** - \[ \] OU-process fit + calibration loop (Section 4 of spec) until simulated vs. real stats are defensibly close. - \[ \] Manipulation persona agents (spoofing, wash trading, pump-and-dump) + manipulation_events logging. - \[ \] FeatureEngine with the full edge-case table (Section 6.1). - \[ \] Round 0: IsolationForestDetector + PersonaDetector (×3) + EnsembleScorer, baseline metrics recorded.

**Weeks 7–8 — Adversarial Loop** - \[ \] mutation_library.py — 3–5 hand-authored variants per persona (spec Section 5.2). - \[ \] RoundOrchestrator — pre-retrain / post-retrain measurement per round, round_metrics.parquet, shap_drift.parquet. - \[ \] Full multi-round run, results table matching spec Section 5.5's expected pattern.

**Week 9 — Module 4 (Agentic Layer) + one other stretch item** - \[ \] LLMClient + prompt templates + .env wiring. - \[ \] 4A Narrative Generator + fact-grounding validator (Section 7.2), wired into case_narratives.jsonl. - \[ \] 4B Adversarial Strategist — at minimum, one full round run _with_ the strategist proposing mutations vs. the same round with hand-authored mutations, compared side by side in the report (this comparison is itself a strong result to show). - \[ \] 4C Analyst Assistant with the four tools from Section 7.4. - \[ \] Price-impact-avoided metric (spec Section 6.2) OR automated mutation search — choose per spec Section 9 if time-constrained; Module 4B already substantially subsumes the "automated search" stretch goal, so price-impact-avoided is the recommended pairing.

**Week 10 — Dashboard + Dockerisation** - \[ \] Streamlit pages per SRS Section 8 (live market, round performance, case review with narratives, analyst chat). - \[ \] scripts/seed_llm_cache.py demo cache (Section 8.5). - \[ \] Dockerfile + docker-compose; verify docker compose up works on a clean machine (not just the dev machine — borrow a teammate's laptop for this check).

**Weeks 11–12 — Report & Viva Prep** - \[ \] Final report sections per spec Section 6.3's reporting checklist, plus a short "what Module 4 adds and why it doesn't compromise the ABM's core guarantee" subsection (draw directly from Section 0 of this document). - \[ \] Viva rehearsal anchored on spec Sections 1–2 (architecture justification) and this document's Section 0 (LLM layer justification) — these are the two questions most likely to be asked first.

# 11\. Risk Register (Extends Spec Section 11)

| Risk                                                                                                                                                     | Why it could happen                                                                                                                 | Mitigation                                                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM narrative states a number not present in the input (hallucination)                                                                                   | Model ignores instruction to only phrase given facts                                                                                | Mandatory fact-grounding validator (Section 7.2) with fallback to raw chart; this is a hard requirement, tested in CI against mocked responses that deliberately hallucinate.                                                                            |
| Adversarial strategist (4B) proposes nonsensical or out-of-bounds parameters                                                                             | LLM reasoning is imperfect, especially under a small tool set                                                                       | Hard parameter-bounds validation before any config write (Section 7.3); guaranteed fallback to hand-authored mutation_library.py.                                                                                                                        |
| LLM API cost overrun during Week 7–9 heavy experimentation                                                                                               | Iterating on prompts/rounds multiplies call volume                                                                                  | Mocked LLMClient for all iterative dev/testing; real API calls reserved for scheduled full-pipeline runs only; hard spend cap on the key (Section 7.5).                                                                                                  |
| Live demo network/API failure during viva                                                                                                                | Conference wifi, API outage, rate limiting                                                                                          | Offline demo cache (Section 8.5) — required deliverable, verified the day before the viva.                                                                                                                                                               |
| Schema drift between modules (someone adds a column and downstream code silently breaks or silently ignores it)                                          | Three people editing independently after Week 2                                                                                     | Pandera schema validation at every write boundary (Section 5.4); schema-level unit test for EnsembleScorer's output specifically, since Module 4 depends on it exactly.                                                                                  |
| Prompt-injection-style content inside logged data reaching the LLM (e.g., a trader_id or a crafted feature value engineered to look like an instruction) | Low real risk here since inputs are numeric/structured, but worth stating for completeness given Module 4C reads live query results | 4C's tool outputs are always inserted as structured JSON in a clearly-delimited data block in the prompt, never string-concatenated into the system prompt; system prompt explicitly instructs the model to treat tool output as data, not instructions. |

All risks already documented in the spec's Section 11 (simulation calibration drift, weak manipulation signal, detector non-recovery, RL non-convergence, Person A/B coordination) remain in force unchanged and are not repeated here.

# 12\. Appendix

## 12.1 Environment Variables Reference

| Variable            | Required                  | Purpose                                                   |
| ------------------- | ------------------------- | --------------------------------------------------------- |
| ANTHROPIC_API_KEY   | Yes, for Module 4         | LLM API access                                            |
| MLFLOW_TRACKING_URI | No (defaults to ./mlruns) | Experiment tracking store location                        |
| LOG_LEVEL           | No (defaults to INFO)     | Python logging verbosity                                  |
| DEMO_MODE           | No (defaults to false)    | Forces LLMClient to prefer cached responses (Section 8.5) |

## 12.2 Config File Schemas (summary — full schemas in src/utils/schemas.py)

- agent_config.yaml: population sizes per archetype, arrival rates (Poisson λ per archetype), order size distributions, mean-reversion strength, fundamental-value drift parameters — all populated from calibration (Section 4).
- persona_config.yaml: per-persona, per-round parameter dicts (size, timing, frequency, stealth) — this file is what Module 4B writes to.
- detector_config.yaml: feature window/step sizes, contamination for IsolationForest, XGBoost hyperparameters, top_k for volume concentration.
- llm_config.yaml: model names per component (4A vs 4B, per Section 7.5), temperature, max tool-call bound for 4B, retry/timeout settings.

## 12.3 Suggested Git Branching for a 3-Person Team

- main — always demoable.
- module1-\*, module2-\*, module3-\*, module4-\* — per-person feature branches, merged via PR once unit tests for that piece pass in CI.
- Tag round-0-baseline, round-N-final etc. at each completed adversarial round so the results table in the final report can cite exact commit states.