Software Requirements Specification (SRS)

Commodities Market Manipulation Detection — Agent-Based Simulation, Detector, Adversarial Loop & Agentic Intelligence Layer

Final-Year B.Tech Data Science Project — v2.0

Table of Contents

**Companion document.** _Implementation Plan v2.0_ — covers environment setup, repository structure, module-by-module build detail, and the week-by-week execution schedule. This document specifies _what_ the system must do and how correctness is judged; the Implementation Plan specifies _how_ it gets built.

# 1\. Introduction

## 1.1 Purpose

This SRS defines the functional and non-functional requirements for a simulated commodities-market surveillance system consisting of an agent-based market simulator, a two-layer manipulation detector, an adversarial evaluation harness, an LLM/agentic intelligence layer, and an analyst-facing dashboard. It is written to the level of detail needed to (a) guide implementation without further design decisions being invented mid-build, (b) support systematic testing against explicit acceptance criteria, and (c) serve as the primary defensible reference document in project viva examination.

## 1.2 Scope

**In scope:** a fully simulated market (no live/real trading, no real exchange connectivity, no real money at any point), manipulation detection trained exclusively on simulated ground truth, an adversarial round-based evaluation methodology, an LLM-based explainability and analyst-assistance layer operating strictly downstream of the simulation, and a local/Dockerised Streamlit dashboard for demonstration.

**Out of scope:** real-time/live market data ingestion, real order routing or execution, regulatory filing or actual Suspicious Activity Report submission, any claim of production-readiness for real market surveillance, multi-user authentication/authorization beyond a single local demo user, and any trading or investment functionality of any kind. Every dashboard artifact that resembles a compliance document (Section 6.4's narratives) is explicitly labelled as simulation output.

## 1.3 Definitions, Acronyms, Abbreviations

| Term         | Meaning                                                                                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ABM          | Agent-Based Model                                                                                                                                             |
| LOB          | Limit Order Book                                                                                                                                              |
| OHLCV        | Open/High/Low/Close/Volume (bar-level price data)                                                                                                             |
| SHAP         | SHapley Additive exPlanations — feature attribution method                                                                                                    |
| FPR          | False Positive Rate                                                                                                                                           |
| CoT          | Commitments of Traders (CFTC report)                                                                                                                          |
| OU process   | Ornstein–Uhlenbeck process (mean-reverting stochastic process)                                                                                                |
| Persona      | A manipulation-agent type (spoofing, wash trading, pump-and-dump, layering, cornering)                                                                        |
| Round        | One iteration of the adversarial evaluation loop (Module 3)                                                                                                   |
| Window       | A fixed-duration slice of simulated time over which detector features are computed                                                                            |
| Agentic      | Software behaviour in which an LLM autonomously selects and sequences tool calls to accomplish a bounded task, rather than producing a single text completion |
| Ground truth | The manipulation_events log — known exactly because it is authored by the simulation itself                                                                   |

## 1.4 References

- _Commodities_ABM_Project_Plan.docx_ — Project Specification v1 (architecture rationale, GAN-vs-ABM justification).
- _Implementation Plan v2.0_ (companion document).
- CFTC Commitments of Traders reports and Enforcement Action summaries (cftc.gov).
- yfinance Python library documentation.

## 1.5 Overview

Section 2 gives the overall product description and personas. Section 3 lists functional requirements by module. Section 4 covers external interfaces including the UI. Section 5 covers non-functional requirements. Section 6 defines data requirements/schemas. Section 7 gives detailed use cases. Section 8 is the full UI/UX specification. Section 9 is a system-wide edge-case matrix. Section 10 gives acceptance criteria. Section 11 is a requirements traceability summary.

# 2\. Overall Description

## 2.1 Product Perspective

The system is a standalone research/academic artifact, not an extension of an existing product. It is composed of five layers (simulator, detector, adversarial harness, agentic layer, dashboard) communicating exclusively through versioned Parquet/CSV files, as detailed in the Implementation Plan Section 1. This file-boundary design is itself a requirement (FR-0.1 below) because it is what enables independent module development, replayability, and the price-impact-avoided counterfactual metric.

## 2.2 Product Functions (Summary)

1. Simulate a synthetic commodities market via an order-book matching engine and rule-based trader agents, calibrated against real commodity statistics.
2. Inject configurable manipulation personas into the same market mechanism and log ground-truth manipulation windows.
3. Detect manipulation using an unsupervised + supervised two-layer model with explainability.
4. Evaluate detector robustness under an adversarial, round-based mutation loop.
5. Generate plain-language, fact-grounded narratives explaining detector flags.
6. Autonomously propose and validate new adversarial mutation strategies using an LLM-driven tool-using agent.
7. Answer analyst natural-language questions about simulation/detection results, grounded strictly in logged data.
8. Present all of the above through an interactive, intuitive dashboard suitable for a live demonstration/viva.

## 2.3 User Classes and Characteristics

| User class                                           | Characteristics                                                                                                          | Primary needs                                                                                                                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project developer (the team)**                     | Technically strong, building and debugging the system                                                                    | Clear module interfaces, fast local iteration, good test coverage, reproducibility                                                                              |
| **Viva examiner / evaluator**                        | Domain-literate but not necessarily hands-on with the codebase; time-constrained (typically a 15–30 minute demo/defense) | An intuitive dashboard that tells a clear story without narration, defensible architecture answers, credible numbers                                            |
| **Simulated "analyst" persona (demo role-play)**     | Represents a compliance/surveillance analyst using the dashboard                                                         | Fast answers to "why was this flagged," ability to ask follow-up questions in plain language, clear visual distinction between confirmed and uncertain findings |
| **Future maintainer / interviewer reading the code** | Assesses engineering quality directly from the repository                                                                | Clean module boundaries, tests, documented edge-case handling, a system that runs from a clean checkout via the setup steps in the Implementation Plan          |

## 2.4 Operating Environment

- Local development: Python 3.11 venv, any OS with Docker Desktop support (Linux/macOS/Windows).
- Deployment/demo: single Docker container running the Streamlit dashboard, docker compose up, no external database — Parquet files on a mounted volume are the persistence layer.
- No real-time/production infrastructure requirement — the system is not designed for concurrent multi-user production load (see NFR Scalability, Section 5.2, for the explicit boundary of what scale is required).

## 2.5 Design and Implementation Constraints

- The order-book/agent layer (Module 1) must contain zero learned/generative components — this is an architectural constraint carried over from the Project Specification and is treated as a hard requirement (FR-1.1) because the entire "why not a GAN" defensibility argument depends on it.
- Module 4 (LLM layer) must never be positioned upstream of price/trade generation — enforced structurally by the file-boundary design (Section 2.1) and explicitly tested (see edge case E5/E6, Implementation Plan Section 8.4).
- All external data sources used are free/non-paywalled (yfinance, CFTC public reports, optional Kaggle datasets) — no proprietary data licensing constraint applies.
- LLM component requires network access and a valid API key at generation time; the system must degrade gracefully (not fail entirely) when this is unavailable — see FR-4.4 and NFR Reliability 5.3.

## 2.6 Assumptions and Dependencies

- The team has (or acquires) a free Anthropic API key with sufficient quota for development and one full pipeline run before the viva.
- yfinance continues to serve free historical futures data for the calibration tickers at build time; if a specific ticker is unavailable, an equivalent liquid commodity future is substituted with the change documented in the calibration notebook.
- The project is evaluated as an academic/research artifact; no claim of regulatory or production-grade surveillance accuracy is made or required.

# 3\. Functional Requirements

Each requirement has an ID, a description, priority (Must/Should/Could — MoSCoW), and explicit edge-case notes. IDs are stable identifiers for the traceability matrix in Section 11.

## 3.1 Module 0 — Cross-Cutting

| ID     | Requirement                                                                                                                                                                                                             | Priority |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| FR-0.1 | All inter-module communication shall occur exclusively via schema-validated files on disk (Parquet/CSV/YAML); no module shall import another module's in-memory objects directly.                                       | Must     |
| FR-0.2 | Every data file written by any module shall be validated against its declared schema before being persisted; a schema violation shall abort the write with a descriptive error, never silently truncate or coerce data. | Must     |
| FR-0.3 | Every simulation run shall be uniquely identified by a run_id and stored under data/runs/&lt;run_id&gt;/, preserving full provenance (config used, seed, timestamp).                                                    | Must     |
| FR-0.4 | Given an identical run_id's seed and configuration, re-running the simulation shall produce byte-identical output logs.                                                                                                 | Must     |

## 3.2 Module 1 — Agent-Based Market Simulator

| ID     | Requirement                                                                                                                                                                                                    | Priority                    | Edge cases                                                                                                                                                                                                                                           |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-1.1 | The system shall implement a limit order book matching engine using price-time priority, with no learned/generative component in the price-formation path.                                                     | Must                        | See Implementation Plan Section 5.2 full table.                                                                                                                                                                                                      |
| FR-1.2 | The system shall support at minimum four legitimate agent archetypes (noise, momentum, mean-reversion, market maker), each implementing a shared decide(book_state) interface.                                 | Must                        | An agent must be able to validly return "no action" on any given activation.                                                                                                                                                                         |
| FR-1.3 | The system shall support at minimum three manipulation personas (spoofing, wash trading, pump-and-dump) as agents submitting real orders into the same order book, each with independently tunable parameters. | Must                        | A persona configured with zero active ticks shall run without error and produce a valid empty manipulation_events log for that persona.                                                                                                              |
| FR-1.4 | The system shall support layering and cornering personas as optional extensions.                                                                                                                               | Should / Could respectively | Cornering requires finite-supply modelling not needed elsewhere — explicitly scoped as lowest priority.                                                                                                                                              |
| FR-1.5 | The system shall log every trade to trade_log and every order lifecycle event (placed/cancelled/filled/partial-fill/rejected-cancel) to order_log.                                                             | Must                        | A cancel referencing a non-existent or already-resolved order_id shall log a rejected_cancel event, not raise an exception.                                                                                                                          |
| FR-1.6 | The system shall log ground-truth manipulation windows (persona, start/end tick, involved trader_ids, parameters) to manipulation_events.                                                                      | Must                        | Overlapping windows from two different personas shall both be logged independently and both be retrievable.                                                                                                                                          |
| FR-1.7 | The system shall resample trade_log into fixed-width OHLCV bars.                                                                                                                                               | Must                        | A bar with zero trades shall be represented with NaN OHLC and zero volume, not forward-filled from the prior bar, unless forward-fill is explicitly selected as a visualization option in the dashboard only (never in the data used for detection). |
| FR-1.8 | The system shall estimate calibration targets (volatility, drift, mean-reversion strength, volume distribution) from real commodity futures data via yfinance and CFTC CoT reports.                            | Must                        | Ticker unavailability shall raise a clear error at fetch time, not proceed silently with an empty dataset.                                                                                                                                           |
| FR-1.9 | The system shall support replaying a simulation from any tick with a modified agent population (for the price-impact-avoided counterfactual, Module 3/6.2).                                                    | Must                        | Requires FR-0.4's reproducibility guarantee to hold for the pre-modification segment exactly.                                                                                                                                                        |

## 3.3 Module 2 — Manipulation Detector

| ID     | Requirement                                                                                                                                                                                                                              | Priority | Edge cases                                                                                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-2.1 | The system shall compute rolling-window, per-trader features from order_log/trade_log: cancellation rate, volume concentration, net-position/gross-volume ratio, price-volume divergence, bid-ask imbalance spikes, order-size variance. | Must     | See Implementation Plan Section 6.1 for the full NaN-vs-zero handling table per feature; these are binding requirements, not implementation suggestions.               |
| FR-2.2 | The system shall fit an unsupervised anomaly model (Isolation Forest, minimum) on engineered features without using ground-truth labels.                                                                                                 | Must     | Must run and produce scores even on a run with zero manipulation events (used as an FPR-on-clean-data check).                                                          |
| FR-2.3 | The system shall train a supervised classifier per manipulation persona (not one lumped multiclass model), using manipulation_events as ground truth.                                                                                    | Must     | A training set with zero positive examples for a persona shall raise an explicit, actionable error rather than silently training a degenerate all-negative classifier. |
| FR-2.4 | The system shall compute SHAP feature attributions for every supervised prediction.                                                                                                                                                      | Must     | —                                                                                                                                                                      |
| FR-2.5 | The system shall combine unsupervised and supervised layers into a single ensemble flag per (window, trader_id), recording which sub-layer(s) triggered and the top contributing features.                                               | Must     | This output schema is a fixed contract consumed by Module 4A — schema changes require an explicit version bump (FR-0.2).                                               |
| FR-2.6 | The system shall report precision, recall, F1, and false-positive rate per manipulation persona, never as one lumped "manipulation" metric.                                                                                              | Must     | —                                                                                                                                                                      |
| FR-2.7 | The system shall measure detection latency (ticks from manipulation onset to first flag) per detected event.                                                                                                                             | Must     | An event never flagged has undefined (not zero) latency and shall be reported separately as a miss, not averaged in as if it had zero latency.                         |

## 3.4 Module 3 — Adversarial Evaluation Loop

| ID     | Requirement                                                                                                                                                                                                        | Priority | Edge cases                                                                                                                                                                                 |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FR-3.1 | The system shall establish a Round 0 baseline: obvious-parameter manipulation personas, trained detector, recorded baseline metrics.                                                                               | Must     | —                                                                                                                                                                                          |
| FR-3.2 | For each subsequent round, the system shall generate new simulation data under mutated persona parameters, evaluate the _existing_ (pre-retrain) detector against it, then retrain and re-evaluate (post-retrain). | Must     | If a mutation degrades recall to zero and it never recovers post-retrain, this shall be reported and flagged as a limitation, not treated as a pipeline failure to be silently suppressed. |
| FR-3.3 | The system shall provide a library of manually-authored mutation parameter sets (3–5 per persona), grounded in documented CFTC enforcement patterns where applicable.                                              | Must     | —                                                                                                                                                                                          |
| FR-3.4 | The system shall log per-round SHAP feature-importance drift (which features gained/lost weight between rounds).                                                                                                   | Must     | —                                                                                                                                                                                          |
| FR-3.5 | The system shall compute the price-impact-avoided metric via deterministic counterfactual replay from the detection tick.                                                                                          | Should   | Requires FR-1.9; if replay reproducibility cannot be guaranteed for a given run, this metric shall be omitted for that run rather than reported with unverified values.                    |

## 3.5 Module 4 — LLM & Agentic Intelligence Layer

| ID     | Requirement                                                                                                                                                                                                              | Priority | Edge cases                                                                                                                                                                                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR-4.1 | The system shall generate a natural-language case narrative for each detector-flagged window, derived only from already-computed numeric SHAP/summary data, never from raw trade/order rows directly.                    | Must     | Every generated narrative shall pass an automated fact-grounding check (every number/trader_id mentioned must appear in the input); on failure, the system shall fall back to displaying the raw SHAP chart, never an unverified narrative.                                                   |
| FR-4.2 | The system shall implement a bounded, tool-using agentic loop that reads round metrics and SHAP drift, and proposes the next round's mutation parameters.                                                                | Should   | Proposals shall be validated against hard parameter bounds before being applied; an invalid or malformed proposal shall trigger one re-prompt, then fall back to the manually-authored mutation library (FR-3.3). The loop shall be capped at a fixed maximum number of tool calls per round. |
| FR-4.3 | The system shall provide a conversational assistant that answers analyst questions about simulation/detection results using function-calling over the logged data exclusively — never by free-generating facts.          | Should   | A question outside the scope of available tools/data (e.g., asking for a future price prediction) shall be explicitly declined by the assistant, not creatively answered. Every answer shall be traceable to the specific tool call(s) that produced it.                                      |
| FR-4.4 | If the LLM API is unavailable or misconfigured, the system shall degrade gracefully: Modules 1–3 continue to function fully, and the dashboard shows non-LLM views (raw SHAP charts, no chat panel) rather than failing. | Must     | Missing/invalid API key shall be detected at Module 4 startup with a clear error, not discovered lazily mid-run.                                                                                                                                                                              |
| FR-4.5 | The system shall cache a full set of Module 4 outputs for a designated demo run_id to allow the dashboard to run in a demo mode without live API calls.                                                                  | Must     | This is required specifically to de-risk the live viva demonstration.                                                                                                                                                                                                                         |
| FR-4.6 | Every LLM call shall be logged (prompt, response, token counts, latency) for auditability and cost accounting.                                                                                                           | Should   | —                                                                                                                                                                                                                                                                                             |

## 3.6 Dashboard

| ID     | Requirement                                                                                                                                      | Priority | Edge cases                                                                                                                     |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| FR-5.1 | The dashboard shall display a live/replayable price chart with manipulation windows visually highlighted and detector flags overlaid.            | Must     | A run_id with zero completed rounds shall show an explicit empty state, not a blank chart.                                     |
| FR-5.2 | The dashboard shall display round-by-round performance curves (precision/recall/F1/FPR per persona, across rounds).                              | Must     | —                                                                                                                              |
| FR-5.3 | The dashboard shall provide a case-review view listing flagged windows with their generated narratives (Module 4A) and underlying SHAP evidence. | Must     | If a narrative failed grounding validation (FR-4.1), the view shall show the SHAP chart with a visible note, not a blank card. |
| FR-5.4 | The dashboard shall provide a chat panel for the analyst assistant (Module 4C), including a per-answer "sources" disclosure.                     | Should   | If Module 4 is unavailable (FR-4.4), this panel shall be hidden or disabled with an explanatory message, not shown broken.     |
| FR-5.5 | The dashboard shall display the aggregated price-impact-avoided figure where available.                                                          | Could    | Shall be omitted (not shown as zero or blank) for runs where FR-3.5's reproducibility precondition wasn't met.                 |
| FR-5.6 | The dashboard shall allow selecting between multiple saved run_ids without restarting the application.                                           | Should   | —                                                                                                                              |

# 4\. External Interface Requirements

## 4.1 User Interfaces

Full detailed UI/UX specification is in Section 8. Summary interface list:

- Streamlit web application, single-page-app style with a left sidebar for navigation and run_id/round selection, accessible at localhost:8501 in local dev and via the Docker-exposed port in deployment.
- Four primary views: Live Market, Round Performance, Case Review, Analyst Chat (mapped to FR-5.1–5.4).
- No native mobile app; the dashboard shall be usable (not necessarily optimised) on a laptop screen in a viva room, minimum resolution 1280×720.

## 4.2 Hardware Interfaces

None beyond standard developer/demo hardware (laptop/desktop, no specialised hardware, no GPU requirement for the core deliverable — GPU is only relevant to the optional RL stretch goal per the Project Specification Section 8, and even there Stable-Baselines3 PPO trains adequately on CPU at this project's scale).

## 4.3 Software Interfaces

| Interface                                | Direction        | Purpose                                     | Failure handling                                                                    |
| ---------------------------------------- | ---------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
| yfinance (Yahoo Finance, unofficial API) | Inbound          | Calibration OHLCV pull                      | Explicit error on empty/failed response (FR-1.8); never proceed silently            |
| CFTC public reports (cftc.gov)           | Inbound          | CoT positioning data, enforcement case text | Manual download step is acceptable; no automated dependency at runtime              |
| Anthropic API                            | Inbound/Outbound | LLM completions + tool calling for Module 4 | Graceful degradation per FR-4.4; retries with backoff; demo cache fallback (FR-4.5) |
| Kaggle API (optional)                    | Inbound          | Appendix credibility-check dataset only     | Not a runtime dependency of the main pipeline                                       |
| Docker / docker-compose                  | N/A              | Deployment/runtime environment              | Standard container lifecycle; data persisted via mounted volume                     |

## 4.4 Communication Interfaces

- All network calls (yfinance, Anthropic API) use HTTPS.
- No inbound network exposure required beyond the dashboard's own port; the system is not designed to accept external/public traffic.

# 5\. Non-Functional Requirements

## 5.1 Performance

| ID     | Requirement                                                                                                                                                                                                                                                                        |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-P1 | A single simulation run of 10,000 ticks with the full default agent population shall complete in under 2 minutes on a standard development laptop (no GPU).                                                                                                                        |
| NFR-P2 | Feature engineering over a completed run's logs shall complete in under 30 seconds for a 10,000-tick run.                                                                                                                                                                          |
| NFR-P3 | Detector training (unsupervised + 3 supervised classifiers) shall complete in under 3 minutes per round on a standard laptop.                                                                                                                                                      |
| NFR-P4 | Dashboard page loads (Live Market, Round Performance) shall render in under 3 seconds for a cached/completed run.                                                                                                                                                                  |
| NFR-P5 | A single Module 4A narrative generation call shall return in under 8 seconds under normal API conditions; batch narrative generation for a full round (up to ~150 flags) shall complete in under 10 minutes, run asynchronously/offline, never blocking the interactive dashboard. |
| NFR-P6 | The Module 4C chat assistant shall return a first-token or full response within 10 seconds for a typical query under normal API conditions.                                                                                                                                        |

## 5.2 Scalability

The system is explicitly scoped for single-user, single-machine, academic-demonstration scale — it is **not** required to support concurrent multi-user access, distributed simulation, or production-scale order volumes. Simulation scale is bounded by tick count and agent population size, both configuration values; the architecture (file-boundary modules, Parquet storage) would extend to larger scale without redesign, but performance at larger scale is explicitly out of scope for testing/acceptance.

## 5.3 Reliability / Availability

| ID     | Requirement                                                                                                                                                                                                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NFR-R1 | A failure in Module 4 (LLM layer) shall never cause data loss or failure in Modules 1–3's already-persisted outputs (FR-4.4).                                                                                                                          |
| NFR-R2 | A crash mid-simulation-run shall leave partial logs in a clearly-identifiable, non-corrupted state (either fully written and valid, or not written at all — no partial/truncated Parquet files); use atomic write-then-rename for all log persistence. |
| NFR-R3 | The system shall be demonstrable without live internet access for every function except Module 4's live (non-cached) chat responses (FR-4.5's demo cache covers this gap).                                                                             |

## 5.4 Security

| ID     | Requirement                                                                                                                                                                                                                                                                                                                       |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-S1 | The Anthropic API key shall never be committed to version control; .env is gitignored and .env.example contains no real secret.                                                                                                                                                                                                   |
| NFR-S2 | LLM prompts shall structurally separate trusted instructions (system prompt) from untrusted/data content (tool outputs, logged values), per the prompt-injection mitigation described in the Implementation Plan Section 7.4/11.                                                                                                  |
| NFR-S3 | No real financial account data, personal data, or real trader identities are used anywhere in the system — all trader_ids are synthetic simulation identifiers with no real-world referent, and this shall be stated explicitly in the dashboard UI (footer disclaimer) to avoid any implication of real surveillance capability. |
| NFR-S4 | Generated compliance-style narratives (FR-4.1) shall be visually and textually labelled as simulation output, never presented in a way that could be mistaken for a real regulatory finding.                                                                                                                                      |

## 5.5 Maintainability

| ID     | Requirement                                                                                                                                          |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-M1 | Every module shall have unit test coverage for its documented edge cases (Implementation Plan Section 8.4) at minimum.                               |
| NFR-M2 | Code shall pass ruff linting and mypy type checking in CI before merge.                                                                              |
| NFR-M3 | Every cross-module data contract (schema) shall be defined once in src/utils/schemas.py and referenced, never duplicated, across modules.            |
| NFR-M4 | LLM prompt templates shall be versioned as separate files (not inline strings scattered through code), enabling prompt changes without code changes. |

## 5.6 Usability

| ID     | Requirement                                                                                                                                                                                                             |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-U1 | A first-time viewer (viva examiner) shall be able to understand what a given dashboard screen is showing within 10 seconds, without verbal narration, via clear titles, legends, and short inline explanatory captions. |
| NFR-U2 | Every numeric metric shown on the dashboard shall have a one-line plain-language explanation available on hover/expand (e.g., what "false positive rate" costs in this domain, per spec Section 6.1).                   |
| NFR-U3 | Colour choices shall not be the sole carrier of meaning (e.g., manipulation-type distinctions use both colour and icon/label) to remain usable for colourblind viewers.                                                 |
| NFR-U4 | Error and empty states shall always include a next-action suggestion (e.g., "no rounds run yet — start one from the sidebar"), never a bare error message.                                                              |

## 5.7 Portability

| ID      | Requirement                                                                                                                                                                   |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-PO1 | The full system shall run from a clean checkout via the setup steps in the Implementation Plan (venv + requirements, or docker compose up) with no undocumented manual steps. |
| NFR-PO2 | The system shall run on Linux, macOS, and Windows (via WSL2 or native Docker Desktop) without OS-specific code paths beyond standard cross-platform Python practice.          |

## 5.8 Compliance / Ethical Requirements

| ID     | Requirement                                                                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NFR-E1 | The system shall not present itself, in any UI text, report text, or generated narrative, as a real or production-ready market surveillance product.                                                    |
| NFR-E2 | The system shall not use any real, identifiable individual's or institution's trading data at any point.                                                                                                |
| NFR-E3 | Any LLM-generated content resembling a compliance/investigative document (FR-4.1) shall carry a visible "simulation output — not a real regulatory finding" label wherever it is displayed or exported. |

# 6\. Data Requirements

## 6.1 Core Log Schemas (authoritative; implemented as pandera schemas per Implementation Plan Section 5.4)

**trade_log** — one row per executed trade | Column | Type | Notes | |—|—|—| | trade_id | string | unique | | timestamp / tick | int | simulation tick | | price | float | > 0 | | quantity | int | > 0 | | buyer_trader_id | string | | | seller_trader_id | string | must differ from buyer, except for the wash-trading persona's intentional self-cross case (flagged separately) |

**order_log** — one row per order lifecycle event | Column | Type | Notes | |—|—|—| | order_id | string | | | trader_id | string | | | side | enum(buy, sell) | | | price | float | | | quantity | int | | | tick | int | | | event_type | enum(placed, cancelled, filled, partial_fill, rejected_cancel) | |

**manipulation_events** — one row per ground-truth manipulation episode | Column | Type | Notes | |—|—|—| | event_id | string | | | persona | enum(spoofing, wash_trading, pump_and_dump, layering, cornering) | | | start_tick / end_tick | int | | | trader_ids | list\[string\] | involved accounts | | parameters | JSON/dict | the exact mutation parameters active for this event, for auditability |

**ohlcv_bars** — one row per resampled bar | Column | Type | Notes | |—|—|—| | bar_start | int/timestamp | | | open, high, low, close | float | NaN if zero trades in the bar | | volume | int | 0 if no trades |

**flags** (Module 2 output, EnsembleScorer contract) | Column | Type | Notes | |—|—|—| | window_start | int | | | trader_id | string | | | final_flag | bool | | | final_score | float | | | triggered_by | list\[string\] | e.g. \["isolation_forest", "spoofing_classifier"\] | | top_shap_features | dict | feature: attribution value, per triggering classifier |

**round_metrics** (Module 3 output) | Column | Type | Notes | |—|—|—| | round_num | int | | | persona | string | | | precision, recall, f1, fpr | float | | | detection_latency_ticks | float | NaN for missed events | | pre_retrain / post_retrain | bool | |

**case_narratives** (Module 4A output, JSONL) | Field | Type | Notes | |—|—|—| | window_start, trader_id | — | join key back to flags | | narrative_text | string | | | grounding_check_passed | bool | if false, dashboard falls back to chart (FR-4.1) | | model_used, tokens, latency_ms | — | audit trail (FR-4.6) |

## 6.2 Configuration Files

agent_config.yaml, persona_config.yaml, detector_config.yaml, llm_config.yaml — summarised in Implementation Plan Section 12.2; each is version-controlled (unlike raw data) since they represent decisions, not fetched data.

## 6.3 Data Retention

All simulation run data is local/ephemeral for this academic project; no requirement for long-term retention policy beyond keeping the demo run_id intact through the viva date.

# 7\. Use Case Specifications

## UC-1: Run a Full Simulation Round

- **Actor:** Developer / demo operator.
- **Preconditions:** Environment set up per Implementation Plan Section 3; agent_config.yaml and persona_config.yaml populated.
- **Main flow:** Operator runs scripts/run_round.py --round N. System generates simulation data (Module 1), computes features and scores with the existing detector (pre-retrain), retrains (post-retrain), logs round_metrics. Dashboard reflects the new round automatically on next load.
- **Alternate flow:** --dry-run flag validates config without writing simulation output, for quick config-checking.
- **Exception flow:** Config validation failure (e.g., an out-of-bounds persona parameter) aborts before any simulation compute is spent, with a specific field-level error message.
- **Postconditions:** New run_id/round directory created with all four Module 1 logs, round_metrics updated.

## UC-2: Analyst Reviews a Flagged Case

- **Actor:** Demo "analyst" role.
- **Preconditions:** At least one completed round with flags exists for the selected run_id.
- **Main flow:** Analyst opens Case Review (FR-5.3), sees a sortable/filterable list of flagged windows, selects one, sees the generated narrative (FR-4.1) alongside the SHAP chart and raw trade context for that window.
- **Alternate flow:** Narrative failed grounding validation → SHAP chart shown with a visible "narrative unavailable for this case" note instead of blank content.
- **Exception flow:** No flags exist for the selected run/round → empty state per NFR-U4.
- **Postconditions:** None (read-only view).

## UC-3: Analyst Asks a Free-Text Question

- **Actor:** Demo "analyst" role.
- **Preconditions:** Module 4 available (API key configured) or demo cache present.
- **Main flow:** Analyst types a question in the chat panel (FR-5.4); system determines relevant tool call(s) (FR-4.3), retrieves grounded data, returns an answer with a "sources" disclosure listing which tool calls were used.
- **Alternate flow:** Question is out of scope of available tools (e.g., a prediction request) → assistant explicitly declines, states what it can answer instead.
- **Exception flow:** Module 4 unavailable → chat panel disabled with explanatory message (FR-4.4/FR-5.4).
- **Postconditions:** Conversation logged for audit (FR-4.6).

## UC-4: Adversarial Strategist Proposes a Mutation

- **Actor:** System (autonomous, triggered by the round orchestrator), observed by developer.
- **Preconditions:** At least one completed round exists with recorded metrics and SHAP drift for the persona being mutated.
- **Main flow:** Orchestrator invokes Module 4B; the agent calls get_round_metrics/get_shap_drift/get_enforcement_notes as needed (≤6 calls), then calls propose_mutation; the proposal is validated against parameter bounds and, if valid, written to persona_config.yaml for the next round.
- **Alternate flow:** Proposal fails validation → one re-prompt with the specific validation error.
- **Exception flow:** Second failure, or Module 4 unavailable → orchestrator falls back to the next hand-authored entry in mutation_library.py (FR-4.2/FR-3.3); the round proceeds either way.
- **Postconditions:** persona_config.yaml updated for the next round; the source of the mutation (agentic vs. hand-authored) is recorded in manipulation_events' parameters field for that round, so the report can compare the two head-to-head.

## UC-5: Examiner Views Round-by-Round Performance During Viva

- **Actor:** Viva examiner.
- **Preconditions:** Demo run_id with all rounds pre-computed and Module 4 cache seeded (Implementation Plan Section 8.5).
- **Main flow:** Examiner (or presenting student) navigates Round Performance view (FR-5.2), sees precision/recall/F1/FPR curves per persona across rounds, with the expected-degrade-then-recover pattern (spec Section 5.5) visible without narration (NFR-U1).
- **Postconditions:** None (read-only).

# 8\. UI/UX Specification

## 8.1 Design Principles

1. **Read without narration.** Every screen must communicate its finding to someone who has never seen the project before, within seconds (NFR-U1) — this drives every layout decision below.
2. **Confirmed vs. uncertain is always visually distinct.** Ground-truth manipulation windows (known because the simulator authored them), detector flags (the system's belief), and LLM narratives (a phrasing of the detector's belief, separately validated) are three different levels of certainty and must never be rendered identically.
3. **Numbers earn their place with a one-line meaning**, not a bare metric name (NFR-U2).
4. **Never a blank screen.** Every view has a defined empty/loading/error state (NFR-U4).
5. **Simulation, not reality — always labelled** (NFR-E1/E3).

## 8.2 Visual Language

| Element                          | Treatment                                                                                                                                                                                                                                                                                                                                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ground-truth manipulation window | Shaded background band on time-series charts, one hue per persona type (e.g., spoofing = amber, wash trading = purple, pump-and-dump = red), with a text label on the band itself — never colour alone (NFR-U3).                                                                                                                                                                                        |
| Detector flag                    | A marker (triangle/diamond) placed on the chart at the flagging tick, distinct shape from the shaded ground-truth band, with a tooltip on hover showing score and top SHAP feature.                                                                                                                                                                                                                     |
| LLM narrative                    | Rendered inside a bordered "case note" card with a small robot/quote icon and a persistent caption: "AI-generated summary of detector output — simulation only." A green "verified" chip if the grounding check passed; if it failed, the card is replaced by the raw SHAP chart with a neutral note, never shown with a warning-red chip that could look like a false alarm on the underlying finding. |
| Chat assistant answers           | Each answer bubble has a small collapsed "Sources (2)" expander beneath it, listing the exact tool calls/queries used — this is not optional chrome, it is the trust mechanism required by FR-4.3.                                                                                                                                                                                                      |
| Price-impact-avoided figure      | Shown as a large single number with a one-sentence plain-language definition directly beneath it (never just a labelled statistic), and hidden entirely (not shown as 0%) when its reproducibility precondition (FR-3.5) wasn't met for that run.                                                                                                                                                       |

## 8.3 Screen Inventory

### Screen 1 — Live Market (FR-5.1)

- **Purpose:** Show the simulated price/volume series for the selected run, with manipulation windows and detector flags overlaid, so a viewer can visually confirm "the detector caught what actually happened" in seconds.
- **Layout:** Full-width price chart (candlestick or line, toggle) on top; volume bars beneath, synced x-axis; a persona legend strip; a scrub/zoom control to move through the simulated timeline tick by tick.
- **Empty state:** If the selected run_id has no completed simulation, show a centered message with a "Run a simulation" call-to-action button, not a blank chart area.
- **Loading state:** Skeleton chart placeholder while a run's logs are being read from disk.

### Screen 2 — Round Performance (FR-5.2)

- **Purpose:** Show the adversarial loop's core result — the degrade/recover pattern across rounds, per persona.
- **Layout:** A small-multiples grid: one line chart per persona (precision/recall/F1 as three lines each, pre-retrain vs. post-retrain marked distinctly, e.g. solid vs. dashed), round number on the x-axis. Below the grid, a compact table of exact numbers (for the examiner who wants the precise figure, not just the shape of the curve) with FPR and detection latency columns. A short auto-generated one-sentence summary above the grid ("Recall for spoofing dropped 41% after Round 2's mutation and recovered to 89% of baseline after retraining") — this summary is computed from the numbers directly (not LLM-generated) to keep it always available even with Module 4 disabled.
- **Empty state:** "No rounds completed yet for this run" with guidance to Screen 1 / CLI.

### Screen 3 — Case Review (FR-5.3)

- **Purpose:** Let an analyst inspect individual flagged windows with full explanation.
- **Layout:** Left panel — filterable/sortable list of flagged windows (filter by persona, round, trader_id, flag confidence). Right panel — selected case detail: narrative card (Section 8.2), SHAP waterfall/bar chart, a mini price chart zoomed to that window's ± N ticks, and the raw order/trade rows for that trader in that window (collapsed by default, expandable — this is the "show your work" affordance for a skeptical examiner).
- **Empty state:** "No flagged cases in this round" is itself a valid, clearly-labelled result (e.g., for a clean baseline run), distinguished from a loading or error state.

### Screen 4 — Analyst Chat (FR-5.4)

- **Purpose:** Free-text Q&A grounded in the run's data.
- **Layout:** Standard chat panel, message input pinned to bottom, suggested-question chips above the input for first-time users ("Why was T-118 flagged in round 3?", "Which persona is hardest to catch after mutation?") to reduce blank-page anxiety and demonstrate scope without narration.
- **Disabled state:** If Module 4 is unavailable, the entire panel is replaced by a single explanatory message and a link back to Screen 3's SHAP-only case review, never a chat box that silently fails on submit.

## 8.4 Navigation

Left sidebar, persistent across all screens: run_id selector (dropdown, FR-5.6) at top, then the four screens as a vertical nav list, then a footer disclaimer (NFR-S3/E1) always visible: "All data is synthetic. No real trading, real accounts, or real regulatory findings are represented in this application."

## 8.5 Accessibility

- All interactive controls reachable via keyboard (Streamlit's default widget behaviour satisfies this; verify no custom component breaks tab order).
- Colour is never the sole differentiator (NFR-U3) — every colour-coded element also carries a text label or distinct shape.
- Minimum text contrast ratio 4.5:1 for body text (standard WCAG AA target) — verify against Streamlit's default theme or override if the default fails this on any custom-styled element (e.g., the shaded manipulation bands).
- Chart tooltips and narrative text use plain language, avoiding unexplained jargon (ties to NFR-U2).

## 8.6 Error, Loading, and Empty States (System-Wide Pattern)

| State                                                                         | Required treatment                                                                                                                                                                           |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loading                                                                       | Skeleton/spinner with a short descriptive label ("Loading round 3 results…"), never a bare spinner with no context.                                                                          |
| Empty (valid — e.g. zero flags in a clean run)                                | Neutral message, explicitly distinguished from an error, with a next-action suggestion where relevant.                                                                                       |
| Error (e.g. corrupted log file, schema validation failure surfaced to the UI) | Plain-language message plus a technical detail expander (for the developer), never a raw Python traceback as the primary display.                                                            |
| Module 4 unavailable                                                          | Explicit, calm degradation message per FR-4.4 — framed as "AI features are temporarily unavailable" rather than as an application error, since the rest of the app remains fully functional. |

# 9\. System-Wide Edge Case Matrix

This consolidates cross-cutting edge cases not already itemised per-module in Section 3 or in the Implementation Plan's Section 8.4 regression table; this table is scenario-first (spans modules) rather than module-first.

| Scenario                                                                                                                       | Expected system behaviour                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Two simulation runs are started with the same run_id (accidental re-run)                                                       | The second run shall not silently overwrite the first; either refuse with a clear "run_id already exists" error, or require an explicit --overwrite flag.                                                                                                                                                                       |
| Dashboard is opened before any simulation has ever been run (fresh clone)                                                      | Screen 1's empty state (Section 8.3) guides the user to run a simulation; no other screen is reachable in a broken state as a result.                                                                                                                                                                                           |
| A round's detector retraining is interrupted (process killed) mid-write                                                        | NFR-R2's atomic write guarantee ensures round_metrics for that round is either fully absent or fully present — never a half-written row that later code would misinterpret as a completed round.                                                                                                                                |
| The demo run_id used for the viva is regenerated after cache-seeding (Implementation Plan Section 8.5), invalidating the cache | scripts/seed_llm_cache.py keys the cache by content hash of the input, not by run_id string alone, so a regenerated-but-identical run still hits cache; a genuinely different run correctly misses cache and (if online) calls the live API, or (if offline) shows the graceful-degradation state rather than stale narratives. |
| A teammate adds a new feature column to FeatureEngine without updating the pandera schema                                      | FR-0.2 causes this to fail loudly at the next write, in that teammate's own test run, not silently three weeks later in someone else's.                                                                                                                                                                                         |
| The Anthropic API changes its response format in a way the SDK doesn't abstract away                                           | LLMClient's response parsing is isolated to one function (Implementation Plan Section 7.1); a parsing failure there triggers FR-4.4's graceful-degradation path rather than propagating an unhandled exception into the dashboard.                                                                                              |
| An examiner asks the chat assistant a question referencing a run_id/round that doesn't exist in the current session            | The assistant's tools return an explicit "not found" result, which the assistant is instructed to relay plainly, not paper over with a plausible-sounding fabricated answer.                                                                                                                                                    |

# 10\. Acceptance Criteria / Definition of Done

A module is considered complete when **all** of the following hold:

1. All "Must"-priority FRs for that module (Section 3) are implemented and covered by a passing test in the appropriate tests/ subdirectory (Implementation Plan Section 8).
2. All edge cases listed for that module (Section 3's per-row edge-case notes, plus the relevant rows of Section 9 and Implementation Plan Section 8.4) have an explicit test, not just a mention in a docstring.
3. The module's output(s) pass pandera schema validation on every code path, including the degenerate/empty-input paths.
4. For UI-facing functionality: the corresponding screen (Section 8.3) has a verified loading, empty, and error state — verified means manually triggered and screenshotted at least once, not just theoretically handled.
5. For Module 4 components specifically: the fact-grounding/validation guardrail (FR-4.1) or parameter-bounds validation (FR-4.2) has a test that deliberately feeds a bad/hallucinated input and confirms the fallback path fires.

**Project-level acceptance (viva readiness):** - A clean checkout, following only the Implementation Plan's setup steps, produces a running dashboard via docker compose up. - The demo run_id has all rounds pre-computed and the Module 4 cache seeded (Implementation Plan Section 8.5), verified functional with network access disabled. - Round Performance (Screen 2) shows the expected degrade/recover pattern (Project Specification Section 5.5) for at least the manually-mutated baseline, with the agentic-strategist comparison (FR-4.2, UC-4) available as a secondary result if Week 9's work completed.

# 11\. Requirements Traceability Summary

| FR range | Primary module (Implementation Plan) | Primary test location                      | Primary UI screen    |
| -------- | ------------------------------------ | ------------------------------------------ | -------------------- |
| FR-0.x   | src/utils/                           | tests/unit, tests/property                 | N/A (infrastructure) |
| FR-1.x   | src/core/                            | tests/unit, tests/integration              | Screen 1             |
| FR-2.x   | src/detector/                        | tests/unit, tests/integration              | Screens 2, 3         |
| FR-3.x   | src/adversarial/                     | tests/integration                          | Screen 2             |
| FR-4.x   | src/agentic/                         | tests/unit (mocked LLM), tests/integration | Screens 3, 4         |
| FR-5.x   | dashboard/                           | Manual verification (Section 10, item 4)   | All screens          |

Non-functional requirements (Section 5) are verified continuously via CI (NFR-M2), the property-based test suite (NFR-M1 evidence), and the manual viva-readiness checklist (Section 10) rather than a single dedicated test file, since they are cross-cutting by nature.

_End of Software Requirements Specification. Read alongside the Implementation Plan v2.0 for build-order detail, and the original Project Specification for the core architectural argument (ABM vs. GAN) that both documents assume as settled._