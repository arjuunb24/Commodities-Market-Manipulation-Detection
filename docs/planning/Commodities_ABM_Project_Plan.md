**Commodities Market Manipulation Detection**

An Agent-Based Market Simulation Approach

_Full Project Specification & Implementation Plan_

Final Year B.Tech Data Science Project

Team of 3 — Module ownership defined in Section 9

# Table of Contents

# 1\. Problem Framing & Core Thesis

## 1.1 The Core Argument

Real commodity exchanges do not publish ground-truth labels for manipulative trading (spoofing, wash trading, pump-and-dump). If a regulator could detect every instance of manipulation with certainty, that manipulation would already be eliminated from the market. This means any supervised "detect manipulation" system cannot be built on purely real, passively-collected data — the labels simply do not exist at the scale a learning system needs. The only way to get reliable, exhaustive ground truth is to construct a system where you control the underlying process and therefore know exactly when manipulation occurred.

This single fact justifies the entire simulation-based approach to any examiner. It is not a workaround being hidden — it is the methodologically correct starting point, and it should be stated explicitly and early in the project report.

## 1.2 Why Simulation Alone Is Not Enough — The Mechanism Problem

A simulator that only outputs a believable-looking price curve is insufficient, because "believable-looking" is a statistical property, not an economic one. A detector trained on data with the wrong underlying mechanism will learn to recognise artefacts of the generation process rather than genuine signatures of manipulative behaviour. This is the central flaw in using a generic generative model (GAN/TimeGAN) for this project, and it is documented, not speculative: recent research on financial time-series generation found that standard GANs and WGAN-GP frequently collapse under backtesting, producing extreme, implausible outcomes (annualised returns exceeding 2000%, volatility approaching 1000%), because the adversarial objective optimises for distributional resemblance, not for any underlying economic consistency.

The fix in that line of research is to bolt economic structure back onto the generator as explicit constraints or conditioning variables. That is a valid path, but it raises an obvious question: if you have to hand-engineer the economic structure anyway, why route it through an unstable adversarial training loop at all? The agent-based approach in this plan answers that question — it builds the economic structure directly into the data-generating mechanism, so price and volume are never independently generated quantities that might disagree with each other; they are the necessary output of orders clearing against each other in a real order book.

## 1.3 Revised Three-Module Structure

The project keeps the same three-module skeleton as the original plan, but Module 1 is now an agent-based market simulator instead of a stochastic-process-plus-GAN simulator.

- Module 1 — Agent-Based Market & Manipulation Simulator: a limit order book matching engine populated by rule-based trading agents, whose interaction produces price and volume as an emergent, mechanically-consistent output; manipulative agents are injected into the same book.
- Module 2 — Manipulation Detector: unsupervised (Isolation Forest / statistical) and supervised (XGBoost + SHAP) layers operating on engineered, window-based features computed from the order/trade log Module 1 produces.
- Module 3 — Adversarial Evaluation Loop: round-based arms race where manipulative agent strategies are mutated to evade the current detector, and the detector is retrained and re-measured each round.

**One-sentence pitch for your proposal / viva**

We built a synthetic commodities market as an agent-based simulation — a limit order book matched against rule-based trader agents — because real exchanges never publish manipulation ground truth, and because price must emerge from genuine order flow rather than be statistically mimicked, so that our detector is learning real mechanism violations and not generative artefacts.

# 2\. Why Agent-Based Modelling Instead of GAN

## 2.1 Side-by-Side Comparison

| **Dimension**                                  | **GAN / TimeGAN approach**                                                                                                                        | **Agent-Based Model (ABM)**                                                                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **What generates price**                       | A neural network trained to statistically resemble historical price curves                                                                        | A matching engine clearing real buy/sell orders — price is a computed consequence, not a learned output                                          |
| **Can it violate supply/demand logic?**        | Yes — the network has no concept of an order book and can output any sequence that fools the discriminator                                        | No — every trade is the result of two real orders crossing; price cannot exist without the orders that produced it                               |
| **Training stability risk**                    | Real and well-documented; GAN/WGAN-GP models are reported to collapse during backtest evaluation, producing implausible return/volatility figures | None — there is no adversarial training loop unless RL agents are deliberately added as a stretch goal                                           |
| **How manipulation interacts with the market** | Pasted on afterward — a synthetic burst added on top of an already-generated curve                                                                | Genuinely competes for execution against real agent orders — price impact is causal, not asserted                                                |
| **Controllability for Module 3 mutation**      | Lower — the learned distribution is a black box, hard to "dial" precisely                                                                         | High — every manipulative agent parameter (size, timing, stealth) is directly tunable code                                                       |
| **Implementation difficulty**                  | Medium–high, particularly for TimeGAN; failure mode is often silent (looks fine on a plot, fails on backtest)                                     | Low–medium — data structures plus simple decision rules; every component is independently inspectable and debuggable                             |
| **Precedent for this exact use case**          | Used mainly for data augmentation and scenario generation, not specifically for manipulation mechanics                                            | Directly precedented — published research uses agent-based models specifically to simulate spoofing and quantify its effect on market efficiency |
| **Defensibility in viva**                      | "We trained a network on a proxy dataset and hoped the learned distribution transfers"                                                            | "We implemented the actual mechanism — a continuous double auction — that real exchanges use to clear orders"                                    |

## 2.2 The Underlying Principle

A GAN learns to imitate what manipulation-free market data looks like, statistically. An agent-based model constructs a system that actually behaves like a market — manipulation is detectable because it represents a genuine deviation in mechanism and incentive, not because it looks different from a learned baseline distribution. For a fraud/manipulation detection project specifically, this distinction is the entire point: your detector's job is to notice when behaviour breaks the implicit rules of legitimate trading, and that statement is only meaningful if "legitimate trading" in your dataset was never itself an approximation.

## 2.3 Where a Learned Component Can Still Fit (Optional, Not Core)

If the team wants a clear machine-learning component inside the simulator itself (beyond the detector in Module 2), the strongest option is not a generative network but reinforcement learning applied to one or two specific agent roles — for example, a market-making agent that learns to manage inventory risk, or a momentum agent that learns position sizing. This keeps the economically-grounded mechanism (the order book) as the source of truth, while still giving the team a genuine RL component to discuss. This is treated as an optional Phase 2 enhancement in Section 8, never as a dependency for the core deliverable.

# 3\. Module 1 — Agent-Based Market & Manipulation Simulator

This module has four components, each described with what to build, how to build it, and what the expected output/behaviour should look like so the team can self-check progress.

## 3.1 Component A — The Limit Order Book (Matching Engine)

This is pure data structures and control flow — no machine learning, no randomness beyond what agents feed into it. It is the foundation everything else sits on.

### What it is

A central object that holds two ordered collections of outstanding orders:

- Bids (buy orders), sorted by price descending, then time ascending (price-time priority)
- Asks (sell orders), sorted by price ascending, then time ascending

Each order is a record with: order_id, trader_id, side (buy/sell), price, quantity, timestamp, and order_type (limit or cancel).

### What it does, step by step

1. Receive an incoming order from an agent.
2. If it is a cancel instruction, remove the referenced order_id from the book if still resting.
3. If it is a new buy order, check whether its price is ≥ the best (lowest) resting ask price.
4. If it crosses, execute a trade at the resting order's price (not the incoming order's price — this is standard price-time priority and matters for realism), for the minimum of the two quantities.
5. Partially filled orders remain on the book with reduced quantity; fully filled orders are removed.
6. If it does not cross, the order rests on the book at its specified price.
7. Record every trade (price, quantity, buy trader_id, sell trader_id, timestamp) to a trade log, and every order event (placed, cancelled, filled, partially filled) to an order log.

### Expected output

Two append-only logs per simulation run:

- trade_log: one row per executed trade — timestamp, price, quantity, buyer_id, seller_id, trade_id
- order_log: one row per order event — timestamp, order_id, trader_id, side, price, quantity, event_type (placed / cancelled / filled / partial_fill)

From trade_log alone you can already reconstruct a standard OHLCV-style price series by resampling into bars (e.g., 1-minute bars: open = first trade price in the window, high/low = max/min, close = last, volume = sum of quantities). This is the bridge back to the familiar format your detector features will be computed from.

**Why this matters for the rest of the project**

Because every later component (manipulation injection, feature engineering, the detector, the adversarial loop) reads from these two logs, the order book is the single source of truth. If the book is implemented correctly, it is structurally impossible for price to move without a real order causing it — this is the property a GAN cannot give you.

## 3.2 Component B — Legitimate Trader Agents

Each agent is a small, explainable rule: given the current state of the book (best bid, best ask, recent price history, its own inventory), decide whether to submit an order, and if so, what kind.

### Agent archetype 1 — Noise Trader

- Behaviour: places small buy or sell limit orders at random prices close to the current best bid/ask, with random order size.
- Purpose: provides baseline liquidity and realistic order-arrival randomness; prevents the book from being too "clean" or predictable.
- Expected output: a steady trickle of small trades throughout the simulation with no directional bias over time.

### Agent archetype 2 — Momentum Trader

- Behaviour: computes the price change over the last N ticks; if positive beyond some threshold, places a buy order; if negative, places a sell order. Order size can scale with the strength of the recent move.
- Purpose: creates trend-following behaviour, a well-documented real effect in commodity markets, and is also the mechanism that pump-and-dump manipulation will deliberately exploit later.
- Expected output: short-lived price trends that emerge and fade as momentum traders pile on and then run out of conviction — this is exactly the kind of organic-looking trend a pump-and-dump scheme will try to imitate, which is what makes the detection problem genuinely hard and interesting.

### Agent archetype 3 — Mean-Reversion (Fundamental Value) Trader

- Behaviour: maintains a notion of "fundamental value" for the commodity (a slowly-drifting reference price representing something like production cost or a long-run anchor); buys when the market price is below this anchor by more than some margin, sells when above.
- Purpose: this is the component that gives you the Ornstein-Uhlenbeck-style mean reversion from the original plan — except now it is an emergent property of agent incentives rather than a formula imposed directly on the price series. This is a substantially stronger statement for a viva: "mean reversion emerges because rational agents trade against deviations from fundamental value," not "we forced mean reversion into our price formula."
- Expected output: price drifts away from the fundamental anchor over short windows (driven by momentum/noise traders) but is pulled back over longer windows, producing the same statistical signature (autocorrelated, mean-reverting returns) that real commodity prices show — but caused by something economically real in the simulation.

### Agent archetype 4 — Market Maker (recommended, not optional)

- Behaviour: continuously quotes both a bid and an ask around the current mid-price, profiting from the bid-ask spread; adjusts its quotes based on its own inventory (if it has bought too much, it skews quotes to encourage selling, and vice versa).
- Purpose: real order books always have liquidity providers; without one, the book can become thin, jumpy, and unrealistic, especially during the manipulation windows you care about most.
- Expected output: a consistently populated book with a visible bid-ask spread at all times, and noticeably wider spreads / more cautious quoting immediately after large or unusual order flow — itself a useful detector feature.

### How agents are activated each tick

At each simulation tick, a subset of agents is activated probabilistically rather than every agent acting every tick (this is what real, continuous markets look like — sparse, asynchronous arrivals, not synchronized batches). A simple Poisson-process arrival rate per agent type is sufficient and standard: noise traders arrive most frequently, momentum and mean-reversion traders less so, and the market maker is effectively always present (updates its quotes very frequently).

## 3.3 Component C — Manipulation Personas (Injected Agents)

This is the key structural improvement over the original plan: manipulation is no longer a synthetic event pasted onto a price curve after generation. It is implemented as additional agents that submit real orders into the same matching engine as everyone else, and therefore their price impact is genuine — caused by the mechanism, not asserted by the simulation author.

### Persona 1 — Spoofing Agent

- Mechanism: places one or more large limit orders on one side of the book (e.g., a large buy order well above where it would expect to actually be filled), which shifts the visible book imbalance and can influence other agents (especially momentum traders, who may react to the apparent demand) — then cancels the order shortly after, before it is filled, once it has had the desired effect.
- Parameters to expose for later mutation: order size (relative to average book depth), cancellation delay (how many ticks before cancelling), frequency (how often the agent repeats this), and price aggressiveness (how far from the mid-price the spoof order is placed).
- Expected output / detectable signature: an unusually high order cancellation rate for this trader_id within short windows, paired with a brief, anomalous shift in book imbalance that does not correspond to any executed volume.

### Persona 2 — Wash Trading Agent Pair

- Mechanism: two trader_ids, controlled as a colluding pair, alternately place matching buy and sell orders that cross with each other (directly, or via the book) at prices very close to the prevailing price, with no net change in either party's true position over time.
- Parameters to expose for later mutation: trade frequency, price deviation from mid (how far from "fair" the wash price is set), and number of colluding account pairs used simultaneously (spreading activity across more pairs is the natural stealth mutation).
- Expected output / detectable signature: elevated trading volume concentrated between a small, recurring set of trader_id pairs, with each party's net inventory position staying flat over the relevant window despite high gross activity — a volume/inventory-change mismatch that legitimate trading does not produce.

### Persona 3 — Pump-and-Dump Coalition

- Mechanism: a coordinated group of trader_ids (controlled together) submits a burst of correlated buy orders over a short window, deliberately timed to trigger momentum traders into following the move (because the order book correctly reflects the coalition's real buying pressure — this is what makes it dangerous and realistic, not a scripted bump). Once price has risen and organic momentum volume has joined in, the coalition reverses and sells into the rally.
- Parameters to expose for later mutation: burst duration (sharp vs. drawn-out accumulation), number of coordinated accounts, and the delay before the dump phase begins.
- Expected output / detectable signature: a price run-up with volume concentrated among a small number of accounts in the early phase, followed by a sharp reversal coinciding with selling concentrated among the same small set of accounts — visible as a spike in volume-concentration-by-account features right before a trend reversal.

**Why this is mechanically stronger than the original (non-ABM) plan**

In the original design, a pump-and-dump was "a coordinated burst of correlated buy orders... followed by a sell-off" added on top of an already-simulated price path. In this design, that burst is competing for execution in the same book as every legitimate trader, so the price actually moves because of real demand — and critically, real momentum traders genuinely (and innocently) join in, which is exactly what makes pump-and-dump schemes work in real markets and exactly what makes them hard to tell apart from organic rallies. That difficulty is a feature for your project: an easy detection problem is not an interesting one.

### Persona 4 — Layering (optional extension of spoofing)

- Mechanism: multiple spoofing orders placed simultaneously at several price levels on the same side, creating a false impression of deep supply or demand without any single order being unusually large.
- Note: implement this only after Persona 1 (spoofing) is working end-to-end — it is a parametric variation (multiple smaller orders across levels instead of one large order) rather than new mechanism, so it is a cheap addition once spoofing exists.

### Persona 5 — Cornering / Squeeze (optional, advanced, lowest priority)

- Mechanism: a single large agent (or coalition) accumulates a dominant share of available supply over a long window, then exploits that position to control price, typically by restricting the supply available to others.
- Note: mark explicitly as a stretch goal only — it requires modelling supply constraints (e.g., a finite deliverable quantity per period) that the simpler personas do not need, and is not necessary to demonstrate the core thesis.

## 3.4 Component D — Calibrating the Simulation to Real Commodities

The agents' behaviour needs realistic parameters (how volatile is normal noise-trading, how strong is momentum, where does fundamental value sit) — these should not be guessed, they should be estimated from real data so the simulation's overall statistical behaviour is grounded.

### Step 1 — Pull real reference data

- Use the yfinance Python library (free, no signup) to pull daily OHLCV data for 2–3 commodity futures with different characters, e.g. Gold (GC=F), Crude Oil (CL=F), Wheat (ZW=F).
- Expected output: a clean daily price/volume history per commodity, several years deep, used only for calibration — not fed directly into the detector.

### Step 2 — Estimate statistical targets from real data

- Daily volatility: standard deviation of log returns.
- Drift: average log return over the sample.
- Mean-reversion strength: fit an Ornstein-Uhlenbeck process to log price (or estimate via an autoregressive model) to get a reversion speed parameter — this number becomes the target your mean-reversion agents' aggressiveness should reproduce, not a formula imposed directly on the simulated price.
- Volume distribution: typical daily volume and its variability, used to scale how large "typical" non-manipulative orders should be relative to manipulative ones.

### Step 3 — Tune agent parameters to match

Run the simulator with an initial guess at agent parameters (arrival rates, order size distributions, momentum/reversion sensitivity), then adjust until the simulated trade_log, resampled into daily bars, reproduces volatility and autocorrelation statistics reasonably close to the real calibration targets from Step 2. This does not need to be exact — it needs to be defensibly close, with the comparison shown in the report.

### Step 4 — Optional grounding from CFTC data

- CFTC Commitments of Traders reports (cftc.gov, free) give aggregate positioning by trader category (commercial/speculative), which can justify realistic proportions between your momentum/noise/mean-reversion agent population sizes.
- CFTC enforcement action summaries (cftc.gov, free, qualitative) describe real spoofing and wash-trading cases, including documented findings that real spoofers learned to randomize order size and timing to evade early surveillance systems — this is the literature grounding for your Module 3 mutation design (Section 5) and should be cited in your literature review.

### Expected output of the whole calibration step

A short calibration report (can be a notebook with plots) showing: real commodity statistics side by side with simulated statistics, the chosen agent parameters, and a brief justification of any deliberate deviations. This single artifact answers the "how do you know your simulation is realistic" question before an examiner asks it.

## 3.5 Putting It Together — The Simulation Loop

This is the orchestration code that ties Components A–C into a runnable simulation.

1. Initialise the order book (empty) and instantiate the agent population: a configurable number of noise traders, momentum traders, mean-reversion traders, one market maker, and zero or more manipulation personas (configurable per run).
2. Set the fundamental value anchor and its drift process (a slow random walk representing changing long-run "cost of production" / supply-demand fundamentals over the simulation horizon).
3. For each tick from 1 to T: sample which agents are activated this tick (Poisson arrivals); for each activated agent, call its decision function with the current book state and let it submit an order or cancellation to the order book; the order book processes the order/cancellation immediately and updates its state and logs.
4. If manipulation personas are configured to run in a specific window (e.g., "spoofing active between tick 2000 and tick 2400"), activate them only within that window, and record the ground-truth label (start tick, end tick, persona type, persona parameters, involved trader_ids) to a separate manipulation_events log — this is your label source for Module 2.
5. After the run completes, resample trade_log into fixed-width bars (e.g., 1-minute) to produce a standard OHLCV-style series for visualisation and for any feature that needs bar-level aggregation.

### Expected output of a complete simulation run

- trade_log.csv — every executed trade
- order_log.csv — every order event (placed/cancelled/filled)
- manipulation_events.csv — ground-truth windows: which manipulation persona was active, when, with which trader_ids and parameters
- ohlcv_bars.csv — resampled bar-level price/volume series, for quick visual sanity-checking and for any bar-level features

**Sanity check before moving to Module 2**

Plot price over time with manipulation windows shaded. A spoofing window should show a brief, unusual book imbalance with no lasting price move; a pump-and-dump window should show a genuine price rise that then reverses; a wash-trading window should show a volume spike with little net price movement. If these don't visually show up as described, the agent parameters need tuning before any detector work begins — building a detector on a simulator that doesn't yet behave as intended just wastes Module 2's time.

## 3.6 Module 1 Technology Stack

| **Purpose**                        | **Tools**                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Order book / matching engine**   | Pure Python (a heap or sorted-list-backed class is sufficient at this scale; no external dependency required)                                                                                                                                                                                                                                      |
| **Agent logic**                    | Pure Python classes, one per archetype, sharing a common interface (e.g., a decide(book_state) method)                                                                                                                                                                                                                                             |
| **Calibration data**               | yfinance for OHLCV; pandas for analysis; NumPy/SciPy for OU-process fitting                                                                                                                                                                                                                                                                        |
| **Logging**                        | pandas DataFrames written to CSV or Parquet at the end of each run                                                                                                                                                                                                                                                                                 |
| **Visualisation / sanity checks**  | matplotlib / plotly for price-with-manipulation-window plots                                                                                                                                                                                                                                                                                       |
| **(Optional) Reference framework** | ABIDES (Agent-Based Interactive Discrete Event Simulation) can be consulted as a reference implementation of a realistic limit order book simulator if the team wants a head start or a benchmark to compare against — but a hand-rolled implementation is recommended as the primary path because it is easier to explain every line of in a viva |

# 4\. Module 2 — Manipulation Detector

This module is largely unchanged in spirit from the original plan — it consumes whatever Module 1 produces (now: trade_log, order_log, manipulation_events) and is agnostic to how that data was generated, provided the logs have the right shape. It has four steps.

## 4.1 Step 1 — Feature Engineering

All features are computed per time-window (e.g., rolling 1–5 minute windows) per trader_id, or per window for the whole market, depending on the feature. This is where the order-book-level richness of Module 1 pays off — these features could not be computed from flat OHLCV rows alone.

- Order cancellation rate: (orders cancelled) / (orders placed) per trader_id per window — the primary spoofing signal.
- Volume concentration: share of total window volume attributable to the top-k most active trader_ids — the primary wash-trading and pump-and-dump-coalition signal.
- Net position vs. gross volume ratio: for a given trader_id, how much their position actually changed vs. how much they traded — a low ratio with high gross volume is a wash-trading signature.
- Price-volume divergence: price change in a window relative to the volume that accompanied it — price moving without proportional genuine participation is a classic manipulation signal (and also a feature that would flag a badly-calibrated simulator, which is a useful side benefit).
- Bid-ask imbalance spikes: sudden, short-lived changes in the resting volume on one side of the book relative to the other — the primary spoofing/layering signal, directly available because Module 1 logs the full book, not just trade prices.
- Order size variance over time, per trader: increasing randomisation of order sizes from a given trader_id over the course of a run is a documented stealth-evasion pattern (see Section 3.4's CFTC grounding) and becomes especially relevant once Module 3's mutation rounds begin.

## 4.2 Step 2 — Unsupervised Layer

- Model: Isolation Forest (or a simpler statistical control-chart / z-score threshold approach as a baseline to compare against) trained on the engineered features, without using the manipulation_events labels.
- Purpose: this layer is meant to catch genuinely novel or unknown manipulation patterns — ones that don't resemble anything the supervised layer was trained on. It is also what you would actually deploy in a setting with no labels at all, which is worth stating explicitly since it mirrors real-world surveillance constraints.
- Expected output: an anomaly score per window per trader_id; a chosen threshold converts this into a binary flag.

## 4.3 Step 3 — Supervised Layer

- Model: XGBoost or LightGBM, trained on the engineered window-level features with labels taken directly from Module 1's manipulation_events log (which window/trader overlaps a known manipulation event, and which persona type).
- Explainability: SHAP values computed per prediction, so that for any flagged window you can show which features drove the flag (e.g., "this window was flagged primarily due to cancellation rate and bid-ask imbalance") — strong demo material and directly useful for the price-impact-avoided narrative in Section 6.
- Important framing choice: train and report per manipulation type separately (spoofing vs. wash trading vs. pump-and-dump), not as one lumped "manipulation/not-manipulation" label. A detector can be excellent at one pattern and blind to another, and that asymmetry is itself a genuinely interesting, reportable finding.

## 4.4 Step 4 — Combining the Layers

- Combination rule: flag a window if either layer fires, or use a weighted ensemble score — this mirrors how real surveillance systems stack multiple independent signals rather than relying on a single model.
- Expected output: a single per-window, per-trader manipulation score and flag, with an audit trail of which sub-layer(s) triggered and why (via SHAP for the supervised layer, via the specific anomalous feature for the unsupervised layer).

**Optional credibility step (not required, but cheap and worthwhile)**

Validate that your engineered features (volume concentration, timing irregularity, cancellation rate analogues) are not meaningless outside your own simulator by testing a version of them against a real labelled fraud dataset such as the Kaggle Credit Card Fraud or IEEE-CIS Fraud datasets. This doesn't validate manipulation detection directly (the domains differ) — it validates that your feature engineering approach generalises at all, which is a reasonable thing to show in an appendix.

## 4.5 Module 2 Technology Stack

| **Purpose**             | **Tools**                                             |
| ----------------------- | ----------------------------------------------------- |
| **Feature engineering** | pandas, NumPy (rolling windows, groupby on trader_id) |
| **Unsupervised layer**  | scikit-learn (IsolationForest)                        |
| **Supervised layer**    | XGBoost or LightGBM                                   |
| **Explainability**      | SHAP                                                  |
| **Evaluation**          | scikit-learn metrics (precision/recall/F1, per class) |

# 5\. Module 3 — Adversarial Evaluation Loop

This is the project's core novelty and is structurally unchanged from the original plan — but it is now more credible, because the manipulation personas being mutated are real agents with tunable parameters inside a real mechanism, not synthetic events being re-scripted.

## 5.1 Step 1 — Establish Baseline (Round 0)

1. Run Module 1 with manipulation personas at their initial ("obvious") parameter settings — e.g., spoofing with large, rare orders; wash trading between a single obvious account pair; pump-and-dump as a sharp, short burst.
2. Train the Module 2 detector on this Round 0 data.
3. Record precision, recall, and F1 per manipulation type (not lumped together) as the baseline.

Expected output: a baseline performance table the team can show improves — and degrades — over subsequent rounds; this is the reference point every later round is measured against.

## 5.2 Step 2 — Design the Mutation Strategy

For each persona, author 3–5 increasingly subtle parameter variants, manually, grounded in documented real-world evasion patterns where possible:

- Spoofing: large, rare orders → smaller, more frequent orders → randomised order sizes and cancellation timing (this last step directly mirrors the CFTC-documented finding, cited in Section 3.4, that real spoofers learned to randomise to evade early surveillance systems).
- Wash trading: a single obvious colluding pair → trades spread across more colluding account pairs → timing varied to look organically spaced rather than mechanically regular.
- Pump-and-dump: a sharp, short coordinated burst → a slower accumulation phase deliberately paced to resemble genuine momentum-trader behaviour → a dump phase staggered across more accounts to avoid a single obvious sell-off signature.

## 5.3 Step 3 — Run the Rounds

For each round (after Round 0):

1. Generate new Module 1 data using the current round's mutated persona parameters.
2. Test the existing (previous round's) detector against this new data without retraining, and record the degradation in precision/recall/F1 per manipulation type — this number is your direct, quantified measure of how much a given evasion tactic actually works against the current detector.
3. Retrain the detector on the new round's data (optionally combined with prior rounds' data) and re-measure performance — this is your recovery measurement.
4. Log everything: round number, persona parameters used, pre-retrain performance, post-retrain performance, and the SHAP feature importances both before and after retraining (a shift in which features matter most, round to round, is itself a reportable finding — it shows the detector adapting its reasoning, not just its threshold).

## 5.4 Step 4 — Optional: Automate the Mutation Search

Once the manual-mutation version works end-to-end, an optional stretch enhancement is to replace hand-authored mutation steps with a simple search procedure — e.g., random search or a small genetic algorithm over persona parameters, with the search objective being explicitly to minimise the current detector's recall for that persona type. This converts the adversarial loop from "we guessed some evasions" to "we searched for evasions automatically," which is a stronger research claim, but it adds real implementation complexity (defining a search space, a fitness function, a budget) and should be attempted only after the manual version is complete and time-boxed strictly, with manual mutation as the guaranteed fallback.

## 5.5 Expected Pattern of Results (what "good" looks like)

| **Round**                                 | **Expected detector behaviour**                                                                                                               | **Why this is the expected pattern**                                                                                                                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Round 0 (baseline)**                    | High precision/recall on obvious manipulation patterns                                                                                        | Obvious spoofing/wash-trading/pump-and-dump signatures are exactly what the engineered features were designed to catch                                                                                       |
| **Round 1 (first mutation), pre-retrain** | Noticeable drop in recall for the mutated persona specifically; other personas largely unaffected                                             | The detector was trained on Round 0's specific signature, and the mutation was designed to move just outside that signature's typical range                                                                  |
| **Round 1, post-retrain**                 | Substantial recovery, though possibly not back to Round 0 levels                                                                              | The detector has now seen the mutated pattern and can learn its signature, but subtler patterns may genuinely overlap more with legitimate trading, capping achievable precision                             |
| **Later rounds (subtler mutations)**      | Smaller absolute drops, but slower/incomplete recovery, and a possible rise in false positives as the detector is pushed to be more sensitive | As manipulation is designed to look more like legitimate trading, the achievable precision-recall trade-off genuinely worsens — this is a real phenomenon in market surveillance, not a flaw in your project |

**If your results don't look like this**

If recall never drops at all across rounds, your mutations probably weren't subtle enough to actually challenge the engineered features — strengthen them. If recall collapses to zero and never recovers even after retraining, check whether the mutated persona's behaviour has become statistically indistinguishable from legitimate trading at the feature level you're using — if so, that's a genuine, reportable limitation, not a bug, and is worth a paragraph in your discussion section.

# 6\. Metrics & Reporting — What to Measure and Why

## 6.1 Core ML Metrics (per manipulation type, per round)

- Precision, Recall, F1 — reported separately for spoofing, wash trading, and pump-and-dump, never lumped into one "fraud" label, since a detector's strengths and blind spots differ meaningfully by pattern type.
- False Positive Rate — weighted carefully in this domain specifically: flagging a legitimate large institutional trade as manipulation carries real reputational and operational cost in a real surveillance system, so this number deserves its own discussion, not just a footnote.
- Detection latency — measured in ticks or minutes: how long after a manipulation event begins does the detector first flag it. This matters because in real surveillance, a manipulation pattern detected after it has already completed is far less useful than one caught mid-execution.

## 6.2 The Economically-Meaningful Metric — Price Impact Avoided

This metric ties the ML performance numbers to something a non-technical examiner or interviewer can immediately understand, the same way a cost-savings figure does for a forecasting project.

1. For each detected manipulation event, identify the tick at which the detector first flagged it (detection latency, from 6.1).
2. Using Module 1's order book (which is deterministic and replayable), simulate two counterfactuals from that point forward: (a) the manipulation persona is allowed to continue unimpeded for N more ticks, and (b) the manipulation persona is halted at the detection tick (representing an intervention).
3. Measure the difference in cumulative price deviation from the fundamental value anchor between the two counterfactuals — this difference is the "price impact avoided" by early detection.
4. Aggregate this across all detected events in a round to get a single, reportable number: "our detector's early intervention avoided an estimated X% additional price deviation across Y detected manipulation episodes in this round."

**Why this metric is uniquely available because of the ABM approach**

This counterfactual replay is only possible because Module 1 is a deterministic, mechanistic simulation you can re-run forward from any point with a modified agent population. A GAN-generated price series has no equivalent — you cannot "remove" a manipulation pattern from a learned curve and ask what would have happened instead, because the curve was never decomposed into causes in the first place. This is one of the clearest practical payoffs of the architecture choice in Section 2, and is worth stating explicitly in the report.

## 6.3 Full Reporting Checklist

- Per-round, per-manipulation-type precision/recall/F1 table
- False positive rate, with a short discussion of its real-world cost
- Detection latency distribution per manipulation type
- Price impact avoided, aggregated per round
- SHAP feature importance comparison across rounds (which features the detector relies on, and how that shifts as manipulation mutates)
- Calibration validation plots (Section 3.4) showing simulated vs. real commodity statistics
- Sanity-check plots (Section 3.5) showing price with manipulation windows shaded

# 7\. Datasets Used — Complete Summary

This project uses no paywalled or proprietary data anywhere. Every external dataset is free and used only for calibration/grounding, never as the primary training signal (which always comes from Module 1's own simulation output, since real manipulation ground truth does not exist).

| **Source**                                                                                 | **What is pulled**                                                                                      | **Used for**                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **yfinance (Python library, free, no signup)**                                             | Daily OHLCV for 2–3 commodity futures (e.g., Gold GC=F, Crude Oil CL=F, Wheat ZW=F), several years deep | Calibrating volatility, drift, and mean-reversion strength targets for Module 1's agent parameters (Section 3.4)                                                                                    |
| **CFTC Commitments of Traders reports (cftc.gov, free)**                                   | Aggregate positioning by trader category (commercial/speculative)                                       | Justifying realistic proportions between agent archetype populations in Module 1                                                                                                                    |
| **CFTC enforcement action summaries (cftc.gov, free, qualitative)**                        | Real, documented spoofing and wash-trading case descriptions                                            | Grounding persona design (Section 3.3) and mutation design (Section 5.2) in real evasion patterns, for the literature review                                                                        |
| **Module 1's own simulation output**                                                       | trade_log, order_log, manipulation_events, ohlcv_bars — generated entirely in-house, round after round  | The primary, and only, source of labelled training data for Module 2 and the entire Module 3 adversarial loop                                                                                       |
| **(Optional, credibility check only) Kaggle Credit Card Fraud or IEEE-CIS Fraud datasets** | Real, labelled (non-market) fraud data                                                                  | Sanity-checking that the team's general feature-engineering approach (timing irregularity, volume concentration) generalises at all outside the simulator — optional appendix material, Section 4.4 |

# 8\. Optional Phase 2 — Reinforcement Learning Agents (Stretch Goal Only)

This is explicitly optional and should only be attempted after Modules 1–3 are working end-to-end with rule-based agents. It is the recommended way to add a deep-learning component to the simulator itself, if the team wants one, without reintroducing the instability risk that ruled out the GAN approach.

## 8.1 What Changes

- One or two specific agent roles (recommended: the market maker, and optionally the momentum trader) are converted from fixed rules into reinforcement learning policies.
- Market-making RL agent: observes book state (best bid/ask, recent volume, its own inventory), chooses bid/ask quote levels and sizes as its action, and is rewarded for profit while penalised for inventory risk and large unrealised losses — this mirrors published market-making RL formulations using algorithms such as PPO.
- Momentum RL agent (optional, second priority): observes recent price/volume trend and order book imbalance, chooses position size as its action, rewarded for realised profit — this is the closer analogue to published work using RL agents trained against a realistic limit order book for execution-style tasks.

## 8.2 Why This Is Safer Than a GAN, If You Want a Deep Learning Component

An RL agent trained inside the ABM is learning a policy (a decision rule) within a mechanism that is already economically valid — it cannot "learn" its way into violating the order book's matching logic, because the matching engine enforces that regardless of what the agent decides. This is fundamentally different from a GAN, which is learning to produce the data itself with no external mechanism enforcing consistency. If RL training is unstable or doesn't converge well, the fallback is simply to keep that one agent rule-based — the rest of the simulation, the detector, and the adversarial loop are entirely unaffected, because they only depend on the order book's logs, not on how any individual agent decided to act.

## 8.3 Implementation Notes

- Algorithm: Proximal Policy Optimisation (PPO) is the standard, stable choice for this kind of continuous-control-flavoured trading task in the published literature.
- Library: Stable-Baselines3 (built on PyTorch) is the standard, well-documented choice for a team with no prior RL experience.
- Training loop: train the RL agent inside many repeated short simulation runs (episodes), with the rest of the agent population (noise/momentum/mean-reversion/manipulation personas) held fixed during training so the RL agent has a stable environment to learn against.
- Time-box: strictly 2–3 weeks, attempted only after the core deliverable (Modules 1–3, rule-based) is complete. If it doesn't converge to sensible behaviour in that window, document the attempt and the specific issue encountered — this is legitimate, examinable content in its own right, and arguably more credible than a black-box result the team can't fully explain.

# 9\. Team Ownership & Division of Work

Same clean three-way separation as the original plan — each person owns an independently strong, independently explainable piece of work, with clear interfaces between modules (defined by the log file formats in Sections 3.1 and 3.5).

| **Person**   | **Owns**                                                                                      | **Concrete outputs**                                                                                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Person A** | Module 1 — Order book engine, legitimate agent archetypes, calibration to real commodity data | Order book class; noise/momentum/mean-reversion/market-maker agent classes; calibration notebook comparing simulated vs. real statistics; trade_log, order_log, ohlcv_bars outputs                               |
| **Person B** | Module 1 (manipulation personas) + Module 2 — feature engineering and the full detector       | Spoofing / wash-trading / pump-and-dump persona agent classes with tunable parameters; manipulation_events label log; engineered feature pipeline; Isolation Forest + XGBoost detector with SHAP                 |
| **Person C** | Module 3 — adversarial round orchestration, mutation design, full metrics suite, dashboard    | Round orchestration code; manually-authored (and optionally searched) mutation parameter sets per persona; per-round metrics logging (precision/recall/F1/FPR/latency/price-impact-avoided); Streamlit dashboard |

Note on the manipulation persona split: Person A and Person B's work meets at the order book interface — manipulation personas are just agents, so Person B can build and test them against Person A's order book independently once the order book's interface (the decide(book_state) pattern from Section 3.2) is fixed early. Agreeing on that interface in Week 1 is the single most important coordination step in the whole project.

# 10\. Week-by-Week Timeline (10–12 Weeks)

| **Weeks** | **Activities & Expected Output**                                                                                                                                                                                                                                                                                                                                                    |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1–2**   | Study real manipulation enforcement cases (CFTC) for grounding; pull and explore yfinance calibration data; design and freeze the order book interface and log schemas (trade_log, order_log, manipulation_events) so all three people can work independently afterward. Output: agreed data schemas, calibration target numbers, initial repo structure.                           |
| **3–4**   | Person A builds the order book engine and the four legitimate agent archetypes; runs first uncalibrated simulations. Person B starts designing manipulation persona logic in parallel against the agreed interface, and begins feature engineering design. Output: a runnable simulation producing plausible-looking (not yet calibrated) price/volume behaviour.                   |
| **5–6**   | Person A finishes calibration against real commodity statistics (Section 3.4). Person B finishes manipulation personas and the full feature pipeline, produces Round 0 labelled data. Person C begins building round orchestration and starts drafting the first mutation parameter sets. Output: calibrated simulator; Round 0 detector trained with baseline precision/recall/F1. |
| **7–8**   | Run the full multi-round adversarial loop (Section 5); refine mutation strategies for subtlety; log all per-round metrics. Output: a complete round-by-round performance table covering all manipulation types, with both pre-retrain degradation and post-retrain recovery measured.                                                                                               |
| **9**     | Stretch work, choose one: the price-impact-avoided metric (Section 6.2), the automated mutation search (Section 5.4), or the RL agent enhancement (Section 8). Output: one additional, clearly-scoped enhancement, fully documented even if only partially successful.                                                                                                              |
| **10**    | Build the Streamlit dashboard (live price chart with manipulation events highlighted, detector alerts overlaid, round-by-round performance curves); Dockerise the full pipeline. Output: a demoable, deployed dashboard.                                                                                                                                                            |
| **11–12** | Write the final report (problem framing, architecture, calibration validation, per-round results, discussion of limitations); prepare viva materials anchored on Sections 1–2 of this document for the architecture justification. Output: final report and viva-ready presentation.                                                                                                |

# 11\. Risk Register

| **Risk**                                                                               | **Why it could happen**                                                                                             | **Mitigation**                                                                                                                                                                                                               |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Simulated market doesn't match real statistics after calibration**                   | Agent parameters chosen without enough iteration against the real targets from Section 3.4                          | Treat calibration as its own milestone with a clear pass/fail check (Section 3.4, Step 4) before any detector work begins; budget Weeks 5–6 specifically for this, not just Weeks 3–4                                        |
| **Manipulation windows don't visibly differ from normal trading even before mutation** | Persona parameters too weak, or legitimate agent population too large/active relative to the persona's impact       | Use the sanity-check plots in Section 3.5 immediately after each persona is implemented, before moving on — catch this early, not after Module 2 is already built around weak signal                                         |
| **Detector recall collapses permanently after mutation and never recovers**            | A mutation made the persona statistically indistinguishable from legitimate trading at the feature level being used | This is a legitimate, reportable finding (see Section 5.5's callout), not necessarily a bug — but verify first that the manipulation_events ground truth itself is still being correctly generated and logged for that round |
| **RL agent (if attempted) doesn't converge**                                           | No prior RL experience on the team; reward shaping for trading tasks is genuinely fiddly                            | Strict 2–3 week time-box (Section 8.3); rule-based fallback for that agent role costs nothing to the rest of the pipeline since other modules only depend on the order book's logs                                           |
| **Coordination breaks down between Person A and Person B's work**                      | Manipulation personas and legitimate agents both depend on the same order book interface                            | Freeze the order book interface and log schemas in Week 1–2 before either person writes agent logic against it (Section 9's coordination note)                                                                               |

# 12\. Full Technology Stack Summary

| **Layer**                                | **Tools**                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Simulation core**                      | Pure Python (order book, agent classes); pandas for logging                                                                                                              |
| **Calibration**                          | yfinance, NumPy, SciPy (OU-process fitting)                                                                                                                              |
| **Detection — unsupervised**             | scikit-learn (IsolationForest)                                                                                                                                           |
| **Detection — supervised**               | XGBoost / LightGBM, SHAP                                                                                                                                                 |
| **Adversarial orchestration & tracking** | Custom Python orchestration scripts; MLflow for experiment logging across rounds (round number, parameters, metrics) — looks strong in both the report and the live demo |
| **Dashboard**                            | Streamlit — live price chart with manipulation events highlighted, detector alerts overlaid, round-by-round performance curves                                           |
| **(Optional) RL stretch goal**           | PyTorch via Stable-Baselines3 (PPO)                                                                                                                                      |
| **Deployment**                           | Docker — same as the team's existing experience                                                                                                                          |

# 13\. How to Talk About This Project

## 13.1 The One-Paragraph Pitch

"We built a synthetic commodities market as an agent-based simulation — a limit order book matched against rule-based trader agents — because real exchanges never publish manipulation ground truth, so any supervised detector has to be trained on simulated data with known labels. Rather than using a generative network to imitate historical price curves, which recent research shows can produce statistically realistic but economically inconsistent data, we modelled the actual mechanism (a continuous double auction) that real exchanges use, so that price and volume emerge from genuine order flow. We then injected known manipulation patterns — spoofing, wash trading, pump-and-dump — as additional agents competing in that same mechanism, grounded in real regulatory enforcement cases, and tested our detector under adversarial pressure as those manipulation strategies evolved to evade it round over round. This taught us that manipulation detection is not a static classification problem — it's a moving target, and we measured exactly how and why our detector's performance degraded, and what it took to recover."

## 13.2 Why This Beats the Generic "Fraud Detection" Pitch

- Less saturated: "credit card fraud detection" is one of the most repeated project phrases in student portfolios; "market manipulation detection via agent-based simulation" reads as deliberate and far less common.
- Real, regulator-recognised taxonomy: spoofing, wash trading, and pump-and-dump are documented in CFTC/SEC enforcement cases, giving the literature review genuine grounding instead of a generic "fraud happens" framing.
- A defensible architecture choice, not just a default one: being able to explain why a GAN was considered and rejected in favour of a mechanistic simulation — with the specific documented GAN failure mode cited — is a much stronger signal of engineering judgement than simply using whichever tool is most discussed online.
- Natural fit for sequence/window-based thinking: order-book and time-window features are a more sophisticated, more interview-relevant skill to demonstrate than flat single-row transaction classification.
- A genuinely novel evaluation contribution: the round-based adversarial loop, especially paired with the price-impact-avoided metric (Section 6.2), is closer to a small research contribution than a standard coursework project.

## 13.3 If Asked "Why Not Just Use a GAN, Everyone Else Does"

Answer directly with the documented failure mode: standard GAN/WGAN-GP models have been shown to collapse under backtesting, producing implausible outcomes precisely because the adversarial objective rewards statistical resemblance, not economic consistency. An agent-based model avoids this by construction — price is the output of a real matching mechanism, not a learned approximation of one. This is a stronger, more current answer than "we didn't have time to learn GANs," and it is true.