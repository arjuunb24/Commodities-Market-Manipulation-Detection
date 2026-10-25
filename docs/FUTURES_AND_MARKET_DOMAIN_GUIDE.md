# Futures and Market-Microstructure Guide for the Veridex Review

This guide is for presenting Veridex to a mixed technical/finance audience. It introduces the futures-market concepts the project relies on, maps them to what the simulator actually implements, explains why the manipulation patterns matter, and makes clear what the prototype does **not** model. It is educational project context, not investment or legal advice.

## 1. The project in one minute

> Veridex is a research prototype for detecting suspicious trading behavior in a simulated commodity-futures-style market. It generates orders and trades from seeded participant agents, converts each account's recent activity into overlapping time-window features, and scores each account-window for manipulation patterns. The simulator includes both suspicious strategies and legitimate look-alikes such as market makers and execution algorithms, so a high cancellation rate or large order alone is not enough. Results are evaluated against simulator-generated labels on separate random-seed markets, plus separately configured historical-case replays. This is not a production surveillance system or a proof of misconduct: it is a controlled test bed for detection logic.

The short version of the pipeline is:

```text
participants -> orders -> order book -> executions/trades
             -> account + market windows -> features -> model scores -> alerts/reviews
```

## 2. Futures: the financial instrument

### What is a futures contract?

A futures contract is a standardized agreement to buy or sell a defined quantity and grade of an underlying commodity at a specified future date, under exchange rules. The contract specification fixes details such as the commodity, unit of measure, quality/location, delivery or cash-settlement terms, expiry month, and minimum price increment (the **tick size**).

Examples of underlying commodities in Veridex are crude oil, natural gas, gold, silver, copper, wheat, and corn. The project uses commodity names and tick-size settings associated with futures markets, but the simulator itself is a stylized electronic order book. It does not instantiate the full exchange contract lifecycle.

### Why participants use futures

- **Hedging:** A producer can sell futures to offset the risk that the price of future output falls. A manufacturer can buy futures to offset the risk that a key input becomes more expensive.
- **Speculation:** A trader can take a view on price direction without owning or storing the physical commodity.
- **Price discovery and liquidity:** Trading aggregates participants' views and creates observable prices and executable depth.
- **Portfolio/risk management:** Futures can be used to adjust exposures efficiently, subject to contract and clearing rules.

Hedging does not mean a trader is risk-free. Futures positions still have price, basis, liquidity, and margin risk; a hedge can be imperfect if the futures contract and the physical exposure do not move together.

### Long, short, and P&L intuition

- A **long** position benefits when the futures price rises.
- A **short** position benefits when the futures price falls.
- Ignoring fees and financing, a simple contract P&L is approximately:

```text
P&L = price change × contract multiplier × number of contracts
```

The **contract multiplier** converts a quoted price move into monetary value. Its size and units differ by contract, so a price change of 1.00 does not mean the same dollar P&L in every commodity.

Veridex mostly tracks inventory as integer quantities and prices as simulation values. It does **not** apply real contract multipliers, exchange fees, brokerage fees, margin requirements, funding, or clearing cash flows. Its pump-and-dump P&L is a simplified mark-to-market calculation from the simulation's trades and a final observed price—not a complete futures account statement.

### Margin and daily settlement in a real futures market

Futures generally require **initial margin** as performance collateral, not the full notional value of the contract. If losses reduce account equity below **maintenance margin**, the clearing broker may require additional funds or reduce the position. Futures are typically marked to market through clearing processes, so gains and losses affect available equity over time.

These mechanisms strongly affect real participant behavior, forced liquidation, and risk. They are not represented in the current Veridex order-book simulation. Do not claim that its agent positions are constrained by realistic margin calls.

### Expiry, delivery, and settlement

Futures contracts expire. Depending on the contract, they may settle financially against a reference price or involve delivery procedures. Many traders close or roll positions before expiry. Settlement windows can be especially sensitive when a benchmark price is determined from trading during a defined period.

Veridex has no contract months, expiry, delivery, roll, clearing, or official settlement process. It simulates ticks in a market session; it does not calculate a daily settlement price. A case replay involving a closing-range or settlement mechanism is therefore a simplified behavioral scenario, not a full reproduction of the exchange's settlement process.

## 3. Prices, ticks, and commodity contract details

### Price versus tick size

The **tick size** is the smallest permitted price increment. For example, a contract quoted in 0.01 increments cannot have a valid order price of 0.005 above the current quote. A one-tick price move has different economic value depending on the contract multiplier.

The simulator snaps order prices onto its configured tick grid. Current values in [`src/core/calibration.py`](../src/core/calibration.py) are:

| Simulated commodity | Tick size configured |
|---|---:|
| WTI crude oil | 0.01 |
| Brent crude | 0.01 |
| Gold | 0.10 |
| Silver | 0.005 |
| Natural gas | 0.001 |
| Copper | 0.0005 |
| Wheat | 0.0025 |
| Corn | 0.0025 |

These are configured price increments, not a complete contract specification. The simulator does not encode contract multipliers, unit conventions, expiry calendars, delivery grades, or exchange session calendars for each product.

### Spot, futures, and basis

- **Spot price:** price for the commodity for near-immediate physical delivery, depending on the market/location/grade.
- **Futures price:** price for a specified future contract month.
- **Basis:** commonly, spot price minus the relevant futures price (definitions can vary by context).

The relationship between spot and futures can reflect storage, financing, insurance, convenience yield, seasonality, delivery location/grade, and supply-demand conditions. A futures curve with later months above nearer months is often called **contango**; one with later months below nearer months is often called **backwardation**. These are market-structure concepts, not necessarily a prediction of future spot prices.

Veridex's simulation uses a per-commodity fundamental price process and a single active book per simulated commodity. It does not create a spot market, a curve of delivery months, storage economics, or a dynamic basis.

## 4. How an electronic order book works

Veridex's [`src/core/order_book.py`](../src/core/order_book.py) implements a **continuous double auction**:

- Buyers submit bids; sellers submit offers/asks.
- Resting bids are ranked from highest to lowest price.
- Resting asks are ranked from lowest to highest price.
- At the same price, earlier arrival has priority (**price-time priority**).
- When a new order crosses the opposite best quote, it trades against the resting order at the resting order's price.
- If the incoming order is larger than the available quantity at that level, it can continue through multiple price levels. Any unfilled remainder rests on the book.
- Orders can be cancelled while still resting; time-to-live (TTL) expiry also generates cancellations for many agents.

### A tiny example

Suppose the best bid is 99.90 and the best ask is 100.00:

| Side | Price | Quantity | Meaning |
|---|---:|---:|---|
| Ask | 100.00 | 10 | Cheapest available seller |
| Bid | 99.90 | 12 | Highest waiting buyer |

The **spread** is `100.00 - 99.90 = 0.10`. A buyer who posts a bid at 99.90 joins the bid queue; that order does not immediately trade. A buyer submitting a limit order priced at 100.00 or higher can trade against the ask. If the buyer submits quantity 15 at a price that crosses, 10 can execute at 100.00 and any remaining 5 may continue to the next ask level or rest at the incoming limit price.

This example describes the simulator's matching convention, not all exchange-specific order types or queue rules.

### Essential order-book vocabulary

| Term | Plain meaning |
|---|---|
| **Limit order** | “Buy/sell this quantity, but no worse than my limit price.” It may rest without trading. |
| **Marketable limit order** | A limit order priced aggressively enough to trade against current quotes; unlike an unbounded market order, its limit constrains execution price. |
| **Resting/passive order** | An order waiting in the book, providing potential liquidity. |
| **Aggressive/taking order** | An incoming order that crosses and consumes resting liquidity. |
| **Best bid / best ask** | Highest current bid / lowest current ask. |
| **Mid-price** | Average of best bid and ask when both exist; a reference, not necessarily an executable price. |
| **Depth** | Quantity resting at one or more price levels. |
| **Liquidity** | How easily a position can be traded without a large price impact; it depends on spread, depth, resilience, and market conditions. |
| **Slippage / market impact** | Difference between an intended/reference price and actual execution prices, often increasing with urgency and size relative to depth. |
| **Partial fill** | Only some of an order's quantity executes; the remainder may remain resting. |
| **Order-to-trade ratio** | A broad activity measure comparing submitted order messages with executions. High values can be normal for electronic market making and are not proof of manipulation. |
| **Aggressor side** | Whether the incoming buyer or seller initiated the match. In Veridex this is recorded on each trade. |

### Why order messages matter more than price charts alone

A trade tape shows executions. It usually does not show all the intent and lifecycle context needed to understand displayed liquidity. Order-message data can show when orders were entered, changed, cancelled, partially filled, or filled. For surveillance, repeated patterns in the full order lifecycle can matter even if a price chart looks ordinary.

Veridex records both order events and trades. That enables features such as order lifetime, cancellation behavior, counterparties, aggressor-side flow, and whether a large cancelled order coincided with the same account's opposite-side execution. In real investigations, intent is assessed from broader evidence and applicable rules; a statistical alert by itself is not a legal conclusion.

## 5. Who is trading? Roles in the project and in markets

These are simplified archetypes. Real firms can have multiple strategies, accounts, products, and motives.

### Hedgers and commercial participants

Producers, processors, merchants, and consumers may use futures to offset business exposure. A grain producer may sell futures to reduce exposure to a possible fall in crop prices; a food manufacturer may buy futures to reduce exposure to rising input prices. Commercial flows may be large, scheduled, and not motivated by short-term speculation.

The simulator's mean-reversion/fundamental agents are only a stylized stand-in for value-sensitive commercial activity. They are not a full model of a producer's physical inventory, hedge ratio, procurement schedule, or basis exposure.

### Speculators and trend followers

Speculators take directional or relative-value risk seeking returns. Trend-followers/CTAs may respond to observed price movement and can reinforce a move. This creates a useful feedback mechanism for the simulation: activity can attract more activity. Veridex's momentum agents are simplified trend-followers, not a full CTA portfolio or risk system.

### Market makers and liquidity providers

Market makers quote both sides and seek to earn spread while managing inventory and adverse-selection risk. They often cancel and replace quotes as the market changes. Electronic liquidity providers may generate many order messages relative to executions.

In Veridex, the market maker quotes two-sided and skews quotes with inventory and visible book imbalance; HFT liquidity-provider agents refresh one- or multi-level quotes and use inventory controls. These behaviors intentionally resemble some superficial spoofing indicators. The detector should not label every large, frequently cancelled quote as manipulation.

### Execution algorithms

An asset manager, pension fund, producer, or hedge fund can have a large **parent order** that it does not want to execute all at once. An execution algorithm slices it into smaller **child orders**, works them over time, and may cancel/repost as quotes move. It can also occasionally take liquidity to stay on schedule.

This behavior can look like repeated large placements and cancellations even when the purpose is legitimate execution. The project's execution-algorithm agent exists as a hard negative: a realistic detector must consider context, order intent evidence, executions, and repeated patterns—not just cancellation count.

### Noise and occasional activity

The simulator includes stochastic noise traders and sparse occasional accounts to avoid an empty or fully deterministic market. They provide heterogeneous flow, but they are stylized random processes, not empirically calibrated retail order flow.

## 6. Market mechanics behind the manipulation patterns

The project studies four synthetic personas. These terms describe broad surveillance concepts; specific legal definitions depend on jurisdiction, facts, and rules.

### Spoofing

At a high level, spoofing concerns displaying an order with intent to cancel it before execution. A common surveillance pattern is displayed interest on one side while the trader seeks an execution on the other side. The relevant evidence is in message-level behavior and context: repeated placement/cancellation, displayed size and depth, timing relative to genuine orders/fills, and the trader's executions.

In Veridex, the spoofing persona creates bait orders and a separate genuine leg. Its features include large-order/cancellation measures and an opposite-side-fill signature. A high cancellation ratio alone is not sufficient: market makers and execution algorithms can cancel heavily for legitimate reasons.

### Layering

Layering is a related displayed-depth pattern involving multiple orders at successive price levels on one side. The suspicious signal is the apparent depth profile and its lifecycle in relation to trading on the other side, not simply the existence of multiple price levels. In the project the layering agent is a separate persona; its feature engine includes a same-account, same-tick, same-side multi-level order-share measurement.

The line between the simulator's spoofing and layering personas is a modeling distinction. In real surveillance the strategies may overlap, and an alert may be classified under a related family rather than a uniquely certain persona.

### Wash trading

Wash trading involves transactions that create apparent market activity without a genuine change in beneficial market risk/ownership, depending on facts and applicable rules. A simple signal is repeated reciprocal trading between related accounts, especially with little net position change. Real markets require ownership/account relationship and intent context; opposite trades alone do not establish a wash.

Veridex's wash persona creates pre-arranged crosses among its own simulated accounts. Its evaluation can measure how much episode volume is between those accounts, and feature engineering tracks reciprocal counterparties and ring-like patterns. The simulator's trader IDs reveal account-level counterparties but do not model a real-world beneficial-ownership registry or broker account structure.

### Pump-and-dump / momentum ignition

At a high level, this pattern involves building exposure, initiating or amplifying a move, and later selling/distributing into resulting demand or reversing the position. In futures context, it is useful to reason about aggressive flow, price response, coordination, inventory build-up and reversal. Any specific allegation still needs case-specific evidence.

Veridex models phases such as accumulation, ignition, distribution, and dump. Its feature set looks at directional execution, aggressor flow, market return, activity relative to baseline, coordination, and reversals. Quiet accumulation and decoys are deliberately not always labelled as active manipulation, because the model is intended to learn suspicious activity rather than the fact that an account belongs to an episode.

### Why these are difficult to distinguish

Several legitimate mechanisms can mimic individual indicators:

- Market makers cancel/repost quotes.
- Execution algos work large parent orders with child-order cancellation.
- HFT firms quote multiple levels and may have high order-message counts.
- Institutional crossing or rebalancing can create reciprocal counterparty patterns.
- Hedgers can trade in the same direction as a price move because of real business needs.
- Momentum traders can be coordinated by the same market signal without coordinating with each other.

Therefore the detector combines multiple kinds of evidence: behavior within the account, comparison with peers, comparison with the account's own history, counterparty structure, and market context.

## 7. What the Veridex simulation represents

### Market state and price formation

Each commodity simulation has one order book. Agents receive a snapshot of visible quotes/depth and decide whether to submit orders. The order book is the source of executions: transaction prices emerge when incoming orders cross resting orders. A separate fundamental-value process influences some agent decisions, but it does not directly dictate the transaction price.

This design gives the project a controllable microstructure loop:

```text
displayed orders affect visible depth
visible depth may affect agent quote/price decisions
crossing orders create executions
executions update observed price and account inventory
```

This loop is useful for studying hypotheses about signals and detection. It is not a calibrated market-impact model for a particular exchange.

### Time scale

The simulator's default tick is one second, and its configured trading day is 23,400 ticks (6.5 hours). A common run length of 10,000 ticks is therefore about 2 hours 46 minutes of simulated time. The default OHLCV bar interval is 100 ticks (100 seconds). These are model-time mappings, not an exchange feed's exact timestamp or session convention.

### How historical data is used

The project can download daily OHLCV histories and fit broad price-process parameters. However, the current adversarial simulation worker uses fallback drift/volatility values; it does not refit every run from the downloaded files. Historical OHLCV also contains no account identities or message-level order behavior, so it cannot estimate actual spoofing, execution-algorithm, or hedger strategies.

Use this distinction in a review:

> Historical prices help motivate/calibrate broad market-scale assumptions, but the trader population and trading rules are hand-designed agent models. We are not replaying historical order flow or learning real participant strategies from OHLCV.

### Ground-truth labels and audit tags

The simulator has privileged knowledge of which controller/account is in a persona episode. It writes episode metadata and uses internal order-ID suffix tags for label/effectiveness accounting. Those are simulator ground truth and are not model input features. In the default activity-label mode, only windows containing active tagged orders are positive; quiet accumulation, decoys, and camouflage can remain unlabelled.

This is useful for controlled experiments but easier than live surveillance, where true intent and account relationships are not directly observable and labels may be delayed, incomplete, or disputed.

## 8. What is intentionally outside the model

The current project should be described as a **stylized, synthetic limit-order-book ABM and detector experiment**. It is not a full commodity-futures exchange simulator. Current important omissions include:

- Contract multiplier and monetary tick value.
- Initial/maintenance margin, collateral, margin calls, and forced liquidation.
- Exchange/broker/clearing fees and transaction-cost models.
- Contract expiry, delivery, cash settlement, calendar spreads, and roll activity.
- Physical inventory, storage, transport, quality/location basis, and supply/demand fundamentals.
- Multiple venues, routing, queue position across venues, hidden/iceberg liquidity, and full exchange order types.
- Network latency, message throttles, exchange gateways, and realistic colocation/priority effects.
- Real participants' beneficial ownership, account hierarchy, and broker/client relationships.
- Actual exchange intraday session calendar, auctions, price limits, and market-specific settlement procedures.
- Empirically calibrated full-depth order-message streams for every commodity.

The exact omissions matter when interpreting P&L, market impact, latency, or manipulation effectiveness. For example, a simulated positive mark-to-market result is not equivalent to realized net profit after real contract multipliers, fees, margin funding, risk limits, and execution costs.

## 9. How to explain the detector in finance language

The detector asks: “Given the observable order/trade behavior of this account in a recent interval, does the pattern resemble any of the labeled behaviors?”

It does not ask: “Is this person guilty?” or “Will the next price move up or down?”

The account-window features roughly correspond to:

- **Order behavior:** order counts, size relative to baseline, cancellation rate, order lifetime, re-pricing and multi-level placement.
- **Execution behavior:** traded quantity, buy/sell imbalance, aggressor activity, price relative to VWAP, fills versus placements.
- **Counterparty behavior:** concentration, reciprocal flow, number of counterparties, ring-like connected groups.
- **Temporal behavior:** buildup, directional reversal, unusual activity versus the account's own past, coordinated bursts.
- **Market context:** return, volatility, range, volume spike, order/trade imbalance, and abnormal return.
- **Peer/self normalization:** whether the behavior differs from current peers or from the same account's previous behavior.

This language is safer and more accurate than saying the model identifies manipulation from one metric. Features are evidence summaries; the model learns combinations from its labelled training set.

## 10. How to talk about evaluation honestly

The adversarial experiment creates a training market and separate seeded test markets each round. The old frozen model's **PRE** result tests whether the current mutation evades the prior detector. The **POST** result evaluates after retraining using current/prior training markets and configured replay-augmentation cases. The held-out synthetic test markets are not used for that round's fit or threshold tuning.

But this is not a full external validation:

- Train and test markets share simulator and persona code.
- Different seeds mean different random realizations, not different behavioral mechanisms.
- Overlapping 3-minute windows stepped every 30 seconds are correlated.
- Activity labels are generated from known simulator tags.
- Case replays included in replay augmentation are training examples, not held-out tests.
- A model can score highly because the simulator produces repeated patterns, even without direct label columns in the feature matrix.

A strong score is evidence that the model separates the generated cases under this protocol. It is not, by itself, evidence of real-world precision, recall, causality, or deployability. Stronger validation would hold out strategies/parameter ranges, test separately sourced order-message data with defensible labels, use temporal and market-regime splits, and report calibration and operational false-alert rates.

## 11. A practical presentation flow (about 8–10 minutes)

1. **Problem (1 min):** Futures markets rely on transparent order books; surveillance needs to distinguish suspicious displayed/trading patterns from legitimate electronic execution.
2. **Instrument basics (1 min):** A futures contract is standardized exposure to a commodity price; long/short positions and hedging explain why legitimate traders have different motives.
3. **Mechanics (1 min):** Explain bid, ask, spread, depth, resting versus aggressive orders, matching and cancellation with the small 99.90/100.00 example.
4. **Why it is hard (1 min):** Market makers and execution algos cancel frequently; no single feature is conclusive.
5. **Project method (2 min):** Seeded participant agents -> order book -> trades/order events -> 3-minute rolling account windows every 30 seconds -> features -> per-persona scores and anomaly review.
6. **Personas (1 min):** High-level spoofing, layering, wash trading, and pump/dump patterns; emphasize simulator labels are ground truth for experiments, not real investigation labels.
7. **Evaluation (1 min):** PRE = frozen detector against mutation; POST = retrained detector; held-out seeds are synthetic market generalization, not live-market validation.
8. **Limitations and next steps (1 min):** No margin, contract expiry/delivery, fees, ownership graph, venue fragmentation, or real order-level data calibration; next step is external/generator-held-out validation.

### Suggested opening

> “This project is not trying to forecast commodity prices or declare legal violations. It is a controlled market-microstructure experiment: we simulate different participant behaviors, observe the order and trade trail, and test whether an account-window detector can separate suspicious patterns from legitimate look-alikes.”

### Suggested closing

> “The main result is about the behavior of this simulator and evaluation protocol. The detector is a research prototype. To make real deployment claims, we would need independently sourced message-level data, reliable labels, contract-specific economics, and validation across venues, regimes, and strategies not used in training.”

## 12. Likely review questions and concise answers

### “What exactly is a future?”

A standardized exchange-traded contract giving exposure to a specified commodity and delivery/settlement month. It lets commercial participants hedge and other participants take price risk. Veridex uses futures-inspired commodity settings but does not simulate contract expiry, delivery, margin, or clearing.

### “How does the price move in the simulation?”

Agents submit limit orders. When an incoming order crosses the opposite best quote, the order book matches it at the resting order's price. The last execution price updates the observed market price. The fundamental-value process informs some agents but does not directly set the transaction price.

### “Why can a big order be suspicious if it did not trade?”

Displayed orders can affect perceived depth and other participants' decisions even before execution. A pattern of large displayed interest, cancellation, and trading on the other side may warrant review. The order alone is not proof of intent.

### “Are all cancellations bad?”

No. Electronic market makers and execution algorithms regularly cancel/reprice orders to manage risk and quote quality. Veridex deliberately includes these legitimate look-alikes. The model relies on combinations and context, not a blanket cancellation rule.

### “Does the model use the hidden persona tag?”

No. The simulator uses audit tags and episode metadata to create ground-truth labels for training/evaluation and effectiveness measurement. Model features are computed from orders, trades, account behavior, and market context; identifiers and audit tags are excluded from model columns.

### “Does 95% performance mean it will get 95% in a live exchange?”

No. It means the reported metric was achieved on that particular synthetic held-out-seed test set and metric definition. The generator is shared between training and test; real deployment needs external, independently labelled, message-level validation.

### “Can an alert identify the exact persona?”

It outputs the highest-margin persona prediction when a classifier fires, and can also produce a generic anomaly alert. Spoofing and layering overlap conceptually, and model classification is uncertain. An alert is a triage signal, not a legal conclusion.

### “Does wash trading mean the simulator knows common ownership?”

The simulator knows which accounts were created by the same persona controller and can label their episode. Real surveillance needs reliable beneficial-ownership/account-linkage data, which this project does not model.

### “Does simulated P&L mean the strategy is profitable?”

Not in a real economic sense. The pump-and-dump calculation is a simplified mark-to-market measure based on simulated buys, sells, remaining inventory, and a final price. It omits contract multipliers, fees, margin, financing, and realistic slippage/market impact.

### “What is the biggest next validation step?”

Test on independently sourced message-level market data with defensible labels, and hold out entire strategy families/parameter ranges and time regimes. Also add contract-specific costs/risk controls before making economic claims.

## 13. Project-specific code and references

- Market matching and order lifecycle: [`src/core/order_book.py`](../src/core/order_book.py)
- Tick sizes and historical calibration routines: [`src/core/calibration.py`](../src/core/calibration.py)
- Agent populations, simulation clock, and run configuration: [`src/core/simulation.py`](../src/core/simulation.py)
- Basic agent behavior: [`src/core/agents/`](../src/core/agents/)
- Execution algorithm: [`src/core/agents/execution_algo.py`](../src/core/agents/execution_algo.py)
- Market maker: [`src/core/agents/market_maker.py`](../src/core/agents/market_maker.py)
- Manipulator personas: [`src/core/agents/manipulation/`](../src/core/agents/manipulation/)
- Case notes and documented-versus-assumed distinctions: [`docs/enforcement_case_notes.md`](./enforcement_case_notes.md)
- Simulation, calibration, and feature pipeline: [`docs/SIMULATION_AND_FEATURE_PIPELINE.md`](./SIMULATION_AND_FEATURE_PIPELINE.md)
- Prediction, thresholds, and evaluation: [`docs/PREDICTION_AND_EVALUATION.md`](./PREDICTION_AND_EVALUATION.md)
