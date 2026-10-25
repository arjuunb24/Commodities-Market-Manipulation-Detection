# Simulation, Calibration, and Detector Feature Pipeline

This guide follows the current code path from historical market data and simulator configuration through agent behavior, generated Parquet files, labels, feature engineering, training, and evaluation. It describes the code as it is implemented now; in particular, it distinguishes historical price calibration from simulated trader behavior.

## 1. At a glance

The adversarial experiment is a seeded agent-based market simulation followed by a supervised and unsupervised detector:

1. A market configuration supplies a starting price, fundamental-value parameters, agent populations, and persona parameters.
2. Seeded legitimate and manipulator agents submit orders to a continuous double-auction order book.
3. The simulator writes order events, trades, OHLCV bars, and manipulation episode metadata.
4. The feature pipeline converts the order and trade logs into one row per `(commodity, window_start, trader_id)`. It writes a separate four-column persona-label table with the same index.
5. A per-persona XGBoost layer and Isolation Forest layers fit on training markets, then score held-out markets.

The detector's unit is an account-window, not an individual order or an entire episode. A single episode therefore contributes multiple window rows and multiple accounts may be involved in one episode.

## 2. Historical-data calibration: what is and is not extracted

### Historical price and volume data

[`scripts/fetch_calibration_data.py`](../scripts/fetch_calibration_data.py) fetches daily OHLCV histories for commodity futures using `yfinance` and stores the raw bars as Parquet. Missing dates are not forward-filled. The project also has [`scripts/fetch_market_data.py`](../scripts/fetch_market_data.py), a separate path for recent anonymous one-minute bars used for live scoring. Those bars do not contain order-book events, trader IDs, or account-level behavior.

[`src/core/calibration.py`](../src/core/calibration.py) provides calibration methods. It fits an AR(1) regression to log prices and expresses the resulting estimates as OU-style parameters:

- `mean_reversion_speed` (`κ`): rate of pull toward the fitted long-run level.
- `long_run_mean` (`θ`): fitted long-run log-price level.
- `volatility` (`σ`): residual diffusion scale.
- `drift`: additional directional drift.
- Volume calibration separately estimates mean daily volume and its coefficient of variation.

These routines can be used to fit and store parameters, but downloading data is not the same as applying a fresh fit to every simulation. **The adversarial simulation worker currently calls `CalibrationEngine.get_fallback(commodity)`**, as shown in [`scripts/run_simulation.py`](../scripts/run_simulation.py). The worker passes the fallback drift and volatility to the simulation. It does not read the downloaded Parquet and refit on each run. Although the fallback call returns OU and volume parameters, the worker ignores the returned volume parameters.

There is also an important boundary between market calibration and trader behavior:

- The historical input is price/volume bars, not trader-level order data.
- The simulation's population sizes, arrival rates, order behavior, and manipulator knobs are configured in code/configuration; they are not inferred from those OHLCV histories.
- The persona baselines are specified in [`configs/persona_config.yaml`](../configs/persona_config.yaml). LLM mutations alter those configuration values in later rounds; they are not historical trader-parameter estimates.
- Current worker setup uses fallback drift/volatility. The fitted `κ` and `θ` are not passed through this worker to the fundamental process.

Thus, the market's broad price/volume scale is historically motivated, but agents are stylized simulated actors rather than reconstructed real accounts.

### Fundamental value versus traded market price

[`src/core/simulation.py`](../src/core/simulation.py) creates a seeded `FundamentalValueProcess`. Fundamental value evolves per tick using drift and random volatility, with configured jump behavior; daily-scale drift and volatility are converted to the simulation's tick scale. [`configs/sim_calibration.yaml`](../configs/sim_calibration.yaml) can apply commodity-specific volatility scaling.

The fundamental value is not itself the exchange price. Agents use it as an input to their decisions, and the observed market price emerges from submitted orders and matches in the order book. Mean-reversion traders respond to the difference between market and fundamental value; the fundamental process does not directly force the order-book price to equal an OU long-run mean.

## 3. How the simulated market is generated

### Order book and deterministic seeds

The simulator uses a continuous double-auction limit order book with price-time priority. Orders can rest, match, partially fill, expire, or be cancelled. Each run has seeded random streams for agents and for the randomized within-tick actor arrival order. Commodity seeds are derived stably from the master seed and commodity name, so Python's process-randomized `hash()` is not used for that purpose.

In a multi-commodity adversarial run, each commodity is simulated separately with its derived seed. The orchestrator also assigns separate seeds to train and test markets. These are separate generated markets, but they use the same simulator and persona implementations.

### Legitimate participant families

Agent populations are instantiated in [`src/core/simulation.py`](../src/core/simulation.py), using the configured counts and parameters. The main legitimate families are:

| Agent family | Purpose / behavior |
|---|---|
| Noise traders | Stochastic arrivals and price-offset orders; represent unstructured flow. |
| Momentum traders | Use recent price history and a threshold to trade with short-term direction. |
| Mean-reversion traders | Trade in response to deviations from the fundamental value, subject to configured limits. |
| Market makers | Maintain two-sided quotes and periodically cancel/replace them. Their trader IDs can be excluded from the detector's trader-level features because their routine requotes otherwise dominate cancellation statistics. |
| HFT liquidity providers | Supply liquidity with configurable quote sizes and levels. |
| Execution algorithms | Break trading activity into executions over time. |
| Institutional crossing desks | Create benign related-account crosses/rebalances, which can produce counterparty concentration or ring-like patterns without manipulation labels. Crossing sessions reach the wash persona's lot sizes and rates. |
| Quote-protecting liquidity providers | Quote large size near the touch (sometimes axed to one side or laddered) and pull the opposite side when one side is hit: legitimate large cancels opposite a fill (spoofing look-alike). |
| Ladder hedgers | Work a large one-sided hedge as a ladder of big orders on consecutive levels, re-placed when the market moves away (layering look-alike). |
| Fund families | One allocation decision over several sub-accounts: passive then urgent buying, a hold, an exit (pump-and-dump look-alike). |
| Occasional traders | Provide sparse background activity. Manipulator accounts also share a background-order generator with occasional traders, so an account is not intended to be detectable solely from its quiet background behavior. |

These mechanisms create some benign behavior that overlaps with individual manipulation indicators. They do not make the synthetic distribution equivalent to real-world order flow.

### Manipulator personas

The current personas are spoofing, layering, wash trading, and pump-and-dump. They are controllers that create one or more `ManipulatorAccount` instances, schedule episodes, and submit tagged orders. The shared account/background conventions are in [`src/core/agents/manipulation/common.py`](../src/core/agents/manipulation/common.py); individual strategies are in the persona modules under [`src/core/agents/manipulation/`](../src/core/agents/manipulation/).

| Persona | Intended simulated behavior | Typical parameter families |
|---|---|---|
| Spoofing | Place displayed bait orders, then cancel them; depending on parameters, split orders, reprice, persist direction, or trade on the opposite side. | Order size, cancellation delay/jitter, split count, account count, price aggressiveness, direction persistence, cover, busy-period bias. |
| Layering | Place multiple same-side orders at separated price levels, then cancel them; may stagger placement or vary/taper size. | Layer count, spacing, size multiplier/taper, cancellation delay/jitter, placement stagger, frequency, cover. |
| Wash trading | Coordinate accounts to trade with each other, potentially in a ring; may vary size, camouflage, or prefer busier periods. | Frequency, group/ring size, size jitter, camouflage rate, busy-period bias. |
| Pump-and-dump | Coordinate accumulation/ignition, a burst or price push, then a delayed exit. Some quiet accumulation/decoy behavior is deliberately not labelled as manipulative activity. | Account count, accumulation size, burst duration, ignition aggressiveness, dump delay, stagger, decoy rate, busy-period bias. |

The exact values active in a run are resolved by [`src/core/persona_params.py`](../src/core/persona_params.py): `round_0` is the baseline and top-level persona values overlay it. The mapping applies those names to `SimConfig` fields. Episode logs also record the effective parameters in the `parameters` column of `manipulation_events.parquet`.

IDs such as `pnd_0_9` are simulator trader IDs: `pnd` is the persona prefix, `0` is the episode/controller index for that simulation, and `9` is the zero-based account index within that controller. The middle component is not an adversarial-loop round number.

## 4. Files generated by a simulation

The simulator assembles four main tables and writes them under the run directory. In an adversarial run, a market directory such as `round_0/test_market_0/` contains one commodity's data; the test directory may aggregate equivalent data across the configured commodities.

| File | One row represents | Main fields / purpose |
|---|---|---|
| `order_log.parquet` | One order lifecycle event (for example, placed or cancelled) | `order_id`, `trader_id`, `side`, `price`, `quantity`, `tick`, `event_type`, `commodity`. This is the source for placement/cancellation features and activity labels. |
| `trade_log.parquet` | One matched trade | `trade_id`, `tick`, `price`, `quantity`, buyer and seller trader IDs, aggressor side, commodity. This is the source for execution, counterparty, and market-price features. |
| `manipulation_events.parquet` | One ground-truth manipulation episode | `event_id`, persona, start/end ticks, involved trader IDs, recorded parameters, commodity. Used to create labels and to evaluate episode effectiveness; not a detector input. |
| `ohlcv_bars.parquet` | One resampled bar | Bar start, open, high, low, close, volume, commodity. Useful for market/replay displays; the detector feature pipeline derives its market features from the trade/order logs. |

All of these files are simulator outputs. The columns `order_id`, `trade_id`, `trader_id`, and `commodity` are identifiers or join keys, not labels. Simulator audit tags embedded in order IDs are available to the label-generation routine, not to the model's feature columns.

### Example table sizes from a stored test market

As a concrete illustration, one test directory in `adversarial_20261021T121523Z_seed888` contained 181,214 feature rows, 920,448 trade rows, 4,438,574 order-event rows, 32 episode rows, and 808 OHLCV bars. When loaded as a pandas DataFrame, the feature table had 62 ordinary columns: 61 numeric features plus `window_end`; `commodity`, `window_start`, and `trader_id` were restored as the three-level row index. Counts vary with run length, market activity, agent configuration, and number of commodities.

> **Schema-count clarification:** The Arrow-visible Parquet schema contains 65 fields: the 61 numeric features, `window_end`, and the three fields used to reconstruct the pandas index. The row index is not included in pandas' ordinary `DataFrame.columns` count. The example row count/schema above is intended to illustrate scale, not a fixed contract.

## 5. Preprocessing: logs to account-window examples

[`src/detector/features.py`](../src/detector/features.py) constructs the feature matrix. The default feature configuration is in [`configs/detector_config.yaml`](../configs/detector_config.yaml):

- Window length: **3 minutes**.
- Step/stride: **30 seconds**.
- Simulator tick: **1 second**.
- Market-maker trader IDs beginning with `mm_` are excluded from trader-level feature rows by default.
- A window is emitted only after a complete feature window is available; startup warm-up rows shorter than three minutes are omitted.

The pipeline maps events to 30-second steps, builds market price/volume summaries, enriches placed orders and cancellations, aggregates trades by account and counterparty, and rolls those aggregates into overlapping 3-minute windows. The result is one row per `(commodity, window_start, trader_id)`. Overlapping windows are intentionally correlated examples; they are not independent samples.

Features and labels are persisted separately as `features.parquet` and `labels.parquet`. The pipeline checks a feature-version marker and the source order-log timestamp before reusing those caches.

### Ground-truth labels

Labels are four binary columns: `spoofing`, `wash_trading`, `pump_and_dump`, and `layering`. With the default `labels.mode: activity`, an account-window is positive for a persona only if:

1. The account is listed in a corresponding manipulation episode, and
2. It placed at least one order with that persona's activity tag in that window.

The tag mapping in [`src/detector/pipeline.py`](../src/detector/pipeline.py) is: spoofing `{s,g,f}`, layering `{l,g,s}`, wash trading `{w}`, and pump-and-dump `{i,x,u,p}`. Quiet pump accumulation (`a`), pump decoys (`d`), and wash camouflage (`c`) are intentionally excluded. The alternative `span` label mode labels windows overlapping the episode by at least 15 ticks, rather than requiring an active tagged order.

Labels come from the simulator's audit trail and episode metadata. The detector does **not** receive the audit tag, episode ID, active-persona parameter dictionary, trader ID, or event label as a numeric feature.

## 6. Feature reference

The current feature engine emits **61 numeric columns**. Ratios generally use safe division and return zero when the denominator is absent unless noted below. The detector excludes identifiers, window timestamps, commodity, and cross-validation group labels before fitting.

### Account order, cancellation, and execution behavior

| Feature | Meaning |
|---|---|
| `n_orders` | Number of placed orders by the account in the window. |
| `n_trades` | Number of execution legs involving the account; one market trade contributes one leg to each participating account. |
| `exec_vol` | Total quantity executed by the account (buys plus sells). |
| `avg_order_size_rel` | Mean placed-order size divided by the pre-window rolling median order-size baseline. |
| `max_order_size_rel` | Largest placed-order size divided by that baseline. |
| `order_size_cv` | Coefficient of variation of placed-order sizes: standard deviation divided by mean. |
| `large_order_share` | Share of placed quantity from orders at least 4× the size baseline. |
| `layered_order_share` | Share of placed quantity in same-account, same-tick, same-side batches with at least three distinct price levels. |
| `cancel_replace_ratio` | Cancelled orders with a same-account, same-tick, same-side replacement divided by cancellations, clipped to `[0,1]`. |
| `cancel_ratio` | Cancelled quantity divided by placed quantity, clipped to `[0,1]`; if there are cancellations but no new placed quantity in the window, it is set to 1. |
| `fast_cancel_ratio` | Quantity cancelled within 10 ticks divided by the larger of placed and cancelled quantity. |
| `large_cancel_ratio` | Quantity cancelled from orders classified as large divided by placed large quantity. |
| `mean_cancel_lifetime` | Mean time in ticks between placement and cancellation; if no cancellations, safe division uses the full window length. |
| `passive_distance_bp` | Order-quantity-weighted signed distance from the previous step's reference price in basis points. A passive quote away from the reference is positive; an aggressive price through it can be negative. |
| `fill_ratio` | Executed quantity divided by placed quantity, clipped to `[0,2]`; fills can include orders placed before the current window. |
| `cancel_side_imbalance` | `(buy-cancel quantity - sell-cancel quantity) / total cancelled quantity`, in `[-1,1]`. |
| `exec_side_imbalance` | `(buy execution quantity - sell execution quantity) / total execution quantity`, in `[-1,1]`. |
| `cancel_exec_opposition` | Negative product of cancellation-side and execution-side imbalance, clipped to `[0,1]`; rises when cancellation and executed-flow directions oppose. |
| `n_large_cancels` | Count of cancelled orders that had been classified as large. |
| `spoof_signature` | Fraction of large cancellations accompanied by an opposite-side fill for the same account while the large order rested or immediately around its cancellation. |
| `net_flatness` | `1 - abs(net executed direction)`, normalized by execution volume; high values indicate buys and sells roughly offset. Zero when there are no executions. |
| `aggressor_ratio` | Share of the account's execution quantity executed as the aggressor. |
| `top_cpty_share` | Gross executed volume with the account's busiest counterparty divided by the account's total execution volume. |
| `reciprocal_ratio` | Reciprocal volume with counterparties, using twice the smaller of bought-from/sold-to volume per counterparty, divided by total execution volume; clipped to `[0,1]`. |
| `n_counterparties` | Number of distinct counterparties with executions in the window. |
| `trade_size_cv` | Coefficient of variation of the account's executed trade sizes. |
| `price_dev_bp` | Execution-volume-weighted absolute distance of execution prices from that step's market VWAP, in basis points. |

### Counterparty-ring structure

| Feature | Meaning |
|---|---|
| `mutual_cpty_share` | Share of account volume with counterparties that qualify as mutual partners: each side's partner-volume share clears the configured mutual-share threshold and the pair's volume exceeds the minimum size-baseline threshold. |
| `ring_size` | Number of accounts in the connected component of qualifying mutual-partner links; defaults to 1 when no ring is found. |
| `ring_volume_share` | Share of the account's executed volume traded with members of its qualifying connected ring. |

These are structural signals, not labels: benign crossing desks can also generate related-account or reciprocal-looking activity.

### Temporal behavior and coordination

| Feature | Meaning |
|---|---|
| `prev_net_ratio` | Net buy/sell imbalance over the two preceding non-overlapping windows, normalized by their gross volume. |
| `long_net_ratio` | Net buy/sell imbalance over the current and three preceding non-overlapping windows (roughly 12 minutes at the default window size). |
| `activity_vs_history` | `log1p(current activity) - log1p(own historical baseline)`, where activity combines executed volume and order count and the baseline uses trimmed earlier history when enabled. Positive values mean higher activity than the account's baseline. |
| `reversal_score` | Positive part of `-prev_net_ratio × current exec_side_imbalance`; high values indicate current trading opposite to the preceding net direction. |
| `coordinated_peers` | Number of other accounts in the same window that are both bursting relative to their own history (`activity_vs_history > 0.7`) and trading directionally (`abs(exec_side_imbalance) > 0.5`) in the same direction. |

### Market context joined onto each account-window

These columns describe the shared market window and are repeated on account rows. The model can therefore use the market regime as context; they do not identify which account caused a market move.

| Feature | Meaning |
|---|---|
| `mkt_return_bp` | Log return from the start to end of the rolling window, converted to basis points. |
| `mkt_volatility_bp` | Standard deviation of per-step log returns within the window, converted to basis points. |
| `mkt_range_bp` | High-low market price range divided by the window low, in basis points. |
| `mkt_volume_spike` | Current window volume divided by its earlier rolling baseline; safe fallback is 1 when no usable baseline exists. |
| `mkt_trade_imbalance` | `(2 × aggressive-buy volume - total volume) / total volume`, in `[-1,1]`. |
| `mkt_order_imbalance` | `(placed buy quantity - placed sell quantity) / total placed quantity`, in `[-1,1]`. |
| `mkt_cancel_ratio` | Number of cancellation events divided by number of placed-order events in the window. |
| `mkt_abnormal_return_z` | Z-score of the absolute window return against earlier absolute-return history; missing/degenerate baselines become zero and values are clipped to `[-10,10]`. |
| `gross_volume_share` | Account executed volume divided by total market traded volume in the window. |
| `signed_flow_return` | Account execution-side imbalance multiplied by market return in basis points. |

### Peer-relative features

Each `peer_z_*` value compares an account with other active accounts in the same window. The implementation uses robust peer-relative statistics (median and an IQR-derived scale with floors), rather than treating raw levels as comparable across all accounts.

| Feature | Meaning |
|---|---|
| `peer_z_cancel_ratio` | Peer-relative cancellation ratio. |
| `peer_z_fast_cancel_ratio` | Peer-relative fast-cancellation ratio. |
| `peer_z_avg_order_size_rel` | Peer-relative average order size versus baseline. |
| `peer_z_max_order_size_rel` | Peer-relative maximum order size versus baseline. |
| `peer_z_gross_volume_share` | Peer-relative share of market volume. |
| `peer_z_reciprocal_ratio` | Peer-relative reciprocal trading ratio. |
| `peer_z_aggressor_ratio` | Peer-relative aggressor share. |
| `peer_z_large_order_share` | Peer-relative large-order share. |

### Account-self-history features

Each `self_z_*` compares the account with its own earlier, non-overlapping history. The baseline is based on earlier windows, robust median/IQR scaling, a minimum active-history requirement, and configured scale floors. Extreme prior windows are iteratively excluded from later baselines; current-window evidence is downweighted when order count is low. Before enough history exists, these features are zero.

| Feature | Meaning |
|---|---|
| `self_z_avg_order_size_rel` | Deviation of average relative order size from the account's earlier baseline. |
| `self_z_max_order_size_rel` | Deviation of maximum relative order size from the account's earlier baseline. |
| `self_z_large_order_share` | Deviation of large-order share from the account's earlier baseline. |
| `self_z_cancel_ratio` | Deviation of cancellation ratio from the account's earlier baseline. |
| `self_z_fast_cancel_ratio` | Deviation of fast-cancellation ratio from the account's earlier baseline. |
| `self_z_n_orders` | Deviation of order count from the account's earlier baseline. |
| `self_z_gross_volume_share` | Deviation of the account's market-volume share from its earlier baseline. |
| `self_z_layered_order_share` | Deviation of layered-order share from the account's earlier baseline. |

`window_start`, `window_end`, `commodity`, and `trader_id` are row metadata, not members of the 61 numeric feature columns. The internal `cv_group` column, when attached during training, is also removed from the model matrix.

## 7. Training, thresholds, and evaluation

[`src/detector/classifiers.py`](../src/detector/classifiers.py) fits one XGBoost binary classifier per persona. It uses class weighting because positive manipulation windows are much rarer than negatives. Thresholds are tuned from out-of-fold predictions using `GroupKFold`; in the adversarial orchestration, groups identify source markets so overlapping rows from one market are kept together rather than randomly split across folds.

The detector also has two unsupervised components, implemented in [`src/detector/anomaly.py`](../src/detector/anomaly.py):

- An account-level Isolation Forest over behavioral/peer/self features, with market-context columns excluded.
- A market-level Isolation Forest over market-context features; its score is contextual evidence, not a standalone account label.

The ensemble combines supervised flags and corroborated anomaly evidence according to `detector_config.yaml`. Thresholds and model artifacts are saved with each round.

In [`src/adversarial/orchestrator.py`](../src/adversarial/orchestrator.py), each round generates one training market and multiple separately seeded test markets. The frozen model's **PRE** score measures how the newly mutated behavior challenged the previous detector. The **POST** score is measured after retraining on training markets and configured replay-augmentation cases. Held-out test markets are not used for fitting or threshold tuning; however, train and test use the same simulator and persona implementations. This is a test of generalization across seeds/markets within this synthetic generator, not independent validation against an unrelated real-market data-generating process.

## 8. Interpreting high scores and known limits

- A high score on a held-out seed does not by itself demonstrate target leakage. It can mean the same simulator repeatedly produces recognizable persona signatures.
- Separating train/test seeds prevents direct reuse of the same simulated market, but does not hold out agent code, parameter families, or the simulator's data-generating process.
- Overlapping windows are correlated. Metrics should be interpreted at the episode and market level as well as at the row level.
- Tags and episode metadata provide synthetic ground truth. Real market bars do not provide equivalent account/persona labels.
- Some features encode direct behavioral patterns, such as same-tick multi-level placement, reciprocal counterparties, and audit-informed spoof signature calculations from observed orders/trades. These are behavioral measurements, not explicit label fields, but their presence can make synthetic strategy patterns easier to recognize than in real deployment.
- Synthetic benign participants and institutional crossing desks add overlap, but the real market has wider strategy diversity, hidden liquidity, venue fragmentation, latency effects, participant incentives, and incomplete observability.
- Robust out-of-generator validation should hold out more than random seeds: vary or hold out persona implementations/parameter ranges, calibrate against independent market regimes, and evaluate on separately sourced labeled replay cases. A held-out replay is not independent if the same case was included in `replay_augmentation`.

## 9. Useful code entry points

- Simulation configuration, agent wiring, seeded loop, and output assembly: [`src/core/simulation.py`](../src/core/simulation.py)
- Historical calibration and fallback values: [`src/core/calibration.py`](../src/core/calibration.py)
- Persona baseline parameters: [`configs/persona_config.yaml`](../configs/persona_config.yaml)
- Persona-to-simulation parameter mapping: [`src/core/persona_params.py`](../src/core/persona_params.py)
- Feature definitions and formulas: [`src/detector/features.py`](../src/detector/features.py)
- Feature caching and ground-truth labels: [`src/detector/pipeline.py`](../src/detector/pipeline.py)
- Model columns, per-persona fit, and threshold tuning: [`src/detector/classifiers.py`](../src/detector/classifiers.py)
- Train/test rounds and pre/post evaluation: [`src/adversarial/orchestrator.py`](../src/adversarial/orchestrator.py)
- Detector defaults, feature windows, labeling mode, and replay configuration: [`configs/detector_config.yaml`](../configs/detector_config.yaml)
