# Veridex — Overhaul Change Log (from → to → why)

This document records every change made in this overhaul: the simulator, the manipulation personas, the detector (prediction methods), the adversarial loop, the LLM/agentic layer, the dashboard, data contracts, configs, scripts and tests. Every entry says **what it was**, **what it is now** and **why**. Where a change was validated, the evidence (measured numbers) is included.

Nothing in this overhaul has been committed yet; all changes are in the working tree on branch `harsh`.

---

## Contents

1. [Summary](#1-summary)
2. [Simulator realism](#2-simulator-realism)
3. [Manipulation personas](#3-manipulation-personas)
4. [Detector (prediction methods)](#4-detector-prediction-methods)
5. [Adversarial loop and metrics](#5-adversarial-loop-and-metrics)
6. [LLM / agentic layer](#6-llm--agentic-layer)
7. [Dashboard](#7-dashboard)
8. [Data contracts (schemas)](#8-data-contracts-schemas)
9. [Configuration files](#9-configuration-files)
10. [Scripts](#10-scripts)
11. [Tests](#11-tests)
12. [Bugs found and fixed](#12-bugs-found-and-fixed)
13. [Results and evidence](#13-results-and-evidence)
14. [Known limitations and next steps](#14-known-limitations-and-next-steps)
15. [File index](#15-file-index)

---

## 1. Summary

| Area | Before | After |
|---|---|---|
| Market realism | Prices trended strongly (lag-1 return autocorrelation +0.23 to +0.79), volatility ~10× real, only manipulators cancelled orders | Near-zero/negative return autocorrelation, fat tails, volatility clustering, realistic intraday volatility, every participant cancels |
| Manipulators | Spoofing had no price impact; "wash" pair mostly traded with strangers; pump-and-dump ran out of buying in a few ticks | Each persona reproduces the documented mechanism of real CFTC cases, with evasion knobs taken from those cases |
| Detector unit | One row per *market window* — could not say *who* manipulated | One row per *(account, window)* — the account is the thing flagged |
| Detector models | Market-level IsolationForest + LOF + EllipticEnvelope; per-persona XGBoost/LightGBM on hand-picked market features; fixed 0.5 threshold | Trader-level IsolationForest (percentile-calibrated, SHAP-explained) + market-regime IsolationForest as context; per-persona XGBoost on all trader, peer and market features; thresholds tuned out-of-fold; SHAP stored with every flag |
| Adversarial loop | Retrained before scoring (no measure of evasion); one test market per round | Scores with the *frozen* detector first (evasion), then retrains (recovery); 2 test markets per round; effectiveness of each manipulation measured |
| LLM | Gemini only; strategist saw only last round's metrics; no bounds enforcement; analyst tools returned invented numbers | One switch for Gemini or Cerebras (gpt-oss-120b); budget-guarded; strategist sees full track record, effectiveness, SHAP reliance and tactic cards; strict bounds; analyst tools read real data |
| Dashboard "Flagged Trades" | Showed simulator ground truth with hard-coded confidence values | Shows real detector flags with stored SHAP, ground-truth confirmation, missed episodes |
| Tests | 69 | 104 (all passing) |

---

## 2. Simulator realism

The simulator is a limit-order-book agent-based model (ABM). Before changing anything, the project's own stylized-facts script (`scripts/verify_stylized_facts.py`) and new diagnostics were used to measure how real the market looked.

### 2.1 What was measured before

| Statistic | Real futures markets | Old simulator |
|---|---|---|
| Lag-1 autocorrelation of bar returns | ≈ 0 or slightly negative | **+0.23 to +0.79** (strong artificial trends) |
| Trade-level return autocorrelation | Negative (bid-ask bounce) | Not checked |
| 30-second return volatility (gold) | ~2–4 bp | ~27–45 bp |
| 1-hour price range (gold) | ~0.3–0.5% | ~2–3% |
| Cancel ratio of normal traders | High (most limit orders are cancelled) | **0%** — only spoofers and the market maker cancelled |
| Avg spread for Brent | Positive | Negative value reported (a reporting bug in the script, see §12) |

### 2.2 Changes

#### 2.2.1 Daily calibration applied per tick → converted to per-tick scale
- **From:** `fundamental_value_vol` (a *daily* volatility, e.g. gold 0.94%) was applied every tick, and 1 tick = 1 second. Over a 5,000-tick run the fundamental drifted ~66%.
- **To:** `SimConfig.ticks_per_day = 23_400` (6.5-hour session at 1 tick/s). Per-tick volatility = daily vol / √23,400, per-tick drift = daily drift / 23,400. Jump shocks: `jumps_per_day = 4`, jump size = 0.25 × daily vol.
- **Why:** the calibrated *values* were right but the *units* were wrong. Calibrated parameters themselves were not touched (per your instruction).
- File: `src/core/simulation.py` (`FundamentalValueProcess`, `run_simulation`).

#### 2.2.2 Real contract tick sizes
- **From:** prices rounded to 4 decimals on an arbitrary grid.
- **To:** `COMMODITY_TICK_SIZES` (CME/ICE minimum increments: crude/brent 0.01, gold 0.10, silver 0.005, natural gas 0.001, copper 0.0005, wheat/corn 0.0025). Every agent snaps prices to the grid via `Agent.snap()` / `round_to_tick()`.
- **Why:** queues form at discrete price levels in real books; tick size also drives the bid-ask bounce seen in real data.
- Files: `src/core/calibration.py`, `src/core/order_book.py`, `src/core/agents/base.py`.

#### 2.2.3 Order expiry for every participant
- **From:** only spoofers and the market maker ever cancelled. Unfilled noise/momentum/hedger orders stayed in the book forever.
- **To:** `Agent.limit()` schedules a geometric lifetime (`mean_order_ttl`: noise 40 ticks, momentum 8, mean-reversion 60; manipulator background accounts 40). Expired orders are cancelled at the start of each tick (`expire_orders`). Personas can set an exact lifetime (`exact_ttl`) when cancel timing is part of the tactic.
- **Why:** in real markets most limit orders are cancelled. When only manipulators cancel, `cancel_rate` becomes a trivially perfect spoofing signal, which overstates detector skill. Normal cancel ratios are now ~10–20%.

#### 2.2.4 Random arrival order within a tick
- **From:** agents always acted in the same fixed order (`noise_0` first every tick).
- **To:** a seeded `order_rng.permutation(...)` each tick; the book snapshot is refreshed after any actor changes the book.
- **Why:** no participant should systematically get first look at the book; deterministic given the seed (FR-0.4).

#### 2.2.5 Order book engine
- **From:** depth snapshot (`_aggregate_depth`) scanned the heap array in storage order, which is not sorted — the "top 5 levels" given to agents could be wrong. Ties at the same price broke on tick only (not strict arrival). Partial fills re-heapified the whole book. Trade ids were `uuid4`.
- **To:** per-side price-level dictionaries kept in sync on rest/fill/cancel, so depth is exact (`heapq.nlargest/nsmallest`). Heap entries carry an arrival sequence number (strict price-time priority, edge case E5). Partial fills mutate quantity in place (heap order unchanged). Trade ids are deterministic (`<prefix>T<n>`). Added `is_resting()`, `round_to_tick()`, `BookState.imbalance()` and band volumes (`bid_band_volume`, `ask_band_volume` within ±0.2% of mid). Trades carry `aggressor_side`.
- **Why:** correctness of what agents see, strict time priority, speed, reproducibility, and a realistic aggressor flag (published by real exchanges and needed by surveillance features).
- File: `src/core/order_book.py`.

#### 2.2.6 Legitimate trader behaviour

| Agent | From | To | Why |
|---|---|---|---|
| **Noise trader** | Price offset absolute; sizes uniform; never cancelled | Offset relative to price (`noise_price_offset_pct = 0.0005`); log-normal sizes; orders centred on the **microprice** (mid shifted toward the heavier side of the visible book by ≤ ~½ spread, weight 1.5); expiry | Log-normal sizes match retail flow. Microprice (Stoikov 2018) is how displayed depth realistically influences where flow trades — and it is the channel through which spoofed depth moves prices. |
| **Momentum (trend follower)** | Aggressive offset of 0.5 price units (absolute — 20% of natural gas's price); lookback 30 *trades* (~3 seconds); no risk limit | Marketable orders priced through the best quote with 0–5 bp slippage; lookback 60 *ticks* sampled once per second; position limit 400; `momentum_lambda 0.05`, threshold 0.1% | Old lookback was so short the trend signal was noise; absolute offsets were meaningless across commodities; real trend followers have risk limits. |
| **Mean-reversion (hedger)** | Sent up to 200 lots priced up to 0.3% through mid (swept several levels) | Trades only at the touch (best bid/ask); threshold 0.1% | Sweeping orders were a major source of excess volatility; hedgers work orders patiently. |
| **Market maker** | Inventory skew = % of price (40 units of inventory moved quotes up to 4%); random "inventory decay" (inventory vanished 25% of ticks) | Skew in half-spreads (Avellaneda–Stoikov style); quotes lean on visible book imbalance (`imbalance_lean 0.6`); quote size 40; half-spread `max(2 ticks, 0.03% of price)`; no magic decay | The old skew was a huge price-moving force; decay was unphysical. Imbalance-aware quoting is what real liquidity providers do. |

#### 2.2.7 Things tried and reverted (and why)
- **Noise traders chasing order-book imbalance** (`imbalance_sensitivity 0.3`): created a self-reinforcing loop (buying thins asks → book looks bid-heavy → more buying). 30 s volatility went to ~40 bp; with it off, 3.3 bp. Left at 0.0 and replaced by the bounded microprice.
- **Position limits on mean-reversion traders** (to let pumps move prices): at a tight limit the market destabilised (30 s autocorrelation +0.56, range 3.9%). Reverted.

#### 2.2.8 Measured after (from `stylized_facts_report.md` and diagnostics)
- 30 s volatility: gold ~4 bp, crude ~6–8 bp; 1-hour range 0.4–0.8% (was 2–3%).
- Raw 30 s return autocorrelation ≈ 0 (e.g. +0.05 average in the final tuning sweep); trade-level autocorrelation ≈ −0.1 (bid-ask bounce).
- Excess kurtosis positive (fat tails) and |return| autocorrelation positive (volatility clustering) in every commodity.
- Large-tick contracts (wheat, corn, natural gas, silver; ~4–6 bp per tick) show strongly negative bar autocorrelation, as real large-tick futures do.
- Market-wide cancel ratio no longer changes during manipulation windows — realistic, since one spoofer barely moves a market-wide average. This is the core reason detection had to move to the account level (§4).

#### 2.2.9 Reproducibility (FR-0.4)
- **From:** `uuid4` for order, trade, event and coalition ids; multi-commodity seeds used Python's `hash()` (salted per process).
- **To:** order ids `<prefix><trader>-<n>`, trade ids `<prefix>T<n>`, event ids `<prefix>E<i>`, where prefix = commodity + seed; `multi_commodity_seed()` uses CRC32. A new test asserts *every* log is identical for the same seed, including ids.

---

## 3. Manipulation personas

All three personas now share `ManipulatorAccount` / `ManipulationController` (`src/core/agents/manipulation/common.py`):
- **Background activity:** every manipulator account trades like a small ordinary trader *outside* its manipulation window. Previously accounts existed only while manipulating, which would let a detector key on "a new account appeared" instead of on behaviour, and made trader-level ground truth trivially separable.
- **Unique account ids per episode** (previously multiple pump-and-dump episodes reused `pumpdump_acct_0..2`, so their inventories collided).
- **One ground-truth event per episode** (wash groups of one episode are merged).
- **Audit tags in order ids** (`s` spoof leg, `g` genuine leg, `w` wash leg, `c` camouflage, `d` decoy). Used only for effectiveness accounting; the detector never parses order ids.

Layering was not changed (per instruction) and stays disabled.

### 3.1 Spoofing (`spoofing.py`)
- **From:** one large order placed 0.3% behind mid on a fixed side, no genuine order, frequency 0.4 per tick (overlapping spoofs). Measured price impact: **−0.03 bp, t = −0.37** (none).
- **To:** the Coscia/Sarao bait-and-switch. Each cycle: a small *genuine* order rests at the best price on one side; a large *spoof* rests a few ticks behind the best price on the other side; the spoof is cancelled after `cancellation_delay_ticks`, the genuine order shortly after if unfilled. One cycle at a time. Direction chosen per cycle (`spoof_side="both"`). Default distance 0.05% from the touch.
- **Evasion knobs** (from `docs/enforcement_case_notes.md`):

| Knob | Tactic | Source |
|---|---|---|
| `size_jitter` | Random spoof size | CFTC 2020 staff advisory #1; Coscia round 2 |
| `cancel_jitter` | Random cancel delay | Advisory #2 |
| `n_split_orders` (1–4) | Spoof split across consecutive levels | Coscia "split into batches"; Tower layering |
| `n_accounts` (1–3) | Genuine and spoof legs on different accounts | JPMorgan cross-trader spoofing |

- **Why:** spoofing only makes sense if displayed depth moves prices. After the microprice change and moving the spoof near the touch, measured impact is **+0.18 bp in the spoof's direction, t = 2.15** — small but significant, as real spoofing profits a tick at a time.

### 3.2 Wash trading (`wash_trading.py`)
- **From:** two accounts each independently sent aggressive orders at mid ± 0.1%. They mostly traded with *other* participants — two directional traders, not a wash. `n_colluding_pairs` multiplied wash volume. `price_deviation_from_mid` was later found to have almost no effect.
- **To:** pre-arranged crosses. One account rests an order strictly inside the spread and its partner takes it at the same price and size in the same tick; unmatched remainders are cancelled next tick. If the spread is one tick (no room inside), they wait rather than trade with strangers. Roles rotate so net positions stay flat. Baseline uses a fixed scripted lot (120).
- **Evasion knobs:**

| Knob | Tactic | Source |
|---|---|---|
| `ring_size` (2–6) | Crosses rotate A→B→C→A | Coinbase cyclical washes |
| `n_groups` (1–5) | Independent groups; the *total* cross rate is split across them (dilutes per-account signals without adding volume) | Advisory #3; Kraft round 2 |
| `size_jitter` | Varied lot sizes | Advisory #1 |
| `camouflage_rate` (0–0.5) | Some cycles replaced by a genuine directional market trade | Kraft round 2; advisory #4 |

- **Measured:** self-crossed share of the pair's trades went from ~31% (first version) to **98–100%** after the "wait when spread is one tick" fix.

### 3.3 Pump-and-dump (`pump_and_dump.py`)
- **From:** "accumulation" was aggressive buying that hit the per-account target (~50 lots) in a few ticks, then a long idle hold, then an aggressive dump. Ground truth covered long stretches with no activity.
- **To:** momentum ignition (how pump-and-dumps appear in futures): **ACCUMULATE** (passive bids at the touch) → **IGNITE** (last 30% of the burst: marketable buys through the ask) → **DISTRIBUTE** (passive offers at the ask, selling into followers) → **DUMP** (aggressive sells of what remains). `accumulation_size` is per account.
- **Evasion knobs:**

| Knob | Tactic | Source |
|---|---|---|
| `stagger` (0–1) | Each account runs the schedule with its own random delay | Nefedova: "staggered accumulation… appearing like independent decisions" |
| `ignition_aggression` | How far through the ask ignition buys go | — |
| `decoy_rate` (0–0.4) | Small opposite-side trades while accumulating/distributing | Advisory #4 |

- **Measured:** the net-flow profile shows the intended shape (e.g. +450/+500 lots of buying, then a −800 to −900 lot dump).

---

## 4. Detector (prediction methods)

### 4.1 Unit of detection
- **From:** one row per *(commodity, market window)*. Market-wide features (cancel rate, top-K concentration, volume spikes) were diluted by ~40 other participants, and the output could not identify the manipulating account. The SRS requires per-trader features (FR-2.1) and per-trader flags (FR-2.5).
- **To:** one row per **(commodity, window, trader_id)** for every account active in the window. Windows: 3 minutes, stepping every 30 s (tick-based, 1 tick = 1 s). Incomplete windows at the start of a run are dropped, not zero-padded.
- **Why:** surveillance scores *accounts*. A spoofer is identified by what its own orders do.

### 4.2 Features (`src/detector/features.py`, rewritten)
Computed fully vectorised (~0.7 s per 5,000-tick run). Per-step aggregates are replicated into the windows that contain them and grouped (exact rolling windows).

**Trader behaviour features**

| Persona targeted | Features |
|---|---|
| Spoofing | `spoof_signature` (share of the account's large cancelled orders during whose life it was filled on the **opposite** side — the Coscia/Sarao signature, computed per order), `fast_cancel_ratio` (cancelled within 10 s), `large_cancel_ratio`, `n_large_cancels`, `max_order_size_rel`, `avg_order_size_rel`, `large_order_share`, `cancel_ratio`, `mean_cancel_lifetime`, `passive_distance_bp`, `cancel_side_imbalance`, `cancel_exec_opposition`, `order_size_cv`, `fill_ratio` |
| Wash trading | `top_cpty_share` (share of volume with one counterparty), `reciprocal_ratio` (volume traded back and forth with the same accounts), `net_flatness`, `n_counterparties`, `trade_size_cv`, `price_dev_bp` |
| Pump-and-dump | `exec_side_imbalance`, `aggressor_ratio`, `signed_flow_return` (own direction × market return), `prev_net_ratio`, `long_net_ratio` (~12-minute position build-up), `reversal_score` (net buyer before, net seller now), `activity_vs_history`, `coordinated_peers` (other accounts bursting in the same direction in the same window) |
| Activity | `n_orders`, `n_trades`, `exec_vol`, `gross_volume_share` |

**Peer-relative features:** robust z-scores (`peer_z_*`, median/MAD) of key features against all other accounts in the same window — normalises for market regime.

**Market context features** (joined to every trader row, prefix `mkt_`): window return, volatility, range, volume spike vs history, aggressor-side trade imbalance, placed-volume order imbalance, cancel ratio, abnormal-return z-score. Plus `mkt_anomaly_score` from the market-regime IsolationForest.

**Removed / fixed:**
- `order_book_imbalance` compared sides to `"BUY"`/`"SELL"` while logs store `"buy"`/`"sell"` → it was **always 0**. Fixed (`mkt_order_imbalance`).
- Market-maker exclusion by `startswith("mm")` hacks inside features → replaced by an explicit, documented rule: designated market makers (`exclude_trader_prefixes: ["mm_"]`) are not scored, as real surveillance monitors registered liquidity providers under separate rules (their requote cancels would dominate cancellation features).
- The O(n) graph/iterrows features (circular volume via networkx per step) → replaced by the reciprocal/counterparty features.

**Measured separation (medians, crude oil run):** spoofing windows `fast_cancel_ratio` 0.97 vs noise traders 0.06; `spoof_signature` 0.87 mean vs 0.02 for legitimate accounts; wash windows `top_cpty_share` 0.98 vs 0.08.

### 4.3 Labels (`pipeline.generate_labels`)
- **From:** a market window was positive if any episode overlapped it by ≥ 15 s.
- **To:** a *(account, window)* row is positive for persona P when the account is one of the episode's accounts **and** the window overlaps the episode by ≥ 15 ticks. Vectorised (explode + merge).

### 4.4 Unsupervised layer (`src/detector/anomaly.py`, rewritten)
- **From:** a market-level IsolationForest plus LocalOutlierFactor (wash) and EllipticEnvelope (pump), used as metric fallbacks; raw scores.
- **To:**
  - **Trader-level IsolationForest** on behaviour and peer features only (market context excluded so "the market was volatile" is not mistaken for "this account behaved oddly"), robust scaling (10–90 quantile range), scores reported as an **empirical percentile** of the training distribution (threshold = 1 − contamination, directly interpretable). **SHAP explanations** via `shap.TreeExplainer` on the forest.
  - **Market-regime IsolationForest** on market features; its score is fed to the classifiers as context, not used to flag accounts.
  - Contamination lowered from 0.03 to **0.01**: at 3% the IsolationForest produced 35–61 unmatched "anomaly" cases per round (mostly manipulator background activity and hedgers' large orders); at 1% this fell to ~1 with 98% precision for that layer.

### 4.5 Supervised layer (`src/detector/classifiers.py`, rewritten)
- **From:** XGBoost for spoofing, LightGBM for wash and pump, each on a **hand-picked subset of market features**; `scale_pos_weight` capped at 15–20; fixed 0.5 threshold; zero-positive personas silently skipped; no SHAP at prediction time (only gain importance saved).
- **To:**
  - One **XGBoost** binary classifier per persona on **all** trader, peer and market-context features (SHAP shows what each model actually uses; subsets hid cross-signals).
  - Class imbalance via `scale_pos_weight = n_neg / n_pos` (cap 50), no resampling — so SHAP explains the true distribution.
  - **Out-of-fold threshold tuning:** `GroupKFold` over time blocks (600 ticks) so overlapping windows never straddle train/validation; threshold chosen to maximise F1, kept within **[0.25, 0.9]** (without the floor, single-episode folds pushed thresholds to 0.1–0.125 and precision collapsed).
  - `InsufficientLabelsError` (FR-2.3): raised in strict mode (`require_all_personas: true`), otherwise an explicit warning.
  - `shap_values()` and `mean_abs_shap()` (global SHAP importance per persona).

### 4.6 Ensemble and flags (`src/detector/pipeline.py`)
- **From:** `ensemble_flag` = IsolationForest OR max supervised probability > 0.5; no persona attribution; no `flags.parquet` ever written; no SHAP stored.
- **To:** `score()` returns per-persona probability and tuned-threshold flag, IsolationForest percentile and flag, market anomaly score, `predicted_persona` (largest margin over its own threshold, or `"anomaly"` if only the IsolationForest fired), `final_score`, `final_flag`, `triggered_by`. `build_flags()` writes **`flags.parquet`** (schema v2) with the **top-5 SHAP attributions of the model that fired** for every flag. Features and labels are cached per run directory (`features.parquet`, `labels.parquet`) so cumulative training doesn't recompute earlier rounds. `shap_importance.json`, `feature_importances.json` and `thresholds.json` are saved per round.

### 4.7 Detector performance (held-out test markets, crude oil, 3 rounds, hand-written mutations; before the evasion knobs were added)

| Persona | Round 0 F1 | Round 1 F1 | Round 2 F1 | FPR |
|---|---|---|---|---|
| Spoofing | 0.88 | 0.97 | 0.97 | ≤ 0.1% |
| Wash trading | 0.89 | 0.98 | 0.96 | ≤ 0.1% |
| Pump-and-dump | 0.60 | 0.65 | 0.84 | ≤ 0.6% |

All episodes detected. For comparison, the old market-level detector's last run had pump-and-dump F1 0.13–0.15 with FPR 0.15–0.40.

---

## 5. Adversarial loop and metrics

### 5.1 Pre- vs post-retrain (FR-3.2)
- **From:** each round retrained on cumulative data *before* scoring, so the loop never measured whether a mutation evaded the existing detector. `round_metrics.parquet` was not schema-validated and had no pre/post distinction.
- **To:** each round scores its test markets first with the detector **frozen** at the end of the previous round (*pre-retrain* = evasion), then retrains on rounds 0..N and scores again (*post-retrain* = recovery). Round 0 has only post metrics.

### 5.2 Event-level metrics and latency (FR-2.7)
- **From:** row-level metrics only; personas with no positives reported 0.0 precision/recall/F1.
- **To:** per persona, row-level precision/recall/F1/FPR plus `events_total`, `events_detected` and `detection_latency_ticks` (first flagged window end − episode start; **NaN for missed episodes**, never averaged as zero). Undefined metrics are NaN, not 0. `compute_unsupervised_metrics()` evaluates the IsolationForest against "any manipulation".

### 5.3 Multiple test markets
- **From:** one test market per round (often one episode per persona → metrics dominated by chance).
- **To:** `n_test_markets` (default 2) per round with distinct seeds and tick offsets, simulated in parallel and concatenated.

### 5.4 Manipulation effectiveness (`src/adversarial/effectiveness.py`, new)
- **Why:** without it, the strategist could "evade" by weakening the manipulation until it did nothing (observed: it moved the spoof 0.5% from the touch, where it has no price impact).
- **What:** per episode, from the test market's own logs:

| Persona | Primary measure | Also reported |
|---|---|---|
| Spoofing | `favorable_move_bp` — price move toward the genuine order while the spoof rested | `genuine_fill_rate` |
| Wash trading | `wash_volume_share` — self-crossed share of market volume in the window | `wash_volume` |
| Pump-and-dump | `ignition_pressure` — peak share of market volume taken by the coalition's marketable buying in any 60-tick window | `price_lift_bp`, `pnl_bp` |

- **Objective:** `score = (1 − frozen recall) × min(1, effectiveness / round-0 effectiveness)`; **0 if effectiveness < 50% of round 0**.
- **Validation:** "neutered" settings (spoof far from the touch, rare washes, tiny pump) fall below 0.5× for all three personas; e.g. wash share 0.039 vs 0.21 baseline, pump pressure 0.024 vs 0.089.
- **Pump measure history:** the first version (90th-percentile price over the episode) was noise-dominated (a neutered pump scored 1.1×). The ignition-span lift was too weak to separate (5.4 vs 3.8 bp). The span-relative share mis-scored a tiny pump. The peak 60-tick pressure is stable (±0.01) and separates correctly.

### 5.5 Persisted outputs per round
`metrics.json` now holds: post metrics (`personas`), `pre_retrain`, `unsupervised`, `thresholds`, `n_flags`, `n_test_markets`, `params_used` (exact parameters in force), `mutation_source` (baseline / agentic / hand_authored / unchanged), `effectiveness`, `effectiveness_ratio`, `mutation_score`. Run level: `round_summary.parquet` (all columns) and schema-validated `round_metrics.parquet`.

---

## 6. LLM / agentic layer

### 6.1 Provider switch and client (`src/agentic/llm_client.py`, rewritten)
- **From:** Gemini only, via a key pool; the request log wrote **API keys** into `llm_calls.jsonl/.yaml`; up to `2 × keys` attempts plus the OpenAI SDK's hidden retries; responses cached only in demo mode.
- **To:**
  - **One switch for every LLM call:** `LLM_PROVIDER` in `.env` (wins) or `active_provider` in `configs/llm_config.yaml` — `cerebras` or `gemini`. With `cerebras`, only `CEREBRAS_API_KEY` is used (model `gpt-oss-120b`); Gemini keys are never read.
  - **Budget ledger** (`data/llm_cache/cerebras_spend.json`, persisted across runs): before every call the worst case (estimated input + `max_tokens` output) must fit in the remaining budget (`budget_usd: 4.50`, pricing $0.35 / $0.75 per million in/out), otherwise no request is sent (`BudgetExceededError`). Actual cost recorded from usage.
  - **Per-process call cap** (60) and **≥ 2 s between calls** (stays under per-minute rate limits).
  - **Response cache** for every provider: identical requests are never re-billed.
  - **Bounded, non-looping retries:** auth/billing errors → one attempt, provider disabled for the process; bad requests → no retry; rate limits/timeouts → at most `max_retries` with backoff. SDK hidden retries disabled (`max_retries=0`).
  - **Message sanitisation:** litellm adds `reasoning_content` / `provider_specific_fields` to assistant messages; Cerebras rejects them when sent back on the next tool-calling turn. Stripped centrally, which fixed multi-turn strategist and analyst calls.
  - **API keys stripped from logs;** usage and cost per call logged.
  - `llm_unavailable_reason()` for fail-fast start-up checks (FR-4.4).
- **Key handling:** the Cerebras key lives only in `.env` (gitignored).

### 6.2 Token and prompt settings (`configs/llm_config.yaml`)

| Component | Output cap (Cerebras) | Reasoning effort | Other limits | Typical / worst cost |
|---|---|---|---|---|
| Strategist | 2,500 | medium | ≤ 4 model turns; only `propose_mutation` offered | ~3,500 input tokens per turn; ~$0.005 per round |
| Analyst chat | 600 | low | ≤ 3 tool turns; each tool result ≤ 3,000 chars; last 3 Q&A kept | < $0.005 per question |
| Case narrative | 450 | low | ≤ 2 calls (1 retry on grounding failure) | < $0.001 |

Measured: with "low" reasoning the strategist thought for ~10 tokens and made a self-contradicting proposal; "medium" (~1,100 reasoning tokens on the first turn) produced coherent reasoning, so only the strategist uses it.

### 6.3 Adversarial Strategist (`src/agentic/adversarial_strategist.py`)
- **From:** saw only the last round's metrics, the last parameter sets (no outcomes), and 10 lines of case notes (effectively only Coscia); proposals were barely validated; the hand-written fallback was indexed off by one (rounds 1 and 2 both got variant 0, and variants wrapped back to the first at round 4).
- **To:**
  - **Objective in the system prompt:** maximise `(1 − frozen recall) × min(1, effectiveness ratio)`; weakening the scheme below 50% is not evasion.
  - **Method:** start from the best-scoring set, change 1–3 knobs aimed at features the detector currently relies on, never resubmit a past set, and switch tactic after a failure.
  - **Context per persona (compact):** knob bounds with current values; the **track record** for every past round (parameters → frozen recall/F1 → retrained recall → effectiveness and its ratio → score, plus who chose them); the detector's current mean |SHAP| reliance and rising features; and **tactic cards** (`docs/tactic_cards.yaml`) mapping each documented tactic to its knobs, the features it weakens and its cost.
  - **Validation:** every proposal is checked against `PARAM_BOUNDS` (persona allow-list, known names, types, ranges). One re-prompt with the exact error; a second rejection, an LLM failure, or an unavailable/over-budget provider → the hand-written variant **for the target round**.
- **Observed in the live run:** all four rounds' mutations came from the LLM except the last round's pump-and-dump (turns ran out → fallback). Out-of-range proposals (`size_jitter` 0.95, `decoy_rate` 0.45) were rejected and corrected on the re-prompt.

### 6.4 Tools (`src/agentic/tools.py`)
- **From:** `get_round_metrics` returned **made-up numbers** (F1 0.88/0.82) when files were missing; "SHAP drift" was precision/recall deltas or gain importance; `propose_mutation` only checked `cancellation_delay_ticks > 0`; analyst `query_flags` / `get_trader_history` returned hard-coded strings ("Found 5 flags…", "placed 40 orders…").
- **To:** missing data → `{"error": ..., "available_rounds": [...]}`, never invented values. `get_shap_drift` uses saved mean-|SHAP| importance. New `get_track_record`, `get_tactic_cards`, `get_round_metrics_full`. Analyst tools read real data: `list_rounds`, `query_flags` (flags.parquet, filterable, SHAP included), `query_round_metrics`, `get_trader_history` (summary of the account's orders/trades across all test markets of a round).

### 6.5 Mutation library (`src/agentic/mutation_library.py`, new file)
- **From:** variants that lowered frequency and pulled spoofs away from the touch (neutering the manipulation) and claimed extra wash pairs "dilute" volume (they multiplied it).
- **To:** three variants per persona following the case progressions (Coscia/Tower/JPMorgan; Coinbase/Kraft; Nefedova), all validated against the bounds; indexed by target round and clamped at the last variant.

### 6.6 Analyst Assistant (`src/agentic/analyst_assistant.py`)
- **From:** mocked tools; no sources; no availability check.
- **To:** fails fast if the provider is unavailable; `ask()` returns `{"answer", "sources"}` listing every tool call used; prompt requires citing tools, relaying errors and declining out-of-scope questions (FR-4.3); bounded turns and context.

### 6.7 Narrative Generator (`src/agentic/narrative_generator.py`)
Input is now the real flag case (account, persona, window, score, triggered layers) plus its stored SHAP values; compact JSON; an empty reply (token limit) stops instead of retrying blindly. The fact-grounding check (every number must appear in the input) is unchanged and still enforced.

---

## 7. Dashboard

| Tab / part | From | To |
|---|---|---|
| **Flagged Trades** | Listed simulator ground-truth events with **hard-coded confidence** (0.88 spoofing, 0.82 wash…) and made-up feature values; placeholder "SHAP" charts | Real detector flags from `flags.parquet`, merged into cases (consecutive windows of one account and persona); stored SHAP chart and feature meanings; badge "ground truth confirms / other persona / no episode"; list of **missed episodes**; price chart with the account's own buys/sells; raw orders/trades expander; LLM case note labelled "AI-generated summary of detector output — simulation only", shown only if it passes the grounding check |
| **Model Performance** | Post-only curves | New "Evasion vs. Recovery" section: pre-retrain (dashed) vs post-retrain (solid) recall/F1 per persona, and a one-sentence summary computed from the numbers (no LLM) |
| **Analyst Chat** | Did not exist | New tab; suggested questions; "Sources (n)" expander per answer; calm "AI features are temporarily unavailable" message when the provider is unavailable |
| Data loading | Read only `round_N/test` | Reads all test markets of a round (`test`, `test_1`, …); null metrics normalised to NaN |
| Footer | — | "All data is synthetic…" disclaimer (NFR-S3/E1) |
| **Test-market selector** | All of a round's test markets were concatenated, so price charts jumped back to the starting price where the second market began | Sidebar "Test market k of n · ticks a–b" selector; Market Overview and Flagged Trades show one market at a time (flags filtered to that market's tick range); Model Performance states that its metrics combine all markets of the round |
| **Persona vs. unsupervised cases** | IsolationForest-only "anomaly" cases were mixed with persona cases (e.g. 467 of 505 unmatched cases in an 8-commodity run were IsolationForest-only) | Headline numbers cover persona-classifier cases; IsolationForest-only cases have their own count and appear when "anomaly" is added in the Persona filter |
| **Interpreter check** | Starting Streamlit from another Python (e.g. Miniconda) crashed with `ModuleNotFoundError` | The app detects missing project packages and shows the correct launch command |
| **Strategist reasoning / parameter evolution** | Parsed the free-text `llm_reasoning.log` with the wrong encoding (garbled `â€'`, `Ã—`), mis-grouped personas (empty round headers, everything under "[SPOOFING]"/"GENERAL"), indented HTML rendered as code blocks; the parameter table read the *current* `persona_config.yaml`, not the selected run | The pipeline writes `round_N/strategist.json` (source, old → new changes, reasoning); the dashboard shows one expander per round with a card per persona: source badge, outcome (frozen recall, retrained recall, effect vs round 0, score), parameter changes and the reasoning as a clean numbered list. Older runs fall back to `metrics.json` + a corrected UTF-8 log parser. The parameter table comes from each round's recorded `params_used` |
| **Colours** | 8-digit hex colours (`#3b82f622`) crashed the radar chart on older Plotly versions | Converted to `rgba(...)` |

Launch command (Streamlit is installed in `.venv` without a launcher `.exe`):
`.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py`

---

## 8. Data contracts (schemas)

`src/utils/schemas.py`:
- **trade_log:** new required column `aggressor_side` (`buy`/`sell`).
- **flags (v2):** adds `window_end`, `iso_score`, `predicted_persona` (nullable). Parquet stores the SHAP dict as a struct padded with the union of keys; `read_flags()` / `clean_shap_dict()` strip the padding on read.
- **round_metrics:** now actually written and validated (`round_num`, `commodity`, `persona`, precision, recall, F1, FPR, `detection_latency_ticks`, `is_post_retrain`).

---

## 9. Configuration files

| File | Change |
|---|---|
| `configs/detector_config.yaml` | Window 3 min / step 30 s; `exclude_trader_prefixes`; IsolationForest contamination 0.01, `market_contamination` 0.05; XGBoost `max_scale_pos_weight`; new `supervised` block (`min_positives`, `require_all_personas`, `cv_folds`, `cv_block_ticks`, `threshold_bounds`); `top_shap_features`; personas limited to the three maintained ones |
| `configs/persona_config.yaml` | Clean `round_0` baseline with every new knob at its "obvious" value; spoof distance 0.0005; unused start/end ticks removed (timing is sampled per commodity); stale top-level mutations removed |
| `configs/llm_config.yaml` | Provider switch, provider profiles, `cerebras_budget`, per-component caps (see §6.2) |
| `.env` / `.env.example` | `LLM_PROVIDER`, `CEREBRAS_API_KEY` (real key only in `.env`, gitignored) |
| `docs/tactic_cards.yaml` | New: evasion tactics → knobs → weakened features → cost |

---

## 10. Scripts

| Script | Change |
|---|---|
| `scripts/run_full_pipeline.py` | `--mutations llm\|hand` (hand = no API, baseline for comparisons), `--test-markets`, `--run-name`, default `--persona all`; strategist given the target round; mutation source recorded; console table shows pre/post, events, latency, effectiveness ratio and score; UTF-8 reasoning log; a syntax error introduced earlier in this work was fixed |
| `scripts/run_simulation.py` | Persona parameters via the shared `src/core/persona_params.py` mapping; deterministic multi-commodity seeds |
| `scripts/verify_stylized_facts.py` | Counts `cancelled` (was `cancel`, so cancel ratios always read 0%); deterministic seeds; uses current simulator defaults |
| `scripts/test_pipeline.py`, `plot_windows.py`, `test_module_4.py` | Updated to trader-level outputs / market feature names / flag format |
| `scripts/visualize_results.py` | Reads `round_summary.parquet` (post rows) |

New shared module: `src/core/persona_params.py` — one mapping from `persona_config.yaml` names to `SimConfig` fields, used by the run script and by the orchestrator to record `params_used`.

---

## 11. Tests

69 → **104** tests, all passing. New or rewritten:

| File | Covers |
|---|---|
| `tests/unit/test_detector.py` | Exact feature values on hand-built logs (spoof signature, fast cancels, wash counterparty features), MM exclusion, no NaN, trader-level labels, IsolationForest percentiles and SHAP, strict zero-positive error, flags schema round-trip, event detection and NaN latency |
| `tests/unit/test_personas.py` | Spoof cycle structure, wash crosses inside the spread (and waiting at one-tick spreads), pump-and-dump phases, ≥90% self-cross in simulation, background activity, unique ids, tick grid |
| `tests/unit/test_agentic.py` | Bounds validation, no write on rejection, fallback indexing/clamping, all library variants valid, one re-prompt then fallback for the target round, LLM failure → fallback, tools report missing data, `query_flags` on real parquet, grounding check rejects invented numbers (all hermetic: real LLM calls would fail the test) |
| `tests/unit/test_llm_client.py` | Routing to Cerebras only, no hidden SDK retries, cost recorded, key not logged, cache prevents re-billing, budget refusal sends nothing, fatal error → single attempt and provider disabled, bounded rate-limit retries, Gemini switch never reads the Cerebras key |
| `tests/unit/test_effectiveness.py` | Neutered manipulation falls below the floor, baseline has an effect, score formula, one event per wash episode, cross-account spoof legs, config mapping |
| `tests/integration/test_simulation_reproducibility.py` | New strict test: every log identical for the same seed, ids included |

---

## 12. Bugs found and fixed

| Bug | Effect | Fix |
|---|---|---|
| Daily volatility applied per tick | Fundamental moved ~150× too fast | Per-tick scaling (§2.2.1) |
| Depth snapshot read heap in storage order | Agents could see wrong top-of-book levels | Price-level dictionaries |
| `"BUY"` vs `"buy"` in features | `order_book_imbalance` always 0 | Fixed |
| Fallback mutation off by one, and wrapping | Rounds 1 and 2 got the same variant; round 4 went back to variant 0 | Indexed by target round, clamped |
| Analyst tools hard-coded | Assistant answered with invented numbers | Real data tools |
| `get_round_metrics` mock fallback | Strategist could read fake metrics | Explicit errors |
| Flagged Trades used ground truth + fixed confidences | Dashboard misrepresented detector output | Rewritten (§7) |
| API keys in LLM logs | Secrets written to run folders | Stripped |
| Pump-and-dump account ids reused across episodes | Inventories collided | Unique ids per episode |
| `uuid4` / `hash()` seeds | Runs not reproducible | Deterministic ids and seeds |
| Wash "pair" traded with strangers | Not a wash | Pre-arranged inside-spread crosses |
| Wash groups logged as separate episodes | Episode counts inflated, effect split | Merged per episode |
| Cerebras rejected echoed assistant fields | Every follow-up strategist/analyst turn failed | Message sanitisation |
| Parquet SHAP structs padded with `None` | Flags read back with dozens of empty keys | `read_flags()` |
| Syntax error in `run_full_pipeline.py` (from an edit in this work) | Pipeline script could not run | Fixed; every Python file now syntax-checked |
| Reasoning log written with Windows default encoding | Crash on a Unicode hyphen → applied mutations recorded as "unchanged" | UTF-8 log, console `errors="replace"` |
| `#` in tactic card sources | YAML treated text as a comment | Quoted |
| Stylized-facts script counted `"cancel"` events | Cancel ratios always 0% | Counts `"cancelled"` |
| Results chart read old metric columns | Auto-chart failed after each run | Reads `round_summary.parquet` |

---

## 13. Results and evidence

### 13.1 LLM strategist vs hand-written mutations
5 rounds each, crude oil, seed 42, 2 test markets per round. Round 0 was identical in both runs (deterministic), so this is a like-for-like comparison. Score: higher is better for the manipulator.

| Persona | Mean frozen recall (hand → LLM) | Mean effectiveness ratio (hand → LLM) | Mean score (hand → LLM) |
|---|---|---|---|
| Wash trading | 0.92 → **0.77** (round 4: 0.89 → **0.51**) | 0.88 → 0.73 | 0.06 → **0.14** |
| Pump-and-dump | 0.44 → 0.51 | 1.38 → 2.26 | 0.56 → 0.49 |
| Spoofing | 0.83 → 0.88 | 1.32 → 0.74 | 0.17 → **0.05** |

- The LLM was clearly better at wash trading, improving every round (rings, then camouflage and more groups) while keeping ≥ 0.5× the wash volume.
- Pump-and-dump: roughly tied; both evade hard in round 1 (frozen recall ~0.17), then the retrained detector recovers.
- Spoofing: the LLM was worse; one mutation lost ~90% of the spoof's price effect (score 0). It then raised the spoof size to restore impact — it used the track record as intended.
- Spoofing stays detectable (~0.85 frozen recall) under both: spoofing inherently requires large, quickly cancelled orders, which are the detector's top SHAP features.
- **Caveat:** one seed, ~2–4 episodes per persona per round. Only the wash-trading gap is clearly larger than noise; 3+ seeds per arm (~$0.02 per LLM run) are needed to settle the others.

### 13.2 Cost
Total Cerebras spend during development and both comparison runs: **$0.028** of the $4.50 cap (19 calls). One full 5-round LLM run: **$0.023**.

---

## 14. Known limitations and next steps

1. **Pump-and-dumps barely move prices (~5 bp) and lose money.** Hedgers absorb the buying and trend followers amplify too little. Effectiveness therefore measures buying *pressure*, not realised lift. Next: strengthen trend-follower response so ignition can produce realistic 0.5–1.5% moves (as in Nefedova) without breaking the stylized facts.
2. **Statistical power.** Results are from one seed. Run several seeds per arm before drawing conclusions about the LLM.
3. **Calibration.** Calibrated values were left as they were (per instruction); `agent_config.yaml` is still not read by the simulator (defaults live in `SimConfig`).
4. **Layering** is unchanged and disabled; it still uses random ids.
5. **Remaining SRS items not built:** Docker/compose, CI, `seed_llm_cache.py` offline demo cache, price-impact-avoided counterfactual metric.
6. **Old runs** in `data/runs` predate `flags.parquet`; the Flagged Trades tab asks to re-run the pipeline for them.
7. **Older run folders may contain Gemini API keys** in `llm_calls.jsonl/.yaml` (written before the log fix). Check before sharing any run folder.

---

## 15. File index

**New files**
- `src/core/agents/manipulation/common.py` — manipulator accounts, controller base
- `src/core/persona_params.py` — persona config ↔ SimConfig mapping
- `src/adversarial/effectiveness.py` — manipulation effectiveness and objective
- `src/agentic/mutation_library.py` — hand-written mutation variants
- `docs/tactic_cards.yaml` — evasion tactic cards
- `dashboard/components/tab_chat.py` — Analyst Chat tab
- `tests/unit/test_detector.py`, `test_personas.py`, `test_agentic.py`, `test_llm_client.py`, `test_effectiveness.py`
- `docs/CHANGES_overhaul.md` — this document

**Substantially rewritten**
- Simulator: `src/core/order_book.py`, `src/core/simulation.py`, `src/core/agents/base.py`, all four legitimate agents, `spoofing.py`, `wash_trading.py`, `pump_and_dump.py`
- Detector: `src/detector/features.py`, `anomaly.py`, `classifiers.py`, `pipeline.py`
- Loop: `src/adversarial/orchestrator.py`, `metrics.py`
- LLM: `src/agentic/llm_client.py`, `adversarial_strategist.py`, `tools.py`, `analyst_assistant.py` (and `narrative_generator.py` updated)
- Dashboard: `dashboard/components/tab_flags.py`, `tab_metrics.py`, `data_loader.py`, `app.py`

**Updated**
- `src/utils/schemas.py`, `src/core/calibration.py` (tick sizes added), all three configs, `.env.example`, the scripts in §10, `stylized_facts_report.md` (regenerated), `tests/unit/test_schemas.py`, `tests/integration/test_simulation_reproducibility.py`
