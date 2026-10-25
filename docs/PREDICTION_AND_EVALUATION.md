# Prediction Windows, Thresholds, and Evaluation

This document explains how Veridex turns simulated order/trade logs into account-window predictions, how each prediction threshold is chosen, how ensemble alerts are formed, and what the adversarial-loop metrics mean. The descriptions follow the current implementation in `src/detector/` and `src/adversarial/`.

## 1. Prediction unit and end-to-end flow

The detector predicts on one row per `(commodity, window_start, trader_id)`. A row summarizes one account's behavior in a completed market window, joined with market context for that same window. It does not make one decision per order, trade, or whole episode.

The runtime path is:

1. Read `order_log.parquet`, `trade_log.parquet`, and `manipulation_events.parquet`.
2. Compute market and account features in `FeatureEngine`.
3. At training/evaluation time, generate labels from the manipulation audit trail. At production scoring time, labels are not needed.
4. Add the trained market-anomaly score as `mkt_anomaly_score`.
5. Generate account-anomaly scores and per-persona probabilities.
6. Apply persona thresholds and the ensemble rule to label rows as `alert`, `review`, or `none`.
7. For the adversarial loop, compare predictions against known test-market labels and episode records.

The main implementation points are [`src/detector/features.py`](../src/detector/features.py), [`src/detector/pipeline.py`](../src/detector/pipeline.py), [`src/detector/classifiers.py`](../src/detector/classifiers.py), [`src/detector/anomaly.py`](../src/detector/anomaly.py), and [`src/adversarial/metrics.py`](../src/adversarial/metrics.py).

## 2. How the rolling windows work

The defaults in [`configs/detector_config.yaml`](../configs/detector_config.yaml) are:

- Window duration: **3 minutes**.
- Step/stride: **30 seconds**.
- Simulator tick: **1 second**.
- Warm-up: omit windows until a complete 3-minute window can be formed.

The feature engine converts ticks into 30-second steps. With the default settings, each 3-minute window contains six steps. Each step contributes to every rolling window that contains it. Adjacent windows overlap by 150 seconds, so most observations appear in multiple rows.

For an illustrative run starting at tick 0, the first full window covers ticks `0–179`; subsequent windows cover `30–209`, `60–239`, and so on. The bounds are inclusive in the stored metadata: `window_start` and `window_end`. In code, for the ending step `w`, the window spans steps `[w - n_steps + 1, w]`; metadata is then calculated as:

```text
window_start = (w - n_steps + 1) * step_ticks
window_end   = (w + 1) * step_ticks - 1
```

The first window in a market whose first observed tick is not zero is aligned to that run's tick grid. A market shorter than one full window produces no trader-window features.

### What one row summarizes

For each account and window, order events and trades are aggregated into counts, quantities, ratios, side imbalances, counterparty relationships, and activity relative to peer/self history. The market-level rolling features (return, volatility, range, volume, order/trade imbalance, cancellation ratio) are joined onto every account row for that window. The account's share of market volume and signed-flow/return interaction are then calculated.

These are completed-window summaries, not tick-by-tick streaming predictions. With a 30-second stride, a new decision point is available every 30 seconds after warm-up. A behavior beginning just after a window boundary can take close to 3 minutes to appear in a full-window aggregate; subsequent overlapping rows refresh the decision every 30 seconds. The feature `window_end` identifies the last tick included, not necessarily the wall-clock time when a downstream consumer receives the prediction.

### Time-history baselines

Some features compare the current window with information from earlier windows:

- Order-size baseline uses earlier step-level median order sizes; the cold start falls back to the current observed median so ordinary first-window orders are not all marked large.
- Market volume/abnormal-return baselines use prior windows, not the current window as its own history.
- `prev_net_ratio`, `long_net_ratio`, and `activity_vs_history` summarize preceding account history.
- Self-history z-scores use earlier, non-overlapping windows. They remain zero until at least ten active historical windows are available.
- Peer z-scores compare accounts within the same current window.

The current window's own order, execution, cancellation, market, and peer observations are intentionally part of its features. Historical baselines and feature versioning are implemented by the feature engine; no labels or audit tags are part of feature calculation.

## 3. Ground truth at prediction time versus evaluation time

Model inference consumes feature rows. Labels are needed only for fitting and evaluation.

The default `activity` labeling mode in [`src/detector/pipeline.py`](../src/detector/pipeline.py) marks an account-window positive for a persona if the account belongs to that episode and placed at least one active persona-tagged order in the window. Quiet accumulation, decoys, and camouflage are intentionally excluded. The alternative `span` mode labels windows that overlap enough of the episode interval.

The labels are stored in a separate `labels.parquet`, aligned with features by the `(commodity, window_start, trader_id)` index. Audit tags, episode metadata, `trader_id`, and timestamps are not model input features. The model-column filter excludes `commodity`, `trader_id`, `window_start`, `window_end`, and `cv_group`.

## 4. Classifier probability and persona threshold

There is one supervised XGBoost binary classifier for each persona (spoofing, wash trading, pump-and-dump, layering). For each row and persona `p`, the model produces a probability `prob_p`. This is a model score, not necessarily a calibrated real-world probability.

The per-persona decision is:

```text
flag_p = prob_p >= threshold_p
```

Thresholds are tuned during training, separately for each persona:

1. Construct feature matrix columns that are numeric and not excluded identifiers/timestamps/group metadata.
2. For each persona, split training rows using `GroupKFold` grouped by source market (`cv_group`). Grouping keeps windows from one market together, which matters because adjacent windows overlap heavily.
3. Generate out-of-fold probabilities: each validation market is scored by a model trained without that market.
4. Evaluate 27 threshold candidates evenly spaced between the configured bounds (`0.25` and `0.90` by default) and choose based on weighted F1.
5. Smooth the F1 curve over neighboring thresholds and choose the center of its near-optimal plateau. The prior threshold can be retained if still within the configured tolerance, for at most two retrains in a row.
6. Fit the final persona model on all supplied training rows.

Because positives are rare, `scale_pos_weight` is based on the negative-to-positive ratio, capped at 50 by default. Threshold tuning may subsample negatives (`cv_negative_fraction: 0.25`) and weights the retained negatives to account for that sampling.

Threshold tuning falls back to the configured default threshold (0.5) when there are fewer than two positive source-market groups, fewer than two groups total, or no usable out-of-fold positive predictions. A persona classifier is skipped if it lacks the minimum positive training rows (five by default), unless strict mode is enabled.

Threshold selection optimizes F1 on these out-of-fold training predictions. It is not tuned on the held-out test markets, and it is not selected to achieve a target FPR.

## 5. Isolation Forest scores and the ensemble

The scoring system also has two unsupervised components:

### Account-level Isolation Forest

The trader-level Isolation Forest scores behavioral features, peer-relative features, and self-history features. Market-context columns are excluded from this forest. Robust scaling uses the training median and a 10th-to-90th percentile range. The raw anomaly score is converted to an empirical percentile against the training-score distribution:

```text
iso_score = fraction of calibrated training scores <= this row's anomaly score
iso_flag  = iso_score >= 1 - contamination
```

With the default trader contamination of `0.01`, the base anomaly cutoff is the 99th percentile of training anomaly scores. This is a percentile threshold, not a promise that exactly one percent of future accounts will be flagged.

### Market-level Isolation Forest

The market forest uses `mkt_*` context features and produces `mkt_anomaly_score`. That score is joined into each account row as context for the supervised classifiers. It does not independently create account alerts.

### Current default ensemble: `or_corroborated`

[`configs/detector_config.yaml`](../configs/detector_config.yaml) currently selects `combination_rule: or_corroborated`. The account is an alert if either:

1. Any persona classifier fires at its own threshold; or
2. The account-level Isolation Forest fires **and** that anomaly is corroborated by either:
   - the largest persona probability is at least 50% of that persona's threshold; or
   - at least three self-history z-score features have absolute value at least 6.

An Isolation Forest flag without corroboration becomes a `review`, not a final alert. A row with neither classifier nor anomaly flag is `none`.

The emitted `final_score` is the maximum persona probability or, when the anomaly is an alert, its `iso_score`; `final_flag` is the ensemble alert decision. Under the alternative `weighted` rule, the score is a weighted combination and the row is flagged if any persona classifier fires or the weighted score exceeds 0.5. That is not the configured default.

If multiple persona classifiers fire, `predicted_persona` is the one with the largest margin `prob_p - threshold_p`. If no classifier fires but the anomaly becomes an alert, the prediction is `"anomaly"`. Review rows do not retain a predicted persona. `triggered_by` records the classifiers and/or Isolation Forest that fired.

## 6. What the evaluation numbers mean

The adversarial loop evaluates held-out test-market windows twice where applicable:

- **PRE**: the detector frozen at the end of the previous round. This asks whether the new mutation evaded the old detector.
- **POST**: after retraining on training markets through the current round plus configured replay-augmentation training cases. This asks how well the updated detector recognizes the held-out markets.

Round 0 has no prior detector, so it has no meaningful PRE result. Its POST result follows the first training.

For each round, the orchestrator simulates one training market and multiple separately seeded test markets. Test markets are not used for fitting or threshold tuning. Test and training markets do share simulator and persona implementations, so these metrics test generalization across generated markets/seeds, not fully independent real-market generalization.

### Per-persona row-level metrics

The usual per-persona Precision, Recall, F1, and FPR use that persona's own `flag_p` against that persona's binary label:

- **Precision** = true-positive rows / all rows flagged for that persona.
- **Recall** = true-positive rows / all positive-labeled rows for that persona.
- **F1** = harmonic mean of precision and recall.
- **FPR** = negative rows for that persona flagged positive / all negative rows for that persona.
- **Support** = number of positive-labeled account-window rows.

Consequently, a spoofing window flagged only by the layering classifier is not a true positive for spoofing's row-level precision/recall. It can still count as an episode caught by the ensemble below. Other-persona positive windows are negative examples for the current persona's one-vs-rest metric.

Overlapping rolling windows are correlated, so row-level counts and scores should not be interpreted as independent-sample statistical evidence. A high row-level score can also coexist with weak episode coverage if most windows are detected for some episodes and many windows are missed for others.

### Episode detection and latency

For each ground-truth episode, `events_detected` counts it as detected when at least one account involved in the event has a `flag_p` in a window overlapping the episode. Detection latency is:

```text
first flagged window_end - episode start_tick
```

Negative values are floored at zero. The reported latency is the mean for detected episodes only; missed episodes are not inserted as zero-delay events. If none are detected, latency is undefined (`NaN`).

### Episode coverage and “caught” metrics

- **`episode_coverage`**: for each episode, calculate the share of that episode's positive-labeled account-window rows flagged by its own persona classifier, then average those shares across episodes with labelled rows.
- **`caught_coverage`**: same share, but using `final_flag`, so any persona classifier or corroborated anomaly alert counts as caught.
- **`caught_events_detected` / `caught_latency_ticks`**: event detection counts and latency using any final alert, regardless of which persona classifier triggered it.
- **`caught_weak_share`**: fraction of episodes whose positive windows have less than 20% final-alert coverage. This distinguishes an episode barely caught once from one consistently detected.
- **`family_coverage`**: for spoofing and layering, coverage using either the spoofing or layering classifier, making cross-classification visible.

The displayed `Events` count is a count of ground-truth episodes detected / total episodes for the metric's selected flag column; it is not a count of alerts, unique traders, or orders. In the usual persona table this is the persona-specific classifier's event count. The `caught_*` fields in `metrics.json` use any final alert.

### Other metrics

- **Unsupervised metrics** compare `iso_flag` with a single target meaning “any persona label is positive.” These describe the Isolation Forest alone, not the ensemble.
- **`n_flags`** in round metadata is the number of account-window rows with `final_flag`, not unique accounts or episodes.
- **Mutation score** is an adversarial objective based on PRE results and manipulation effectiveness, not classifier accuracy. Its evasion components include lower any-alert episode coverage, practically missed episodes, and detection delay; the result is multiplied by retained manipulation effectiveness.
- Replay checks score configured held-out case simulations separately and report whether the expected persona/family is recognized. Cases included in `replay_augmentation` are training data and must not be described as held out.

## 7. Reading a round table correctly

For example:

```text
spoofing   pre   Precision ... Recall ... F1 ... Events ... Score ...
spoofing   post  Precision ... Recall ... F1 ... Events ...
```

- `pre` is frozen-model mutation evasion on the current round's test markets.
- `post` is after retraining on current and previous round training data plus training replay augmentation.
- Precision/recall/F1/FPR are persona-classifier, row-level metrics.
- `Events` in a persona row uses that persona's classifier, not necessarily the `final_flag` catch metric.
- A separate `caught_coverage` / `caught_latency_ticks` value in `metrics.json` answers whether any alert caught the episode.
- A strong POST result does not prove that the mutation did not challenge the detector: the PRE result is the measure of the new mutation against the frozen detector.
- `r0` (from round 2) scores the round's test markets with the round-0 detector: how far the personas have moved from the baseline.
- `bench` scores the just-retrained detector on a fixed benchmark of unseen random persona variants (`benchmark/market_*`, simulated once per run, never trained on). This is the transfer curve and should rise over rounds. PRE cannot be: the strategist picks each mutation to be the frozen detector's worst case.
- The `Caught` column is any-alert episode coverage. A split spoof flagged by the layering (or `fake_depth` family) classifier, or only by the isolation forest, is caught but scores 0 own-label F1.
- Each round also trains on `round_N/train_variant` (personas from that round with 3-6 knobs redrawn inside the bounds; `loop.train_variants`), so the classifiers learn the schemes rather than one parameter set.

Always check `round_N/metrics.json` for `pre_retrain`, `personas`, `thresholds`, `evaluation_protocol`, and `n_flags` rather than relying on a single printed table.

## 8. Important interpretation limits

1. **A threshold is not a real-world risk probability.** XGBoost probability outputs are used as ranking scores and thresholded; the current procedure does not establish calibration to the real prevalence of manipulation.
2. **The Isolation Forest cutoff is training-relative.** `contamination: 0.01` sets a training-score percentile boundary; future score distributions can differ.
3. **Windows overlap heavily.** Adjacent 3-minute windows share most of their observations. Grouped threshold CV avoids splitting a source market across folds, but it does not make window rows statistically independent.
4. **Seed-held-out is not generator-held-out.** Test seeds are separate, but simulator code and persona mechanisms are shared with training.
5. **Activity labels exclude quiet/cover behavior.** A model is evaluated on the labelled manipulative activity, not necessarily every phase or account in an episode.
6. **Episode detection and row metrics answer different questions.** One early flagged window can make an episode “detected” while its coverage remains low.
7. **Anomaly reviews are not final alerts.** Counting `iso_flag` directly can overstate actionable alerts under the default corroborated ensemble.

## 9. Useful outputs

| Output | What to inspect |
|---|---|
| `round_N/metrics.json` | POST `personas`, PRE `pre_retrain`, episode/caught metrics, unsupervised metrics, thresholds, test seeds, protocol notes, mutation scores. |
| `round_N/thresholds.json` | Fitted threshold per available persona classifier. |
| `round_N/flags.parquet` | One scored account-window row: final flag/score, persona scores, alert tier, trigger sources, SHAP explanation. |
| `features.parquet` | Numeric feature rows before model scoring; index is `(commodity, window_start, trader_id)`. |
| `labels.parquet` | Separate per-persona binary ground truth used for training/evaluation. |
| `round_N/replay_check.json` | Recognition results on configured held-out case replays when that round runs the replay check. |

