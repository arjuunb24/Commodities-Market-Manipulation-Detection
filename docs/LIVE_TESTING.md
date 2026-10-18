# Testing the detector on real market data

This guide covers three tools:

1. **Saved models.** Every adversarial round now saves the models it trained.
2. **Fetching real data.** A script downloads and saves recent real 1-minute bars for the eight commodities.
3. **Prediction and checking.** A separate script scores a saved dataset with a round's models. It then checks each flagged window against how the four manipulation personas actually move a market, using rules and, optionally, an LLM.

Read [What this can and cannot find](#what-this-can-and-cannot-find) before trusting any output.

---

## The one thing to know first

The project's main detector scores **accounts**. It reads each trader's own orders, cancels and counterparties from the simulator's order and trade logs. **Public market data has none of that.** Exchanges publish anonymous prices and volumes; who placed which order is visible only to the exchange and regulators.

So on real data the tools work at the level of **windows, not traders**. They answer "did gold between 15:24 and 15:27 UTC trade the way a spoofing episode leaves its mark on 1-minute bars?", never "which account was spoofing".

---

## 1. Saved models

After every round, `src/adversarial/orchestrator.py` writes:

```
data/runs/<run>/round_N/models/
    detector.joblib    the trader-level detector, retrained that round
    bar_model.joblib   the bar-level model, for anonymous 1-minute bars
    manifest.json      what was saved, thresholds, settings, and the bar model's skill
```

| Model                | Works on                                                                   | Use it for                                                                   |
| -------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `detector.joblib`  | Order and trade logs **with account ids**, in the simulator's format | New simulated sessions; any future data source that has account-level orders |
| `bar_model.joblib` | Anonymous 1-minute OHLCV bars                                              | Real exchange data from `fetch_market_data.py`                             |

The bar model is trained each round on the simulator's training markets for rounds 0..N, plus the test markets of rounds 0..N-1 (all labelled, none used to judge round N). It is then scored on round N's held-out test markets. That held-out skill is saved in `manifest.json` and printed with every real-data prediction.

### Runs made before this existed

Backfill them. This simulates nothing and calls no LLM:

```powershell
.\.venv\Scripts\python scripts\save_round_models.py --run adversarial_20261016T140752Z_seed42
.\.venv\Scripts\python scripts\save_round_models.py --run latest --bar-only     # bar model only, fast
```

The detector is retrained round by round from the run's cached features, using the current `configs/detector_config.yaml`. Each manifest records `threshold_match`: whether the rebuilt thresholds equal the ones the run originally saved. `false` means the detector settings have changed since that run.

---

## 2. Fetch real data

```powershell
.\.venv\Scripts\python scripts\fetch_market_data.py --hours 6
```

| Option                        | Meaning                                                                                                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--hours N`                 | How many hours of 1-minute bars (up to 168; Yahoo keeps about 7 days at 1-minute resolution)                                            |
| *(default)*                 | The **last N hours of trading**, counted back from the newest bar. On a weekend this returns Friday's session instead of nothing. |
| `--wall-clock`              | The N hours before now, even if markets were closed for part of it                                                                      |
| `--commodities gold,silver` | A subset (default: all eight)                                                                                                           |

Each fetch is saved under a name that records the source, interval and the UTC window it covers, so you can re-use it later:

```
data/live/yf_1m_20261016T1500Z_20261016T2100Z/
    bars.parquet     commodity, ticker, ts_utc, open, high, low, close, volume
    manifest.json    window, rows per commodity, fetch time, notes
```

Symbols are Yahoo's front-month continuous futures: CL=F (WTI), BZ=F (Brent), NG=F (natural gas), GC=F (gold), SI=F (silver), HG=F (copper), ZC=F (corn), ZW=F (wheat). `data/live/` is git-ignored.

---

## 3. Predict and check

```powershell
.\.venv\Scripts\python scripts\predict_live.py                       # latest dataset, latest saved models
.\.venv\Scripts\python scripts\predict_live.py --data yf_1m_20261016T1500Z_20261016T2100Z --round 3
.\.venv\Scripts\python scripts\predict_live.py --llm --llm-max 5      # add an LLM second opinion
.\.venv\Scripts\python scripts\predict_live.py --list                 # saved datasets and model rounds
```

What it does:

1. Computes 50 scale-free features for every minute: returns in units of local volatility over 1-30 minutes, volume against its own median, range, wicks, run-ups and give-backs, and moves shared with other commodities. It also computes episode-length (centred 5-15 minute) aggregates and microstructure estimators that work on bars: order-flow imbalance by bulk volume classification, price impact (Kyle-lambda and Amihud style), Roll and Corwin-Schultz spreads, variance ratio, jump share, and volume left unexplained by the price move.
2. Scores each minute for each persona with the bar model, smooths the scores over 5 minutes (episodes last 3-15 minutes), and flags minutes above the persona's threshold. Thresholds hold to an **alert budget** of about 2% of minutes per persona.
3. Merges nearby flagged minutes into windows, ranked by how far they clear the threshold.
4. Runs the **mechanism checker** on the top windows (`--top`, default 20).
5. With `--llm`, sends the measured evidence for the top `--llm-max` windows to the LLM for a written opinion.

Output goes to `data/live/<dataset>/predictions/<run>_r<N>_<timestamp>/`:

| File                      | Contents                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `report.md`             | Readable summary: model skill, every checked window, test results, ordinary explanations, LLM opinion |
| `cases.json`            | The same, with all measured evidence                                                                  |
| `minute_scores.parquet` | Every minute: per-persona probability, flag, margin                                                   |

### The mechanism checker

For each window it measures the move, volume, range, wicks, how much of the move was given back, which other commodities moved with it, and which scheduled events it overlaps. It then tests each persona:

| Persona       | Tests (all must pass)                                                                                            | Can bars prove it?                                     |
| ------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Pump and dump | Run-up of at least 2.5 sigma; volume at least 1.5x normal; at least half given back within 30 min; price went up | Pattern only. Proof needs account positions over time. |
| Wash trading  | Volume at least 2.5x normal; price moved at most 1 sigma; bars stayed narrow                                     | No. Needs buyer and seller account ids.                |
| Spoofing      | Push of at least 2 sigma; on light volume; at least 60% undone within 5 min; long wicks                          | No. Needs the order book (placements and cancels).     |
| Layering      | Stepped push of at least 1.5 sigma over at least 3 minutes; light volume; half undone within 5 min               | No. Needs order-book depth by level.                   |

It looks for ordinary explanations first:

- **Scheduled events:** data releases (8:30 and 10:00 ET), EIA crude (Wednesday 10:30 ET) and natural-gas storage (Thursday 10:30 ET), settlement windows, LBMA gold and silver auctions, session opens, the daily halt and reopen, and usual USDA report times.
- **Sector or market-wide moves:** if other commodities moved the same way, it points to news rather than one manipulator.
- **Thin markets:** many minutes with no trades, so small orders move the price.

Verdicts:

| Verdict                       | Meaning                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `consistent`                | Every test passes and nothing ordinary explains it (possible only for pump and dump)  |
| `consistent (unverifiable)` | Every test passes, but the persona can only be proven with order-book or account data |
| `partial`                   | Some tests pass                                                                       |
| `likely benign`             | The pattern is there, but a scheduled event or a shared move explains it              |
| `not consistent`            | The window does not behave like this persona                                          |

The checker's thresholds are in `RULES` at the top of `src/live/checker.py`.

### The LLM reviewer (`--llm`)

It reads only the evidence the checker measured and returns a JSON verdict, a confidence level, 3–5 sentences of reasoning, ordinary explanations, and what data would settle the question. It is told the model is weak and that most flags are expected to be ordinary activity, and it is never allowed to call manipulation proven.

- It uses the project's budget-capped LLM client and the `live_checker` block in `configs/llm_config.yaml`, at about one call per window.
- Its output is labelled as an AI opinion in the report.
- If no key is set or the budget is spent, the rule-based checker still runs.

---

## What this can and cannot find

**Model skill.** On held-out **simulated** markets, where the true answer is known, the bar model's mean AUC is about 0.6-0.67 by round (0.5 = coin flip, 1.0 = perfect). It is uneven across personas. In round 5, spoofing reaches 0.81 and pump and dump 0.70 (26% of its flags are real episodes against a 9% base rate). Wash trading (0.51) and layering (0.63) show no reliable signal: during an episode the manipulators are only 3-13% of the market's volume, and wash trades are indistinguishable from ordinary trades in a bar. For comparison, the trader-level detector on the same markets reaches an F1 of about 0.87. The difference is the account-level data, not the modelling. Even in the simulator, where manipulation is certain, its footprint in anonymous 1-minute bars is faint.

**On real data it is weaker still:**

- The model learned from simulated manipulators, not real ones.
- Real markets are deeper, and big moves are usually news.
- Real manipulation is rare. Over a few hours of eight liquid futures, the expected number of real episodes is close to zero, so nearly every flag will be ordinary activity. The checker is designed to say so.

**Realistic expectations per persona:**

- **Spoofing and layering** do happen in metals and energy futures. Large enforcement cases ran for years and involved thousands of spoof orders. They are invisible in bars, though: finding them takes the exchange's order-by-order data with account ids. A window here can only say "the price moved on little volume and snapped back".
- **Wash trading** in these contracts is rare and is caught by matching buyer and seller accounts, which public data does not have.
- **Pump and dump** is very unlikely in deep markets like crude or gold. It is much more common in thin markets (small caps, small crypto tokens). A real crude or gold rally that reverses is almost always news.

Use this as a research tool that shows the gap between simulated and real surveillance, and as a way to surface unusual windows worth a look. It is not a way to accuse anyone. Nothing it produces is a finding about real traders.
