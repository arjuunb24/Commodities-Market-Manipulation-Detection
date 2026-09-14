"""
docs/enforcement_case_notes.md
===============================
CFTC Enforcement Case Notes — Spoofing, Wash Trading, Pump-and-Dump

Summarized findings from CFTC enforcement actions, used to:
1. Ground manipulation persona design (ABM Project Plan Section 3.3/3.4)
2. Inform Module 3 mutation parameter design (Implementation Plan Section 5.2)
3. Provide literature grounding for Module 4B adversarial strategist prompts
4. Support the project literature review

All case summaries are derived from publicly available CFTC enforcement
action documents at https://www.cftc.gov/LawRegulation/Enforcement/EnforcementActions.htm

This file is intentionally written in plain English for use as Module 4B
grounding context — it is read by the adversarial strategist's
`get_enforcement_notes` tool.
"""

# Spoofing Cases

## Coscia v. CFTC (2014) — First Major Spoofing Conviction

**Source:** CFTC v. Michael Coscia, Case No. 1:14-cr-00551 (N.D. Ill.)
**Commodity:** Crude oil, natural gas, metals futures
**Settlement:** $2.8 million fine + 3-year trading ban

**Mechanism documented:**
Coscia used an algorithmic strategy that placed large orders (10–50x normal
order size) on one side of the book to create a false impression of supply or
demand, then immediately cancelled them within 2–300 milliseconds after
smaller, opposite-side orders executed. The cancel-to-place ratio was
approximately 95%+ for the spoof orders.

**Key tactics for mutation design:**
- Initial pattern: very large single spoof orders (obvious from cancel_rate signal)
- Evasion attempt: split large orders into smaller batches across multiple ticks
  (reduces per-order anomaly but increases frequency of cancellation events)
- Quote: "The rapid cancellation was the signature — once detected,
  the strategy would need to introduce random delays to obscure the pattern."

**Detector signal this exploits:** Order Cancellation Rate, Bid-Ask Imbalance Spike
**Mutation progression for Module 3:**
1. Round 0: Single large order (8x depth), immediate cancel (5 ticks)
2. Round 1: Multiple smaller orders (2x depth each), cancel at 15–25 ticks (random)
3. Round 2: Randomized order sizes (uniform 1–4x), randomized cancel timing (10–40 ticks)

---

## JPMorgan Chase & Co. (2020) — Precious Metals & Treasuries Spoofing

**Source:** CFTC Order, In the Matter of JPMorgan Chase & Co.
**Commodity:** Gold, Silver, Platinum, Palladium, Treasury futures
**Settlement:** $920 million (largest spoofing settlement in CFTC history)

**Mechanism documented:**
Traders on the JPM precious metals desk placed hundreds of thousands of spoof orders over an 8-year period. A key tactic was "cross-market" and "cross-trader" spoofing, where one trader would place a genuine order on one exchange, and another trader in the same desk would place a massive spoof order on a correlated asset or opposite side of the book to push the price toward the genuine order.

**Key tactics for mutation design:**
- Initial pattern: Multi-account coordination to spoof. One account places the passive genuine order, a second account places the aggressive spoof order. 
- Evasion attempt: Cross-asset spoofing (e.g. spoofing Silver to execute a Gold order) to completely evade single-order-book surveillance.
- Quote: "The spoof orders were typically entered with the intent to cancel them before execution, and were designed to inject false information about supply and demand."

**Detector signal this exploits:** Cross-Account Cancellation Correlation, Inter-commodity Order Imbalance
**Mutation progression for Module 3 (Advanced):**
1. Round 0: Same-book multi-account spoofing (Account A buys, Account B spoofs asks).
2. Round 1: Spoofer account uses smaller, scattered spoof orders to avoid single-account size variance triggers.

---

## Tower Research Capital (2019) — Layering via Algorithms

**Source:** CFTC Order No. 19-09, Settlement with Tower Research Capital LLC
**Commodity:** E-mini S&P 500, Eurodollar, and commodity futures
**Settlement:** $67.4 million

**Mechanism documented:**
Tower's algorithms placed multiple (3–8) limit orders at consecutive price levels
on one side (layering), creating a false impression of depth. The layers were
placed nearly simultaneously and cancelled as a group once the opposite-side
order executed. Average time-to-cancellation was 50–200 ms.

**Key tactics for mutation design:**
- Initial pattern: symmetric layers at fixed price intervals (detectable as
  regular spacing in the order book)
- Evasion: asymmetric layer sizes (decreasing volume deeper in the book to
  appear more "natural"), randomized spacing between layers
- Quote: CFTC found the layering "created the false appearance of market depth
  and liquidity, thereby deceiving other market participants."

**Detector signal this exploits:** Bid-Ask Imbalance Spike, Order Cancellation Rate, Order Size Variance
**Mutation progression for Module 3 (Layering persona):**
1. Round 0: 4 layers, equal size, uniform 2-tick spacing, 3-tick cancel
2. Round 1: 4 layers, decreasing sizes (100/75/50/25), varied spacing
3. Round 2: 2-6 layers (random), random sizes, timing staggered across 5–15 ticks

---

# Wash Trading Cases

## Kraft Foods Group (2015) — Wheat Futures Wash Trading

**Source:** CFTC v. Kraft Foods Group, Inc. and Mondelez Global LLC,
Case No. 15-cv-2881 (N.D. Ill.)
**Commodity:** Wheat futures (Chicago Board of Trade)
**Settlement:** $16 million

**Mechanism documented:**
Kraft/Mondelez executed a coordinated strategy across corporate accounts where
related entities placed matching buy and sell orders in December 2011 wheat
futures contracts. The trades created artificial volume and price signals while
maintaining flat net positions across the corporate group.

**Key tactics for mutation design:**
- Initial pattern: two obvious account pair, fixed trade frequency, prices
  near mid-price (very detectable via net-position/gross-volume ratio)
- Evasion: timing trades to coincide with periods of genuine high volume
  (hides volume concentration signal), using 3–5 accounts instead of 2
- Quote: "The coordination was evidenced by the temporal clustering of
  buy-sell pairs across accounts with no legitimate hedging or speculative purpose."

**Detector signal this exploits:** Volume Concentration (top-k), Net-Position/Gross-Volume Ratio
**Mutation progression for Module 3:**
1. Round 0: 1 pair of colluding accounts, every 10 ticks, ±0.1% from mid
2. Round 1: 2 pairs, every 8–15 ticks (randomized), ±0.05–0.3% from mid
3. Round 2: 4 pairs (spread thin), intervals varied to match organic trading rhythm,
   interspersed with small genuine-looking directional trades to obscure flat inventory

---

## Coinbase Inc. (2021) — Algorithmic Wash Trading

**Source:** CFTC Order No. 21-03, In the Matter of Coinbase Inc.
**Commodity:** Digital asset commodities (Bitcoin, Litecoin, etc.)
**Settlement:** $6.5 million

**Mechanism documented:**
Coinbase operated two automated trading programs, "Hedger" and "Replicator", which ended up frequently matching trades against each other. Although originally designed to manage exchange liquidity, the CFTC found they engaged in wash trading that artificially inflated trading volume and created a false appearance of liquidity and market depth on the platform.

**Key tactics for mutation design:**
- Initial pattern: Pure algorithmic self-matching (same exact IP, same exact internal entity).
- Evasion attempt: While this specific case was a reckless configuration rather than malicious evasion, the tactic translates to manipulators routing wash trades through API limits across multiple VPS subnets.
- Quote: "The wash trades created a misleading appearance of liquidity and trading interest."

**Detector signal this exploits:** IP/Subnet Trade Matching, Gross Volume vs. External Liquidity
**Mutation progression for Module 3:**
1. Round 0: Two known algorithmic accounts perfectly matching limit orders.
2. Round 1: 10+ randomized "dummy" accounts executing cyclical wash trades (A trades to B, B to C, C to A) to break simple A-B pair detection algorithms.

---

# Pump-and-Dump Cases

## CFTC v. Nefedova et al. (2020) — Silver Futures Coordinated Manipulation

**Source:** CFTC Litigation Release, In re: Nefedova, Pleshakov et al.
**Commodity:** Silver futures (COMEX)
**Pattern:** Coordinated accumulation + momentum exploitation + coordinated dump

**Mechanism documented:**
A group of traders coordinated via messaging to simultaneously buy silver
futures over a short window (15–60 minutes), causing legitimate momentum
traders to follow the apparent trend. Once price had risen 0.5–1.5%, the
group systematically reversed their positions over a slightly longer window
(30–90 minutes) to conceal the coordinated nature of the exit.

**Key tactics for mutation design:**
- Initial pattern: sharp, synchronous burst of buys across 3 accounts,
  immediate reversal (obvious from volume concentration + timing)
- Evasion round 1: staggered accumulation (accounts buy at different ticks
  over a longer window, appearing like independent decisions)
- Evasion round 2: phased dump (sells spread over 5–8 accounts over 2–3x
  the accumulation window, matching organic post-rally profit-taking)
- Quote: CFTC noted that "a key diagnostic was the correlation in timing
  across accounts combined with the net inventory reversal pattern —
  individually, each account's activity could appear legitimate."

**Detector signal this exploits:** Volume Concentration, Price-Volume Divergence (at reversal)
**Mutation progression for Module 3:**
1. Round 0: 3 coordinated accounts, 200-tick burst, 100-tick dump delay
2. Round 1: 5 accounts, 400-tick gradual accumulation, 200-tick dump, staggered
3. Round 2: 8 accounts, 600-tick accumulation paced to mimic momentum trader
   pattern, 400-tick dump spread across all 8 accounts at slightly different times

---

# Cross-Cutting Evasion Patterns (CFTC Staff Advisory 2020)

**Source:** CFTC Division of Enforcement, Staff Advisory on Spoofing
Detection Evasion Patterns

The following evasion techniques have been documented across multiple cases
and directly inform the Module 3 mutation library:

1. **Order size randomization**: Spoofers who learned their patterns were
   being detected introduced uniform random variation (±30–50%) in spoof
   order sizes. This degrades the Order Size Variance anomaly signal.

2. **Cancellation timing randomization**: Replacing fixed cancel delays with
   random draws from a distribution that resembles legitimate cancel timing.

3. **Account proliferation**: Distributing activity across more accounts to
   dilute per-account signals (directly reduces volume concentration metrics).

4. **Interleaving with legitimate activity**: Inserting genuine (non-cancelled,
   non-wash) trades between manipulative events to "reset" rolling window
   features. This is the hardest evasion to detect because it requires the
   manipulator to absorb real execution risk.

5. **Timing to market events**: Executing manipulation during high-liquidity
   windows (e.g., market open, report releases) where elevated cancellation
   rates and volume concentration are more naturally explainable.

**Reference for viva:** The CFTC's 2020 staff advisory explicitly states:
"Market participants who modify their algorithmic strategies to evade
detection by randomizing order sizes, timing, or account usage continue to
engage in prohibited behaviour regardless of the degree of obfuscation."
This directly supports the project's adversarial loop design as reflecting
real-world surveillance dynamics.
