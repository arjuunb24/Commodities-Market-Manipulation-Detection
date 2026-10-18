"""
docs/enforcement_case_notes.md
===============================
Enforcement case notes: spoofing, layering, wash trading, pump-and-dump and price pushes.

Used to ground the manipulation personas, the replayed real cases
(configs/case_replays.yaml) and the adversarial strategist, which reads one section
per persona through its `get_enforcement_notes` tool.

Rules for this file (revised Oct 2026):
- "Documented" lines state only what the cited public source says (CFTC/DOJ releases
  and orders, court opinions, reputable reporting). Each case lists its sources.
- Anything added for the simulator (sizes, tick counts, evasion knobs) is marked
  "Design note" or listed under "Tactics without a documented case".
- An earlier version cited a "CFTC 2020 staff advisory on spoofing evasion" and a
  "CFTC v. Nefedova et al. (2020)" silver case. Neither could be found in CFTC records,
  so both were removed; tactics that relied on them are now design assumptions.
"""

# Spoofing Cases

Pattern documented across cases: a small genuine order on one side, larger orders on the other side
placed with intent to cancel, pulled once the genuine order fills or after seconds; repeated many times.
- Coscia/Panther (2011): algorithm placed a small order, then large opposite-side orders it cancelled automatically after a short interval (milliseconds); configured so they would not all fill.
- Walleye (2022 order): small order of 10 or fewer contracts, then 50+ contracts on the opposite side, cancelled within 10 seconds.
- Tower Research (2012-2014): large orders pulled after smaller genuine opposite orders filled; an order splitter entered several smaller, randomly-sized orders to obscure the scheme.
- Sarao (2010-2015): one-sided campaigns (sell side) for long stretches; orders repriced to stay several levels behind the touch.
- Bank desks (JPMorgan 2008-2016, Merrill 2008-2014, Deutsche Bank, Scotiabank): several traders on one precious-metals desk; Pacilio typically one large order, Bases many smaller orders.
- Oystacher/3Red (2011-2014): false depth in five products on four exchanges.

## Michael Coscia / Panther Energy Trading (2011)
**Sources:** CFTC Release 6649-13 (22 Jul 2013); United States v. Coscia, 866 F.3d 782 (7th Cir. 2017), No. 16-3017.
**Documented:** algorithmic trading on CME Globex across a broad range of futures, 8 Aug - 18 Oct 2011.
A small order on one side, then large orders on the opposite side; the algorithm was configured so the
large orders would not all be filled and cancelled them automatically after a short interval
(milliseconds). First CFTC penalty under the Dodd-Frank spoofing prohibition: $1.4M penalty, $1.4M
disgorgement, one-year trading ban (2013). Criminal conviction upheld by the 7th Circuit (2017).

## Walleye Trading (2022 CFTC order)
**Source:** CFTC order, 12 Dec 2022 (cftc.gov/media/7981).
**Documented:** a trader repeatedly (1) placed a small order of ten or fewer contracts on one side,
(2) while it was pending placed an order of fifty or more contracts on the opposite side that he
intended to cancel, and (3) cancelled the larger order within ten seconds.

## Tower Research Capital (2012-2014)
**Sources:** CFTC Release 8074-19 and order (7 Nov 2019); DOJ guilty pleas of Gandhi and Mohan (2018).
**Documented:** E-mini equity index futures. Traders placed large orders on one side and cancelled them
once smaller genuine orders on the opposite side filled. They "often used an order splitter to enter
several smaller, randomly-sized orders in an attempt to obscure their scheme". $67.4M total
($32.6M restitution, $10.5M disgorgement, $24.4M penalty).

## Navinder Sarao / Nav Sarao Futures (2010-2015)
**Sources:** CFTC Release 7156-15 (21 Apr 2015); CFTC Release 7480-16 (2016); CNBC reporting on the complaint.
**Documented:** a modified trading platform placed several large SELL orders one price level apart in the
E-mini S&P 500 book and repriced them as the market moved to stay at least 3-4 levels above the best
ask; most were cancelled without executing. Used on 400+ trading days. On 6 May 2010 the orders were
about 20-29% of the sell-side book (up to ~40% on some days, per reporting). Guilty plea (spoofing,
wire fraud); more than $38M in sanctions sought in the CFTC consent order.

## JPMorgan precious metals desk (2008-2016)
**Sources:** CFTC Release 8260-20 (29 Sep 2020); DOJ, Smith and Nowak convicted (Aug 2022).
**Documented:** traders placed orders they intended to cancel before execution to move prices toward
orders they intended to execute on the opposite side; gold, silver, platinum and palladium futures
(NYMEX/COMEX) and Treasury futures; hundreds of thousands of spoof orders over about eight years;
$920.2M in total. Several traders on the desk took part.

## Merrill Lynch precious metals (Bases and Pacilio, 2008-2014)
**Sources:** United States v. Bases and Pacilio, 18-cr-48 (N.D. Ill.), convicted Aug 2021; trial testimony reported by Bloomberg.
**Documented:** large buy and sell orders cancelled quickly to push gold, silver and platinum prices.
Testimony: Pacilio typically used a single large order; Bases favoured many smaller orders.

## Deutsche Bank (Vorley and Chanu, 2008-2013)
**Source:** DOJ, convicted Sept 2020.
**Documented:** orders in COMEX gold and silver futures they did not intend to execute, to create a
false impression of supply or demand and trade on the opposite side.

## Bank of Nova Scotia (Corey Flaum and others)
**Sources:** CFTC order (19 Aug 2020); DOJ plea of Flaum (2019).
**Documented:** thousands of gold and silver futures orders placed with intent to cancel, by four traders
on the precious-metals desk; earlier $800K CFTC penalty (2018) for 2013-2016 conduct.

## Igor Oystacher / 3Red Trading (2011-2014)
**Sources:** CFTC Release 7264-15 (19 Oct 2015) and complaint; consent order Dec 2016.
**Documented:** spoofing on at least 51 days, Dec 2011 - Jan 2014, in E-mini S&P 500, copper, crude oil,
natural gas and VIX futures on four exchanges, creating the appearance of false market depth.
$2.5M penalty and a three-year independent monitor.
**Not verified here:** the commonly reported "flip" (cancel, then trade aggressively the other way).

---

# Layering Cases

Layering = several non-bona-fide orders stacked on consecutive price levels of one side to fake depth.
- Sarao (2010-2015): several large sell orders one level apart, repriced to stay 3-4+ levels above the best ask, resting for long stretches, mostly cancelled; ~20-29% of sell-side depth on 6 May 2010.
- Merrill/Bases: many smaller orders rather than one large order (testimony).
- Tower: an order splitter producing several smaller, randomly-sized orders.
No case found that documents layer-size tapering, staggered placement or randomised layer spacing:
those are design assumptions (see the last section).

---

# Wash Trading Cases

Pattern documented across cases: equal and opposite orders between accounts with common ownership,
matched in product, quantity, price and timing, often within seconds.
- Cargill de Mexico (2010-2014): equal and opposite orders between two accounts it owned (corn, soybean, wheat), to move hedges between accounts.
- Shinhan Securities (2025): 127 wash trades across its accounts, offsetting orders placed within seconds.
- Aralia/Vintage (2021): accounts with the same ultimate beneficial owner placed offsetting orders in the same product.
- Coinbase (2015-2018): two internal trading programs matched each other; one employee washed Litecoin/Bitcoin for six weeks in 2016.
- CME rule: orders for accounts with common beneficial ownership, entered to negate market risk or price competition, are wash trades.

## Cargill de Mexico (2010-2014)
**Sources:** CFTC Release 7241-15 (24 Sep 2015) and order.
**Documented:** wash sales and non-competitive transactions in CBOT corn, soybean and wheat and KCBT hard
red wheat futures, Mar 2010 - Aug 2014: employees entered equal and opposite orders for another
account it also owned, matching product, quantity, price and timing, to transfer hedging positions.
$500K penalty; undertaking to use self-match prevention.

## Shinhan Securities (2025)
**Source:** CFTC administrative order (17 Sep 2025).
**Documented:** a trader executed 127 wash trades across Shinhan accounts at a US futures commission
merchant, placing offsetting orders within seconds, knowing they would fill against each other.

## Aralia / Vintage (2021)
**Source:** CFTC order (27 Dec 2021).
**Documented:** trading accounts of two entities with the same ultimate beneficial owner placed
offsetting orders in the same futures product; also treated as non-competitive transactions.

## Coinbase Inc. (2021 order)
**Sources:** CFTC Release 8369-21 (19 Mar 2021); Coinbase 10-Q.
**Documented:** Jan 2015 - Sept 2018, two trading programs ("Hedger" and "Replicator") generated orders
that at times matched each other, and the platform did not disclose that Coinbase ran more than one
program or traded through multiple accounts; over six weeks in Aug-Sept 2016 a former employee
placed Litecoin/Bitcoin orders that matched each other as wash trades. $6.5M penalty.
**Not documented:** rotating ("cyclical") washes through rings of accounts; that is a design assumption.

## Kraft Foods / Mondelez (wheat, 2011)
**Sources:** CFTC complaint, No. 15-cv-2881 (N.D. Ill., Apr 2015); consent order May 2022 ($16M, paid by Mondelez); Reuters.
**Documented (allegations):** about $90M of December 2011 wheat futures bought without intent to take
delivery, to move cash-wheat prices; position limits exceeded; non-competitive trading of both sides
of exchange-for-physical wheat transactions between related accounts.

---

# Pump-and-Dump and Price-Push Cases

Classic pump-and-dumps (accumulate, promote or ignite, sell into the rise) are documented mainly in
thin markets (small caps, crypto tokens: e.g. Telegram-coordinated pumps, Kamps & Kleinberg 2018).
In large commodity futures the documented price pushes are mostly concentrated trading around the
settlement ("banging the close") to move a settlement price that other positions reference.
- Optiver (2007): about 20-30% of its trades just before the close and the rest during it; NYMEX crude, heating oil, gasoline.
- Moore Capital (2007-2008): banging the close to push platinum and palladium settlement prices up.
- Amaranth (2006): heavy selling in the 2:00-2:30 pm closing range on expiry days to push natural gas settlement down, benefiting short swaps.
No futures case found that documents a multi-account coordinated pump with staggered entries, decoy
trades or a phased exit: the pump_and_dump persona's evasion knobs are design assumptions.

## Optiver (March 2007)
**Sources:** CFTC Release 5521-08 (24 Jul 2008); CFTC Release 6239-12 (consent order, Apr 2012).
**Documented:** 19 attempted manipulations over 11 days in NYMEX crude oil, heating oil and gasoline
futures (at least five successful): a substantial position acquired leading up to the closing
period, about 20-30% of the trades just before the close and the remainder during it. $13M penalty
and $1M disgorgement.

## Moore Capital (Christopher Pia, 2007-2008)
**Sources:** CFTC Release 5815-10 (29 Apr 2010); CFTC Release 6079-11 (Pia, 2011).
**Documented:** a portfolio manager attempted to push NYMEX platinum and palladium settlement prices up by
banging the close, about Nov 2007 - May 2008. $25M and a two-year restriction on closing-period
trading for the firm; Pia $1M and a permanent ban from closing-period trading.

## Amaranth Advisors (Brian Hunter, 2006)
**Sources:** CFTC complaint (July 2007); consent order ($7.5M, 2009); Risk.net.
**Documented:** attempted manipulation of NYMEX natural gas settlement prices on expiry days (Feb and Apr
2006): futures shorts reversed to long holdings during the day, then large volumes sold in the
closing range (2:00-2:30 pm, whose volume-weighted price sets the settlement) while holding large
short swaps on ICE that settle on that price.

---

# Tactics Without a Documented Case (design assumptions)

These knobs make the personas harder targets for the detector. They are plausible evasions but no
case above documents them; the strategist should treat their realism as unproven.
- Randomised cancel timing (spoofing, layering): documented cancels are fast and automatic (Coscia, Walleye).
- Trading mainly in busy periods (all personas).
- Rotating wash trades through rings of 3-6 accounts; camouflage directional trades among the washes.
- Pump-and-dump: more coordinated accounts, staggered entries, gentler ignition, decoy trades, phased exit.
- Layering: tapered layer sizes, staggered placement, wider spacing, fewer layers.
- Cover trading alongside the scheme: Sarao and Coscia traded genuinely as well, but the sizes and
  rates used here are assumptions.
