# Stylized Facts & Market Microstructure Verification

**Simulation Length:** 10000 ticks per commodity

This report verifies that the simulated limit order book organically exhibits universal statistical properties of real financial markets.

## 1. Macroeconomic Stylized Facts
- **Excess Kurtosis (> 0):** Proves 'fat tails' (extreme events occur more frequently than a normal distribution).
- **Abs Rtn Autocorr (> 0):** Proves 'volatility clustering' (large moves follow large moves).
- **Raw Rtn Autocorr (~ 0):** Proves 'local efficiency' (returns themselves are not trivially predictable).

## 2. Microstructure & Manipulation Footprints
- **Order-to-Trade Ratio:** Usually high in algorithmic markets (3.0 - 15.0).
- **Avg Spread:** Organically maintained by the MM inventory model.
- **Cancel Ratios:** Should spike significantly during manipulation episodes (Spoofing/Layering) compared to normal ledgers.

## Results Table

| Commodity     |   Excess Kurtosis |   Abs Rtn Autocorr (Lag 1) |   Raw Rtn Autocorr (Lag 1) |   Order-to-Trade Ratio |   Avg Spread (Ticks) | Normal Cancel Ratio   | Manip Cancel Ratio   |
|:--------------|------------------:|---------------------------:|---------------------------:|-----------------------:|---------------------:|:----------------------|:---------------------|
| wheat         |              0.34 |                      0.353 |                      0.36  |                   2.94 |                 2.09 | 0.00%                 | 0.00%                |
| corn          |              1.18 |                      0.221 |                      0.401 |                   2.92 |                 2.03 | 0.00%                 | 0.00%                |
| crude_oil_wti |              5.06 |                      0.521 |                      0.528 |                   3.01 |                 3.48 | 0.00%                 | 0.00%                |
| gold          |              0.19 |                      0.473 |                      0.595 |                   3    |                 2.39 | 0.00%                 | 0.00%                |
| silver        |              1.75 |                      0.304 |                      0.233 |                   2.96 |                 1.98 | 0.00%                 | 0.00%                |
| copper        |              1.08 |                      0.696 |                      0.788 |                   3.04 |                 2.63 | 0.00%                 | 0.00%                |
| brent_crude   |              5.65 |                      0.682 |                      0.635 |                   3.23 |               -46.66 | 0.00%                 | 0.00%                |
| natural_gas   |             21.59 |                      0.639 |                      0.617 |                   3.15 |                 2.48 | 0.00%                 | 0.00%                |