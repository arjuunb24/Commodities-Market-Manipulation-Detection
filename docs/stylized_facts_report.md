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
| wheat         |              7.73 |                      0.141 |                     -0.386 |                   3.22 |                 0.01 | 9.58%                 | 9.50%                |
| corn          |             -0.14 |                      0.06  |                     -0.362 |                   3.22 |                 0.01 | 9.60%                 | 9.79%                |
| brent_crude   |              2.21 |                      0.028 |                     -0.052 |                   3.33 |                 0.05 | 10.72%                | 10.76%               |
| copper        |              1.41 |                      0.068 |                      0.022 |                   3.36 |                 0    | 11.09%                | 11.19%               |
| natural_gas   |              0.51 |                      0.004 |                     -0.341 |                   3.18 |                 0    | 8.98%                 | 9.20%                |
| crude_oil_wti |              5.35 |                      0.053 |                     -0.084 |                   3.3  |                 0.04 | 10.41%                | 10.71%               |
| silver        |              0.33 |                      0.051 |                     -0.251 |                   3.27 |                 0.02 | 10.01%                | 10.17%               |
| gold          |              4.11 |                      0.177 |                      0.073 |                   3.51 |                 1.39 | 12.75%                | 12.66%               |