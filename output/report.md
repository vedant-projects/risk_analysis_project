# Japan Bond Shock Risk Report

## Sources

- **curve_source**: built-in demo curve
- **scenario_loss_pct**: {'japan_baseline': -3.4896944479647094, 'japan_severe': -8.654614811612364, 'japan_extreme': -16.79988047064024}
- **returns_source**: book_history.parquet missing - VaR tables skipped

## Portfolio

Current book composition, DV01, and key-rate exposure table.

|         | sleeve     |   weight |   notional_usd |   price |   mod_dur |   convexity |         dv01_usd |       krd_2y_usd |       krd_5y_usd |      krd_10y_usd |   krd_30y_usd |
|:--------|:-----------|---------:|---------------:|--------:|----------:|------------:|-----------------:|-----------------:|-----------------:|-----------------:|--------------:|
| UST_2Y  | treasuries |     0.06 |        6e+08   |  100.04 |     1.899 |         4.6 | 114010           | 113993           |     -0           |     -0           |  -0           |
| UST_5Y  | treasuries |     0.1  |        1e+09   |  100    |     4.45  |        23.2 | 445045           |     -0           | 445893           |     -0           |  -0           |
| UST_10Y | treasuries |     0.14 |        1.4e+09 |  100    |     7.869 |        74.6 |      1.10162e+06 |     -0           |     -1.98952e-07 |      1.11472e+06 |  -0           |
| UST_30Y | treasuries |     0.1  |        1e+09   |  100    |    15.28  |       346   |      1.52798e+06 |     -1.42109e-07 |     -1.42109e-07 |     -1.42109e-07 |   1.52527e+06 |
| IG_7Y   | ig_corp    |     0.3  |        3e+09   |  103.88 |     5.847 |        40.5 |      1.82216e+06 |   9208.91        |      1.05993e+06 | 695109           |  -0           |
| HY_4Y   | hy         |     0.1  |        1e+09   |  111.66 |     3.484 |        14.7 | 389001           | 123839           | 224951           |     -0           |  -0           |
| MBS_5Y5 | mbs        |     0.1  |        1e+09   |  104.85 |     4.746 |        26.7 | 497591           |   5001.83        | 424356           |  46508.2         |  -0           |
| CASH    | cash       |     0.1  |        1e+09   |  nan    |     0     |         0   |      0           |      0           |      0           |      0           |   0           |
| TOTAL   | TOTAL      |     1    |        1e+10   |  nan    |   nan     |       nan   |      5.89741e+06 | 252043           |      2.15513e+06 |      1.85634e+06 |   1.52527e+06 |

## Scenarios

Japan shock scenarios and portfolio-level loss estimates.

| scenario       |   rates_$M |   spread_$M |   haircut_$M |   total_$M |   total_%NAV |   dur_only_$M |   taylor2_$M |
|:---------------|-----------:|------------:|-------------:|-----------:|-------------:|--------------:|-------------:|
| japan_baseline |   -266.601 |    -52.3681 |          -30 |   -348.969 |     -3.48969 |      -274.536 |     -264.45  |
| japan_severe   |   -618.357 |   -157.104  |          -90 |   -865.461 |     -8.65461 |      -664.997 |     -604.715 |
| japan_extreme  |  -1185.78  |   -314.208  |         -180 |  -1679.99  |    -16.7999  |     -1365.04  |    -1118.47  |

## VaR

VaR section not yet available because return-history exports are missing.

## Backtest

Backtest section not yet available.

## Worst days

Worst-days section not yet available.

## Scenario vs history

One-day scenario/history comparison not yet available.

Five-day scenario/history comparison not yet available.
