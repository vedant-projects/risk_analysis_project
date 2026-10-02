# Japan Bond Shock Risk Report

## Sources

- **curve_source**: real yields_dec panel (last row)
- **scenario_loss_pct**: {'japan_baseline': -3.4533884323377935, 'japan_severe': -8.573000034269377, 'japan_extreme': -16.65536974278053}
- **returns_source**: book_history.parquet

## Portfolio

Current book composition, DV01, and key-rate exposure table.

|         | sleeve     |   weight |   notional_usd |   price |   mod_dur |   convexity |         dv01_usd |   krd_2y_usd |       krd_5y_usd |      krd_10y_usd |   krd_30y_usd |
|:--------|:-----------|---------:|---------------:|--------:|----------:|------------:|-----------------:|-------------:|-----------------:|-----------------:|--------------:|
| UST_2Y  | treasuries |     0.06 |        6e+08   |   98.72 |     1.892 |         4.6 | 112092           |   113774     |     -0           |     -0           |  -0           |
| UST_5Y  | treasuries |     0.1  |        1e+09   |   96.76 |     4.425 |        23   | 428203           |    -3979.48  | 447724           |     -0           |  -0           |
| UST_10Y | treasuries |     0.14 |        1.4e+09 |   96.21 |     7.806 |        73.7 |      1.05139e+06 |    -2868.86  | -11000.9         |      1.11309e+06 |  -0           |
| UST_30Y | treasuries |     0.1  |        1e+09   |   92.18 |    14.673 |       325.8 |      1.35262e+06 |     -784.492 |  -3008.21        | -34311           |   1.51409e+06 |
| IG_7Y   | ig_corp    |     0.3  |        3e+09   |  100.06 |     5.806 |        40.1 |      1.74276e+06 |      140.92  |      1.04926e+06 | 699454           |  -0           |
| HY_4Y   | hy         |     0.1  |        1e+09   |  108.81 |     3.465 |        14.6 | 376992           |   121568     | 225667           |     -0           |  -0           |
| MBS_5Y5 | mbs        |     0.1  |        1e+09   |  101.36 |     4.716 |        26.4 | 477989           |     1426.66  | 424546           |  46993.1         |  -0           |
| CASH    | cash       |     0.1  |        1e+09   |  nan    |     0     |         0   |      0           |        0     |      0           |      0           |   0           |
| TOTAL   | TOTAL      |     1    |        1e+10   |  nan    |   nan     |       nan   |      5.54205e+06 |   229277     |      2.13319e+06 |      1.82523e+06 |   1.51409e+06 |

## Scenarios

Japan shock scenarios and portfolio-level loss estimates.

| scenario       |   rates_$M |   spread_$M |   haircut_$M |   total_$M |   total_%NAV |   dur_only_$M |   taylor2_$M |
|:---------------|-----------:|------------:|-------------:|-----------:|-------------:|--------------:|-------------:|
| japan_baseline |   -263.316 |    -52.0231 |          -30 |   -345.339 |     -3.45339 |      -269.371 |     -259.743 |
| japan_severe   |   -611.231 |   -156.069  |          -90 |   -857.3   |     -8.573   |      -652.36  |     -594.844 |
| japan_extreme  |  -1173.4   |   -312.139  |         -180 |  -1665.54  |    -16.6554  |     -1339.52  |    -1104.08  |

## VaR

Latest historical VaR and expected shortfall estimates.

| metric         |   value_pct |
|:---------------|------------:|
| Latest 95% VaR |    0.361793 |
| Latest 95% ES  |    0.52002  |
| Latest 99% VaR |    0.672618 |
| Latest 99% ES  |    0.746417 |
| Worst day      |   -1.57891  |
| Std dev        |    0.305764 |

## Backtest

Historical exceedance counts for the VaR series.

|   alpha |   n_obs |   expected_breaches |   actual_breaches |   breach_rate |   expected_rate |
|--------:|--------:|--------------------:|------------------:|--------------:|----------------:|
|    0.95 |    5421 |              271.05 |               304 |     0.0560782 |            0.05 |
|    0.99 |    5421 |               54.21 |                70 |     0.0129127 |            0.01 |

## Worst days

Worst realized book-return days in the available history.

| date       |   book_return |
|:-----------|--------------:|
| 2022-06-13 |    -0.0157891 |
| 2008-09-19 |    -0.0157268 |
| 2020-03-17 |    -0.0140074 |
| 2008-09-30 |    -0.0137882 |
| 2008-03-24 |    -0.0130754 |
| 2020-03-10 |    -0.0122222 |
| 2009-06-01 |    -0.0119819 |
| 2010-12-07 |    -0.011196  |
| 2013-07-05 |    -0.0111045 |
| 2024-04-10 |    -0.0109933 |

## Scenario vs history

### One-day comparison

| scenario       |   loss_pct |   days_worse_or_equal |   share_of_history_pct |   approx_return_period_days |   approx_return_period_years |
|:---------------|-----------:|----------------------:|-----------------------:|----------------------------:|-----------------------------:|
| japan_baseline |    3.45339 |                     0 |                      0 |                         inf |                          inf |
| japan_severe   |    8.573   |                     0 |                      0 |                         inf |                          inf |
| japan_extreme  |   16.6554  |                     0 |                      0 |                         inf |                          inf |

### Five-day comparison

| scenario       |   loss_pct |   days_worse_or_equal |   share_of_history_pct |   approx_return_period_days |   approx_return_period_years |
|:---------------|-----------:|----------------------:|-----------------------:|----------------------------:|-----------------------------:|
| japan_baseline |    3.45339 |                     0 |                      0 |                         inf |                          inf |
| japan_severe   |    8.573   |                     0 |                      0 |                         inf |                          inf |
| japan_extreme  |   16.6554  |                     0 |                      0 |                         inf |                          inf |
