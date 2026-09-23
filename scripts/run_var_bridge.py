"""scripts/run_var_bridge.py - build rolling historical VaR/ES tables from book_history.

If data/processed/book_history.parquet is missing, falls back to a SYNTHETIC
return series so the VaR machinery can be inspected end-to-end. Every number
printed under the fallback is fake - re-run the real chain (fetchers ->
processors -> build_history) before using any output in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.risk.var_bridge import (load_book_returns, multi_day_losses,
                                 report_table, scenario_vs_history)

SCENARIOS = {
    'japan_baseline': 0.035,
    'japan_severe': 0.087,
    'japan_extreme': 0.168,
}


def synthetic_returns(n: int = 1500) -> pd.Series:
    """Clearly-fake daily returns: calm-ish vol with a 2022-style cluster."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range('2020-01-01', periods=n)
    r = rng.normal(0, 0.0014, n)
    r[640] = -0.0081
    r[641] = -0.0052
    r[642] = 0.0033
    r[643] = -0.0044
    return pd.Series(r, index=idx)


def main() -> None:
    try:
        r = load_book_returns()
        source = 'book_history.parquet'
    except FileNotFoundError:
        r = synthetic_returns()
        source = 'SYNTHETIC DEMO (fake data - do not use in report)'
        print('WARNING: book_history.parquet not found - using synthetic demo series.')
        print('         Run the real chain first for real numbers.\n')

    print(f'return series : {source}')
    print(f'observations  : {len(r)}, {r.index.min().date()} -> {r.index.max().date()}')

    latest, bt, worst = report_table(r)
    print('\nlatest risk metrics (%):')
    print(latest.to_string(float_format=lambda x: f'{x:,.3f}'))

    print('\nbacktest summary:')
    print(bt.to_string(float_format=lambda x: f'{x:,.4f}'))

    print('\nworst single days (%):')
    print((100 * worst).to_string(float_format=lambda x: f'{x:,.3f}'))

    print('\nscenario vs single-day history:')
    print(scenario_vs_history(r, SCENARIOS).to_string(
        float_format=lambda x: 'inf' if x == float('inf') else f'{x:,.3f}'))

    m5 = multi_day_losses(r, 5)
    print('\nscenario vs 5-day-window history (Japan scenarios are multi-day):')
    print(scenario_vs_history(m5, SCENARIOS).to_string(
        float_format=lambda x: 'inf' if x == float('inf') else f'{x:,.3f}'))


if __name__ == '__main__':
    main()