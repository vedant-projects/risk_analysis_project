import numpy as np
import pandas as pd

from src.models.history_bridge import build_book_history, var_series_input
from src.models.portfolio import default_book, NAV


def _curve_panel(dates, base=0.05, bump_bp=0.0, missing_30y=False):
    cols = ['DGS1MO','DGS3MO','DGS6MO','DGS1','DGS2','DGS3','DGS5','DGS7','DGS10','DGS20','DGS30']
    ten = np.array([1/12,0.25,0.5,1,2,3,5,7,10,20,30])
    rows = []
    for i, _ in enumerate(dates):
        row = base + (i * bump_bp / 1e4) * np.sqrt(ten / 10)
        rows.append(row)
    df = pd.DataFrame(rows, index=dates, columns=cols)
    if missing_30y:
        df.loc[dates[1], 'DGS30'] = np.nan
    return df


def _spreads_panel(dates, start=1.00, daily_change=0.0, first_nan=False):
    s = pd.Series([start + i * daily_change for i in range(len(dates))], index=dates, name='BAMLC0A0CM')
    if first_nan:
        s.iloc[0] = np.nan
    return s.to_frame()


def test_zero_curve_and_zero_spread_change_give_zero_pnl():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates), _spreads_panel(dates), require_spreads=True)
    assert len(hist) == 4
    assert hist['total_pnl_usd'].abs().sum() < 1e-9


def test_parallel_up_move_produces_negative_rates_pnl():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates, bump_bp=10), _spreads_panel(dates), require_spreads=True)
    assert (hist['rates_pnl_usd'] < 0).all()
    avg_daily = -hist['rates_pnl_usd'].mean()
    assert 4e7 < avg_daily < 8e7


def test_spread_only_move_hits_credit_not_treasuries():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates), _spreads_panel(dates, daily_change=0.05), require_spreads=True)
    assert hist['rates_pnl_usd'].abs().sum() < 1e-9
    assert (hist['spread_pnl_usd'] < 0).all()


def test_missing_30y_tenor_does_not_break_history():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates, bump_bp=5, missing_30y=True), _spreads_panel(dates), require_spreads=True)
    assert len(hist) >= 3
    assert hist['n_positions'].min() == 7


def test_require_spreads_drops_pre_oas_rows():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates), _spreads_panel(dates, first_nan=True), require_spreads=True)
    assert hist.index.min() == dates[2]


def test_var_series_input_returns_clean_series():
    dates = pd.bdate_range('2021-01-01', periods=5)
    hist = build_book_history(default_book(), _curve_panel(dates, bump_bp=3), _spreads_panel(dates), require_spreads=True)
    s = var_series_input(hist)
    assert isinstance(s, pd.Series)
    assert s.name == 'book_return'
    assert len(s) == len(hist)