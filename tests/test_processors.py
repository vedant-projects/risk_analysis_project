"""Offline tests for processors. No network, no data/raw dependency."""
import numpy as np
import pandas as pd

from src.data.processors import (adf_check, apply_gap_policy, enforce_series_windows,
                                 gap_report, log_returns, yield_changes_bp)


def test_gap_policy_fills_short_holes_only():
    idx = pd.bdate_range("2020-01-01", periods=30)
    s = pd.Series(1.0, index=idx)
    s.iloc[5:7] = np.nan    # 2-day hole -> must fill
    s.iloc[15:20] = np.nan  # 5-day hole -> 3 days fill, rest survive
    out = apply_gap_policy(s.to_frame("x"))
    assert out["x"].iloc[5:7].notna().all()
    assert out["x"].iloc[15:18].notna().all() and out["x"].iloc[18:20].isna().all()
    rep = gap_report(out)
    assert len(rep) == 1 and rep.loc[0, "days"] == 2


def test_leading_nans_are_availability_not_gaps():
    idx = pd.bdate_range("2000-01-03", periods=20)
    s = pd.Series(np.nan, index=idx)
    s.iloc[10:] = 1.0
    assert len(gap_report(s.to_frame("DGS1MO"))) == 0


def test_series_windows_dgs30_blackout_and_short_tenors():
    idx = pd.bdate_range("2000-06-01", "2000-06-30")
    df = pd.DataFrame({"DGS1MO": 1.0, "DGS6MO": 1.0, "DGS10": 1.0, "DGS30": 1.0},
                      index=idx)
    out = enforce_series_windows(df)
    assert out["DGS1MO"].isna().all() and out["DGS6MO"].isna().all()  # start 2001-07
    assert out["DGS10"].notna().all()                                   # exists since 1962
    idx04 = pd.bdate_range("2004-06-01", "2004-06-30")
    df04 = pd.DataFrame({"DGS30": 1.0}, index=idx04)
    assert enforce_series_windows(df04)["DGS30"].isna().all()           # blackout


def test_yield_changes_bp_units():
    yi = pd.DataFrame({"DGS10": [0.05, 0.0501, 0.05, 0.0525]},
                      index=pd.bdate_range("2020-01-01", periods=4))
    chg = yield_changes_bp(yi)
    assert abs(chg["DGS10"].iloc[0] - 1.0) < 1e-9    # +1bp
    assert abs(chg["DGS10"].iloc[2] - 25.0) < 1e-9   # +25bp


def test_log_returns_guard_on_bad_prices():
    pr = pd.DataFrame({"X": [100.0, 0.0, 101.0, -5.0]},
                      index=pd.bdate_range("2020-01-01", periods=4))
    r = log_returns(pr)
    assert len(r) == 3 and r["X"].isna().all()  # poisoned diffs -> NaN, no crash


def test_adf_check_on_short_series_returns_none():
    s = pd.Series(np.arange(10, dtype=float))
    res = adf_check(s)
    assert res["stationary"] is None