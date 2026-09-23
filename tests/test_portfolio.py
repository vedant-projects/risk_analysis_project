"""Offline tests for the portfolio book and Japan-liquidation scenarios."""
import numpy as np
import pandas as pd
import pytest

from src.models.bond_math import curve_from_row
from src.models.portfolio import (SCENARIOS, NAV, apply_scenario, book_metrics,
                                  default_book, rates_shock, scenario_table)

ROW = pd.Series({"DGS1MO": 0.0402, "DGS3MO": 0.0408, "DGS6MO": 0.0410,
                 "DGS1": 0.0412, "DGS2": 0.0418, "DGS3": 0.0425,
                 "DGS5": 0.0435, "DGS7": 0.0455, "DGS10": 0.0480,
                 "DGS20": 0.0530, "DGS30": 0.0510})


@pytest.fixture(scope="module")
def curve():
    return curve_from_row(ROW)


def test_book_weights_sum_to_one():
    assert abs(sum(p.weight for p in default_book()) - 1.0) < 1e-12


def test_zero_shock_gives_exactly_zero_pnl(curve):
    tenors, par = curve
    res = apply_scenario(default_book(), tenors, par,
                         {"dy_10y_bp": 0, "dy_30y_bp": 0, "ig_bp": 0, "haircut": 0.0})
    assert res["total_pnl_usd"] == 0.0


def test_losses_deepen_across_scenarios(curve):
    tbl = scenario_table(default_book(), *curve)
    assert (tbl.loc["japan_baseline", "total_$M"]
            > tbl.loc["japan_severe", "total_$M"]
            > tbl.loc["japan_extreme", "total_$M"] < 0)


def test_convexity_softens_loss_vs_duration_only(curve):
    ext = apply_scenario(default_book(), *curve,
                         {**SCENARIOS["japan_extreme"], "name": "ext"})
    assert ext["taylor_dur_only_usd"] < ext["rates_pnl_usd"] < 0


def test_spread_pnl_isolated_to_credit_sleeves(curve):
    res = apply_scenario(default_book(), *curve,
                         {"dy_10y_bp": 0, "dy_30y_bp": 0, "ig_bp": 20, "haircut": 0.0})
    bp = res["by_position"]
    assert bp.loc[["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y", "CASH"],
                  "spread_pnl_usd"].abs().sum() == 0.0
    assert (bp.loc["IG_7Y", "spread_pnl_usd"]
            < bp.loc["HY_4Y", "spread_pnl_usd"]
            < bp.loc["MBS_5Y5", "spread_pnl_usd"] < 0)


def test_cash_never_loses(curve):
    res = apply_scenario(default_book(), *curve, SCENARIOS["japan_severe"])
    assert res["by_position"].loc["CASH", "pnl_usd"] == 0.0


def test_dollar_krds_sum_to_dv01(curve):
    m = book_metrics(default_book(), *curve)
    krd_cols = [c for c in m.columns if c.startswith("krd_")]
    assert abs(m.loc["TOTAL", krd_cols].sum() - m.loc["TOTAL", "dv01_usd"]) \
        / m.loc["TOTAL", "dv01_usd"] < 0.05


def test_book_dv01_at_institutional_scale(curve):
    m = book_metrics(default_book(), *curve)
    assert 4e6 < m.loc["TOTAL", "dv01_usd"] < 8e6   # $4-8M/bp on $10bn


def test_rates_shock_shape_anchors_and_monotone():
    t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    sh = rates_shock(t, 50, 65)
    assert sh[np.where(t == 10)[0][0]] == 50
    assert sh[np.where(t == 30)[0][0]] == 65
    assert np.all(np.diff(sh) > 0)