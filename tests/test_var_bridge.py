import numpy as np
import pandas as pd

from src.risk.var_bridge import (backtest_exceedances, report_table,
                                 rolling_hist_var_es, scenario_vs_history,
                                 worst_days, multi_day_losses)


def test_rolling_hist_var_es_shapes_and_signs():
    idx = pd.bdate_range('2020-01-01', periods=400)
    r = pd.Series(np.r_[np.full(200, -0.001), np.full(200, 0.001)], index=idx)
    out = rolling_hist_var_es(r, window=252)
    assert {'var_95', 'es_95', 'var_99', 'es_99', 'return'} <= set(out.columns)
    assert (out[['var_95','es_95','var_99','es_99']] >= 0).all().all()


def test_backtest_counts_breaches_correctly():
    idx = pd.bdate_range('2020-01-01', periods=5)
    returns = pd.Series([-0.03, -0.01, 0.00, -0.05, 0.01], index=idx)
    var = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=idx)
    bt = backtest_exceedances(returns, var, 0.95)
    assert bt['actual_breaches'] == 2
    assert bt['n_obs'] == 5


def test_worst_days_sorts_ascending_losses():
    idx = pd.bdate_range('2020-01-01', periods=6)
    r = pd.Series([0.01, -0.02, -0.05, 0.00, -0.03, 0.02], index=idx)
    w = worst_days(r, 3)
    assert list(np.round(w.values, 3)) == [-0.05, -0.03, -0.02]


def test_scenario_vs_history_handles_zero_hits():
    idx = pd.bdate_range('2020-01-01', periods=100)
    r = pd.Series(np.linspace(-0.02, 0.01, 100), index=idx)
    out = scenario_vs_history(r, {'baseline': 0.03})
    assert out.loc['baseline', 'days_worse_or_equal'] == 0
    assert np.isinf(out.loc['baseline', 'approx_return_period_days'])


def test_report_table_returns_three_tables():
    idx = pd.bdate_range('2020-01-01', periods=400)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    latest, bt, worst = report_table(r, window=252)
    assert latest.shape[0] == 6
    assert set(bt.index) == {0.95, 0.99}
    assert len(worst) == 10

def test_multi_day_losses_rollup():
    idx = pd.bdate_range('2020-01-01', periods=7)
    r = pd.Series([0.01, -0.02, -0.03, 0.00, -0.01, 0.02, 0.005], index=idx)
    m5 = multi_day_losses(r, 5)
    assert len(m5) == 3
    assert abs(m5.iloc[0] - (-0.05)) < 1e-9