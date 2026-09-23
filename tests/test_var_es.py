"""Property tests for the VaR/ES engine. All synthetic - no data files needed."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.models.var_es import (VaREngine, christoffersen_independence,
                               kupiec_pof, portfolio_returns, traffic_light)

RNG = np.random.default_rng(7)
COLS = list("ABCDE")
W = np.full(5, 0.2)
X = pd.DataFrame(RNG.normal(0, 0.01, (2000, 5)), columns=COLS,
                 index=pd.bdate_range("2018-01-01", periods=2000))
ENG = VaREngine()


def test_parametric_var_matches_theory():
    """Equal-weight iid normal: VaR99 = 2.326 * 0.01 / sqrt(5)."""
    theory = stats.norm.ppf(0.99) * 0.01 / np.sqrt(5)
    assert abs(ENG.parametric_var(X, W, 0.99) - theory) / theory < 0.05


def test_es_exceeds_var_at_same_confidence():
    assert ENG.expected_shortfall(X, W, 0.975) > ENG.historical_var(X, W, 0.975)


def test_student_t_fatter_than_normal_on_fat_tails():
    Xt = pd.DataFrame(stats.t.rvs(4, size=(2000, 5), random_state=1), columns=COLS)
    assert ENG.parametric_var(Xt, W, 0.99, dist="t") > ENG.parametric_var(Xt, W, 0.99)


def test_cornish_fisher_widens_on_left_skew():
    Xs = pd.DataFrame(stats.skewnorm.rvs(-6, size=(2000, 5), random_state=2), columns=COLS)
    assert ENG.cornish_fisher_var(Xs, W) > ENG.parametric_var(Xs, W)


def test_horizon_scaling_is_sqrt():
    assert (ENG.parametric_var(X, W, 0.99, horizon=10)
            == pytest.approx(ENG.parametric_var(X, W, 0.99) * np.sqrt(10)))


def test_kupiec_zero_at_exact_expectation():
    lr, p = kupiec_pof(8, 800, 0.99)
    assert abs(lr) < 1e-12 and abs(p - 1.0) < 1e-12


def test_kupiec_monotone_in_exceptions():
    assert kupiec_pof(16, 800, 0.99)[0] > kupiec_pof(12, 800, 0.99)[0] > 0


@pytest.mark.parametrize("n,zone", [(3, "green"), (4, "green"), (5, "yellow"),
                                    (9, "yellow"), (10, "red")])
def test_traffic_light_zones(n, zone):
    assert traffic_light(n) == zone


@pytest.mark.parametrize("conf,n", [(0.99, 252), (0.995, 500)])
def test_power_guard_raises_on_small_samples(conf, n):
    rp = pd.Series(RNG.normal(0, 1, n), index=pd.bdate_range("2020-01-01", periods=n))
    with pytest.raises(ValueError, match="underpowered"):
        ENG.backtest(rp, var_series=rp * 0 + 2.0, confidence=conf)


def test_unsupported_confidence_raises():
    rp = pd.Series(RNG.normal(0, 1, 2000), index=pd.bdate_range("2020-01-01", periods=2000))
    with pytest.raises(ValueError, match="unsupported"):
        ENG.backtest(rp, confidence=0.90)


def test_full_backtest_is_coherent():
    big = pd.Series(RNG.normal(0, 0.01, 1500),
                    index=pd.bdate_range("2020-01-01", periods=1500))
    res = ENG.backtest(big, confidence=0.99)
    assert res["observations"] == 1500 - 250 + 1
    assert res["power_ok"] and res["basel_zone_last250"] in {"green", "yellow", "red"}
    assert res["kupiec_LR"] >= 0


def test_zero_exceptions_edge_case():
    rp = pd.Series(RNG.normal(0, 0.01, 800),
                   index=pd.bdate_range("2020-01-01", periods=800))
    res = ENG.backtest(rp, var_series=rp.abs() + 1.0, confidence=0.99)
    assert res["exceptions"] == 0
    assert res["christoffersen_p"] is None   # untestable, not an error
    assert res["kupiec_LR"] > 0              # 0 exceptions vs ~8 expected


def test_christoffersen_rejects_clustered_exceptions():
    e = np.zeros(500, dtype=int)
    e[:20] = 1                              # block of consecutive exceptions
    lr, p = christoffersen_independence(e)
    assert lr is not None and p < 0.05


@pytest.mark.parametrize("bad", [np.array([0.5, 0.5]), np.array([0.2] * 4)])
def test_weight_validation(bad):
    with pytest.raises(ValueError):
        portfolio_returns(X, bad)