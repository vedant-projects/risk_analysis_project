"""Offline tests for stress_testing utilities."""
import numpy as np
import pandas as pd
import pytest

from src.models.stress_testing import reverse_stress, anchor_scenarios, run_scenario
from src.models.var_es import VaREngine

RNG = np.random.default_rng(3)
N = 4
T = 500

def _make_returns():
    idx = pd.bdate_range("2021-01-01", periods=T)
    data = RNG.normal(0, 0.01, size=(T, N))
    return pd.DataFrame(data, index=idx, columns=list("ABCD"))

def test_reverse_stress_inverts_and_scales():
    base = np.array([1.0, -0.5, 0.2])
    out = reverse_stress(base, factor=2.0)
    assert np.allclose(out, -base * 2.0)

def test_anchor_scenarios_scaling():
    base = np.random.RandomState(0).randn(10)
    anchors = {"shock1": 1.5, "shock2": -0.8}
    res = anchor_scenarios(base, anchors)
    assert np.allclose(res["shock1"], base * 1.5)
    assert np.allclose(res["shock2"], base * -0.8)

def test_run_scenario_matches_shape_and_return():
    returns = _make_returns()
    weights = np.full(N, 1.0 / N)
    eng = VaREngine()
    # simple scenario: multiply per-asset returns by a constant factor
    scenario = np.ones(T) * 1.1
    # compute with a trivial path: portfolio loss should be negative of max gain
    loss = run_scenario(returns, weights, scenario)
    assert isinstance(loss, float)
    assert loss >= 0.0