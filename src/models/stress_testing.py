"""stress_testing.py - Historical scenario replay and stress tests.

This module provides utilities to replay historical scenarios and generate stressed
loss paths for backtests. It assumes returns are provided as a 2-D array or
DataFrame with time on the index and assets on the columns, and a portfolio
weight vector.

Usage:
- from stress_testing import reverse_stress, anchor_scenarios, run_scenario
- All paths return a 1-D array of portfolio losses (positive numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .var_es import VaREngine, portfolio_returns


def reverse_stress(losses: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Invert sign of losses to create a mirrored stress path.

    This is a simple utility used for sensitivity checks in backtesting.
    """
    arr = np.asarray(losses, dtype=float)
    return -arr * factor


def anchor_scenarios(base_losses: np.ndarray, anchors: dict[str, float]) -> dict[str, np.ndarray]:
    """Create anchored stress paths by scaling base losses with anchors.

    anchors: mapping from scenario name to multiplier.
    """
    base = np.asarray(base_losses, dtype=float)
    out = {}
    for name, mult in anchors.items():
        out[name] = base * float(mult)
    return out


def run_scenario(returns: pd.DataFrame, weights, scenario: np.ndarray) -> float:
    """Run a scenario and return the portfolio loss (VaR input).

    - returns: (T, N) DataFrame of per-asset returns
    - weights: (N,) portfolio weights
    - scenario: (T,) a stress-adjusted return path aligned with returns index
    Returns the portfolio loss as a positive number.
    """
    rp = portfolio_returns(returns, weights)
    # Align scenario with rp index if possible
    if len(scenario) != len(rp):
        raise ValueError("Scenario length must match number of observations in returns")
    # compute portfolio path under scenario
    port_scenario = rp.values * scenario
    # loss = deepest single-period drop, expressed as a positive number
    return float(max(0.0, -np.min(port_scenario)))