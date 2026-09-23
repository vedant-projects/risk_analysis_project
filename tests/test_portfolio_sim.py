"""Offline tests for portfolio_sim utilities."""
import numpy as np
import pandas as pd
import pytest

from src.models.portfolio_sim import gbm_paths, correlated_paths, portfolio_losses_from_paths

def test_gbm_paths_shape():
    paths = gbm_paths(n_paths=3, n_steps=10, s0=100.0, mu=0.05, sigma=0.2)
    assert paths.shape == (11, 3)

def test_correlated_paths_shape_and_type():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    paths = correlated_paths(n_paths=2, n_steps=5, n_assets=2, corr_matrix=corr, s0=100.0)
    assert paths.shape == (6, 2, 2)

def test_portfolio_losses_from_paths_basic():
    # simple synthetic paths with two assets
    t = np.array([[100.0, 100.0],
                  [110.0, 90.0]])  # 2 steps
    weights = np.array([0.6, 0.4])
    losses = portfolio_losses_from_paths(t, weights)
    assert isinstance(losses, pd.Series)
    assert len(losses) == 2