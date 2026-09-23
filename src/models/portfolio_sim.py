"""portfolio_sim.py - synthetic path generators for tests and demos.

gbm_paths                       exact GBM paths (cumulative log-returns), seeded
correlated_paths                multi-asset GBM with target correlation via Cholesky
portfolio_losses_from_paths     path matrix -> portfolio loss Series

These generators exist for smoke tests and visual demos only - real analysis
always runs on fetched market data (src/data/fetchers.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gbm_paths(n_paths: int, n_steps: int, s0: float,
              mu: float, sigma: float, dt: float = 1.0,
              seed: int | None = None) -> np.ndarray:
    """Exact GBM: S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*W_t).

    Returns shape (n_steps + 1, n_paths).
    """
    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma**2) * dt
    log_inc = drift + sigma * np.sqrt(dt) * rng.normal(size=(n_steps, n_paths))
    path = np.empty((n_steps + 1, n_paths))
    path[0] = s0
    path[1:] = s0 * np.exp(np.cumsum(log_inc, axis=0))
    return path


def correlated_paths(n_paths: int, n_steps: int, n_assets: int,
                     corr_matrix: np.ndarray, s0: float = 1.0,
                     mu: float = 0.0, sigma: float = 0.1,
                     seed: int | None = None) -> np.ndarray:
    """Correlated GBM-like paths: z @ L.T imposes Cov = corr_matrix.

    Returns shape (n_steps + 1, n_paths, n_assets).
    """
    if corr_matrix.shape != (n_assets, n_assets):
        raise ValueError("corr_matrix must be (n_assets, n_assets)")
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix)
    # independent standard normals per asset, then impose correlation:
    # if z ~ N(0, I), then z @ L.T has covariance L @ L.T = corr_matrix
    z = rng.normal(size=(n_steps, n_paths, n_assets))
    shocks = z @ L.T
    paths = np.empty((n_steps + 1, n_paths, n_assets))
    paths[0] = s0
    drift = mu - 0.5 * sigma**2
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp(drift + sigma * shocks[t - 1])
    return paths


def portfolio_losses_from_paths(paths: np.ndarray, weights: np.ndarray) -> pd.Series:
    """Path matrix -> portfolio losses (positive = loss).

    - 3-D paths (n_steps+1, n_paths, n_assets): final-vs-first return per
      path, weighted -> one loss per path.
    - 2-D paths (T, N): row-wise returns vs first row -> loss per row.
    """
    arr = np.asarray(paths)
    if arr.ndim == 3:
        final, start = arr[-1], arr[0]
        ret = (final - start) / start
    else:
        ret = (arr - arr[0]) / arr[0]
    port = ret @ weights
    return -pd.Series(port)