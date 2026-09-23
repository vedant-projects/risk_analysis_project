"""var_es.py - VaR / Expected Shortfall suite with statistically-guarded backtests.

Conventions
-----------
* VaR and ES are returned as POSITIVE loss numbers (fraction of portfolio value).
* Horizon scaling is sqrt(h) - an approximation documented in methodology.md.
* Backtesting floors exist because Kupiec is a low-power test at small samples:
  at 99% confidence with 252 observations, P(N=0) = 8% even for a correct
  model, i.e. the test cannot distinguish good from broken. Floors below are
  the minimum observation counts for the test to have usable power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BACKTEST_MIN_OBS = {0.95: 250, 0.99: 750, 0.995: 1000}
WINDOW = 250  # rolling estimation window for series-producing methods


def portfolio_returns(returns, weights) -> pd.Series:
    """(T, N) returns x (N,) weights -> (T,) portfolio return Series."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    R = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns, float)
    if R.shape[1] != len(w):
        raise ValueError(f"returns has {R.shape[1]} columns, weights has {len(w)}")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum():.6f}")
    rp = R @ w
    idx = returns.index if isinstance(returns, pd.DataFrame) else None
    return pd.Series(rp, index=idx, name="portfolio")


def _var_es_historical(rp: np.ndarray, confidence: float) -> tuple[float, float]:
    q = np.quantile(rp, 1 - confidence)
    tail = rp[rp <= q]
    es = -tail.mean() if tail.size else -q
    return float(-q), float(es)


def kupiec_pof(n_exceptions: int, n_obs: int, confidence: float) -> tuple[float, float]:
    """Kupiec unconditional-coverage LR test. LR=0 iff N equals its expectation."""
    T, N, p = n_obs, n_exceptions, 1 - confidence
    ph = N / T
    if N in (0, T):
        lr = -2 * (T - N) * np.log(1 - p) - 2 * N * np.log(p)
        if N == T:
            lr = -2 * T * np.log(p)
    else:
        lr = (-2 * ((T - N) * np.log(1 - p) + N * np.log(p))
              + 2 * ((T - N) * np.log(1 - ph) + N * np.log(ph)))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_independence(exceptions: np.ndarray) -> tuple[float | None, float | None]:
    """Independence test on the exception indicator series.
    Returns (None, None) when untestable (no exceptions, all exceptions,
    or a transition count of zero)."""
    e = exceptions.astype(int)
    if e.sum() in (0, len(e)):
        return None, None
    n00 = int(((e[:-1] == 0) & (e[1:] == 0)).sum())
    n01 = int(((e[:-1] == 0) & (e[1:] == 1)).sum())
    n10 = int(((e[:-1] == 1) & (e[1:] == 0)).sum())
    n11 = int(((e[:-1] == 1) & (e[1:] == 1)).sum())

    def pi(a, b):
        return min(max(b / (a + b), 1e-10), 1 - 1e-10) if a + b else None

    pi0, pi1 = pi(n00, n01), pi(n10, n11)
    if pi0 is None or pi1 is None or pi0 in (0, 1) or pi1 in (0, 1):
        return None, None
    pig = min(max((n01 + n11) / (n00 + n01 + n10 + n11), 1e-10), 1 - 1e-10)

    def ll(p, a, b):
        return a * np.log(1 - p) + b * np.log(p)

    lr = -2 * (ll(pig, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(max(lr, 0.0)), float(1 - stats.chi2.cdf(max(lr, 0.0), 1))


def traffic_light(n: int) -> str:
    """Basel zones for 99% VaR over 250 days: <=4 green, 5-9 yellow, 10+ red."""
    return "green" if n <= 4 else ("yellow" if n <= 9 else "red")


class VaREngine:
    """(T, N) returns matrix in, positive-loss scalars/series out."""

    # ---- scalar estimators -------------------------------------------------
    def parametric_var(self, returns, weights, confidence=0.99, horizon=1,
                       dist="normal") -> float:
        rp = portfolio_returns(returns, weights).values
        mu, sd = rp.mean(), rp.std(ddof=1)
        if dist == "normal":
            q = mu + sd * stats.norm.ppf(1 - confidence)
        elif dist == "t":
            nu, loc, scale = stats.t.fit(rp)
            q = loc + scale * stats.t.ppf(1 - confidence, nu)
        else:
            raise ValueError("dist must be 'normal' or 't'")
        return float(-q * np.sqrt(horizon))

    def historical_var(self, returns, weights, confidence=0.99, horizon=1,
                       window: int | None = None) -> float:
        rp = portfolio_returns(returns, weights).values
        if window:
            rp = rp[-window:]
        return _var_es_historical(rp, confidence)[0] * np.sqrt(horizon)

    def expected_shortfall(self, returns, weights, confidence=0.975,
                           method="historical", window: int | None = None) -> float:
        rp = portfolio_returns(returns, weights).values
        if window:
            rp = rp[-window:]
        if method == "historical":
            return _var_es_historical(rp, confidence)[1]
        if method == "normal":
            mu, sd = rp.mean(), rp.std(ddof=1)
            z = stats.norm.ppf(confidence)
            return float(-(mu - sd * stats.norm.pdf(z) / (1 - confidence)) * -1)
        if method == "t":
            nu, loc, scale = stats.t.fit(rp)
            c = stats.t.ppf(1 - confidence, nu)
            es = loc - scale * stats.t.pdf(c, nu) / (1 - confidence) * (nu + c**2) / (nu - 1)
            return float(-es)
        raise ValueError("method must be 'historical', 'normal' or 't'")

    def cornish_fisher_var(self, returns, weights, confidence=0.99,
                           horizon=1) -> float:
        rp = portfolio_returns(returns, weights).values
        mu, sd = rp.mean(), rp.std(ddof=1)
        S = stats.skew(rp)                  # negative skew -> bigger VaR
        K = stats.kurtosis(rp)              # excess kurtosis
        z = stats.norm.ppf(1 - confidence)  # negative far-tail quantile
        q_cf = (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24
                - (2 * z**3 - 5 * z) * S**2 / 36)
        return float(-(mu + q_cf * sd) * np.sqrt(horizon))

    # ---- series (for backtesting) ------------------------------------------
    def rolling_historical_var(self, returns, weights, confidence=0.99,
                               window: int = WINDOW) -> pd.Series:
        rp = portfolio_returns(returns, weights)
        out = rp.rolling(window).quantile(1 - confidence).mul(-1.0).dropna()
        return out.rename(f"VaR_{confidence:.3f}_rolling")

    # ---- backtests ----------------------------------------------------------
    def backtest(self, returns, weights=None, var_series: pd.Series | None = None,
                 confidence: float = 0.99) -> dict:
        if confidence not in BACKTEST_MIN_OBS:
            raise ValueError(f"confidence {confidence} unsupported; floors exist for "
                             f"{sorted(BACKTEST_MIN_OBS)}")
        rp = portfolio_returns(returns, weights) if weights is not None else pd.Series(returns)
        if var_series is None:
            var_series = self.rolling_historical_var(rp.to_frame("p"), np.array([1.0]),
                                                     confidence=confidence)
        aligned = rp.loc[var_series.index]
        n = len(aligned)
        floor = BACKTEST_MIN_OBS[confidence]
        if n < floor:
            raise ValueError(
                f"Backtest underpowered at {confidence:.1%} with {n} obs "
                f"(minimum {floor}). Extend data or lower confidence.")
        exc = (aligned.values < -var_series.values)
        N = int(exc.sum())
        lr_uc, p_uc = kupiec_pof(N, n, confidence)
        lr_ind, p_ind = christoffersen_independence(exc)
        zone = traffic_light(int(exc[-250:].sum())) if confidence == 0.99 else None
        return {"observations": n, "exceptions": N,
                "expected_exceptions": round(n * (1 - confidence), 2),
                "kupiec_LR": round(lr_uc, 4), "kupiec_p": round(p_uc, 4),
                "christoffersen_LR": None if lr_ind is None else round(lr_ind, 4),
                "christoffersen_p": None if p_ind is None else round(p_ind, 4),
                "basel_zone_last250": zone,
                "power_floor": floor, "power_ok": True}