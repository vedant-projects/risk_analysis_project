"""var_bridge.py - rolling historical VaR / ES and simple exceedance backtests.

Consumes data/processed/book_history.parquet from scripts/build_history.py and
produces rolling 95%/99% historical VaR + ES series, plus a compact summary
for report tables. This module is intentionally self-contained so it does not
block on any existing VaREngine API mismatch.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_book_history(path='data/processed/book_history.parquet') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p.resolve()}. Run python scripts/build_history.py first.")
    return pd.read_parquet(p)


def load_book_returns(path='data/processed/book_history.parquet', col='book_return') -> pd.Series:
    df = load_book_history(path)
    if col not in df.columns:
        raise KeyError(f"{col!r} not in {list(df.columns)}")
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    s.name = col
    return s


def _hist_var(x: pd.Series, alpha: float) -> float:
    return float(-np.quantile(np.asarray(x, float), 1 - alpha))


def _hist_es(x: pd.Series, alpha: float) -> float:
    arr = np.asarray(x, float)
    q = np.quantile(arr, 1 - alpha)
    tail = arr[arr <= q]
    return float(-tail.mean())


def rolling_hist_var_es(returns: pd.Series,
                        window: int = 252,
                        levels=(0.95, 0.99)) -> pd.DataFrame:
    r = pd.to_numeric(returns, errors='coerce').dropna().copy()
    out = pd.DataFrame(index=r.index)
    for a in levels:
        tag = int(round(a * 100))
        out[f'var_{tag}'] = r.rolling(window).apply(lambda x: _hist_var(pd.Series(x), a), raw=False)
        out[f'es_{tag}'] = r.rolling(window).apply(lambda x: _hist_es(pd.Series(x), a), raw=False)
    out['return'] = r
    return out.dropna()


def backtest_exceedances(returns: pd.Series, var_series: pd.Series, alpha: float) -> dict:
    aligned = pd.concat([pd.to_numeric(returns, errors='coerce'),
                         pd.to_numeric(var_series, errors='coerce')], axis=1).dropna()
    aligned.columns = ['return', 'var']
    breaches = aligned['return'] < -aligned['var']
    n = int(len(aligned))
    x = int(breaches.sum())
    p = 1 - alpha
    exp = n * p
    return {
        'alpha': alpha,
        'n_obs': n,
        'expected_breaches': exp,
        'actual_breaches': x,
        'breach_rate': x / n if n else np.nan,
        'expected_rate': p,
    }

def multi_day_losses(returns: pd.Series, h: int = 5) -> pd.Series:
    """Rolling h-day cumulative book returns (worst windows for tail analysis)."""
    r = pd.to_numeric(returns, errors='coerce').dropna()
    return r.rolling(h).sum().dropna()
    
def worst_days(returns: pd.Series, n: int = 10) -> pd.Series:
    r = pd.to_numeric(returns, errors='coerce').dropna()
    return r.nsmallest(n)


def scenario_vs_history(returns: pd.Series,
                        scenarios: dict[str, float]) -> pd.DataFrame:
    r = pd.to_numeric(returns, errors='coerce').dropna()
    rows = []
    for name, loss in scenarios.items():
        frac_worse = float((r <= -abs(loss)).mean())
        rows.append({
            'scenario': name,
            'loss_pct': 100 * abs(loss),
            'days_worse_or_equal': int((r <= -abs(loss)).sum()),
            'share_of_history_pct': 100 * frac_worse,
            'approx_return_period_days': np.inf if frac_worse == 0 else 1 / frac_worse,
            'approx_return_period_years': np.inf if frac_worse == 0 else 1 / (252 * frac_worse),
        })
    return pd.DataFrame(rows).set_index('scenario')


def report_table(returns: pd.Series, window: int = 252) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roll = rolling_hist_var_es(returns, window=window, levels=(0.95, 0.99))
    latest = pd.DataFrame({
        'metric': ['Latest 95% VaR', 'Latest 95% ES', 'Latest 99% VaR', 'Latest 99% ES',
                   'Worst day', 'Std dev'],
        'value_pct': [100 * roll['var_95'].iloc[-1], 100 * roll['es_95'].iloc[-1],
                      100 * roll['var_99'].iloc[-1], 100 * roll['es_99'].iloc[-1],
                      100 * returns.min(), 100 * returns.std()]
    }).set_index('metric')
    bt = pd.DataFrame([
        backtest_exceedances(roll['return'], roll['var_95'], 0.95),
        backtest_exceedances(roll['return'], roll['var_99'], 0.99),
    ]).set_index('alpha')
    return latest, bt, worst_days(returns, 10).to_frame('book_return')