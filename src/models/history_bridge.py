"""history_bridge.py - turn processed curve/spread history into daily book P&L.

This is the bridge between the market-data layer and the VaR layer:
- consumes processors.py artifacts (yields_dec.parquet, spreads_daily.parquet)
- reprices yesterday's book on today's curve to get daily rates P&L
- adds a spread P&L term for IG/HY/MBS sleeves using the IG OAS history
- emits a daily return series ready for VaREngine / backtests
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bond_math import build_curve, curve_from_row, modified_duration, price, ytm
from .portfolio import NAV, Position, default_book

UNIT_SCALE = 0.01   # FRED percent / OAS percent -> decimal


def load_processed(processed_dir='data/processed'):
    p = Path(processed_dir)
    yields_dec = pd.read_parquet(p / 'yields_dec.parquet')
    spreads_daily = pd.read_parquet(p / 'spreads_daily.parquet')
    return yields_dec, spreads_daily


def _series_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce')
    return pd.Series(np.nan, index=df.index, name=name)


def build_book_history(book: list[Position] | None,
                       yields_dec: pd.DataFrame,
                       spreads_daily: pd.DataFrame,
                       nav: float = NAV,
                       ig_series: str = 'BAMLC0A0CM',
                       require_spreads: bool = True) -> pd.DataFrame:
    """Daily book P&L from exact reprice + spread approximation.

    Rates P&L: for each date t, price positions off curve[t-1] and curve[t].
    Spread P&L: -D_mod(prev) * dOAS * spread_mult * notional.

    If require_spreads=True, rows before the first valid IG spread change are
    dropped, because half the book (IG/HY/MBS) would otherwise have missing
    spread risk. If False, missing spread changes are treated as 0.
    """
    book = default_book() if book is None else book
    idx = yields_dec.index.intersection(spreads_daily.index)
    y = yields_dec.loc[idx].sort_index()
    s = spreads_daily.loc[idx].sort_index()

    ig = _series_or_nan(s, ig_series)
    d_ig = ig.diff() * UNIT_SCALE   # raw percent points -> decimal spread change

    rows = []
    for i in range(1, len(idx)):
        d0, d1 = idx[i - 1], idx[i]
        row0, row1 = y.loc[d0], y.loc[d1]
        try:
            ten0, par0 = curve_from_row(row0)
            ten1, par1 = curve_from_row(row1)
        except ValueError:
            continue

        if require_spreads and pd.isna(d_ig.loc[d1]):
            continue
        ds = 0.0 if pd.isna(d_ig.loc[d1]) else float(d_ig.loc[d1])

        rates_pnl = 0.0
        spread_pnl = 0.0
        n_valid = 0
        for pos in book:
            if pos.spec is None:
                continue
            px0 = price(pos.spec, build_curve(ten0, par0)[0])
            px1 = price(pos.spec, build_curve(ten1, par1)[0])
            w_nav = pos.weight * nav
            rates_pnl += w_nav * (px1 / px0 - 1.0)
            y0 = ytm(pos.spec, px0)
            md0 = modified_duration(pos.spec, y0)
            spread_pnl += -w_nav * md0 * ds * pos.spread_mult
            n_valid += 1

        total = rates_pnl + spread_pnl
        rows.append({'date': d1,
                     'rates_pnl_usd': rates_pnl,
                     'spread_pnl_usd': spread_pnl,
                     'total_pnl_usd': total,
                     'book_return': total / nav,
                     'ig_oas_change_dec': ds,
                     'n_positions': n_valid})

    out = pd.DataFrame(rows).set_index('date').sort_index()
    out.attrs['nav'] = nav
    out.attrs['ig_series'] = ig_series
    out.attrs['require_spreads'] = require_spreads
    return out


def var_series_input(history: pd.DataFrame) -> pd.Series:
    """Return the daily book return series for VaREngine."""
    s = pd.to_numeric(history['book_return'], errors='coerce').dropna()
    s.name = 'book_return'
    return s