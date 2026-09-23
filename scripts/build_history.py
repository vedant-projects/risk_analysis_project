"""scripts/build_history.py - build daily book P&L history from processed data.

Run from the project root:
    python scripts/build_history.py
or
    python -m scripts.build_history
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.history_bridge import build_book_history, load_processed, var_series_input
from src.models.portfolio import default_book

OUT = ROOT / "data" / "processed" / "book_history.parquet"


def main() -> None:
    yields_dec, spreads_daily = load_processed()
    print(f"yields panel : {len(yields_dec)} rows, {yields_dec.index.min().date()} -> {yields_dec.index.max().date()}")
    print(f"spreads panel: {len(spreads_daily)} rows; columns: {list(spreads_daily.columns)[:8]}")

    hist = build_book_history(default_book(), yields_dec, spreads_daily)
    rets = var_series_input(hist)

    print(f"\nbook history : {len(hist)} rows, {hist.index.min().date()} -> {hist.index.max().date()}")
    print("\nlast 5 days:")
    print(hist[["rates_pnl_usd", "spread_pnl_usd", "total_pnl_usd", "book_return"]].tail())
    print("\ndaily book return stats:")
    print(rets.describe())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(OUT)
    print(f"\nsaved -> {OUT}  (the VaR step reads this file)")


if __name__ == "__main__":
    main()