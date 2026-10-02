"""Download Treasury yields and build an explicitly approximate book proxy."""
from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"

START = "2005-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
NAV = 10_000_000_000.0

SERIES = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
    "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]

# Approximate your existing sleeve weights.
# Credit/MBS sleeves use Treasury-rate proxies only: no credit/OAS risk.
POSITIONS = [
    ("UST_2Y", 0.06, "DGS2", 1.9),
    ("UST_5Y", 0.10, "DGS5", 4.4),
    ("UST_10Y", 0.14, "DGS10", 7.8),
    ("UST_30Y", 0.10, "DGS30", 16.0),
    ("IG_PROXY", 0.30, "DGS7", 6.0),
    ("HY_PROXY", 0.10, "DGS3", 3.2),
    ("MBS_PROXY", 0.10, "DGS5", 4.0),
]


def download(series):
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series, "cosd": START, "coed": END},
        timeout=90,
    )
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if series not in df.columns:
        raise ValueError(f"Unexpected FRED response for {series}")

    date = pd.to_datetime(df.iloc[:, 0])
    values = pd.to_numeric(df[series], errors="coerce")
    result = pd.Series(values.to_numpy(), index=date, name=series)
    result = result.sort_index()
    result.to_csv(RAW / f"{series}.csv")
    return result


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    parts = []
    for series in SERIES:
        print(f"Downloading {series}...")
        parts.append(download(series))

    panel = pd.concat(parts, axis=1).sort_index()
    panel = panel.loc[~panel.index.duplicated()]

    # Remove weekends, forward-fill short gaps only; never backfill.
    panel = panel.reindex(pd.bdate_range(panel.index.min(), panel.index.max()))
    panel = panel.ffill(limit=5).dropna()

    # FRED yields are percentage points; pricing modules use decimals.
    yields = panel / 100.0
    yields.index.name = "date"
    yields.to_parquet(PROCESSED / "yields_dec.parquet")

    previous = yields.shift(1)
    changes = yields.diff()

    book_return = 0.10 * previous["DGS3MO"] / 252

    for name, weight, tenor, duration in POSITIONS:
        sleeve_return = previous[tenor] / 252 - duration * changes[tenor]
        book_return = book_return + weight * sleeve_return

    history = pd.DataFrame({
        "book_return": book_return,
        "cash_return": previous["DGS3MO"] / 252,
        "yield_10y": yields["DGS10"],
        "slope_10y_2y": yields["DGS10"] - yields["DGS2"],
        "dy_10y": changes["DGS10"],
        "total_pnl_usd": NAV * book_return,
    }).dropna()

    history.index.name = "date"
    history.to_parquet(PROCESSED / "book_history.parquet")
    history.to_csv(PROCESSED / "learning_history.csv")

    metadata = {
        "source": "FRED Treasury yields",
        "history_type": "APPROXIMATE_RATE_ONLY_BOOK_PROXY",
        "start": str(history.index.min()),
        "end": str(history.index.max()),
        "rows": len(history),
        "limitations": [
            "Fixed-duration first-order valuation approximation",
            "Approximate yield carry, not actual coupon/roll-down accounting",
            "No credit-spread, MBS optionality, FX or liquidity losses",
            "Not exact historical P&L of the original portfolio",
        ],
    }
    (PROCESSED / "learning_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nBuilt {len(history):,} observations.")
    print("WARNING: approximate rate-only book proxy, not exact portfolio P&L.")


if __name__ == "__main__":
    main()