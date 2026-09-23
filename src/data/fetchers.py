"""fetchers.py - Download and cache market data.

Sources:
- FRED daily series for Treasury yields, spreads, macro, and VIX.
- Yahoo Finance for prices and futures.
- TIC MFH for Japan holdings of Treasuries.

All downloads are cached locally as CSV so the project can run offline after the first pull.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests
import yfinance as yf

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_SERIES = {
    "DGS1MO": "DGS1MO",
    "DGS3MO": "DGS3MO",
    "DGS6MO": "DGS6MO",
    "DGS1": "DGS1",
    "DGS2": "DGS2",
    "DGS3": "DGS3",
    "DGS5": "DGS5",
    "DGS7": "DGS7",
    "DGS10": "DGS10",
    "DGS20": "DGS20",
    "DGS30": "DGS30",
    "BAMLC0A0CM": "BAMLC0A0CM",
    "BAMLC0A4CBBB": "BAMLC0A4CBBB",
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",
    "FEDFUNDS": "FEDFUNDS",
    "CPIAUCSL": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "VIXCLS": "VIXCLS",
    "T10Y2Y": "T10Y2Y",
    "T10Y3M": "T10Y3M",
    "DEXJPUS": "DEXJPUS",
}

YAHOO_TICKERS = {
    "NIFTY": "^NSEI",
    "SPX": "^GSPC",
    "GOLD": "GC=F",
    "BRENT": "BZ=F",
    "USDINR": "INR=X",
    "JPYUSD": "JPY=X",
    "VIX_YF": "^VIX",
    "TY_F": "TY=F",
    "TU_F": "TU=F",
    "FV_F": "FV=F",
    "US_F": "US=F",
    "WN_F": "WN=F",
}

TIC_URL = "https://ticdata.treasury.gov/Publish/mfh.txt"


def _fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _read_fred_csv_text(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    value_col = df.columns[1]
    s = pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s.sort_index()


def fetch_fred_series(series_id: str, start: str = "1995-01-02") -> pd.Series:
    cache_path = RAW_DIR / f"fred_{series_id}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=[0], index_col=0).iloc[:, 0]
        cached.name = series_id
    else:
        cached = None

    if cached is not None and not cached.empty:
        last_date = cached.index.max() + pd.Timedelta(days=1)
        start = max(pd.Timestamp(start), last_date).strftime("%Y-%m-%d")

    resp = requests.get(_fred_url(series_id), params={"cosd": start}, timeout=30)
    resp.raise_for_status()
    new = _read_fred_csv_text(resp.text, series_id)

    if cached is not None and not cached.empty:
        out = pd.concat([cached, new[~new.index.isin(cached.index)]])
    else:
        out = new

    out = out.sort_index()
    out.to_csv(cache_path)
    return out


def fetch_yahoo_ticker(ticker: str, start: str = "2000-01-03") -> pd.Series:
    cache_path = RAW_DIR / f"yahoo_{ticker.replace('=', '_').replace('^', '_')}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=[0], index_col=0).iloc[:, 0]
        cached.name = ticker
    else:
        cached = None

    hist = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No Yahoo data for {ticker}")

    new = hist["Close"].copy()
    new.index = pd.to_datetime(new.index).tz_localize(None)
    new.name = ticker

    if cached is not None and not cached.empty:
        out = pd.concat([cached, new[~new.index.isin(cached.index)]])
    else:
        out = new

    out = out.sort_index()
    out.to_csv(cache_path)
    return out


def fetch_tic_mfh(country: str = "Japan") -> pd.Series:
    cache_path = RAW_DIR / "tic_japan_mfh.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=[0], index_col=0).iloc[:, 0]
        cached.name = country
    else:
        cached = None

    resp = requests.get(TIC_URL, timeout=30)
    resp.raise_for_status()
    text = resp.text

    # Very simple parse: find the Japan row and the date header row.
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Japan"):
            data_line_idx = i
            break
    else:
        raise RuntimeError("Japan row not found in TIC file")

    # A real TIC parser will be more robust; this is a first version.
    # Here we only cache the raw text version for now.
    with open(RAW_DIR / "tic_raw.txt", "w", encoding="utf-8") as f:
        f.write(text)

    raise NotImplementedError("TIC parser will be added in the next step")


def fetch_all(start_fred: str = "1995-01-02", start_yahoo: str = "2000-01-03") -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    for sid in FRED_SERIES.values():
        try:
            out[sid] = fetch_fred_series(sid, start=start_fred)
        except Exception as e:
            print(f"[WARN] FRED failed for {sid}: {e}")

    for name, ticker in YAHOO_TICKERS.items():
        try:
            out[name] = fetch_yahoo_ticker(ticker, start=start_yahoo)
        except Exception as e:
            print(f"[WARN] Yahoo failed for {name}: {e}")

    try:
        out["TIC_JAPAN_MFH"] = fetch_tic_mfh("Japan")
    except Exception as e:
        print(f"[WARN] TIC failed: {e}")

    return out