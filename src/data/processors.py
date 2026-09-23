"""processors.py - raw CSV cache -> cleaned analysis panels.

Pipeline position: fetchers.py writes data/raw/*.csv; this module turns
them into aligned, break-aware panels in data/processed/.

Rules (docs/methodology.md):
  R1  History is context, windows are estimates - store 1995->today.
  R2  Gaps <= 3 business days: forward-fill. Longer: leave NaN + report.
  R3  Series availability is a data fact: DGS1MO/6MO start 2001-07-31;
      DGS30 is blacked out 2002-02-19..2006-02-08 (issuance discontinued).
  R4  Risk math runs on yield CHANGES in bp, never levels (levels are I(1));
      the ADF check enforces this on real data.
  R5  Monthly series (FEDFUNDS, CPIAUCSL, UNRATE) never get interpolated
      onto the daily grid - they keep their own monthly panel.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RAW_DIR = pathlib.Path("data/raw")
PROC_DIR = pathlib.Path("data/processed")

UNIT_SCALE = 0.01
MAX_FFILL_DAYS = 3
HISTORY_START = "1995-01-02"

CURVE = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
         "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
OTHER_DAILY = ["BAMLC0A0CM", "BAMLC0A4CBBB", "BAMLH0A0HYM2", "BAA10Y",
               "T10Y2Y", "T10Y3M", "VIXCLS", "DEXJPUS"]
MONTHLY = ["FEDFUNDS", "CPIAUCSL", "UNRATE"]

SERIES_WINDOWS = {
    "DGS1MO": {"start": "2001-07-31"},
    "DGS6MO": {"start": "2001-07-31"},
    "DGS20":  {"start": "1993-10-01"},
    "DGS30":  {"start": "1977-02-15",
               "blackouts": [("2002-02-19", "2006-02-08")]},
}


def load_raw(data_dir=RAW_DIR):
    """Read every cached CSV (fred_<id>.csv / yahoo_<ticker>.csv) -> Series."""
    out = {}
    for p in sorted(pathlib.Path(data_dir).glob("*.csv")):
        stem = p.stem
        if "_" not in stem or stem.startswith("tic"):
            continue
        name = stem.split("_", 1)[1].lstrip("_")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty:
            continue
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        if not s.empty:
            s = s[~s.index.duplicated(keep="last")]
            out[name] = s.rename(name).sort_index()
    return out


def enforce_series_windows(df: pd.DataFrame, windows=SERIES_WINDOWS) -> pd.DataFrame:
    """NaN everything outside a series' documented availability window."""
    out = df.copy()
    for col, w in windows.items():
        if col not in out.columns:
            continue
        if "start" in w:
            out.loc[out.index < pd.Timestamp(w["start"]), col] = np.nan
        for lo, hi in w.get("blackouts", []):
            m = (out.index >= pd.Timestamp(lo)) & (out.index <= pd.Timestamp(hi))
            out.loc[m, col] = np.nan
    return out


def apply_gap_policy(df: pd.DataFrame, max_ffill_days=MAX_FFILL_DAYS) -> pd.DataFrame:
    """Forward-fill holes up to max_ffill_days; longer holes survive as NaN."""
    return df.ffill(limit=max_ffill_days)


def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    """List every surviving NaN run (per series, after first valid date)."""
    records = []
    for col in df.columns:
        s = df[col]
        first = s.first_valid_index()
        if first is None:
            continue
        tail = s.loc[first:]
        na = tail.isna().to_numpy()
        idx = tail.index
        start = None
        for i, flag in enumerate(na):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                records.append({"series": col, "gap_start": str(idx[start].date()),
                                "gap_end": str(idx[i - 1].date()), "days": i - start})
                start = None
        if start is not None:
            records.append({"series": col, "gap_start": str(idx[start].date()),
                            "gap_end": str(idx[-1].date()), "days": len(na) - start})
    return pd.DataFrame(records, columns=["series", "gap_start", "gap_end", "days"])


def build_daily_panel(raw: dict, names: list, start=HISTORY_START) -> pd.DataFrame:
    """Outer-join series onto one business-day index, apply R2 + R3."""
    idx = pd.bdate_range(start, pd.Timestamp.today().normalize())
    panel = pd.DataFrame({n: raw[n].reindex(idx) for n in names if n in raw},
                         index=idx)
    return apply_gap_policy(enforce_series_windows(panel))


def yield_changes_bp(yields_dec: pd.DataFrame) -> pd.DataFrame:
    """Decimal yield levels -> daily changes in basis points."""
    return (yields_dec.diff() * 1e4).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Guard against zero/negative prints before the log."""
    return np.log(prices.where(prices > 0)).diff().iloc[1:]


def build_monthly_panel(raw: dict, names: list) -> pd.DataFrame:
    """Monthly series keep their own calendar - never interpolated (R5)."""
    have = [n for n in names if n in raw]
    if not have:
        return pd.DataFrame()
    return pd.concat([raw[n] for n in have], axis=1).sort_index()


def adf_check(series: pd.Series) -> dict:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"p_value": float("nan"), "stationary": None}
    s = series.dropna()
    if len(s) < 100:
        return {"p_value": float("nan"), "stationary": None}
    stat, p, *_ = adfuller(s, regression="c", autolag="AIC")
    return {"adf_stat": round(float(stat), 3), "p_value": round(float(p), 4),
            "stationary": bool(p < 0.05), "n_obs": len(s)}


def process_all(raw_dir=RAW_DIR, processed_dir=PROC_DIR, start=HISTORY_START) -> dict:
    """Full processing pipeline. Writes parquet artifacts, returns them too."""
    raw = load_raw(raw_dir)
    if not raw:
        raise FileNotFoundError(
            f"No cached CSVs in {raw_dir}. Run the fetchers first:\n"
            "  python -c 'from src.data.fetchers import fetch_all; fetch_all()'")

    yields_dec = build_daily_panel(raw, CURVE, start) * UNIT_SCALE
    ychg = yield_changes_bp(yields_dec)
    spreads = build_daily_panel(raw, OTHER_DAILY, start)
    skip = set(CURVE) | set(OTHER_DAILY) | set(MONTHLY)
    prices = build_daily_panel(raw, [n for n in raw if n not in skip], start)
    rets = log_returns(prices)
    macro = build_monthly_panel(raw, MONTHLY)

    gaps = pd.concat([gap_report(x) for x in (yields_dec, spreads, prices)
                      if not x.empty], ignore_index=True)

    out = pathlib.Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {"yields_dec": yields_dec, "yield_changes_bp": ychg,
                 "spreads_daily": spreads, "prices_daily": prices,
                 "log_returns": rets, "macro_monthly": macro,
                 "gap_report": gaps}
    for name, df in artifacts.items():
        if not df.empty:
            df.to_parquet(out / f"{name}.parquet")

    print(f"panel span: {yields_dec.index[0].date()} -> {yields_dec.index[-1].date()}")
    print(f"curves {yields_dec.shape} | spreads {spreads.shape} | "
          f"prices {prices.shape} | monthly {macro.shape}")
    if "DGS10" in yields_dec:
        lv, ch = adf_check(yields_dec["DGS10"]), adf_check(ychg["DGS10"])
        print(f"ADF DGS10 levels : p={lv['p_value']} stationary={lv['stationary']}")
        print(f"ADF DGS10 changes: p={ch['p_value']} stationary={ch['stationary']}")
    if len(gaps):
        print(f"{len(gaps)} surviving gaps > {MAX_FFILL_DAYS}d -> gap_report.parquet")
    return artifacts


if __name__ == "__main__":
    process_all()