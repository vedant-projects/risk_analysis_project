"""Offline tests for the fetchers' parsing layer. No network."""
import pandas as pd

from src.data.fetchers import _read_fred_csv_text

FRED_CSV = "observation_date,DGS10\n2026-09-14,4.79\n2026-09-15,.\n2026-09-16,4.81\n"


def test_read_fred_csv_parses_dates_and_missing_sentinels():
    s = _read_fred_csv_text(FRED_CSV, "DGS10")
    assert s.index[0] == pd.Timestamp("2026-09-14")
    assert s.iloc[0] == 4.79
    assert pd.isna(s.iloc[1])      # '.' must become NaN, never 0
    assert s.iloc[2] == 4.81
    assert s.name == "DGS10"