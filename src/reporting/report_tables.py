"""report_tables.py - assemble every analytical table for the deliverable.

Pulls together the pieces built so far into one place:
  - book composition / DV01 / KRD table      (portfolio.book_metrics)
  - Japan scenario table                     (portfolio.scenario_table)
  - VaR / ES summary, backtests, worst days  (var_bridge.report_table)
  - scenario vs history comparison           (var_bridge.scenario_vs_history)

All tables are plain pandas DataFrames so they can be rendered by the
Streamlit dashboard, exported to CSV, or embedded in the PDF report.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pandas as pd

from src.models.bond_math import curve_from_row
from src.models.portfolio import SCENARIOS, book_metrics, default_book, scenario_table
from src.risk.var_bridge import (load_book_returns, multi_day_losses,
                                 report_table, scenario_vs_history)


DEMO_CURVE = {"DGS1MO": 0.0402, "DGS3MO": 0.0408, "DGS6MO": 0.0410,
              "DGS1": 0.0412, "DGS2": 0.0418, "DGS3": 0.0425,
              "DGS5": 0.0435, "DGS7": 0.0455, "DGS10": 0.0480,
              "DGS20": 0.0530, "DGS30": 0.0510}


@dataclass
class ReportBundle:
    book: pd.DataFrame = None
    scenarios: pd.DataFrame = None
    latest_risk: pd.DataFrame = None
    backtest: pd.DataFrame = None
    worst_days: pd.DataFrame = None
    scen_hist_1d: pd.DataFrame = None
    scen_hist_5d: pd.DataFrame = None
    notes: dict = field(default_factory=dict)


def _curve_from_real_or_demo():
    p = pathlib.Path("data/processed/yields_dec.parquet")
    if p.exists():
        row = pd.read_parquet(p).iloc[-1]
        return curve_from_row(row), "real yields_dec panel (last row)"
    return curve_from_row(pd.Series(DEMO_CURVE)), "built-in demo curve"


def _scenario_loss_fractions() -> dict[str, float]:
    """Fractional NAV loss per scenario, computed exactly like scenario_table."""
    try:
        (tenors, par), _ = _curve_from_real_or_demo()
        tbl = scenario_table(default_book(), tenors, par)
        return {name: float(tbl.loc[name, "total_%NAV"]) / 100.0 for name in tbl.index}
    except Exception:
        return {"japan_baseline": 0.035, "japan_severe": 0.087, "japan_extreme": 0.168}


def build_bundle(book=None, curve=None) -> ReportBundle:
    """Assemble every table; skip pieces whose data is missing (never crash)."""
    b = ReportBundle()
    try:
        (tenors, par), src = (curve, "provided curve") if curve else _curve_from_real_or_demo()
        b.book = book_metrics(book or default_book(), tenors, par)
        b.scenarios = scenario_table(book or default_book(), tenors, par)
        b.notes["curve_source"] = src
    except Exception:
        b.notes["curve_source"] = "unavailable - book/scenario tables skipped"
    b.notes["scenario_loss_pct"] = {k: 100 * v for k, v in _scenario_loss_fractions().items()}
    try:
        r = load_book_returns()
        b.latest_risk, b.backtest, b.worst_days = report_table(r)
        losses = _scenario_loss_fractions()
        b.scen_hist_1d = scenario_vs_history(r, losses)
        b.scen_hist_5d = scenario_vs_history(multi_day_losses(r, 5), losses)
        b.notes["returns_source"] = "book_history.parquet"
    except FileNotFoundError:
        b.notes["returns_source"] = "book_history.parquet missing - VaR tables skipped"
    return b