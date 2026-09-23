"""portfolio.py - institutional book, aggregate risk, Japan-liquidation shock engine.

Pipeline position: consumes a par-yield row (from processors' yields_dec
panel via curve_from_row) plus bond_math analytics; produces the book's
aggregate DV01 / KRD profile and full-reprice P&L under the three calibrated
Japan-liquidation scenarios from config/settings.yaml.

Modeling assumptions (docs/methodology.md):
  A1 Credit sleeves (IG/HY/MBS) are modeled as treasury-curve bonds plus a
     separate OAS-spread P&L term: dP/P = -D_mod*dy_treasury - D_mod*ds.
     Price LEVELS are treasury-equivalent; the split is the standard
     rates/spread decomposition.
  A2 Spread beta to the scenario ig_bp shock: IG 1.0, HY 1.8, MBS 0.5
     (HY widens ~1.8x IG in stress; MBS option-adjusted spreads move less).
  A3 Rates shock shape: sqrt(t/10) ramp to the 10Y anchor (short end held
     down by policy rates), linear 10Y->30Y between the two calibrated
     anchors. Anchors trace to stress_calibration_notes in settings.yaml.
  A4 Haircut is a liquidity concession on the 15% Japan-overlay sleeve if
     the fund must sell into the stressed market (margin/redemption flow).
  A5 MBS is a 5.5Y bullet proxy - real MBS negative convexity (extension
     risk when yields rise) is NOT captured, so extreme-scenario losses are
     understated. Flagged in reporting limitations.
  A6 dfx_pct / dvix are scenario metadata consumed by later layers (FX
     overlay, regime model); they do not enter bond-book P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .bond_math import (BondSpec, build_curve, convexity, curve_from_row,
                            key_rate_durations, modified_duration, price, ytm)
except ImportError:
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    from src.models.bond_math import (BondSpec, build_curve, convexity,
                                      curve_from_row, key_rate_durations,
                                      modified_duration, price, ytm)

NAV = 10_000_000_000.0
JAPAN_OVERLAY_PCT = 0.15


@dataclass(frozen=True)
class Position:
    name: str
    sleeve: str
    weight: float
    spec: BondSpec | None
    spread_mult: float = 0.0


def default_book() -> list[Position]:
    """40/30/10/10/10 institutional allocation (settings.yaml)."""
    return [
        Position("UST_2Y", "treasuries", 0.40 * 0.15, BondSpec(coupon=0.042, maturity=2.0)),
        Position("UST_5Y", "treasuries", 0.40 * 0.25, BondSpec(coupon=0.0435, maturity=5.0)),
        Position("UST_10Y", "treasuries", 0.40 * 0.35, BondSpec(coupon=0.048, maturity=10.0)),
        Position("UST_30Y", "treasuries", 0.40 * 0.25, BondSpec(coupon=0.051, maturity=30.0)),
        Position("IG_7Y", "ig_corp", 0.30, BondSpec(coupon=0.052, maturity=7.0), 1.0),
        Position("HY_4Y", "hy", 0.10, BondSpec(coupon=0.075, maturity=4.0), 1.8),
        Position("MBS_5Y5", "mbs", 0.10, BondSpec(coupon=0.054, maturity=5.5), 0.5),
        Position("CASH", "cash", 0.10, None, 0.0),
    ]


SCENARIOS = {
    "japan_baseline": {"dy_10y_bp": 50, "dy_30y_bp": 65, "dfx_pct": -4,
                       "dvix": 10, "ig_bp": 20, "haircut": 0.02},
    "japan_severe":   {"dy_10y_bp": 120, "dy_30y_bp": 160, "dfx_pct": -10,
                       "dvix": 25, "ig_bp": 60, "haircut": 0.06},
    "japan_extreme":  {"dy_10y_bp": 250, "dy_30y_bp": 320, "dfx_pct": -18,
                       "dvix": 50, "ig_bp": 120, "haircut": 0.12},
}


def rates_shock(tenors, dy_10y_bp: float, dy_30y_bp: float) -> np.ndarray:
    """Tenor-shaped yield shock in bp (A3): sqrt ramp to 10Y, linear to 30Y."""
    t = np.asarray(tenors, float)
    return np.where(t <= 10.0,
                    dy_10y_bp * np.sqrt(t / 10.0),
                    dy_10y_bp + (dy_30y_bp - dy_10y_bp) * (t - 10.0) / 20.0)


def book_metrics(book, tenors, par, nav: float = NAV) -> pd.DataFrame:
    """Per-position and total DV01 / KRD dollar profile."""
    rows, all_keys = [], set()
    krd_usd = {}
    for pos in book:
        if pos.spec is None:
            rows.append({"position": pos.name, "sleeve": pos.sleeve,
                         "weight": pos.weight, "notional_usd": pos.weight * nav,
                         "price": np.nan, "mod_dur": 0.0, "convexity": 0.0,
                         "dv01_usd": 0.0})
            krd_usd[pos.name] = {}
            continue
        curve = build_curve(tenors, par)[0]
        px = price(pos.spec, curve)
        y = ytm(pos.spec, px)
        md, cv = modified_duration(pos.spec, y), convexity(pos.spec, y)
        notional = pos.weight * nav
        rows.append({"position": pos.name, "sleeve": pos.sleeve,
                     "weight": pos.weight, "notional_usd": notional,
                     "price": round(px, 2), "mod_dur": round(md, 3),
                     "convexity": round(cv, 1),
                     "dv01_usd": md * (px / 100.0) * notional * 1e-4})
        krd_usd[pos.name] = {k: d * notional * 1e-4
                             for k, d in key_rate_durations(pos.spec, tenors, par).items()}
        all_keys |= set(krd_usd[pos.name])

    df = pd.DataFrame(rows).set_index("position")
    for k in sorted(all_keys):
        df[f"krd_{k}y_usd"] = [krd_usd[n].get(k, 0.0) for n in df.index]
    total = {"sleeve": "TOTAL", "weight": df["weight"].sum(),
             "notional_usd": df["notional_usd"].sum(), "price": np.nan,
             "mod_dur": np.nan, "convexity": np.nan,
             "dv01_usd": df["dv01_usd"].sum()}
    for k in sorted(all_keys):
        total[f"krd_{k}y_usd"] = df[f"krd_{k}y_usd"].sum()
    return pd.concat([df, pd.DataFrame([total], index=["TOTAL"])])


def apply_scenario(book, tenors, par, scenario: dict, nav: float = NAV) -> dict:
    """Full-reprice scenario P&L: rates (exact reprice) + spreads + haircut."""
    shock = rates_shock(tenors, scenario["dy_10y_bp"], scenario["dy_30y_bp"]) / 1e4
    c0 = build_curve(tenors, par)[0]
    c1 = build_curve(tenors, par + shock)[0]

    rows = []
    for pos in book:
        if pos.spec is None:
            rows.append({"position": pos.name, "sleeve": pos.sleeve,
                         "weight": pos.weight, "rates_pnl_usd": 0.0,
                         "spread_pnl_usd": 0.0, "taylor_dur_usd": 0.0,
                         "taylor2_usd": 0.0, "pnl_usd": 0.0})
            continue
        px0, px1 = price(pos.spec, c0), price(pos.spec, c1)
        y = ytm(pos.spec, px0)
        md, cv = modified_duration(pos.spec, y), convexity(pos.spec, y)
        w_nav = pos.weight * nav
        rates_pnl = w_nav * (px1 / px0 - 1.0)
        dy_own = float(np.interp(pos.spec.maturity, tenors, shock))
        taylor_dur = w_nav * (-md * dy_own)
        taylor2 = taylor_dur + w_nav * 0.5 * cv * dy_own ** 2
        spread_pnl = -w_nav * md * (scenario["ig_bp"] * 1e-4) * pos.spread_mult
        rows.append({"position": pos.name, "sleeve": pos.sleeve,
                     "weight": pos.weight, "rates_pnl_usd": rates_pnl,
                     "spread_pnl_usd": spread_pnl, "taylor_dur_usd": taylor_dur,
                     "taylor2_usd": taylor2, "pnl_usd": rates_pnl + spread_pnl})

    by_pos = pd.DataFrame(rows).set_index("position")
    haircut = -scenario.get("haircut", 0.0) * JAPAN_OVERLAY_PCT * nav
    total = by_pos["pnl_usd"].sum() + haircut
    out = {"scenario": scenario.get("name", "custom"), "nav": nav,
           "by_position": by_pos,
           "rates_pnl_usd": by_pos["rates_pnl_usd"].sum(),
           "spread_pnl_usd": by_pos["spread_pnl_usd"].sum(),
           "haircut_pnl_usd": haircut,
           "total_pnl_usd": total, "total_pnl_pct": total / nav,
           "taylor_dur_only_usd": by_pos["taylor_dur_usd"].sum(),
           "taylor_second_order_usd": by_pos["taylor2_usd"].sum()}
    for k in ("dy_10y_bp", "dy_30y_bp", "ig_bp", "haircut", "dfx_pct", "dvix"):
        if k in scenario:
            out[k] = scenario[k]
    return out


def scenario_table(book, tenors, par, nav: float = NAV,
                   scenarios=SCENARIOS) -> pd.DataFrame:
    rows = []
    for name, sc in scenarios.items():
        sc = {**sc, "name": name}
        r = apply_scenario(book, tenors, par, sc, nav)
        rows.append({"scenario": name,
                     "rates_$M": r["rates_pnl_usd"] / 1e6,
                     "spread_$M": r["spread_pnl_usd"] / 1e6,
                     "haircut_$M": r["haircut_pnl_usd"] / 1e6,
                     "total_$M": r["total_pnl_usd"] / 1e6,
                     "total_%NAV": r["total_pnl_pct"] * 100,
                     "dur_only_$M": r["taylor_dur_only_usd"] / 1e6,
                     "taylor2_$M": r["taylor_second_order_usd"] / 1e6})
    return pd.DataFrame(rows).set_index("scenario")


if __name__ == "__main__":
    import pathlib
    p = pathlib.Path("data/processed/yields_dec.parquet")
    if p.exists():
        row = pd.read_parquet(p).iloc[-1]
        tenors, par = curve_from_row(row)
        print(f"curve source: REAL panel, last row {row.name}")
    else:
        row = pd.Series({"DGS1MO": 0.0402, "DGS3MO": 0.0408, "DGS6MO": 0.0410,
                         "DGS1": 0.0412, "DGS2": 0.0418, "DGS3": 0.0425,
                         "DGS5": 0.0435, "DGS7": 0.0455, "DGS10": 0.0480,
                         "DGS20": 0.0530, "DGS30": 0.0510})
        tenors, par = curve_from_row(row)
        print("curve source: built-in demo row (run fetchers+processors for real data)")
    book = default_book()
    m = book_metrics(book, tenors, par)
    print(m.to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\nblended duration: {m.loc['TOTAL','dv01_usd']/(NAV*1e-4):.2f} | "
          f"DV01: ${m.loc['TOTAL','dv01_usd']/1e6:.2f}M/bp")
    print()
    print(scenario_table(book, tenors, par).to_string(float_format=lambda v: f"{v:,.1f}"))