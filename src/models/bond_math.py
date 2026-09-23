"""bond_math.py - discount-curve construction and single-bond risk analytics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import brentq

TENOR_YEARS = {"DGS1MO": 1/12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
               "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0,
               "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0}

@dataclass(frozen=True)
class BondSpec:
    face: float = 100.0
    coupon: float = 0.05
    maturity: float = 10.0
    freq: int = 2

def bond_cashflows(spec):
    n = int(round(spec.maturity * spec.freq))
    times = np.arange(1, n + 1) / spec.freq
    amounts = np.full(n, spec.face * spec.coupon / spec.freq)
    amounts[-1] += spec.face
    return times, amounts

def curve_from_row(row):
    pts = [(TENOR_YEARS[k], float(v)) for k, v in row.items()
           if k in TENOR_YEARS and pd.notna(v)]
    if len(pts) < 3:
        raise ValueError(f"need >=3 tenors, got {len(pts)}")
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

def _par_grid(tenors, par_yields, step=0.5):
    m, y = np.asarray(tenors, float), np.asarray(par_yields, float)
    if m.min() > step:
        m, y = np.concatenate([[step], m]), np.concatenate([[y[0]], y])
    grid = np.arange(step, m.max() + 1e-9, step)
    return grid, np.interp(grid, m, y)

def bootstrap_discounts(grid, par):
    d = np.empty(len(grid))
    d[0] = 1.0 / (1.0 + par[0] / 2.0)
    for n in range(1, len(grid)):
        c = par[n] / 2.0
        d[n] = (1.0 - c * d[:n].sum()) / (1.0 + c)
    return d

class ZeroCurve:
    def __init__(self, times, discounts):
        self.t, self.d = np.asarray(times, float), np.asarray(discounts, float)
        if len(self.t) < 2:
            raise ValueError("curve needs >=2 points")
        self.logd = np.log(self.d)

    def discount(self, t):
        t = np.asarray(t, float)
        scalar = (t.ndim == 0)
        tv = np.atleast_1d(t).astype(float)
        out = np.interp(tv, self.t, self.logd)
        f0 = (self.logd[1] - self.logd[0]) / (self.t[1] - self.t[0])
        f1 = (self.logd[-1] - self.logd[-2]) / (self.t[-1] - self.t[-2])
        hi, lo = tv > self.t[-1], tv < self.t[0]
        out[hi] = self.logd[-1] + f1 * (tv[hi] - self.t[-1])
        out[lo] = self.logd[0] + f0 * (tv[lo] - self.t[0])
        res = np.exp(out)
        return float(res[0]) if scalar else res

    def cont_zero(self, t):
        return -np.log(self.discount(t)) / np.asarray(t, float)

def build_curve(tenors, par_yields):
    grid, gy = _par_grid(tenors, par_yields)
    return ZeroCurve(grid, bootstrap_discounts(grid, gy)), grid, gy

def _curve_from_grid(grid, gy):
    return ZeroCurve(grid, bootstrap_discounts(grid, gy))

def price(spec, curve):
    times, cfs = bond_cashflows(spec)
    return float((cfs * curve.discount(times)).sum())

def ytm(spec, px):
    times, cfs = bond_cashflows(spec)
    def f(y):
        return float((cfs / (1 + y / spec.freq) ** (spec.freq * times)).sum() - px)
    return float(brentq(f, -0.5, 1.0, xtol=1e-12))

def macaulay_duration(spec, y):
    times, cfs = bond_cashflows(spec)
    pvs = cfs / (1 + y / spec.freq) ** (spec.freq * times)
    return float((times * pvs).sum() / pvs.sum())

def modified_duration(spec, y):
    return macaulay_duration(spec, y) / (1 + y / spec.freq)

def convexity(spec, y):
    times, cfs = bond_cashflows(spec)
    r, n = y / spec.freq, spec.freq * times
    pvs = cfs / (1 + r) ** n
    return float((n * (n + 1) * cfs / (1 + r) ** (n + 2)).sum()
                 / (pvs.sum() * spec.freq ** 2))

def _tent_weights(grid, keys):
    keys = [k for k in keys if k <= grid.max() + 1e-9] or [grid.max()]
    W = np.zeros((len(keys), len(grid)))
    for i, k in enumerate(keys):
        left = keys[i - 1] if i > 0 else None
        right = keys[i + 1] if i < len(keys) - 1 else None
        if left is None:
            W[i] = np.clip((right - grid) / (right - k), 0, 1)
        elif right is None:
            W[i] = np.clip((grid - left) / (k - left), 0, 1)
        else:
            W[i] = (np.clip((grid - left) / (k - left), 0, 1)
                    * np.clip((right - grid) / (right - k), 0, 1))
    return keys, W

def effective_duration(spec, tenors, par_yields, bump=1e-4):
    c0, _, _ = build_curve(tenors, par_yields)
    p0 = price(spec, c0)
    up, _, _ = build_curve(tenors, np.asarray(par_yields) + bump)
    dn, _, _ = build_curve(tenors, np.asarray(par_yields) - bump)
    return (price(spec, dn) - price(spec, up)) / (2 * bump * p0)

def key_rate_durations(spec, tenors, par_yields, keys=(2, 5, 10, 30), bump=1e-4):
    """Tent-based key-rate durations via par-yield bumps + re-bootstrap.

    Semantics note (documented for interviews): bumps are applied to PAR
    yields with a tent that dies at neighboring keys, holding other par
    yields fixed. Holding, say, the 5Y par yield fixed while raising the 2Y
    forces implied forwards beyond 2Y DOWN, so discount factors there rise.
    A bond whose cashflows sit beyond the bumped region can therefore show a
    small NEGATIVE KRD at a distant key. That is a genuine, textbook feature
    of par-bump KRDs (sum of KRDs still equals effective duration, which the
    tests assert), not a bug.
    """
    c0, grid, gy = build_curve(tenors, par_yields)
    p0 = price(spec, c0)
    keys, W = _tent_weights(grid, list(keys))
    krd = {}
    for k, w in zip(keys, W):
        cb = _curve_from_grid(grid, gy + bump * w)
        krd[k] = -(price(spec, cb) - p0) / (p0 * bump)
    return krd

def risk_summary(spec, tenors, par_yields):
    curve, _, _ = build_curve(tenors, par_yields)
    px = price(spec, curve)
    y = ytm(spec, px)
    return {"price": px, "ytm": y,
            "mac_dur": macaulay_duration(spec, y),
            "mod_dur": modified_duration(spec, y),
            "convexity": convexity(spec, y),
            "dv01": modified_duration(spec, y) * px * 1e-4,
            "eff_dur": effective_duration(spec, tenors, par_yields),
            "krd": key_rate_durations(spec, tenors, par_yields)}