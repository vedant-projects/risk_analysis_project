"""Offline tests for bond_math: closed forms, identities, KRD semantics."""
import numpy as np
import pandas as pd
import pytest

from src.models.bond_math import (BondSpec, build_curve, curve_from_row,
                                  key_rate_durations, macaulay_duration,
                                  modified_duration, price, risk_summary, ytm)

T = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
FLAT = np.full(10, 0.05)


def test_flat_curve_bootstraps_to_flat_zeros_and_par_bonds_price_at_100():
    curve, grid, gy = build_curve(T, FLAT)
    zeros = curve.cont_zero(grid)
    assert np.all(zeros > 0.049) and np.all(zeros < 0.0495)
    for g, y in zip(grid, gy):
        assert abs(price(BondSpec(coupon=y, maturity=g), curve) - 100.0) < 1e-8


def test_par_bond_macaulay_matches_closed_form():
    b = BondSpec(coupon=0.05, maturity=10.0)
    r, n = 0.025, 20
    closed = ((1 + r) / r * (1 - (1 + r) ** -n)) / 2
    assert abs(macaulay_duration(b, 0.05) - closed) < 1e-10


def test_zero_coupon_duration_is_maturity():
    z = BondSpec(coupon=0.0, maturity=5.0)
    assert abs(macaulay_duration(z, 0.05) - 5.0) < 1e-12
    assert abs(modified_duration(z, 0.05) - 5.0 / 1.025) < 1e-12


def test_dv01_matches_curve_bump():
    b = BondSpec(coupon=0.05, maturity=10.0)
    c0, _, _ = build_curve(T, FLAT)
    dn, _, _ = build_curve(T, FLAT - 1e-4)
    px0 = price(b, c0)
    bump_dv01 = price(b, dn) - px0          # lower yield -> higher price
    analytic = modified_duration(b, ytm(b, px0)) * px0 * 1e-4
    assert abs(bump_dv01 - analytic) / analytic < 1e-3


def test_krd_sums_to_effective_duration_and_concentrates():
    b = BondSpec(coupon=0.035, maturity=10.0)   # discount bond, non-par
    krd = key_rate_durations(b, T, FLAT)
    eff = risk_summary(b, T, FLAT)["eff_dur"]
    assert abs(sum(krd.values()) - eff) < 0.01
    assert max(krd, key=krd.get) == 10
    # par-bump semantics: small negative distant KRDs are legitimate
    assert all(v > -0.5 for v in krd.values())


def test_curve_from_row_drops_nan_tenors_and_rejects_sparse():
    row = pd.Series({"DGS3MO": 0.05, "DGS1": 0.05, "DGS2": 0.05, "DGS3": 0.05,
                     "DGS5": 0.05, "DGS7": 0.05, "DGS10": 0.05, "DGS20": 0.05,
                     "DGS30": np.nan, "SPX": 4000.0})
    tenors, yields = curve_from_row(row)
    assert tenors.max() == 20.0 and 30.0 not in tenors
    with pytest.raises(ValueError):
        curve_from_row(pd.Series({"DGS10": 0.05, "DGS30": 0.05}))


def test_risk_summary_on_realistic_curve():
    row = pd.Series({"DGS1MO": 0.0402, "DGS3MO": 0.0408, "DGS6MO": 0.0410,
                     "DGS1": 0.0412, "DGS2": 0.0418, "DGS3": 0.0425,
                     "DGS5": 0.0435, "DGS7": 0.0455, "DGS10": 0.0480,
                     "DGS20": 0.0530, "DGS30": 0.0510})
    tenors, yields = curve_from_row(row)
    s = risk_summary(BondSpec(coupon=0.045, maturity=30.0), tenors, yields)
    assert 80 < s["price"] < 100
    assert 14 < s["mod_dur"] < 20
    assert 0.10 < s["dv01"] < 0.20