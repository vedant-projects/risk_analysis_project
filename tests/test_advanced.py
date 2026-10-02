from types import SimpleNamespace

import numpy as np
import pandas as pd

from advanced.engine import (
    TENORS, FACTORS, price, values, sensitivities,
    historical_revaluation, payment_between,
)
from advanced.pipeline import coverage_tests


def bond():
    return SimpleNamespace(
        id="TEST",
        issuer="TEST",
        sector="Government",
        credit_bucket="GOV",
        issue_date="2020-01-01",
        maturity_date="2030-01-01",
        coupon=0.04,
        frequency=2,
        face=1_000_000,
    )


def market(rate=0.04):
    return np.r_[np.full(len(TENORS), rate), 0.01, 0.04]


def test_price_falls_when_rates_rise():
    b = bond()
    assert price(b, "2026-01-02", market(0.05)) < price(
        b, "2026-01-02", market(0.04)
    )


def test_cash_payment_is_positive():
    assert payment_between(
        bond(), "2025-12-31", "2026-01-01"
    ) > 0


def test_credit_spread_does_not_change_government_price():
    m = market()
    stressed = m.copy()
    stressed[-2:] += 0.01

    assert np.isclose(
        price(bond(), "2026-01-02", m),
        price(bond(), "2026-01-02", stressed),
    )


def test_sensitivity_matches_parallel_bump():
    book = pd.DataFrame([vars(bond())])
    m = market()
    risk = sensitivities(book, "2026-01-02", m)[0]

    up = m.copy()
    down = m.copy()
    up[:len(TENORS)] += 0.0001
    down[:len(TENORS)] -= 0.0001

    actual = (
        values(book, "2026-01-02", down).sum()
        - values(book, "2026-01-02", up).sum()
    ) / 2

    assert np.isclose(
        risk[:len(TENORS)].sum(), actual, rtol=0.001
    )


def test_attribution_reconciles():
    book = pd.DataFrame([vars(bond())])
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    markets = pd.DataFrame(
        [market(0.04), market(0.0405)],
        index=dates, columns=FACTORS,
    )

    history, _ = historical_revaluation(book, markets)
    row = history.iloc[0]

    explained = (
        row["carry_roll"] + row["rate_explain"]
        + row["IG_explain"] + row["HY_explain"]
        + row["residual"]
    )

    assert np.isclose(row["total_pnl"], explained)


def test_zero_exception_coverage_is_finite():
    result = coverage_tests(np.zeros(250, dtype=int), 0.01)
    assert np.isfinite(result["Kupiec_p"])
    assert np.isnan(result["Christoffersen_ind_p"])


def test_matured_bond_has_no_remaining_value():
    assert price(bond(), "2030-01-02", market()) == 0