from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

TENORS = np.array([
    1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30
])

RATE_COLUMNS = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
    "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
]

FACTORS = RATE_COLUMNS + ["IG", "HY"]


@lru_cache(maxsize=2048)
def schedule(issue, maturity, frequency, coupon):
    issue = pd.Timestamp(issue).normalize()
    maturity = pd.Timestamp(maturity).normalize()
    months = 12 // int(frequency)

    dates = []
    current = maturity

    while current > issue:
        dates.append(current)
        current = current - pd.DateOffset(months=months)

    dates = sorted(dates)
    previous = issue
    flows = []

    for date in dates:
        # Explicit convention: ACT/365 Fixed coupon accrual.
        amount = float(coupon) * (date - previous).days / 365
        if date == maturity:
            amount += 1.0
        flows.append(amount)
        previous = date

    return tuple(dates), tuple(flows)


def cashflows(bond):
    dates, flows = schedule(
        str(bond.issue_date),
        str(bond.maturity_date),
        int(bond.frequency),
        float(bond.coupon),
    )
    return pd.DatetimeIndex(dates), np.asarray(flows)


def accrued_per_face(bond, asof):
    asof = pd.Timestamp(asof).normalize()
    dates, _ = cashflows(bond)

    if asof < pd.Timestamp(bond.issue_date) or asof >= dates[-1]:
        return 0.0

    earlier = dates[dates <= asof]
    previous = (
        earlier[-1] if len(earlier)
        else pd.Timestamp(bond.issue_date)
    )

    return float(bond.coupon) * (asof - previous).days / 365


def price(bond, asof, market):
    asof = pd.Timestamp(asof).normalize()
    dates, flows = cashflows(bond)
    remaining = dates > asof

    if not remaining.any():
        return 0.0

    years = (dates[remaining] - asof).days.to_numpy() / 365.0
    rates = np.interp(
        years,
        TENORS,
        np.asarray(market, dtype=float)[:len(TENORS)],
    )

    spread = 0.0
    if bond.credit_bucket == "IG":
        spread = float(market[len(TENORS)])
    elif bond.credit_bucket == "HY":
        spread = float(market[len(TENORS) + 1])

    # Continuous discounting; yields and OAS are explicitly proxy inputs.
    return float(
        np.dot(flows[remaining], np.exp(-(rates + spread) * years))
    )


def payment_between(bond, previous_date, current_date):
    dates, flows = cashflows(bond)
    mask = (
        (dates > pd.Timestamp(previous_date))
        & (dates <= pd.Timestamp(current_date))
    )
    return float(flows[mask].sum())


def values(book, asof, market):
    return np.array([
        float(bond.face) * price(bond, asof, market)
        for bond in book.itertuples(index=False)
    ])


def sensitivities(book, asof, market, bump=0.0001):
    market = np.asarray(market, dtype=float)
    result = np.zeros((len(book), len(market)))

    for factor in range(len(market)):
        up = market.copy()
        down = market.copy()
        up[factor] += bump
        down[factor] -= bump

        # Positive sensitivity = loss for an upward 1 bp move.
        result[:, factor] = (
            values(book, asof, down) - values(book, asof, up)
        ) / 2

    return result


def parallel_metrics(bond, asof, market, bump=0.0001):
    market = np.asarray(market, dtype=float)

    up = market.copy()
    down = market.copy()
    up[:len(TENORS)] += bump
    down[:len(TENORS)] -= bump

    p = price(bond, asof, market)
    pu = price(bond, asof, up)
    pdn = price(bond, asof, down)

    duration = (pdn - pu) / (2 * bump * p)
    convexity = (pdn + pu - 2 * p) / (bump ** 2 * p)

    return {
        "dirty_price_per_100": 100 * p,
        "clean_price_per_100": 100 * (
            p - accrued_per_face(bond, asof)
        ),
        "effective_rate_duration": duration,
        "effective_rate_convexity": convexity,
        "rate_DV01": float(bond.face) * (pdn - pu) / 2,
    }


def historical_revaluation(book, markets):
    records = []
    instruments = []

    rows = list(markets.iterrows())

    for i in range(1, len(rows)):
        previous_date, previous_market = rows[i - 1]
        date, market = rows[i]

        old = previous_market.to_numpy(dtype=float)
        new = market.to_numpy(dtype=float)

        old_values = values(book, previous_date, old)
        roll_values = values(book, date, old)
        new_values = values(book, date, new)

        payments = np.array([
            float(b.face) * payment_between(b, previous_date, date)
            for b in book.itertuples(index=False)
        ])

        cash_carry = roll_values - old_values + payments
        market_pnl = new_values - roll_values
        total_pnl = cash_carry + market_pnl

        # Sensitivities evaluated at the end-date with unchanged factors.
        risk = sensitivities(book, date, old)
        explained = -risk * ((new - old) / 0.0001)
        residual = market_pnl - explained.sum(axis=1)

        previous_value = old_values.sum()

        record = {
            "date": date,
            "previous_book_value": previous_value,
            "book_value": new_values.sum(),
            "coupon_cash": payments.sum(),
            "carry_roll": cash_carry.sum(),
            "market_pnl": market_pnl.sum(),
            "total_pnl": total_pnl.sum(),
            "normalized_hpl": total_pnl.sum() / previous_value,
            "rate_explain": explained[:, :len(TENORS)].sum(),
            "IG_explain": explained[:, len(TENORS)].sum(),
            "HY_explain": explained[:, len(TENORS) + 1].sum(),
            "residual": residual.sum(),
        }
        records.append(record)

        for j, bond in enumerate(book.itertuples(index=False)):
            instruments.append({
                "date": date,
                "id": bond.id,
                "value": new_values[j],
                "total_pnl": total_pnl[j],
                "market_pnl": market_pnl[j],
                "carry_roll": cash_carry[j],
                "residual": residual[j],
            })

    history = pd.DataFrame(records).set_index("date")
    details = pd.DataFrame(instruments)

    return history, details