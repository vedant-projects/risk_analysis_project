from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scipy.optimize import linprog, minimize
from scipy.special import xlogy
from scipy.stats import chi2, norm, t
from sklearn.decomposition import PCA

from advanced.engine import (
    FACTORS, RATE_COLUMNS, TENORS,
    values, sensitivities, parallel_metrics,
    historical_revaluation,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/advanced"
OUT = ROOT / "output/advanced"

WINDOW = 250
CONFIDENCE = 0.99
SEED = 42

DISCLAIMER = (
    "Illustrative-book research unless a verified instrument master is "
    "supplied. Treasury CMT yields are zero-rate proxies; IG/HY index OAS "
    "are additive spread proxies. No market-price calibration, business-day "
    "adjustment, default, MBS optionality, FX or regulatory compliance."
)


def save(frame, name, index=False):
    frame.to_csv(OUT / name, index=index)


def download_spread(series, start, end):
    path = DATA / f"{series}.csv"

    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series, "cosd": start, "coed": end},
        timeout=90,
    )
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))

    if series not in frame.columns:
        raise ValueError(f"Unexpected FRED response for {series}")

    result = pd.Series(
        pd.to_numeric(frame[series], errors="coerce").to_numpy() / 100,
        index=pd.to_datetime(frame.iloc[:, 0]),
        name=series,
    ).sort_index()

    result.to_csv(path)
    return result


def load_markets(max_rows):
    path = ROOT / "data/processed/yields_dec.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "Run your original build_learning_history.py first."
        )

    rates = pd.read_parquet(path)[RATE_COLUMNS].sort_index()
    start = str(rates.index.min().date())
    end = str(rates.index.max().date())

    ig = download_spread("BAMLC0A0CM", start, end).rename("IG")
    hy = download_spread("BAMLH0A0HYM2", start, end).rename("HY")

    market = rates.join(ig).join(hy)
    market = market.ffill(limit=3).dropna()
    market = market.loc[~market.index.duplicated()].tail(max_rows)

    if len(market) < 600:
        raise ValueError("Need at least 600 aligned market observations.")

    if not np.isfinite(market.to_numpy()).all():
        raise ValueError("Nonfinite market observations.")

    if not market.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted.")

    market.index.name = "date"
    market.to_csv(DATA / "market_factors.csv")

    return market


def demo_book(asof):
    rows = []

    specifications = [
        ("DEMO_UST2", 2, "GOV", 0.040, 15_000_000),
        ("DEMO_UST5", 5, "GOV", 0.040, 20_000_000),
        ("DEMO_UST10", 10, "GOV", 0.045, 20_000_000),
        ("DEMO_UST30", 30, "GOV", 0.045, 15_000_000),
        ("DEMO_IG7", 7, "IG", 0.055, 20_000_000),
        ("DEMO_HY5", 5, "HY", 0.075, 10_000_000),
    ]

    for name, years, bucket, coupon, face in specifications:
        rows.append({
            "id": name,
            "issuer": "ILLUSTRATIVE",
            "sector": "Government" if bucket == "GOV" else "Corporate",
            "credit_bucket": bucket,
            "issue_date": "2000-01-01",
            "maturity_date": str(
                (asof + pd.DateOffset(years=years)).date()
            ),
            "coupon": coupon,
            "frequency": 2,
            "face": face,
            "daily_capacity_usd": 10_000_000 if bucket == "GOV" else 1_000_000,
            "stressed_bidask_bps": 10 if bucket == "GOV" else 75,
        })

    return pd.DataFrame(rows)


def validate_book(book, earliest_date, asof):
    required = {
        "id", "issuer", "sector", "credit_bucket", "issue_date",
        "maturity_date", "coupon", "frequency", "face",
        "daily_capacity_usd", "stressed_bidask_bps",
    }

    if not required <= set(book.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(book.columns))}")

    if book.isna().any().any():
        raise ValueError("Instrument master contains missing values.")

    if book["id"].duplicated().any():
        raise ValueError("Instrument IDs must be unique.")

    if not book["credit_bucket"].isin(["GOV", "IG", "HY"]).all():
        raise ValueError("Only GOV, IG and HY buckets are supported.")

    if not book["frequency"].isin([1, 2, 4, 12]).all():
        raise ValueError("Frequency must be 1, 2, 4 or 12.")

    if (book["face"] <= 0).any() or (book["coupon"] < 0).any():
        raise ValueError("Require positive face and nonnegative coupon.")

    if (book["daily_capacity_usd"] <= 0).any():
        raise ValueError("Liquidity capacities must be positive.")

    if (book["stressed_bidask_bps"] < 0).any():
        raise ValueError("Bid-ask assumptions cannot be negative.")

    issue = pd.to_datetime(book["issue_date"])
    maturity = pd.to_datetime(book["maturity_date"])

    if (issue >= maturity).any():
        raise ValueError("Issue date must precede maturity.")

    if (issue > earliest_date).any():
        raise ValueError(
            "This fixed-book historical study requires every bond to "
            "exist at the start of the selected market history."
        )

    if (maturity <= asof).any():
        raise ValueError("This version requires all bonds to remain outstanding.")


def bernoulli_loglik(successes, observations, probability):
    return float(
        xlogy(successes, probability)
        + xlogy(observations - successes, 1 - probability)
    )


def coverage_tests(exceptions, expected_probability):
    e = np.asarray(exceptions, dtype=int)
    n = len(e)
    k = int(e.sum())

    empirical = k / n

    ll_null = bernoulli_loglik(k, n, expected_probability)
    ll_alt = bernoulli_loglik(k, n, empirical)
    lr_uc = max(0.0, 2 * (ll_alt - ll_null))

    n00 = int(((e[:-1] == 0) & (e[1:] == 0)).sum())
    n01 = int(((e[:-1] == 0) & (e[1:] == 1)).sum())
    n10 = int(((e[:-1] == 1) & (e[1:] == 0)).sum())
    n11 = int(((e[:-1] == 1) & (e[1:] == 1)).sum())

    if n00 + n01 == 0 or n10 + n11 == 0:
        lr_ind = np.nan
        p_ind = np.nan
        p_cc = np.nan
    else:
        p01 = n01 / (n00 + n01)
        p11 = n11 / (n10 + n11)
        pooled = (n01 + n11) / (n - 1)

        independent = (
            bernoulli_loglik(n01, n00 + n01, pooled)
            + bernoulli_loglik(n11, n10 + n11, pooled)
        )
        dependent = (
            bernoulli_loglik(n01, n00 + n01, p01)
            + bernoulli_loglik(n11, n10 + n11, p11)
        )

        lr_ind = max(0.0, 2 * (dependent - independent))
        p_ind = chi2.sf(lr_ind, 1)
        p_cc = chi2.sf(lr_uc + lr_ind, 2)

    return {
        "observations": n,
        "exceptions": k,
        "exception_rate": empirical,
        "Kupiec_LR": lr_uc,
        "Kupiec_p": chi2.sf(lr_uc, 1),
        "Christoffersen_ind_LR": lr_ind,
        "Christoffersen_ind_p": p_ind,
        "conditional_coverage_p": p_cc,
    }


def rolling_risk(history):
    r = history["normalized_hpl"]
    rows = []

    for i in range(WINDOW, len(r)):
        sample = r.iloc[i - WINDOW:i].to_numpy()
        loss = -sample

        mean_loss = loss.mean()
        sigma = loss.std(ddof=1)
        sigma = max(sigma, 1e-12)
        z = norm.ppf(CONFIDENCE)

        recent = sample[-60:].std(ddof=1)
        filtered_loss = (
            -sample / sigma * max(recent, 1e-12)
        )

        # Fixed df is a documented challenger assumption, not a fit.
        df = 5
        scale = sigma * np.sqrt((df - 2) / df)
        q = t.ppf(CONFIDENCE, df)

        models = {
            "Historical": (
                np.quantile(loss, CONFIDENCE),
                None,
            ),
            "Gaussian": (
                mean_loss + sigma * z,
                mean_loss + sigma * norm.pdf(z) / (1 - CONFIDENCE),
            ),
            "Student_t_df5": (
                mean_loss + scale * q,
                mean_loss
                + scale * (df + q * q) / (df - 1)
                * t.pdf(q, df) / (1 - CONFIDENCE),
            ),
            "Vol_scaled_historical": (
                np.quantile(filtered_loss, CONFIDENCE),
                None,
            ),
        }

        for name, (var, es) in models.items():
            distribution = filtered_loss if name.startswith("Vol") else loss

            if es is None:
                es = distribution[distribution >= var].mean()

            realized_loss = -r.iloc[i]

            # Pinball loss for the upper loss quantile.
            error = realized_loss - var
            quantile_loss = (
                CONFIDENCE * error if error >= 0
                else (CONFIDENCE - 1) * error
            )

            rows.append({
                "date": r.index[i],
                "model": name,
                "VaR": var,
                "ES": es,
                "realized_loss": realized_loss,
                "exception": int(realized_loss > var),
                "quantile_loss": quantile_loss,
            })

    forecasts = pd.DataFrame(rows)
    tests = []

    for name, group in forecasts.groupby("model"):
        result = coverage_tests(group["exception"], 1 - CONFIDENCE)
        result.update({
            "model": name,
            "mean_quantile_loss": group["quantile_loss"].mean(),
            # This is a descriptive tail diagnostic, not an ES test.
            "mean_loss_on_exceptions": group.loc[
                group["exception"] == 1, "realized_loss"
            ].mean(),
            "mean_ES_on_exceptions": group.loc[
                group["exception"] == 1, "ES"
            ].mean(),
        })
        tests.append(result)

    save(forecasts, "risk_forecasts.csv")
    save(pd.DataFrame(tests), "risk_validation.csv")


def hedge_optimizer(book, asof, market, portfolio_risk, nav):
    hedge = book.loc[book["credit_bucket"] == "GOV"].copy()
    hedge["face"] = 1_000_000.0

    h = sensitivities(hedge, asof, market).T
    dimension = len(portfolio_risk)
    number = len(hedge)

    # x = short hedge lots >= 0; u = absolute residual by factor.
    # Restrict to short GOV bonds: cannot hedge credit with these instruments.
    matrix = np.block([
        [-h, -np.eye(dimension)],
        [h, -np.eye(dimension)],
    ])
    rhs = np.r_[-portfolio_risk, portfolio_risk]

    gross_row = np.r_[np.ones(number), np.zeros(dimension)]
    matrix = np.vstack([matrix, gross_row])
    rhs = np.r_[rhs, nav / 1_000_000]

    objective = np.r_[
        np.full(number, 0.1),
        np.ones(dimension),
    ]

    result = linprog(
        objective,
        A_ub=matrix,
        b_ub=rhs,
        bounds=[(0, None)] * (number + dimension),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"Hedge optimizer failed: {result.message}")

    lots = result.x[:number]
    residual = portfolio_risk - h @ lots

    trades = hedge[["id"]].copy()
    trades["short_face_usd"] = lots * 1_000_000
    trades["implementation"] = "Continuous cash-bond short; not futures"

    comparison = pd.DataFrame({
        "factor": FACTORS,
        "before_usd_per_bp": portfolio_risk,
        "after_usd_per_bp": residual,
    })

    save(trades, "hedge_trades.csv")
    save(comparison, "hedge_risk.csv")


def scenarios_and_reverse(book, asof, market, nav, changes):
    base = values(book, asof, market).sum()

    templates = {
        "Rates_up_100bp": np.r_[np.full(len(TENORS), 0.01), 0, 0],
        "Rates_down_100bp": np.r_[np.full(len(TENORS), -0.01), 0, 0],
        "Steepener": np.r_[np.linspace(-0.005, 0.01, len(TENORS)), 0, 0],
        "Flattener": np.r_[np.linspace(0.01, -0.005, len(TENORS)), 0, 0],
        "Credit_stress": np.r_[np.zeros(len(TENORS)), 0.01, 0.03],
        "Combined_stress": np.r_[np.full(len(TENORS), 0.01), 0.01, 0.03],
    }

    rows = []
    for name, shock in templates.items():
        pnl = values(book, asof, market + shock).sum() - base
        rows.append({
            "scenario": name,
            "pnl_usd": pnl,
            "pnl_pct": 100 * pnl / nav,
            "type": "Hypothetical; no assigned probability",
        })

    save(pd.DataFrame(rows), "stress_results.csv")
    pd.DataFrame(templates, index=FACTORS).T.to_csv(
        OUT / "scenario_shocks.csv", index_label="scenario"
    )

    scales = np.maximum(changes.std().to_numpy(), 0.00001)

    def shock_from_parameters(z):
        rate_shock = (
            z[0] * 0.01
            + z[1] * np.linspace(-0.01, 0.01, len(TENORS))
        )
        return np.r_[rate_shock, z[2] * 0.01, z[3] * 0.01]

    target = 0.10 * nav

    def loss(z):
        shock = shock_from_parameters(z)
        return base - values(book, asof, market + shock).sum()

    def distance(z):
        shock = shock_from_parameters(z)
        return float(np.sum((shock / scales) ** 2))

    result = minimize(
        distance,
        x0=np.array([1.0, 0.0, 1.0, 2.0]),
        method="SLSQP",
        bounds=[(-3, 3), (-2, 2), (0, 5), (0, 10)],
        constraints=[{
            "type": "ineq",
            "fun": lambda z: loss(z) / nav - target / nav,
        }],
        options={"maxiter": 300, "ftol": 1e-8},
    )

    feasible = bool(result.success and loss(result.x) >= target * 0.999)

    save(pd.DataFrame({
        "factor": FACTORS,
        "shock": shock_from_parameters(result.x),
    }), "reverse_stress_shock.csv")

    save(pd.DataFrame([{
        "target_loss_usd": target,
        "achieved_loss_usd": loss(result.x),
        "solver_success": result.success,
        "constraint_satisfied": feasible,
        "message": result.message,
        "interpretation": (
            "Local constrained solution; diagonal standardized distance "
            "is not a probability or global minimum guarantee"
        ),
    }]), "reverse_stress_result.csv")


def monte_carlo(book, asof, market, nav, changes):
    rng = np.random.default_rng(SEED)
    factors = changes.to_numpy()
    base = values(book, asof, market).sum()
    risk = sensitivities(book, asof, market).sum(axis=0)

    summaries = []
    convergence = []

    # These paths hold the valuation date fixed:
    # terminal market-shock losses, excluding horizon carry.
    for horizon in (1, 10):
        terminal = np.zeros((1000, len(FACTORS)))

        for offset in range(0, horizon, 5):
            width = min(5, horizon - offset)
            starts = rng.integers(0, len(factors) - width + 1, len(terminal))

            for j in range(width):
                terminal += factors[starts + j]

        pnl = np.array([
            values(book, asof, market + shock).sum() - base
            for shock in terminal
        ])

        linear = -(terminal / 0.0001) @ risk
        losses = -pnl / nav

        paths = pd.DataFrame({
            "path": np.arange(len(pnl)),
            "pnl_usd": pnl,
            "linear_pnl_usd": linear,
            "loss": losses,
        })
        save(paths, f"mc_{horizon}d_paths.csv")

        for confidence in (0.95, 0.975, 0.99):
            var = np.quantile(losses, confidence)
            es = losses[losses >= var].mean()
            summaries.append({
                "horizon": horizon,
                "confidence": confidence,
                "VaR": var,
                "ES": es,
                "paths": len(pnl),
                "excluded": "Carry, time roll, defaults and transaction costs",
            })

        for n in (100, 250, 500, 1000):
            sample = losses[:n]
            var = np.quantile(sample, 0.99)
            convergence.append({
                "horizon": horizon,
                "paths": n,
                "VaR99": var,
                "ES99": sample[sample >= var].mean(),
            })

        uncertainty = []
        for _ in range(200):
            sample = rng.choice(losses, len(losses), replace=True)
            var = np.quantile(sample, 0.99)
            uncertainty.append([
                var, sample[sample >= var].mean()
            ])

        uncertainty = np.asarray(uncertainty)
        save(pd.DataFrame({
            "measure": ["VaR99", "ES99"],
            "bootstrap_lower": np.quantile(uncertainty, 0.025, axis=0),
            "bootstrap_upper": np.quantile(uncertainty, 0.975, axis=0),
            "scope": "Conditional path-sampling uncertainty only",
        }), f"mc_{horizon}d_uncertainty.csv")

    save(pd.DataFrame(summaries), "mc_summary.csv")
    save(pd.DataFrame(convergence), "mc_convergence.csv")


def liquidity(book, asof, market, nav):
    frame = book[[
        "id", "daily_capacity_usd", "stressed_bidask_bps"
    ]].copy()

    frame["market_value"] = values(book, asof, market)
    frame["days_to_liquidate"] = np.ceil(
        frame["market_value"] / frame["daily_capacity_usd"]
    )
    frame["liquidation_cost"] = (
        frame["market_value"] * frame["stressed_bidask_bps"] / 10_000
    )

    save(frame, "liquidity_positions.csv")

    rows = []
    for outflow_fraction in (0.10, 0.25, 0.50):
        required = nav * outflow_fraction

        for horizon in (1, 5, 10):
            gross = np.minimum(
                frame["market_value"],
                horizon * frame["daily_capacity_usd"],
            )
            costs = gross * frame["stressed_bidask_bps"] / 10_000
            available = (gross - costs).sum()

            rows.append({
                "outflow_fraction": outflow_fraction,
                "days": horizon,
                "required_cash": required,
                "net_liquidation_capacity": available,
                "shortfall": max(required - available, 0),
                "assumption": (
                    "User/demo capacities; constant prices; simultaneous "
                    "liquidation; no impact feedback"
                ),
            })

    save(pd.DataFrame(rows), "liquidity_stress.csv")


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(book_path):
    inputs = [
        book_path,
        DATA / "market_factors.csv",
        ROOT / "data/processed/yields_dec.parquet",
    ]

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unavailable"

    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn"):
        packages[name] = importlib.metadata.version(name)

    result = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": sys.version,
        "packages": packages,
        "seed": SEED,
        "risk_window": WINDOW,
        "risk_confidence": CONFIDENCE,
        "input_hashes": {
            str(path.relative_to(ROOT)): hash_file(path)
            for path in inputs if path.exists()
        },
        "disclaimer": DISCLAIMER,
        "historical_design": (
            "Retrospective fixed-current-book study, not historical holdings"
        ),
    }

    (OUT / "manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--rows", type=int, default=800)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    market = load_markets(args.rows)
    asof = market.index[-1]
    book_path = DATA / "instruments.csv"

    if args.demo:
        book_path = DATA / "illustrative_instruments.csv"
        demo_book(asof).to_csv(book_path, index=False)

    if not book_path.exists():
        raise FileNotFoundError(
            "Supply data/advanced/instruments.csv, or explicitly use --demo."
        )

    book = pd.read_csv(book_path)
    validate_book(book, market.index[0], asof)
    save(book, "instrument_master.csv")

    current = market.iloc[-1].to_numpy()
    valuation = values(book, asof, current)
    nav = valuation.sum()

    metrics = []
    for bond in book.itertuples(index=False):
        metrics.append({
            "id": bond.id,
            **parallel_metrics(bond, asof, current),
        })

    save(pd.DataFrame(metrics), "bond_metrics.csv")

    risk = sensitivities(book, asof, current)
    risk_table = pd.DataFrame(risk, columns=FACTORS)
    risk_table.insert(0, "id", book["id"].to_numpy())
    save(risk_table, "sensitivities.csv")

    concentration = book[["id", "issuer", "sector", "credit_bucket"]].copy()
    concentration["market_value"] = valuation
    concentration["weight"] = valuation / nav
    save(concentration, "concentration.csv")

    history, detail = historical_revaluation(book, market)
    save(history, "full_revaluation_history.csv", index=True)
    save(detail, "instrument_history.csv")

    changes = market.diff().dropna()

    # Descriptive PCA only. Learning must not use these full-sample scores.
    pca = PCA(n_components=3)
    scores = pca.fit_transform(changes[RATE_COLUMNS])

    loadings = pd.DataFrame(
        pca.components_.T, index=RATE_COLUMNS,
        columns=["PC1", "PC2", "PC3"],
    )
    save(loadings, "pca_loadings.csv", index=True)
    save(pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "explained_variance": pca.explained_variance_ratio_,
    }), "pca_variance.csv")
    save(pd.DataFrame(
        scores, index=changes.index,
        columns=["PC1", "PC2", "PC3"],
    ), "pca_scores.csv", index=True)

    rolling_risk(history)
    hedge_optimizer(book, asof, current, risk.sum(axis=0), nav)
    scenarios_and_reverse(book, asof, current, nav, changes)
    monte_carlo(book, asof, current, nav, changes)
    liquidity(book, asof, current, nav)
    manifest(book_path)

    inventory = pd.DataFrame([
        {
            "model": name,
            "owner": "Project author",
            "status": "Research / not independently validated",
            "intended_use": use,
            "prohibited_use": "Live execution or regulatory capital",
        }
        for name, use in [
            ("Cashflow valuation", "Model-based fixed-income research"),
            ("Rolling VaR", "Historical forecast comparison"),
            ("Factor bootstrap", "Hypothetical market-shock risk"),
            ("Cash hedge optimizer", "Sensitivity reduction study"),
            ("Reverse stress", "Local loss-threshold scenario search"),
            ("ML quantiles", "Volatility interval research"),
            ("MC hedge RL", "Experimental policy evaluation"),
        ]
    ])
    save(inventory, "model_inventory.csv")

    (OUT / "LIMITATIONS.txt").write_text(
        DISCLAIMER + "\n"
        "Liquidity capacities and bid-ask inputs are assumptions.\n"
        "Cash-short hedges are continuous, not integer futures trades.\n"
        "PCA is full-sample descriptive analysis only.\n"
        "ES diagnostics are not a formal ES backtest.\n"
        "No production or regulatory approval is implied.\n",
        encoding="utf-8",
    )

    print(f"Advanced analytical outputs: {OUT}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()