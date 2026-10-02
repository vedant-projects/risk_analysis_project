from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit

from advanced.engine import historical_revaluation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/advanced"
DATA = ROOT / "data/advanced"

HORIZON = 5
SEEDS = (7, 42, 123)
EPISODES = 1500
EPISODE_LENGTH = 40

TRADING_COST = 0.0002
SHORT_FINANCING = 0.005
COLLATERAL_FRACTION = 0.10
MAX_COLLATERAL_FRACTION = 0.15


def quantile_model(alpha):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=25,
        random_state=42,
    )


def build_learning_data():
    history = pd.read_csv(
        OUT / "full_revaluation_history.csv",
        index_col=0, parse_dates=True,
    )
    markets = pd.read_csv(
        DATA / "market_factors.csv",
        index_col=0, parse_dates=True,
    )

    r = history["normalized_hpl"]

    features = pd.DataFrame(index=history.index)
    features["return"] = r
    features["lag_return"] = r.shift(1)
    features["rms20"] = r.pow(2).rolling(20).mean().pow(0.5)
    features["rms60"] = r.pow(2).rolling(60).mean().pow(0.5)
    features["mean20"] = r.rolling(20).mean()
    features["slope"] = markets["DGS10"] - markets["DGS2"]
    features["IG"] = markets["IG"]
    features["HY"] = markets["HY"]
    features["dy10"] = markets["DGS10"].diff()

    future = pd.concat(
        [r.shift(-i).pow(2) for i in range(1, HORIZON + 1)],
        axis=1,
    )

    features["target"] = future.mean(
        axis=1, skipna=False
    ).pow(0.5)

    # At date t, state uses t-close information; outcome is t+1.
    features["next_return"] = r.shift(-1)
    features["next_previous_value"] = history["previous_book_value"].shift(-1)

    return features.dropna(), markets


def train_ml(data):
    feature_names = [
        name for name in data.columns
        if name not in {
            "target", "next_return", "next_previous_value"
        }
    ]

    # Predeclared chronology; no selection on final test.
    train_end = int(len(data) * 0.60)
    validation_end = int(len(data) * 0.80)

    train = data.iloc[:train_end - HORIZON].copy()
    validation = data.iloc[
        train_end:validation_end - HORIZON
    ].copy()
    test = data.iloc[validation_end:].copy()

    oof = pd.Series(np.nan, index=train.index)

    for fit_idx, predict_idx in TimeSeriesSplit(
        n_splits=4, gap=HORIZON
    ).split(train):
        model = quantile_model(0.5)
        model.fit(
            train.iloc[fit_idx][feature_names],
            train.iloc[fit_idx]["target"],
        )
        oof.iloc[predict_idx] = model.predict(
            train.iloc[predict_idx][feature_names]
        )

    train["forecast"] = oof

    evaluations = []

    for alpha in (0.05, 0.50, 0.95):
        model = quantile_model(alpha)
        model.fit(train[feature_names], train["target"])

        for name, frame in (("validation", validation), ("test", test)):
            frame[f"q{alpha:.2f}"] = model.predict(frame[feature_names])

            evaluations.append({
                "period": name,
                "quantile": alpha,
                "pinball_loss": mean_pinball_loss(
                    frame["target"],
                    frame[f"q{alpha:.2f}"],
                    alpha=alpha,
                ),
            })

        joblib.dump(model, OUT / f"quantile_{alpha:.2f}.joblib")

    interval_rows = []

    for name, frame in (("validation", validation), ("test", test)):
        raw = frame[["q0.05", "q0.50", "q0.95"]].to_numpy()
        crossing = (np.diff(raw, axis=1) < 0).any(axis=1)

        # Explicit postprocessing; record quantile crossings.
        ordered = np.sort(raw, axis=1)
        frame["lower"] = ordered[:, 0]
        frame["forecast"] = ordered[:, 1]
        frame["upper"] = ordered[:, 2]

        interval_rows.append({
            "period": name,
            "nominal_coverage": 0.90,
            "observed_coverage": (
                (frame["target"] >= frame["lower"])
                & (frame["target"] <= frame["upper"])
            ).mean(),
            "mean_width": (frame["upper"] - frame["lower"]).mean(),
            "raw_crossing_fraction": crossing.mean(),
            "baseline_RMSE": np.sqrt(
                np.mean((frame["rms20"] - frame["target"]) ** 2)
            ),
            "ML_RMSE": np.sqrt(
                np.mean((frame["forecast"] - frame["target"]) ** 2)
            ),
        })

    drift = []
    for feature in feature_names:
        result = ks_2samp(train[feature], test[feature])
        drift.append({
            "feature": feature,
            "KS_statistic": result.statistic,
            "KS_pvalue": result.pvalue,
            "interpretation": (
                "Descriptive temporal drift; serial dependence and "
                "multiple testing limit p-value interpretation"
            ),
        })

    pd.DataFrame(evaluations).to_csv(
        OUT / "quantile_scores.csv", index=False
    )
    pd.DataFrame(interval_rows).to_csv(
        OUT / "interval_evaluation.csv", index=False
    )
    pd.DataFrame(drift).to_csv(OUT / "feature_drift.csv", index=False)

    train.dropna(subset=["forecast"]).to_csv(OUT / "ml_train_oof.csv")
    validation.to_csv(OUT / "ml_validation.csv")
    test.to_csv(OUT / "ml_test.csv")

    return train.dropna(subset=["forecast"]), validation, test


def hedge_returns(data, markets):
    book = pd.read_csv(OUT / "instrument_master.csv")

    government = book.loc[book["credit_bucket"] == "GOV"].copy()
    government["years"] = (
        pd.to_datetime(government["maturity_date"])
        - markets.index[-1]
    ).dt.days / 365

    reference_face = float(book["face"].sum())

    # Four auditable templates: none, short belly, short long, blended.
    templates = [[], [], [], []]

    belly = government.iloc[
        (government["years"] - 5).abs().argmin()
    ].copy()
    long = government.iloc[
        (government["years"] - 10).abs().argmin()
    ].copy()

    belly["face"] = reference_face * 0.25
    long["face"] = reference_face * 0.25
    templates[1] = [belly]
    templates[2] = [long]

    blend1 = belly.copy()
    blend2 = long.copy()
    blend1["face"] = reference_face * 0.125
    blend2["face"] = reference_face * 0.125
    templates[3] = [blend1, blend2]

    pnl = pd.DataFrame(0.0, index=data.index, columns=range(4))
    gross = np.zeros(4)
    template_rows = []

    for action in range(1, 4):
        hedge_book = pd.DataFrame(templates[action])
        gross[action] = hedge_book["face"].sum()

        hedge_history, _ = historical_revaluation(hedge_book, markets)

        # Shorting the cash bonds reverses total modeled bond P&L.
        # Financing surcharge is an explicit fixed assumption.
        short_pnl = (
            -hedge_history["total_pnl"]
            - gross[action] * SHORT_FINANCING / 252
        )

        pnl[action] = (
            short_pnl.shift(-1).reindex(data.index)
            / data["next_previous_value"]
        )

        for _, bond in hedge_book.iterrows():
            template_rows.append({
                "action": action,
                "id": bond["id"],
                "short_face": bond["face"],
            })

    pd.DataFrame(template_rows).to_csv(
        OUT / "rl_hedge_templates.csv", index=False
    )

    if pnl.isna().any().any():
        raise ValueError("Hedge outcomes are not aligned with state dates.")

    return pnl, gross


def train_rl(train, hedge_pnl, gross, bins, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((3, 2, 4, 4))
    counts = np.zeros_like(q)
    allowed = np.flatnonzero(
        gross * COLLATERAL_FRACTION
        <= train["next_previous_value"].min() * MAX_COLLATERAL_FRACTION
    )

    def state(row, previous):
        return (
            int(np.digitize(row.forecast, bins)),
            int(row.slope > 0),
            previous,
        )

    def choose(s):
        scores = q[s][allowed]
        candidates = allowed[scores == scores.max()]
        return int(candidates[0])

    dates = train.index.to_numpy()

    starts = [
        i for i in range(len(train) - EPISODE_LENGTH + 1)
        if np.max(np.diff(dates[i:i + EPISODE_LENGTH]))
        <= np.timedelta64(7, "D")
    ]

    if not starts:
        raise ValueError("Not enough contiguous OOF history for RL.")

    for episode_number in range(EPISODES):
        start = int(rng.choice(starts))
        episode = train.iloc[start:start + EPISODE_LENGTH]

        trajectory = []
        previous = 0

        for row in episode.itertuples():
            s = state(row, previous)
            action = (
                int(rng.choice(allowed))
                if rng.random() < 0.15 else choose(s)
            )

            # Conservative switch cost: close old and open new template.
            traded = 0 if action == previous else gross[action] + gross[previous]
            cost = TRADING_COST * traded / row.next_previous_value

            net = (
                row.next_return
                + hedge_pnl.loc[row.Index, action]
                - cost
            )

            reward = net - 0.5 * max(-net, 0)
            trajectory.append((s, action, reward))
            previous = action

        # Charge terminal hedge close to the final reward.
        last = episode.iloc[-1]
        s, a, reward = trajectory[-1]
        close_cost = TRADING_COST * gross[previous] / last.next_previous_value
        trajectory[-1] = (s, a, reward - close_cost)

        returns = np.zeros(len(trajectory))
        cumulative = 0.0

        for i in range(len(trajectory) - 1, -1, -1):
            cumulative += trajectory[i][2]
            returns[i] = cumulative

        seen = set()
        for i, (s, action, _) in enumerate(trajectory):
            key = s + (action,)
            if key in seen:
                continue

            seen.add(key)
            counts[key] += 1
            q[key] += (returns[i] - q[key]) / counts[key]

    return q, counts, allowed


def evaluate(frame, hedge_pnl, gross, bins, q, allowed, strategy):
    previous = 0
    rows = []

    for row in frame.itertuples():
        s = (
            int(np.digitize(row.forecast, bins)),
            int(row.slope > 0),
            previous,
        )

        if strategy == "MC_RL":
            permitted = [
                a for a in allowed
                if gross[a] * COLLATERAL_FRACTION
                <= row.next_previous_value * MAX_COLLATERAL_FRACTION
            ]
            action = int(max(permitted, key=lambda a: q[s][a]))
        elif strategy == "No_hedge":
            action = 0
        elif strategy == "Static_long":
            action = 2 if 2 in allowed else 0
        else:
            action = 2 if row.forecast > bins[-1] and 2 in allowed else 0

        traded = 0 if action == previous else gross[action] + gross[previous]
        cost = TRADING_COST * traded / row.next_previous_value
        hedge = float(hedge_pnl.loc[row.Index, action])
        net = row.next_return + hedge - cost

        rows.append({
            "date": row.Index,
            "action": action,
            "base_return": row.next_return,
            "hedge_return": hedge,
            "cost": cost,
            "net_normalized_hpl": net,
            "turnover_usd": traded,
            "downside_penalty": 0.5 * max(-net, 0),
        })
        previous = action

    path = pd.DataFrame(rows).set_index("date")

    # Terminal closure for fair benchmark comparison.
    close = TRADING_COST * gross[previous] / frame.iloc[-1]["next_previous_value"]
    path.iloc[-1, path.columns.get_loc("cost")] += close
    path.iloc[-1, path.columns.get_loc("net_normalized_hpl")] -= close

    # Sum normalized HPL: not an investable compounded equity curve.
    path["cumulative_normalized_hpl"] = path["net_normalized_hpl"].cumsum()

    metrics = {
        "strategy": strategy,
        "sum_normalized_hpl": path["net_normalized_hpl"].sum(),
        "daily_std": path["net_normalized_hpl"].std(),
        "worst_day": path["net_normalized_hpl"].min(),
        "total_cost": path["cost"].sum(),
        "turnover_usd": path["turnover_usd"].sum(),
        "interpretation": "Normalized fixed-book HPL, not a funded NAV return",
    }

    return path, metrics


def main():
    data, markets = build_learning_data()
    train, validation, test = train_ml(data)

    # Validation is reported, not used here to tune predeclared parameters.
    hedge_pnl, gross = hedge_returns(data, markets)
    bins = np.quantile(train["forecast"], [1 / 3, 2 / 3])
    all_metrics = []

    for seed in SEEDS:
        q, counts, allowed = train_rl(
            train, hedge_pnl, gross, bins, seed
        )

        np.savez(
            OUT / f"hedge_agent_{seed}.npz",
            q=q, counts=counts, bins=bins, allowed=allowed,
        )

        for period, frame in (("validation", validation), ("test", test)):
            for strategy in (
                "MC_RL", "No_hedge", "Static_long", "Vol_rule"
            ):
                path, metrics = evaluate(
                    frame, hedge_pnl, gross, bins,
                    q, allowed, strategy,
                )
                metrics.update({"period": period, "seed": seed})
                all_metrics.append(metrics)
                path.to_csv(
                    OUT / f"policy_{strategy}_{seed}_{period}.csv"
                )

    pd.DataFrame(all_metrics).to_csv(
        OUT / "advanced_policy_metrics.csv", index=False
    )

    (OUT / "learning_model_card.json").write_text(
        json.dumps({
            "target": "Next-five-day RMS normalized hypothetical P&L",
            "decision_time": "Current close; outcomes begin next observation",
            "train_end": str(train.index.max()),
            "validation_start": str(validation.index.min()),
            "validation_end": str(validation.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
            "purge": HORIZON,
            "RL": "First-visit Monte Carlo control",
            "actions": "Fixed cash-bond short templates",
            "short_financing": SHORT_FINANCING,
            "trading_cost": TRADING_COST,
            "limitations": [
                "Zero-rate and spread-proxy valuation",
                "Not a Markov-state guarantee",
                "No futures basis, rolls, borrow recalls or variation margin",
                "Fixed collateral rule, not broker margin",
                "No actual portfolio NAV or investable return claim",
                "Intervals are evaluated, not guaranteed calibrated",
                "No independent model validation",
            ],
        }, indent=2),
        encoding="utf-8",
    )

    print("Advanced ML and hedge-RL outputs created.")


if __name__ == "__main__":
    main()