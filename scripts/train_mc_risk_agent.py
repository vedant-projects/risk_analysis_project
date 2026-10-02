"""First-visit Monte Carlo exposure control, evaluated on held-out history."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/rl"

ACTIONS = np.array([0.50, 0.75, 1.00])
SEEDS = (7, 42, 123)
EPISODES = 3000
EPISODE_LENGTH = 60
EPSILON = 0.15
COST = 0.0002
DOWNSIDE_PENALTY = 0.5


def net_return(row, action, previous):
    exposure = ACTIONS[action]
    turnover = abs(exposure - ACTIONS[previous])
    value = (
        exposure * row.next_return
        + (1 - exposure) * row.next_cash
        - COST * turnover
    )
    return value, turnover


def state(row, previous, drawdown, bins):
    vol = int(np.digitize(row.forecast, bins))
    slope = int(row.slope > 0)
    stress = int(drawdown < -0.02)
    return vol, slope, previous, stress


def greedy(q, s):
    # Default unseen/tied states to full exposure.
    return int(np.flatnonzero(q[s] == q[s].max())[-1])


def train_agent(train, bins, seed):
    rng = np.random.default_rng(seed)
    q = np.zeros((3, 2, 3, 2, 3))
    counts = np.zeros_like(q)
    curve = []

    dates = train.index.to_numpy()
    # Exclude episode starts crossing gaps between OOF blocks.
    starts = [
        i for i in range(len(train) - EPISODE_LENGTH + 1)
        if np.max(np.diff(dates[i:i + EPISODE_LENGTH]))
        <= np.timedelta64(7, "D")
    ]
    if not starts:
        raise ValueError("Not enough contiguous OOF training history.")

    for episode_number in range(EPISODES):
        start = int(rng.choice(starts))
        episode = train.iloc[start:start + EPISODE_LENGTH]

        previous = 2
        nav = peak = 1.0
        trajectory = []

        for row in episode.itertuples():
            s = state(row, previous, nav / peak - 1, bins)
            action = (
                int(rng.integers(3))
                if rng.random() < EPSILON else greedy(q, s)
            )

            value, _ = net_return(row, action, previous)
            reward = value - DOWNSIDE_PENALTY * max(-value, 0)

            trajectory.append((s, action, reward))
            nav *= 1 + value
            peak = max(peak, nav)
            previous = action

        returns = np.zeros(len(trajectory))
        total = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            total = trajectory[t][2] + total
            returns[t] = total

        seen = set()
        for t, (s, action, _) in enumerate(trajectory):
            pair = (s, action)
            if pair in seen:
                continue
            seen.add(pair)
            key = s + (action,)
            counts[key] += 1
            q[key] += (returns[t] - q[key]) / counts[key]

        curve.append({
            "episode": episode_number,
            "net_return": nav - 1,
            "total_reward": returns[0],
        })

    return q, counts, pd.DataFrame(curve)


def evaluate(test, q, bins, target_vol, strategy):
    previous = 2
    nav = peak = 1.0
    rows = []

    for row in test.itertuples():
        if strategy == "MC_RL":
            action = greedy(q, state(row, previous, nav / peak - 1, bins))
        elif strategy == "Full_book":
            action = 2
        elif strategy == "Fixed_75pct":
            action = 1
        else:
            desired = np.clip(target_vol / max(row.forecast, 1e-10), 0.5, 1)
            action = int(np.argmin(abs(ACTIONS - desired)))

        value, turnover = net_return(row, action, previous)
        nav *= 1 + value
        peak = max(peak, nav)

        rows.append({
            "date": row.Index,
            "net_return": value,
            "nav": nav,
            "drawdown": nav / peak - 1,
            "exposure": ACTIONS[action],
            "turnover": turnover,
        })
        previous = action

    path = pd.DataFrame(rows).set_index("date")
    losses = -path["net_return"]
    var = losses.quantile(0.99)
    years = len(path) / 252

    metrics = {
        "strategy": strategy,
        "total_return_pct": 100 * (path["nav"].iloc[-1] - 1),
        "annualized_return_pct": 100 * (
            path["nav"].iloc[-1] ** (1 / years) - 1
        ),
        "max_drawdown_pct": 100 * path["drawdown"].min(),
        "daily_VaR99_pct": 100 * var,
        "daily_ES99_pct": 100 * losses[losses >= var].mean(),
        "average_exposure": path["exposure"].mean(),
        "total_turnover": path["turnover"].sum(),
    }
    return path, metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        ROOT / "output/ml/train_oof.csv", index_col=0, parse_dates=True
    )
    test = pd.read_csv(
        ROOT / "output/ml/test_forecasts.csv", index_col=0, parse_dates=True
    )

    bins = np.quantile(train["forecast"], [1 / 3, 2 / 3])
    target_vol = float(train["forecast"].median())

    results = []
    for seed in SEEDS:
        q, counts, learning = train_agent(train, bins, seed)
        np.savez(OUT / f"agent_{seed}.npz", q=q, counts=counts, bins=bins)
        learning.to_csv(OUT / f"training_{seed}.csv", index=False)

        for strategy in ("MC_RL", "Full_book", "Fixed_75pct", "Vol_target"):
            path, metrics = evaluate(test, q, bins, target_vol, strategy)
            metrics["seed"] = seed
            results.append(metrics)
            path.to_csv(OUT / f"{strategy}_{seed}_test.csv")

    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "policy_metrics.csv", index=False)

    metadata = {
        "history": "APPROXIMATE_RATE_ONLY_BOOK_PROXY",
        "algorithm": "First-visit on-policy Monte Carlo control",
        "training": "Random contiguous historical OOF episodes",
        "evaluation": "Frozen policy on chronological held-out history",
        "actions": ACTIONS.tolist(),
        "cost_per_unit_exposure_change": COST,
        "downside_penalty": DOWNSIDE_PENALTY,
        "limitations": [
            "Approximate discretized state; not guaranteed Markov",
            "Overlapping historical training episodes are not independent",
            "No explicit instrument-level hedging",
            "No automatic hyperparameter tuning on the test period",
            "Terminal liquidation costs are not included",
        ],
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()