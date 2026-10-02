"""Historical block-bootstrap Monte Carlo, plus an ML-scaled experiment."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/monte_carlo"

N_PATHS = 10_000
BLOCK = 5
HORIZONS = (5, 20)
SEED = 42


def bootstrap(values, horizon, rng):
    chunks = []
    for _ in range((horizon + BLOCK - 1) // BLOCK):
        starts = rng.integers(0, len(values) - BLOCK + 1, N_PATHS)
        chunks.append(values[starts[:, None] + np.arange(BLOCK)])
    return np.concatenate(chunks, axis=1)[:, :horizon]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history = pd.read_parquet(ROOT / "data/processed/book_history.parquet")
    forecasts = pd.read_csv(
        ROOT / "output/ml/test_forecasts.csv",
        index_col=0,
        parse_dates=True,
    )

    asof = forecasts.index[-1]
    # Exclude anything beyond the forecast's decision date.
    observed = history.loc[:asof, "book_return"].dropna().to_numpy()
    target_rms = float(forecasts["forecast"].iloc[-1])
    historical_rms = np.sqrt(np.mean(observed ** 2))
    scale = target_rms / max(historical_rms, 1e-10)

    rng = np.random.default_rng(SEED)
    summaries = []

    for horizon in HORIZONS:
        base = bootstrap(observed, horizon, rng)
        mean = observed.mean()

        variants = {
            "historical_bootstrap": base,
            "ML_scaled_experiment": mean + (base - mean) * scale,
        }

        for name, daily in variants.items():
            terminal = np.prod(1 + daily, axis=1) - 1
            losses = -terminal

            pd.DataFrame({
                "path": np.arange(N_PATHS),
                "terminal_return": terminal,
                "terminal_loss": losses,
            }).to_csv(OUT / f"{name}_{horizon}d_paths.csv", index=False)

            for alpha in (0.95, 0.99):
                var = np.quantile(losses, alpha)
                es = losses[losses >= var].mean()
                summaries.append({
                    "method": name,
                    "horizon_days": horizon,
                    "confidence": alpha,
                    "VaR_pct": 100 * var,
                    "ES_pct": 100 * es,
                    "prob_loss_gt_3pct": np.mean(losses > 0.03),
                    "mean_return_pct": 100 * terminal.mean(),
                    "paths": N_PATHS,
                    "asof": str(asof.date()),
                    "ML_RMS_scale": scale if name.startswith("ML") else 1.0,
                })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "risk_summary.csv", index=False)

    (OUT / "notes.txt").write_text(
        "Approximate rate-only book proxy.\n"
        "Historical results are descriptive, not an out-of-sample MC backtest.\n"
        "ML scaling is an experimental volatility adjustment.\n"
        "The 5-day forecast is held constant even for the 20-day simulation.\n"
        "No unseen crisis shocks or explicit hedge pricing are modeled.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()