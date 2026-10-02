"""Chronological ML forecasts with out-of-fold predictions for RL training."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/ml"
HORIZON = 5
SEED = 42


def make_model():
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=30,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data/processed/book_history.parquet")
    r = df["book_return"]

    features = pd.DataFrame(index=df.index)
    features["return_today"] = r
    features["return_lag1"] = r.shift(1)
    features["mean_5"] = r.rolling(5).mean()
    features["rms_5"] = np.sqrt(r.pow(2).rolling(5).mean())
    features["rms_20"] = np.sqrt(r.pow(2).rolling(20).mean())
    features["rms_60"] = np.sqrt(r.pow(2).rolling(60).mean())
    features["yield_10y"] = df["yield_10y"]
    features["slope"] = df["slope_10y_2y"]
    features["dy_10y"] = df["dy_10y"]

    # Prediction made after today's close for days t+1 ... t+5.
    future = pd.concat(
        [r.shift(-i).pow(2) for i in range(1, HORIZON + 1)],
        axis=1,
    )
    target = np.sqrt(future.mean(axis=1, skipna=False))

    data = features.copy()
    data["target"] = target
    data["next_return"] = r.shift(-1)
    data["next_cash"] = df["cash_return"].shift(-1)
    data = data.dropna()

    if len(data) < 1000:
        raise ValueError("Need at least 1,000 usable observations.")

    columns = features.columns.tolist()
    cut = int(len(data) * 0.70)

    # Purge overlapping target windows at the train/test boundary.
    train = data.iloc[:cut - HORIZON].copy()
    test = data.iloc[cut:].copy()

    oof = pd.Series(np.nan, index=train.index)
    splits = TimeSeriesSplit(n_splits=5, gap=HORIZON)

    for number, (fit_idx, valid_idx) in enumerate(splits.split(train), 1):
        model = make_model()
        model.fit(train.iloc[fit_idx][columns], train.iloc[fit_idx]["target"])
        oof.iloc[valid_idx] = model.predict(train.iloc[valid_idx][columns])
        print(f"Finished chronological fold {number}")

    model = make_model()
    model.fit(train[columns], train["target"])

    train["forecast"] = oof
    test["forecast"] = model.predict(test[columns])

    train.dropna(subset=["forecast"]).to_csv(OUT / "train_oof.csv")
    test.to_csv(OUT / "test_forecasts.csv")
    joblib.dump(model, OUT / "volatility_model.joblib")

    rows = []
    for name, prediction in {
        "Trailing_20d": test["rms_20"],
        "RandomForest": test["forecast"],
    }.items():
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(test["target"], prediction),
            "RMSE": np.sqrt(mean_squared_error(test["target"], prediction)),
        })

    pd.DataFrame(rows).to_csv(OUT / "forecast_metrics.csv", index=False)
    pd.Series(
        model.feature_importances_, index=columns, name="importance"
    ).sort_values(ascending=False).to_csv(OUT / "feature_importance.csv")

    metadata = {
        "target": "Next 5 trading days daily RMS return",
        "decision_time": "After current close",
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "purge_rows": HORIZON,
        "history_type": "APPROXIMATE_RATE_ONLY_BOOK_PROXY",
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()