from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

st.set_page_config(
    page_title="Bond Risk Research",
    layout="wide",
)

st.title("Bond Risk Research")
st.warning(
    "Experimental proxy-based results: historical P&L is a simplified "
    "rate-only book approximation. RL changes exposure, not explicit hedges."
)

st.sidebar.header("Display")
theme = st.sidebar.selectbox("Chart theme", ["Dark", "Light"])
template = "plotly_dark" if theme == "Dark" else "plotly_white"

if st.sidebar.button("Refresh saved results"):
    st.rerun()

st.sidebar.caption(
    "This app displays saved results. Refresh does not retrain models."
)


def read(relative, indexed=False, dates=False):
    path = OUTPUT / relative
    if not path.exists():
        st.info(f"Not available: {relative}")
        return None

    try:
        return pd.read_csv(
            path,
            index_col=0 if indexed else None,
            parse_dates=[0] if dates else False,
        )
    except Exception as error:
        st.error(f"Could not read {relative}: {error}")
        return None


def chart(fig):
    fig.update_layout(template=template)
    st.plotly_chart(fig, use_container_width=True)


def show_table(frame, name):
    if frame is None:
        return

    st.dataframe(frame, use_container_width=True)
    st.download_button(
        f"Download {name} CSV",
        data=frame.to_csv().encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
        key=f"download_{name}",
    )


portfolio, historical, ml, mc, rl, downloads = st.tabs([
    "Portfolio", "Historical risk", "ML",
    "Monte Carlo", "RL", "Reports & sources",
])

with portfolio:
    book = read("tables/book.csv", indexed=True)
    scenarios = read("tables/scenarios.csv", indexed=True)

    if book is not None:
        allocation = book.loc[book.index != "TOTAL"].reset_index()
        position = allocation.columns[0]

        chart(px.bar(
            allocation, x=position, y="weight", color="sleeve",
            title="Current book allocation",
        ))

    if scenarios is not None:
        scenario_data = scenarios.reset_index()
        chart(px.bar(
            scenario_data,
            x=scenario_data.columns[0],
            y="total_%NAV",
            title="Scenario P&L (% NAV)",
        ))

    show_table(book, "book")
    show_table(scenarios, "scenarios")

with historical:
    st.caption(
        "These historical estimates refer to the proxy return series, "
        "not exact historical P&L of the priced scenario book."
    )

    for name in ("latest_risk", "backtest"):
        st.subheader(name.replace("_", " ").title())
        show_table(read(f"tables/{name}.csv", indexed=True), name)

    worst = read("tables/worst_days.csv", indexed=True, dates=True)

    if worst is not None:
        data = worst.reset_index()
        data["return_pct"] = 100 * data["book_return"]

        chart(px.bar(
            data, x=data.columns[0], y="return_pct",
            title="Worst proxy-return days",
        ))
        show_table(worst, "worst_days")

    st.caption(
        "Zero historical scenario exceedances do not establish "
        "an infinite real-world return period."
    )

    for name in ("scenario_history_1d", "scenario_history_5d"):
        show_table(read(f"tables/{name}.csv", indexed=True), name)

with ml:
    st.subheader("Volatility forecast evaluation")
    show_table(read("ml/forecast_metrics.csv"), "ml_metrics")

    data = read("ml/test_forecasts.csv", indexed=True, dates=True)

    if data is not None:
        plot = data[["target", "forecast", "rms_20"]].copy() * 100
        plot.columns = ["Realized RMS", "ML forecast", "Trailing 20-day"]

        chart(px.line(
            plot,
            title="Held-out daily RMS return (%)",
        ))

        # Diagnostic only: these five-day targets overlap.
        squared_error = pd.DataFrame({
            "ML": (data["forecast"] - data["target"]) ** 2,
            "Trailing": (data["rms_20"] - data["target"]) ** 2,
        })
        rolling_rmse = squared_error.rolling(60).mean().pow(0.5)

        chart(px.line(
            rolling_rmse,
            title="Rolling 60-observation RMSE — diagnostic",
        ))

    importance = read("ml/feature_importance.csv", indexed=True)

    if importance is not None:
        chart(px.bar(
            importance.sort_values("importance"),
            x="importance", y=importance.index,
            orientation="h",
            title="Model feature importance — not causal effects",
        ))

with mc:
    st.subheader("Simulated risk")
    summary = read("monte_carlo/risk_summary.csv")
    show_table(summary, "monte_carlo_summary")

    st.caption(
        "Descriptive bootstrap results. ML scaling is experimental "
        "and is not an out-of-sample VaR coverage test."
    )

    horizon = st.selectbox("Simulation horizon", [5, 20])
    simulation_frames = []

    for method in ("historical_bootstrap", "ML_scaled_experiment"):
        path = OUTPUT / "monte_carlo" / f"{method}_{horizon}d_paths.csv"
        if path.exists():
            data = pd.read_csv(path)
            simulation_frames.append(pd.DataFrame({
                "Return_pct": 100 * data["terminal_return"],
                "Method": method,
            }))

    if simulation_frames:
        data = pd.concat(simulation_frames, ignore_index=True)
        chart(px.histogram(
            data, x="Return_pct", color="Method",
            nbins=70, barmode="overlay", opacity=0.55,
            title=f"{horizon}-day terminal return distributions",
        ))

with rl:
    st.subheader("Frozen exposure policies")
    metrics = read("rl/policy_metrics.csv")
    show_table(metrics, "rl_metrics")

    if metrics is not None:
        seeds = sorted(metrics["seed"].astype(int).unique().tolist())
        seed = st.selectbox(
            "Training seed — inspect all, do not cherry-pick",
            seeds,
            index=seeds.index(42) if 42 in seeds else 0,
        )

        paths = {}
        rl_path = None

        for strategy in ("MC_RL", "Full_book", "Fixed_75pct", "Vol_target"):
            path = OUTPUT / "rl" / f"{strategy}_{seed}_test.csv"

            if path.exists():
                data = pd.read_csv(
                    path, index_col=0, parse_dates=True
                )
                paths[strategy] = data["nav"]
                if strategy == "MC_RL":
                    rl_path = data

        if paths:
            chart(px.line(
                pd.DataFrame(paths),
                title="Held-out net equity — growth of 1",
            ))

        if rl_path is not None:
            chart(px.line(
                rl_path[["exposure"]],
                title="RL exposure to the original proxy book",
            ))
            chart(px.line(
                100 * rl_path[["drawdown"]],
                title="RL drawdown (%)",
            ))

        aggregate = metrics.groupby("strategy").agg(
            mean_return=("total_return_pct", "mean"),
            mean_drawdown=("max_drawdown_pct", "mean"),
            mean_exposure=("average_exposure", "mean"),
            mean_turnover=("total_turnover", "mean"),
        )
        show_table(aggregate, "rl_seed_averages")

    st.caption(
        "Reducing exposure can reduce losses without improving the "
        "investment policy. Compare return, drawdown and turnover together."
    )

with downloads:
    st.subheader("Generated reports")

    for name, mime in {
        "risk_report.pdf": "application/pdf",
        "risk_results.xlsx":
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "risk_deliverables.zip": "application/zip",
    }.items():
        path = OUTPUT / "reports" / name
        if path.exists():
            st.download_button(
                f"Download {name}",
                path.read_bytes(),
                file_name=name,
                mime=mime,
                key=name,
            )
        else:
            st.info(f"Run the reporting pipeline to create {name}.")

    st.subheader("Provenance")

    metadata_paths = [
        ROOT / "data/processed/learning_metadata.json",
        OUTPUT / "tables/manifest.json",
        OUTPUT / "ml/metadata.json",
        OUTPUT / "rl/metadata.json",
        OUTPUT / "reports/manifest.json",
    ]

    for path in metadata_paths:
        if path.exists():
            with st.expander(str(path.relative_to(ROOT))):
                st.json(json.loads(path.read_text(encoding="utf-8")))