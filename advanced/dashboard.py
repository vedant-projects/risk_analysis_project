from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/advanced"

st.set_page_config(page_title="Advanced Bond Risk", layout="wide")
st.title("Advanced Bond Risk Laboratory")

st.warning(
    "Model-based fixed-book research. CMT zero-rate proxies and index "
    "spread proxies are not calibrated market prices. No live-trading "
    "or regulatory-compliance claims."
)


def load(name, dated=False):
    path = OUT / name
    if not path.exists():
        st.info(f"Missing: {name}")
        return None

    if dated:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.read_csv(path)


def table(name):
    frame = load(name)
    if frame is not None:
        st.dataframe(frame, use_container_width=True)
        st.download_button(
            f"Download {name}",
            frame.to_csv(index=False).encode(),
            file_name=name,
            key=name,
        )


valuation, attribution, risk, stress, hedges, learning, audit = st.tabs([
    "Valuation", "P&L explain", "Risk validation",
    "Stress & liquidity", "Hedges", "ML & RL", "Audit",
])

with valuation:
    table("bond_metrics.csv")
    table("concentration.csv")
    table("sensitivities.csv")

with attribution:
    history = load("full_revaluation_history.csv", dated=True)
    if history is not None:
        columns = [
            "carry_roll", "rate_explain", "IG_explain",
            "HY_explain", "residual",
        ]
        st.plotly_chart(
            px.line(
                history[columns].cumsum(),
                title="Cumulative modeled P&L explain",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(history[["residual"]], title="Daily attribution residual"),
            use_container_width=True,
        )

with risk:
    table("risk_validation.csv")
    table("mc_summary.csv")

    forecasts = load("risk_forecasts.csv")
    if forecasts is not None:
        forecasts["date"] = pd.to_datetime(forecasts["date"])
        model = st.selectbox("Risk model", forecasts["model"].unique())
        subset = forecasts.loc[forecasts["model"] == model]

        st.plotly_chart(
            px.line(
                subset, x="date",
                y=["VaR", "ES", "realized_loss"],
                title="One-day forecasts and realized normalized loss",
            ),
            use_container_width=True,
        )

    convergence = load("mc_convergence.csv")
    if convergence is not None:
        st.plotly_chart(
            px.line(
                convergence, x="paths", y="VaR99",
                color="horizon", markers=True,
                title="Monte Carlo convergence diagnostic",
            ),
            use_container_width=True,
        )

    st.caption(
        "ES tail summaries are descriptive diagnostics, not formal ES tests."
    )

with stress:
    table("stress_results.csv")
    table("reverse_stress_result.csv")
    table("reverse_stress_shock.csv")
    table("liquidity_positions.csv")
    table("liquidity_stress.csv")

with hedges:
    table("hedge_trades.csv")
    risk_table = load("hedge_risk.csv")

    if risk_table is not None:
        st.plotly_chart(
            px.bar(
                risk_table, x="factor",
                y=["before_usd_per_bp", "after_usd_per_bp"],
                barmode="group",
                title="Cash-short hedge sensitivity comparison",
            ),
            use_container_width=True,
        )

    st.caption(
        "Continuous cash-bond shorts; no futures conversion, borrow "
        "availability, basis risk or execution guarantee."
    )

with learning:
    table("interval_evaluation.csv")
    table("quantile_scores.csv")
    table("feature_drift.csv")
    table("advanced_policy_metrics.csv")

    forecasts = load("ml_test.csv", dated=True)
    if forecasts is not None:
        st.plotly_chart(
            px.line(
                forecasts[["target", "forecast", "lower", "upper"]],
                title="Held-out RMS forecast and interval",
            ),
            use_container_width=True,
        )

    seed = st.selectbox("RL seed", [7, 42, 123])
    paths = {}

    for strategy in ("MC_RL", "No_hedge", "Static_long", "Vol_rule"):
        name = f"policy_{strategy}_{seed}_test.csv"
        path = OUT / name

        if path.exists():
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            paths[strategy] = frame["cumulative_normalized_hpl"]

    if paths:
        st.plotly_chart(
            px.line(
                pd.DataFrame(paths),
                title="Cumulative normalized HPL — not funded NAV",
            ),
            use_container_width=True,
        )

with audit:
    table("model_inventory.csv")

    for name in ("manifest.json", "learning_model_card.json"):
        path = OUT / name
        if path.exists():
            with st.expander(name):
                st.json(json.loads(path.read_text()))

    for name in ("advanced_report.pdf", "advanced_results.xlsx"):
        path = OUT / "reports" / name
        if path.exists():
            st.download_button(
                f"Download {name}",
                path.read_bytes(),
                file_name=name,
                key=f"report_{name}",
            )

    path = OUT / "LIMITATIONS.txt"
    if path.exists():
        st.text(path.read_text())