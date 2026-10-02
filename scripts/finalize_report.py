from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, LongTable,
    TableStyle, Image, PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
REPORTS = OUTPUT / "reports"
CHARTS = OUTPUT / "charts"

WARNING = (
    "EXPERIMENTAL RESEARCH OUTPUT. Historical portfolio returns are an "
    "approximate rate-only book proxy, not exact historical portfolio P&L. "
    "Credit spreads, MBS optionality, FX and liquidity risks are omitted "
    "from that proxy. Scenario repricing is a separate calculation. "
    "RL controls exposure; it does not implement an instrument-level hedge."
)

LIMITATIONS = [
    WARNING,
    "ML predicts next-five-day daily RMS returns, not expected returns.",
    "The ML-scaled Monte Carlo model is an experimental adjustment.",
    "The five-day volatility forecast is held constant for 20-day paths.",
    "Monte Carlo distributions are descriptive, not coverage backtests.",
    "RL evaluation uses held-out history and simplified transaction costs.",
    "Multiple RL seeds are shown; no best seed is selected for reporting.",
    "No guarantee of predictive skill or profitable deployment is implied.",
]

TABLE_FILES = {
    "Portfolio": ("tables/book.csv", True),
    "Scenarios": ("tables/scenarios.csv", True),
    "Historical risk": ("tables/latest_risk.csv", True),
    "VaR backtest": ("tables/backtest.csv", True),
    "Worst days": ("tables/worst_days.csv", True),
    "Scenario history 1d": ("tables/scenario_history_1d.csv", True),
    "Scenario history 5d": ("tables/scenario_history_5d.csv", True),
    "ML metrics": ("ml/forecast_metrics.csv", False),
    "ML importance": ("ml/feature_importance.csv", True),
    "Monte Carlo": ("monte_carlo/risk_summary.csv", False),
    "RL metrics": ("rl/policy_metrics.csv", False),
}

METADATA_FILES = [
    ROOT / "data/processed/learning_metadata.json",
    OUTPUT / "tables/manifest.json",
    OUTPUT / "ml/metadata.json",
    OUTPUT / "rl/metadata.json",
]


def read_tables():
    tables = {}
    missing = []

    for title, (relative, indexed) in TABLE_FILES.items():
        path = OUTPUT / relative

        if not path.exists():
            missing.append(relative)
            continue

        frame = pd.read_csv(path, index_col=0 if indexed else None)

        if frame.empty:
            missing.append(relative)
            continue

        if indexed:
            frame = frame.reset_index()

        tables[title] = frame

    return tables, missing


def save_chart(fig, name, data):
    png = CHARTS / f"{name}.png"
    csv = CHARTS / f"{name}.csv"

    fig.tight_layout()
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(csv, index=False)

    return png, csv


def make_charts(tables):
    artifacts = []

    if "Scenarios" in tables:
        data = tables["Scenarios"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(data.iloc[:, 0].astype(str), data["total_%NAV"],
               color="#f97316")
        ax.set_title("Scenario P&L — current curve")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("P&L (% NAV)")
        ax.axhline(0, color="black", linewidth=0.7)
        artifacts.extend(save_chart(fig, "scenario_pnl", data))

    forecast_path = OUTPUT / "ml/test_forecasts.csv"

    if forecast_path.exists():
        data = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        plot_data = data[["target", "forecast"]].copy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_data.index, 100 * plot_data["target"],
                label="Realized 5-day RMS", alpha=0.65)
        ax.plot(plot_data.index, 100 * plot_data["forecast"],
                label="ML forecast", color="#f97316")
        ax.set_title("ML forecast — held-out proxy history")
        ax.set_xlabel("Decision date")
        ax.set_ylabel("Daily RMS (%)")
        ax.legend()

        artifacts.extend(
            save_chart(fig, "ml_forecast", plot_data.reset_index())
        )

    paths = sorted((OUTPUT / "monte_carlo").glob("*_20d_paths.csv"))

    if paths:
        fig, ax = plt.subplots(figsize=(9, 4))
        combined = []

        for path in paths:
            data = pd.read_csv(path)
            label = path.stem.replace("_20d_paths", "")
            ax.hist(
                100 * data["terminal_return"],
                bins=70,
                alpha=0.45,
                density=True,
                label=label,
            )
            combined.append(data.assign(method=label))

        ax.set_title("20-day simulated returns — proxy portfolio")
        ax.set_xlabel("Terminal return (%)")
        ax.set_ylabel("Density")
        ax.legend()

        artifacts.extend(
            save_chart(
                fig, "mc_distribution",
                pd.concat(combined, ignore_index=True),
            )
        )

    equity_files = sorted((OUTPUT / "rl").glob("*_test.csv"))

    if equity_files:
        fig, ax = plt.subplots(figsize=(10, 4))
        combined = []

        for path in equity_files:
            name = path.stem.replace("_test", "")
            strategy, seed = name.rsplit("_", 1)
            data = pd.read_csv(path, index_col=0, parse_dates=True)

            # Show all RL seeds; identical baselines need only one line.
            if strategy != "MC_RL" and seed != "42":
                continue

            ax.plot(
                data.index, data["nav"],
                label=name if strategy == "MC_RL" else strategy,
                alpha=0.8,
            )
            combined.append(
                data.reset_index().assign(strategy=strategy, seed=seed)
            )

        ax.set_title("Frozen policies — held-out proxy history")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of 1")
        ax.legend()

        artifacts.extend(
            save_chart(
                fig, "policy_equity",
                pd.concat(combined, ignore_index=True),
            )
        )

    return artifacts


def make_findings(tables):
    findings = []

    if "ML metrics" in tables:
        metrics = tables["ML metrics"].set_index("model")
        if {"RandomForest", "Trailing_20d"} <= set(metrics.index):
            ml = float(metrics.loc["RandomForest", "RMSE"])
            baseline = float(metrics.loc["Trailing_20d", "RMSE"])
            outcome = "lower" if ml < baseline else "not lower"
            findings.append(
                f"ML held-out RMSE is {ml:.6f}; trailing-volatility "
                f"RMSE is {baseline:.6f}. ML error is {outcome}."
            )

    if "RL seed summary" in tables:
        findings.append(
            "RL seed summaries are descriptive averages and standard "
            "deviations across training seeds, not confidence intervals."
        )

    findings.append(
        "Reduced exposure can lower drawdown without demonstrating "
        "a superior policy; compare return, downside risk and turnover."
    )
    return findings


def formatted(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(value)
        return f"{value:,.6g}"
    return str(value)


def make_excel(tables, chart_files, findings):
    path = REPORTS / "risk_results.xlsx"

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book

        warning_format = workbook.add_format({
            "bold": True,
            "font_color": "#9C0006",
            "bg_color": "#FFC7CE",
            "text_wrap": True,
            "valign": "top",
        })

        overview = workbook.add_worksheet("Overview")
        overview.set_column("A:A", 105)
        overview.write("A1", "Experimental Bond Risk Research")
        overview.merge_range("A3:D7", WARNING, warning_format)
        overview.write("A9", "Findings")

        for row, text in enumerate(findings, 10):
            overview.write(row, 0, text)

        overview.write(
            16, 0,
            "Static export: rerun the pipeline to refresh results."
        )

        if chart_files:
            overview.insert_image(
                "A20", str(chart_files[0]),
                {"x_scale": 0.6, "y_scale": 0.6},
            )

        for title, frame in tables.items():
            sheet = title[:31]
            frame.to_excel(
                writer, sheet_name=sheet,
                index=False, inf_rep="inf",
            )
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            ws.set_column(0, len(frame.columns) - 1, 20)

            for col, name in enumerate(frame.columns):
                if pd.api.types.is_numeric_dtype(frame[name]):
                    ws.conditional_format(
                        1, col, len(frame), col,
                        {"type": "3_color_scale"},
                    )

    return path


def make_pdf(tables, chart_files, findings, missing):
    path = REPORTS / "risk_report.pdf"
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"].clone("SmallCell")
    cell_style.fontSize = 7
    cell_style.leading = 9

    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4),
        leftMargin=30, rightMargin=30,
        topMargin=30, bottomMargin=30,
    )

    story = [
        Paragraph("Bond Risk, ML and Monte Carlo RL", styles["Title"]),
        Paragraph(
            escape(datetime.now(timezone.utc).isoformat()),
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(escape(WARNING), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Findings", styles["Heading2"]),
    ]

    for finding in findings:
        story.extend([
            Paragraph(escape(finding), styles["BodyText"]),
            Spacer(1, 6),
        ])

    story.append(Paragraph("Method limitations", styles["Heading2"]))

    for text in LIMITATIONS:
        story.extend([
            Paragraph(escape(text), styles["BodyText"]),
            Spacer(1, 6),
        ])

    if missing:
        story.append(
            Paragraph(
                escape("Missing outputs: " + ", ".join(missing)),
                styles["BodyText"],
            )
        )

    story.append(PageBreak())

    # Split wide tables into column groups instead of shrinking unreadably.
    for title, frame in tables.items():
        story.append(Paragraph(escape(title), styles["Heading1"]))

        columns = frame.columns.tolist()
        groups = (
            [columns] if len(columns) <= 7 else
            [[columns[0]] + columns[i:i + 6]
             for i in range(1, len(columns), 6)]
        )

        for group in groups:
            subset = frame[group]
            rows = [group] + subset.values.tolist()

            cells = [
                [
                    Paragraph(escape(formatted(value)), cell_style)
                    for value in row
                ]
                for row in rows
            ]

            table = LongTable(
                cells,
                repeatRows=1,
                colWidths=[doc.width / len(group)] * len(group),
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FED7AA")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))

            story.extend([table, Spacer(1, 12)])

        story.append(PageBreak())

    for png in chart_files:
        story.extend([
            Paragraph(
                escape(png.stem.replace("_", " ").title()),
                styles["Heading1"],
            ),
            Image(str(png), width=650, height=280, kind="proportional"),
            PageBreak(),
        ])

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(30, 15, "Experimental proxy-based research")
        canvas.drawRightString(
            landscape(A4)[0] - 30, 15,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    tables, missing = read_tables()
    if not tables:
        raise RuntimeError("No results found. Run the analytical scripts first.")

    if "RL metrics" in tables:
        metrics = tables["RL metrics"]
        columns = [
            name for name in metrics.select_dtypes(include="number").columns
            if name != "seed"
        ]
        summary = metrics.groupby("strategy")[columns].agg(["mean", "std"])
        summary.columns = [
            f"{metric}_{stat}" for metric, stat in summary.columns
        ]
        tables["RL seed summary"] = summary.reset_index()
        tables["RL seed summary"].to_csv(
            REPORTS / "rl_seed_summary.csv", index=False
        )

    findings = make_findings(tables)
    artifacts = make_charts(tables)
    pngs = [path for path in artifacts if path.suffix == ".png"]

    excel = make_excel(tables, pngs, findings)
    pdf = make_pdf(tables, pngs, findings, missing)

    provenance = {}
    for path in METADATA_FILES:
        if path.exists():
            provenance[str(path.relative_to(ROOT))] = json.loads(
                path.read_text(encoding="utf-8")
            )

    manifest = REPORTS / "manifest.json"
    manifest.write_text(
        json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "warning": WARNING,
            "findings": findings,
            "missing": missing,
            "provenance": provenance,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    sources = [
        OUTPUT / relative
        for relative, _ in TABLE_FILES.values()
        if (OUTPUT / relative).exists()
    ]

    files = [
        excel, pdf, manifest,
        *artifacts, *sources,
        *[path for path in METADATA_FILES if path.exists()],
    ]

    summary_path = REPORTS / "rl_seed_summary.csv"
    if "RL seed summary" in tables:
        files.append(summary_path)

    archive = REPORTS / "risk_deliverables.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as zipped:
        for path in dict.fromkeys(files):
            zipped.write(path, arcname=str(path.relative_to(ROOT)))

    print(f"PDF:   {pdf}")
    print(f"Excel: {excel}")
    print(f"ZIP:   {archive}")

    if missing:
        print("Missing outputs:", ", ".join(missing))


if __name__ == "__main__":
    main()