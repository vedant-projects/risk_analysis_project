from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, LongTable,
    TableStyle, PageBreak, Image,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/advanced"
REPORTS = OUT / "reports"

REPORT_TABLES = [
    "bond_metrics.csv",
    "concentration.csv",
    "risk_validation.csv",
    "mc_summary.csv",
    "hedge_trades.csv",
    "hedge_risk.csv",
    "stress_results.csv",
    "reverse_stress_result.csv",
    "liquidity_stress.csv",
    "interval_evaluation.csv",
    "advanced_policy_metrics.csv",
    "model_inventory.csv",
]


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    tables = {
        name: pd.read_csv(OUT / name)
        for name in REPORT_TABLES if (OUT / name).exists()
    }

    if not tables:
        raise RuntimeError("No advanced outputs to report.")

    warning = (OUT / "LIMITATIONS.txt").read_text(encoding="utf-8")

    with pd.ExcelWriter(
        REPORTS / "advanced_results.xlsx", engine="xlsxwriter"
    ) as writer:
        pd.DataFrame({"limitations": warning.splitlines()}).to_excel(
            writer, sheet_name="Limitations", index=False
        )

        for name, frame in tables.items():
            sheet = name.replace(".csv", "")[:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 1)
            ws.set_column(0, len(frame.columns) - 1, 22)
            ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)

    chart_paths = []

    history = pd.read_csv(
        OUT / "full_revaluation_history.csv",
        index_col=0, parse_dates=True,
    )

    data = history[[
        "carry_roll", "rate_explain", "IG_explain",
        "HY_explain", "residual"
    ]].cumsum()

    data.to_csv(REPORTS / "attribution_chart_data.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax)
    ax.set_title("Cumulative P&L explain — modeled fixed book")
    ax.set_ylabel("USD")
    fig.tight_layout()
    path = REPORTS / "attribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    chart_paths.append(path)

    if (OUT / "ml_test.csv").exists():
        forecasts = pd.read_csv(
            OUT / "ml_test.csv", index_col=0, parse_dates=True
        )
        forecasts[["target", "forecast", "lower", "upper"]].to_csv(
            REPORTS / "forecast_chart_data.csv"
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(forecasts.index, forecasts["target"], label="Realized RMS")
        ax.plot(forecasts.index, forecasts["forecast"], label="Median forecast")
        ax.fill_between(
            forecasts.index,
            forecasts["lower"].to_numpy(),
            forecasts["upper"].to_numpy(),
            alpha=0.2,
            label="Nominal 90% interval",
        )
        ax.set_title("Held-out normalized-HPL RMS")
        ax.legend()
        fig.tight_layout()
        path = REPORTS / "forecast.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        chart_paths.append(path)

    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("Small")
    small.fontSize = 7
    small.leading = 9

    doc = SimpleDocTemplate(
        str(REPORTS / "advanced_report.pdf"),
        pagesize=landscape(A4),
        leftMargin=30, rightMargin=30,
        topMargin=30, bottomMargin=30,
    )

    story = [
        Paragraph("Advanced Bond Risk Research", styles["Title"]),
        Paragraph(escape(warning).replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            "Research results require interpretation; model inventory "
            "entries are not approval or independent validation.",
            styles["BodyText"],
        ),
        PageBreak(),
    ]

    for name, frame in tables.items():
        story.append(
            Paragraph(name.replace("_", " "), styles["Heading1"])
        )

        columns = frame.columns.tolist()
        groups = (
            [columns] if len(columns) <= 6 else
            [[columns[0]] + columns[i:i + 5]
             for i in range(1, len(columns), 5)]
        )

        for group in groups:
            subset = frame[group].fillna("")
            raw = [group] + subset.astype(str).values.tolist()

            cells = [
                [Paragraph(escape(value), small) for value in row]
                for row in raw
            ]

            table = LongTable(
                cells, repeatRows=1,
                colWidths=[doc.width / len(group)] * len(group),
            )
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.extend([table, Spacer(1, 12)])

        story.append(PageBreak())

    for path in chart_paths:
        story.extend([
            Image(str(path), width=650, height=270, kind="proportional"),
            PageBreak(),
        ])

    doc.build(story)

    hashes = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_hashes.json":
            hashes[str(path.relative_to(OUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    (REPORTS / "output_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )

    print(f"Advanced reports created: {REPORTS}")


if __name__ == "__main__":
    main()