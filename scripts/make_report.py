from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"
OUT_PATH = ROOT / "output" / "report.md"


def read_table(name: str) -> pd.DataFrame | None:
    path = TABLE_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_Not available._"
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown()


def read_notes() -> dict[str, str]:
    notes = {}
    path = TABLE_DIR / "notes.txt"
    if not path.exists():
        return notes
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            notes[k.strip()] = v.strip()
    return notes


def main() -> None:
    book = read_table("book.csv")
    scenarios = read_table("scenarios.csv")
    latest_risk = read_table("latest_risk.csv")
    backtest = read_table("backtest.csv")
    worst_days = read_table("worst_days.csv")
    scen_1d = read_table("scenario_history_1d.csv")
    scen_5d = read_table("scenario_history_5d.csv")
    notes = read_notes()

    lines: list[str] = []
    lines.append("# Japan Bond Shock Risk Report")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    if notes:
        for k, v in notes.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- Notes not available.")
    lines.append("")

    lines.append("## Portfolio")
    lines.append("")
    lines.append("Current book composition, DV01, and key-rate exposure table.")
    lines.append("")
    lines.append(df_to_md(book))
    lines.append("")

    lines.append("## Scenarios")
    lines.append("")
    lines.append("Japan shock scenarios and portfolio-level loss estimates.")
    lines.append("")
    lines.append(df_to_md(scenarios))
    lines.append("")

    lines.append("## VaR")
    lines.append("")
    if latest_risk is not None:
        lines.append("Latest historical VaR and expected shortfall estimates.")
        lines.append("")
        lines.append(df_to_md(latest_risk))
    else:
        lines.append("VaR section not yet available because return-history exports are missing.")
    lines.append("")

    lines.append("## Backtest")
    lines.append("")
    if backtest is not None:
        lines.append("Historical exceedance counts for the VaR series.")
        lines.append("")
        lines.append(df_to_md(backtest))
    else:
        lines.append("Backtest section not yet available.")
    lines.append("")

    lines.append("## Worst days")
    lines.append("")
    if worst_days is not None:
        lines.append("Worst realized book-return days in the available history.")
        lines.append("")
        lines.append(df_to_md(worst_days, max_rows=10))
    else:
        lines.append("Worst-days section not yet available.")
    lines.append("")

    lines.append("## Scenario vs history")
    lines.append("")
    if scen_1d is not None:
        lines.append("### One-day comparison")
        lines.append("")
        lines.append(df_to_md(scen_1d))
        lines.append("")
    else:
        lines.append("One-day scenario/history comparison not yet available.")
        lines.append("")

    if scen_5d is not None:
        lines.append("### Five-day comparison")
        lines.append("")
        lines.append(df_to_md(scen_5d))
        lines.append("")
    else:
        lines.append("Five-day scenario/history comparison not yet available.")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote report -> {OUT_PATH}")


if __name__ == "__main__":
    main()