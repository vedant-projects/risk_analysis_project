from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.report_tables import build_bundle


def main() -> None:
    out_dir = ROOT / "output" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle()

    tables = {
        "book.csv": bundle.book,
        "scenarios.csv": bundle.scenarios,
        "latest_risk.csv": bundle.latest_risk,
        "backtest.csv": bundle.backtest,
        "worst_days.csv": bundle.worst_days,
        "scenario_history_1d.csv": bundle.scen_hist_1d,
        "scenario_history_5d.csv": bundle.scen_hist_5d,
    }

    exported = []
    skipped = []

    for filename, df in tables.items():
        if df is not None:
            df.to_csv(out_dir / filename)
            exported.append(filename)
        else:
            skipped.append(filename)

    print(f"exported {len(exported)} table(s) to {out_dir}")
    for name in exported:
        print(f"  OK   {name}")

    if skipped:
        print("\nskipped (source data unavailable):")
        for name in skipped:
            print(f"  SKIP {name}")

    notes_path = out_dir / "notes.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        for k, v in bundle.notes.items():
            f.write(f"{k}: {v}\n")

    print(f"\nnotes -> {notes_path}")


if __name__ == "__main__":
    main()