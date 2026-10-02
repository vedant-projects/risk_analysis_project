from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.report_tables import build_bundle


def main():
    os.chdir(ROOT)
    out = ROOT / "output/tables"
    out.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle()

    tables = {
        "book": bundle.book,
        "scenarios": bundle.scenarios,
        "latest_risk": bundle.latest_risk,
        "backtest": bundle.backtest,
        "worst_days": bundle.worst_days,
        "scenario_history_1d": bundle.scen_hist_1d,
        "scenario_history_5d": bundle.scen_hist_5d,
    }

    exported = []
    skipped = []

    for name, frame in tables.items():
        path = out / f"{name}.csv"

        if frame is None:
            path.unlink(missing_ok=True)
            skipped.append(name)
        else:
            frame.to_csv(path)
            exported.append(name)

    metadata_path = ROOT / "data/processed/learning_metadata.json"
    history_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exported": exported,
        "skipped": skipped,
        "bundle_notes": bundle.notes,
        "history_metadata": history_metadata,
        "warning": (
            "Historical risk, ML, Monte Carlo and RL results use an "
            "approximate rate-only book proxy unless explicitly replaced. "
            "Current scenario repricing is a separate calculation."
        ),
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    (out / "notes.txt").write_text(
        "\n".join(
            [manifest["warning"]]
            + [f"{key}: {value}" for key, value in bundle.notes.items()]
        ),
        encoding="utf-8",
    )

    print(f"Exported {len(exported)} tables to {out}")
    print("Skipped:", ", ".join(skipped) or "none")


if __name__ == "__main__":
    main()