from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS = [
    "build_learning_history.py",
    "train_volatility_ml.py",
    "run_learning_mc.py",
    "train_mc_risk_agent.py",
]

REPORTING = [
    "export_tables.py",
    "finalize_report.py",
]


def run(name):
    path = ROOT / "scripts" / name

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\n{'=' * 60}\nRunning {name}", flush=True)

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download history and rerun ML, Monte Carlo and RL.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit after reports are generated.",
    )
    args = parser.parse_args()

    scripts = ANALYSIS + REPORTING if args.refresh else REPORTING

    for name in scripts:
        run(name)

    print("\nReporting pipeline completed.")
    print(f"Deliverables: {ROOT / 'output/reports'}")

    if args.dashboard:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(ROOT / "dashboard/app.py"),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()