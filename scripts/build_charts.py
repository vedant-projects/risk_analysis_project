"""build_charts.py - smoke test chart construction without PNG export."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.charts import build_all_figures

if __name__ == '__main__':
    figs = build_all_figures()
    print(f'built {len(figs)} figure(s)')
    for name in figs:
        print(name)