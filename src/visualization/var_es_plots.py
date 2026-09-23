"""var_es_plots.py - Lightweight plots for VaR/ES diagnostics.

This module provides a simple plotting helper that renders:
- VaR vs ES curves for a given asset/portfolio path
- Backtest result bar for quick quality check

Dependencies are optional; import-time errors are handled gracefully so you can run tests without a display.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Optional: use matplotlib if available
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

def plot_var_es_curve(rp: pd.Series, vaR: float, es: float, title: Optional[str] = None):
    """Plot a tiny VaR/ES diagnostic if matplotlib is available."""
    if plt is None:  # pragma: no cover
        return None
    plt.figure(figsize=(6, 3))
    plt.plot(rp.index, rp.values, label="Portfolio returns")
    plt.axhline(-vaR, color="r", linestyle="--", label="VaR")
    plt.axhline(-es, color="g", linestyle=":", label="ES")
    plt.title(title or "VaR/ES Diagnostics")
    plt.legend()
    plt.tight_layout()
    return plt