"""Offline tests for var_es_plots utilities (plotting is optional)."""
import numpy as np
import pandas as pd

from src.visualization.var_es_plots import plot_var_es_curve

def test_plot_var_es_curve_no_error_when_no_plotting_backend():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=50)
    rp = pd.Series(rng.normal(0, 0.02, size=50), index=idx)
    # dummy values
    plt_back = plot_var_es_curve(rp, vaR=0.03, es=0.04, title="test")
    # If matplotlib is available, it returns a plt object, otherwise None
    # We accept either
    assert plt_back is None or hasattr(plt_back, "plot")