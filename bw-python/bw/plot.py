"""Plotting utilities for body weight model outputs."""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for safety
import matplotlib.pyplot as plt

from bw.stats import _get_numeric_vars, _model_array


def model_plot(model, vars=None, title="Hall's model results",
               ncol=2, figsize=None):
    """Plot model trajectories as a grid of subplots.

    Parameters
    ----------
    model : dict
        Output from ``adult_weight()`` or ``child_weight()``.
    vars : list of str, optional
        Variables to plot.  Default: all numeric 2-D array keys.
    title : str
        Overall figure title.
    ncol : int
        Number of columns in the subplot grid.
    figsize : tuple of (width, height), optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if vars is None:
        vars = _get_numeric_vars(model)

    nplots = len(vars)
    if nplots == 0:
        raise ValueError("No plottable variables found in model.")

    if nplots == 1:
        ncol = 1

    nrow = math.ceil(nplots / ncol)

    if figsize is None:
        figsize = (5 * ncol, 4 * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    fig.suptitle(title)

    time = model["Time"]

    for idx, var in enumerate(vars):
        row, col = divmod(idx, ncol)
        ax = axes[row][col]

        arr = _model_array(model, var)  # (nind, ntime)

        if arr.ndim == 1:
            ax.plot(time, arr)
        else:
            for i in range(arr.shape[0]):
                ax.plot(time, arr[i, :])

        ax.set_xlabel("Time")
        ax.set_ylabel(var.replace("_", " "))

    # Hide unused axes
    for idx in range(nplots, nrow * ncol):
        row, col = divmod(idx, ncol)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig
