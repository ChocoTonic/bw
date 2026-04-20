"""Summary statistics for body weight model outputs."""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# Keys to exclude when auto-detecting numeric variable names
_EXCLUDE_KEYS = {"Time", "Age", "BMI_Category", "Correct_Values", "Model_Type"}

# BMI category labels and thresholds
_BMI_CATEGORIES = ["Underweight", "Normal", "Pre-Obese", "Obese"]
_BMI_THRESHOLDS = [18.5, 25.0, 30.0]


def _get_numeric_vars(model):
    """Return list of keys in *model* that hold 2-D numeric arrays."""
    out = []
    for k, v in model.items():
        if k in _EXCLUDE_KEYS:
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            out.append(k)
    return out


def _model_array(model, var):
    """Return a 2-D array with shape (n_individuals, n_time).

    Adult model stores arrays as (nsteps, nind) while the child model
    stores them as (nind, nsteps).  We normalise to (nind, ntime) here.
    """
    arr = model[var]
    time_len = len(model["Time"])
    # If shape[0] == time_len and shape[1] != time_len => adult layout
    if arr.shape[0] == time_len and arr.shape[1] != time_len:
        return arr.T  # transpose to (nind, ntime)
    if arr.shape[1] == time_len:
        return arr  # already (nind, ntime)
    # Square matrix -- fall back to checking Model_Type
    if model.get("Model_Type") == "Adult":
        return arr.T
    return arr


def _default_days(time_arr, n=25):
    """Return *n* evenly spaced integer day indices."""
    max_idx = len(time_arr) - 1
    return np.unique(np.round(np.linspace(0, max_idx, n)).astype(int))


# -----------------------------------------------------------------------
# model_mean
# -----------------------------------------------------------------------

def model_mean(model, vars=None, days=None, group=None,
               weights=None, confidence=0.95):
    """Compute (optionally weighted) means with confidence intervals.

    Parameters
    ----------
    model : dict
        Output from ``adult_weight()`` or ``child_weight()``.
    vars : list of str, optional
        Variable names to summarise.  Default: all numeric 2-D array keys
        except *Time*.
    days : array-like of int, optional
        Time-step indices to evaluate.  Default: 25 evenly spaced from 0
        to ``len(Time)-1``.
    group : array-like, optional
        Group label per individual (length = n_individuals).
        Default: single group for everyone.
    weights : array-like, optional
        Per-individual sampling weights.  Default: equal weights.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    pd.DataFrame
        Columns: ``variable``, ``day``, ``group``, ``mean``, ``SE_mean``,
        ``lower``, ``upper``.
    """
    if confidence <= 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    if vars is None:
        vars = _get_numeric_vars(model)

    if "BMI_Category" in vars:
        raise ValueError(
            "Cannot compute mean of BMI_Category; use adult_bmi() instead."
        )

    time_arr = model["Time"]

    if days is None:
        day_indices = _default_days(time_arr)
    else:
        day_indices = np.asarray(days, dtype=int)

    # Determine n_individuals from the first var
    sample_arr = _model_array(model, vars[0])
    nind = sample_arr.shape[0]

    if group is None:
        group = np.ones(nind, dtype=int)
    else:
        group = np.asarray(group)

    if weights is None:
        weights = np.ones(nind, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    z = sp_stats.norm.ppf((1.0 + confidence) / 2.0)

    rows = []
    unique_groups = np.unique(group)

    for var in vars:
        arr = _model_array(model, var)  # (nind, ntime)
        for didx in day_indices:
            for g in unique_groups:
                mask = group == g
                vals = arr[mask, didx]
                w = weights[mask]
                w_sum = w.sum()

                wmean = np.average(vals, weights=w)

                # Weighted variance (reliability weights)
                n_g = mask.sum()
                if n_g > 1:
                    wvar = np.average((vals - wmean) ** 2, weights=w)
                    # Effective sample size correction
                    wse = np.sqrt(wvar / n_g)
                else:
                    wse = 0.0

                rows.append({
                    "variable": var,
                    "day": time_arr[didx],
                    "group": g,
                    "mean": wmean,
                    "SE_mean": wse,
                    "lower": wmean - z * wse,
                    "upper": wmean + z * wse,
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# adult_bmi
# -----------------------------------------------------------------------

def _classify_bmi_value(val):
    if val < 18.5:
        return "Underweight"
    elif val < 25.0:
        return "Normal"
    elif val < 30.0:
        return "Pre-Obese"
    return "Obese"


def adult_bmi(model, days=None, group=None, weights=None, confidence=0.95):
    """Compute BMI category prevalence with confidence intervals.

    Parameters
    ----------
    model : dict
        Output from ``adult_weight()`` (must contain ``BMI`` key).
    days : array-like of int, optional
        Time-step indices.  Default: 25 evenly spaced.
    group : array-like, optional
        Group labels per individual.
    weights : array-like, optional
        Per-individual sampling weights.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    pd.DataFrame
        Columns: ``day``, ``group``, ``category``, ``proportion``,
        ``se``, ``lower``, ``upper``.
    """
    if confidence <= 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")

    time_arr = model["Time"]

    if days is None:
        day_indices = _default_days(time_arr)
    else:
        day_indices = np.asarray(days, dtype=int)

    # Get BMI array (numeric, not category)
    bmi_arr = _model_array(model, "BMI")  # (nind, ntime)
    nind = bmi_arr.shape[0]

    if group is None:
        group = np.ones(nind, dtype=int)
    else:
        group = np.asarray(group)

    if weights is None:
        weights = np.ones(nind, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    z = sp_stats.norm.ppf((1.0 + confidence) / 2.0)
    unique_groups = np.unique(group)

    rows = []
    for didx in day_indices:
        for g in unique_groups:
            mask = group == g
            bmi_vals = bmi_arr[mask, didx]
            w = weights[mask]
            w_sum = w.sum()
            n_g = mask.sum()

            cats = np.array([_classify_bmi_value(v) for v in bmi_vals])

            for cat in _BMI_CATEGORIES:
                cat_mask = cats == cat
                prop = w[cat_mask].sum() / w_sum if w_sum > 0 else 0.0

                # SE of proportion (binomial approximation)
                if n_g > 1:
                    se = np.sqrt(prop * (1.0 - prop) / n_g)
                else:
                    se = 0.0

                rows.append({
                    "day": time_arr[didx],
                    "group": g,
                    "category": cat,
                    "proportion": prop,
                    "se": se,
                    "lower": max(0.0, prop - z * se),
                    "upper": min(1.0, prop + z * se),
                })

    return pd.DataFrame(rows)
