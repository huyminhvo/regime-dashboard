"""
regime_labeling.py
------------------
Functions for computing volatility and labeling market regimes.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal

def compute_returns(prices: pd.Series, log: bool = True) -> pd.Series:
    """
    Compute daily returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price data indexed by datetime.
    log : bool, default=True
        If True, compute log returns, else percentage returns.

    Returns
    -------
    pd.Series
        Series of daily returns (log or pct), aligned to input index.
    """
    s = pd.Series(prices).astype(float)
    if log:
        r = np.log(s / s.shift(1))
    else:
        r = s.pct_change()
    return r.dropna()


def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (std dev) of returns.

    Parameters
    ----------
    returns : pd.Series
        Return series (daily).
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling volatility values.
    """
    r = pd.Series(returns)
    return r.rolling(window).std()


def label_regimes(vol: pd.Series,
                  method: Literal["median", "quantile"] = "median",
                  q: float = 0.33) -> pd.Series:
    """
    Label regimes based on volatility.

    Parameters
    ----------
    vol : pd.Series
        Volatility series.
    method : {"median", "quantile"}
        Thresholding method.
    q : float
        Quantile cutoff if method="quantile".

    Returns
    -------
    pd.Series
        Regime labels as strings with the same index as `vol`.
    """
    v = pd.Series(vol)
    if method == "median":
        threshold = v.median()
        labels = np.where(v < threshold, "Low Vol", "High Vol")
    elif method == "quantile":
        low_th = v.quantile(q)
        high_th = v.quantile(1 - q)
        labels = np.select(
            [v <= low_th, v >= high_th],
            ["Low Vol", "High Vol"],
            default="Mid Vol"
        )
    else:
        raise ValueError("Invalid method. Use 'median' or 'quantile'.")

    out = pd.Series(labels, index=v.index, name="regime").astype("string")
    # Optional, consistent order for plotting
    order = pd.CategoricalDtype(categories=["Low Vol", "Mid Vol", "High Vol"], ordered=True)
    try:
        out = out.astype(order)
    except Exception:
        pass
    return out
