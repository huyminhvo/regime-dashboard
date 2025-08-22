"""
regime_labeling.py
------------------
Functions for computing volatility and labeling market regimes.
"""

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
        Series of returns.
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()

def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling volatility (std dev) of returns.

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling volatility values.
    """
    return returns.rolling(window).std()

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
        Regime labels.
    """
    if method == "median":
        threshold = vol.median()
        return np.where(vol < threshold, "Low Vol", "High Vol")

    elif method == "quantile":
        low_th = vol.quantile(q)
        high_th = vol.quantile(1 - q)
        return np.select(
            [vol <= low_th, vol >= high_th],
            ["Low Vol", "High Vol"],
            default="Mid Vol"
        )

    else:
        raise ValueError("Invalid method. Use 'median' or 'quantile'.")
