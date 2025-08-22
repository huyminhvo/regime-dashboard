"""
metrics.py
----------
Functions for computing performance statistics per regime.
"""

import pandas as pd
import numpy as np

def _sharpe_ratio(returns: pd.Series) -> float:
    """
    Compute Sharpe ratio (rf ≈ 0 assumption).
    """
    mu = returns.mean()
    sigma = returns.std()
    return mu / sigma if sigma != 0 else np.nan

def _max_drawdown(returns: pd.Series) -> float:
    """
    Compute max drawdown from a log-return series.

    Parameters
    ----------
    returns : pd.Series (log)

    Returns
    -------
    float
        Maximum drawdown (negative number).
    """
    # Equity curve from log returns
    cumulative = np.exp(returns.cumsum())

    # Rolling peak equity
    rolling_max = cumulative.cummax()

    # Drawdowns (always <= 0)
    drawdown = cumulative / rolling_max - 1

    return drawdown.min()


def compute_regime_metrics(df: pd.DataFrame,
                           return_col: str = "return",
                           regime_col: str = "regime") -> pd.DataFrame:
    """
    Compute summary metrics per regime.
    Assumes log returns in return_col.

    Parameters
    ----------
    df : pd.DataFrame
        Must include return_col and regime_col.
    return_col : str
        Column with log returns.
    regime_col : str
        Column with regime labels.

    Returns
    -------
    pd.DataFrame
        Metrics per regime.
    """
    def _metrics(sub: pd.DataFrame) -> pd.Series:
        r = sub[return_col].dropna()
        return pd.Series({
            "Days": len(r),
            "Mean Log Return": r.mean(),
            "Volatility": r.std(),
            "Sharpe": _sharpe_ratio(r),
            "Hit Rate": (r > 0).mean(),
            "Max Drawdown": _max_drawdown(r),
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis()
        })

    return df.groupby(regime_col).apply(_metrics)

