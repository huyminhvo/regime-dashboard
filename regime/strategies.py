"""
strategies.py
-------------
Simple regime-based strategies for backtesting.
"""

import pandas as pd
import numpy as np

def long_in_low_vol(df: pd.DataFrame,
                    return_col: str = "return",
                    regime_col: str = "regime") -> pd.Series:
    """
    Strategy: Long only in Low Vol regimes, flat otherwise.
    Works with log returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must include return_col (log returns) and regime_col.
    return_col : str, default="return"
        Column with log returns.
    regime_col : str, default="regime"
        Column with regime labels.

    Returns
    -------
    pd.Series
        Equity curve indexed by datetime.
    """
    # Keep log returns only when in "Low Vol", else 0
    strat_log_returns = df[return_col].where(df[regime_col] == "Low Vol", 0.0)

    # Convert log returns to equity curve
    equity_curve = np.exp(strat_log_returns.cumsum())

    return equity_curve

