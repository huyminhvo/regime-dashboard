"""
metrics.py
----------
Functions for computing performance statistics per regime.

Conventions:
- All returns are assumed to be DAILY LOG returns unless noted.
- Annualization uses TRADING_DAYS = 252.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

TRADING_DAYS = 252


# ===== Low-level helpers =====

def _is_dt_index(idx: pd.Index) -> bool:
    return isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex))


def _span_years(index: pd.Index) -> float:
    """
    Estimate span in years for a time-indexed Series/DataFrame.
    - Prefers calendar time if DatetimeIndex is available.
    - Falls back to len(index)/252 otherwise.
    """
    if len(index) == 0:
        return np.nan
    if _is_dt_index(index):
        # inclusive span; avoid zero by max(1 day)
        delta_days = max(1, (index[-1] - index[0]).days)
        return delta_days / 365.25
    return len(index) / TRADING_DAYS


def _equity_from_log_returns(r: pd.Series) -> pd.Series:
    """
    Convert daily log returns to equity curve with base 1.0.
    """
    r = r.dropna()
    eq = np.exp(r.cumsum())
    eq.iloc[0] = 1.0  # normalize start exactly at 1
    return eq


def _sharpe_ratio_annualized(returns_log: pd.Series) -> float:
    """
    Annualized Sharpe using daily LOG returns (rf≈0).
    """
    r = returns_log.dropna()
    sigma = r.std()
    return (r.mean() / sigma * np.sqrt(TRADING_DAYS)) if sigma and sigma > 0 else np.nan


def _downside_vol_annualized(returns_log: pd.Series) -> float:
    """
    Annualized downside volatility using daily LOG returns.
    """
    r = returns_log.dropna()
    dn = r[r < 0]
    if len(dn) == 0:
        return np.nan
    return dn.std() * np.sqrt(TRADING_DAYS)


def _sortino_ratio_annualized(returns_log: pd.Series) -> float:
    r = returns_log.dropna()
    dv = _downside_vol_annualized(r)
    mu = r.mean() * TRADING_DAYS  # annualize mean for Sortino numerator
    return (mu / dv) if dv and dv > 0 else np.nan


def _max_drawdown_from_equity(equity: pd.Series) -> float:
    """
    Max drawdown (negative) computed from equity curve.
    """
    eq = equity.dropna()
    peaks = eq.cummax()
    dd = eq / peaks - 1.0
    return float(dd.min()) if len(dd) else np.nan


def _max_drawdown(returns_log: pd.Series) -> float:
    """
    Backwards-compatible MDD from log returns.
    """
    return _max_drawdown_from_equity(_equity_from_log_returns(returns_log))


def _cagr_from_equity(equity: pd.Series, index: Optional[pd.Index] = None) -> float:
    """
    Compute CAGR given an equity curve and its index (for span).
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return np.nan
    yrs = _span_years(index if index is not None else eq.index)
    if not yrs or yrs <= 0:
        return np.nan
    return (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / yrs) - 1.0


# ===== Public API used by RegimeAnalyzer =====

def compute_regime_metrics(df: pd.DataFrame,
                           return_col: str = "return",
                           regime_col: str = "regime") -> pd.DataFrame:
    """
    Compute summary metrics per regime from a DataFrame that contains
    daily LOG returns and regime labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must include [return_col, regime_col] and ideally be time-indexed.
    return_col : str
        Name of the log-returns column.
    regime_col : str
        Name of the regime label column.

    Returns
    -------
    pd.DataFrame
        One row per regime with the following columns:
        - Days
        - Mean Log Return (daily)
        - Volatility (daily)
        - Volatility (ann)
        - Sharpe (ann)
        - Downside Vol (ann)
        - Sortino (ann)
        - Hit Rate
        - Max Drawdown
        - CAGR
        - Calmar
        - Skewness
        - Kurtosis
    """
    def _metrics(sub: pd.DataFrame) -> pd.Series:
        r = sub[return_col].dropna()
        if len(r) == 0:
            return pd.Series({
                "Days": 0,
                "Mean Log Return": np.nan,
                "Volatility": np.nan,
                "Volatility (ann)": np.nan,
                "Sharpe (ann)": np.nan,
                "Downside Vol (ann)": np.nan,
                "Sortino (ann)": np.nan,
                "Hit Rate": np.nan,
                "Max Drawdown": np.nan,
                "CAGR": np.nan,
                "Calmar": np.nan,
                "Skewness": np.nan,
                "Kurtosis": np.nan
            })

        # Daily stats
        mu_d = r.mean()
        sigma_d = r.std()
        hit = (r > 0).mean()

        # Annualized variants & path-dependent stats
        sharpe = _sharpe_ratio_annualized(r)
        dvol_ann = _downside_vol_annualized(r)
        sortino = _sortino_ratio_annualized(r)

        eq = _equity_from_log_returns(r)
        mdd = _max_drawdown_from_equity(eq)
        cagr = _cagr_from_equity(eq, r.index)
        calmar = (cagr / abs(mdd)) if (mdd is not None and mdd != 0 and not np.isnan(mdd)) else np.nan

        return pd.Series({
            "Days": int(len(r)),
            "Mean Log Return": mu_d,
            "Volatility": sigma_d,
            "Volatility (ann)": sigma_d * np.sqrt(TRADING_DAYS) if sigma_d == sigma_d else np.nan,
            "Sharpe (ann)": sharpe,
            "Downside Vol (ann)": dvol_ann,
            "Sortino (ann)": sortino,
            "Hit Rate": hit,
            "Max Drawdown": mdd,
            "CAGR": cagr,
            "Calmar": calmar,
            "Skewness": r.skew(),
            "Kurtosis": r.kurtosis()
        })

    # Group and compute
    out = df.groupby(regime_col, dropna=True).apply(_metrics)
    # Ensure a plain DataFrame (pandas >= 2.0 changes groupby semantics)
    if isinstance(out, pd.Series):
        out = out.to_frame().T
    return out


# Backwards-compatible exports used in other modules
_sharpe_ratio = lambda returns: returns.mean() / returns.std() if returns.std() != 0 else np.nan  # not annualized
_max_drawdown = _max_drawdown  # keep name for imports in existing code
