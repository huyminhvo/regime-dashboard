from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Dict, Iterable, Tuple

from .metrics import compute_regime_metrics, _max_drawdown
from .regime_labeling import compute_returns, compute_volatility, label_regimes


def _to_series(x, index=None) -> pd.Series:
    """
    Ensure input is a 1D pandas Series with an optional index alignment.
    Accepts: Series, DataFrame(1 col), ndarray, list/tuple.
    """
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected 1D data, got multi-column DataFrame.")
        s = x.iloc[:, 0]
    elif isinstance(x, (np.ndarray, list, tuple)):
        arr = np.asarray(x).ravel()
        s = pd.Series(arr)
    else:
        # attempt squeeze-style behavior
        s = pd.Series(x)

    if index is not None and len(s) == len(index):
        s.index = index
    return s


def _default_regime_colors(labels: Iterable[str]) -> Dict[str, str]:
    palette = {"Low Vol": "tab:green", "Mid Vol": "tab:orange", "High Vol": "tab:red"}
    return {lab: palette.get(str(lab), "tab:gray") for lab in labels}


class RegimeAnalyzer:
    """
    Framework for volatility-based regime analysis and quick strategy checks.
    """

    def __init__(self, prices: pd.Series, log: bool = True):
        self.prices = _to_series(prices).dropna()
        self.log = log
        self.returns = _to_series(compute_returns(self.prices, log=log), index=self.prices.index)
        self.vol: pd.Series | None = None
        self.regimes: pd.Series | None = None
        self.summary: pd.DataFrame | None = None

    # ===== Core computation =====

    def compute_volatility(self, window: int = 20) -> pd.Series:
        self.vol = _to_series(compute_volatility(self.returns, window=window), index=self.returns.index)
        return self.vol

    def label_regimes(self,
                      method: Literal["median", "quantile"] = "median",
                      q: float = 0.33) -> pd.Series:
        if self.vol is None:
            raise ValueError("Must compute volatility first with compute_volatility().")
        self.regimes = _to_series(label_regimes(self.vol, method=method, q=q), index=self.vol.index)
        return self.regimes

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return an aligned DataFrame with columns: ['price','return','vol','regime'].
        """
        frames = {"price": self.prices, "return": self.returns}
        if self.vol is not None:
            frames["vol"] = self.vol
        if self.regimes is not None:
            frames["regime"] = self.regimes

        df = pd.concat(frames, axis=1, join="inner").dropna()
        return df

    def compute_summary(self) -> pd.DataFrame:
        """
        Compute per-regime metrics via metrics.compute_regime_metrics (no duplication).
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")
        df = self.to_dataframe()
        self.summary = compute_regime_metrics(df, return_col="return", regime_col="regime")
        return self.summary

    # ===== Visualization =====

    def plot_regimes(self) -> None:
        """
        Overlay shaded regime spans on top of the price series.
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        price = self.prices
        regs = self.regimes.reindex(price.index).ffill()  # align & fill if needed
        labels = pd.unique(regs.astype(str).values)
        colors = _default_regime_colors(labels)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(price.index, price.values, color="black", linewidth=1.5, label="Price")

        current_regime, start_idx = None, 0
        labeled = set()

        for i in range(len(regs)):
            regime = regs.iloc[i]
            if i == 0:
                current_regime, start_idx = regime, 0
                continue
            if regime != current_regime:
                ax.axvspan(price.index[start_idx], price.index[i - 1],
                           color=colors.get(str(current_regime), "tab:gray"),
                           alpha=0.18,
                           label=str(current_regime) if str(current_regime) not in labeled else None)
                labeled.add(str(current_regime))
                current_regime, start_idx = regime, i

        # close last span
        ax.axvspan(price.index[start_idx], price.index[-1],
                   color=colors.get(str(current_regime), "tab:gray"),
                   alpha=0.18,
                   label=str(current_regime) if str(current_regime) not in labeled else None)

        ax.set_title("Price with Volatility Regimes")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        plt.show()

    def plot_equity_by_regime(self) -> pd.DataFrame:
        """
        Plot equity curves for 'hold only during X' for each regime label.
        Returns a DataFrame of equities (one column per regime).
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        df = self.to_dataframe()
        labels = pd.unique(df["regime"].astype(str).values)
        colors = _default_regime_colors(labels)

        eq = {}
        for lab in labels:
            strat_log = df["return"].where(df["regime"].astype(str) == lab, 0.0)
            eq_lab = np.exp(strat_log.cumsum())
            eq_lab.iloc[0] = 1.0
            eq[lab] = eq_lab

        eq_df = pd.DataFrame(eq, index=df.index)

        fig, ax = plt.subplots(figsize=(12, 6))
        for lab in labels:
            eq_df[lab].plot(ax=ax, linewidth=2, label=f"Only {lab}", color=colors.get(str(lab), None))
        ax.set_title("Equity Curves by Regime (Hold Only During Regime)")
        ax.set_ylabel("Equity (Growth of $1)")
        ax.legend()
        plt.show()
        return eq_df

    # ===== Simple strategy & benchmark =====

    def strategy_long_low_vol(self) -> pd.Series:
        """
        Simple strategy: long only in Low Vol regimes, flat otherwise.
        Returns equity curve (base 1.0) indexed by date.
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")
        strat_log_returns = self.returns.where(self.regimes.astype(str) == "Low Vol", 0.0)
        equity_curve = np.exp(strat_log_returns.cumsum())
        equity_curve.iloc[0] = 1.0
        return equity_curve

    def compare_to_benchmark(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Plot Strategy vs. Buy & Hold and return:
          - equity_df: columns ['Strategy', 'Benchmark']
          - stats_df : rows ['CAGR','Volatility','Sharpe','Sortino','Hit Rate','Avg Win','Avg Loss','Max Drawdown','Calmar']
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        strat_equity = self.strategy_long_low_vol()
        bench_equity = np.exp(self.returns.cumsum())
        bench_equity.iloc[0] = 1.0

        # Convert to simple daily returns from equities
        strat_returns = strat_equity.pct_change().dropna()
        bench_returns = bench_equity.pct_change().dropna()

        equity_df = pd.DataFrame({
            "Strategy": strat_equity,
            "Benchmark": bench_equity
        }).dropna()

        fig, ax = plt.subplots(figsize=(12, 6))
        equity_df["Strategy"].plot(ax=ax, label="Strategy (Long Low Vol)", linewidth=2)
        equity_df["Benchmark"].plot(ax=ax, label="Benchmark (Buy & Hold)", linestyle="--", linewidth=1.5)
        ax.set_title("Strategy vs. Benchmark")
        ax.set_ylabel("Equity (Growth of $1)")
        ax.legend()
        plt.show()

        # Perf stats (annualized) on simple returns
        def _perf_stats(r: pd.Series, eq: pd.Series) -> pd.Series:
            total_days = len(r)
            years = total_days / 252.0 if total_days else np.nan

            # CAGR from equity
            cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan
            vol = r.std() * np.sqrt(252) if len(r) else np.nan
            sharpe = (r.mean() / r.std() * np.sqrt(252)) if (r.std() and r.std() > 0) else np.nan
            downside = r[r < 0]
            downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
            sortino = (r.mean() * 252 / downside_vol) if (downside_vol and downside_vol > 0) else np.nan
            hit_rate = (r > 0).mean() if len(r) else np.nan
            avg_win = r[r > 0].mean() if (r > 0).any() else np.nan
            avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
            mdd = (eq / eq.cummax() - 1).min() if len(eq) else np.nan
            calmar = cagr / abs(mdd) if (mdd and mdd != 0) else np.nan

            return pd.Series({
                "CAGR": cagr,
                "Volatility": vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Hit Rate": hit_rate,
                "Avg Win": avg_win,
                "Avg Loss": avg_loss,
                "Max Drawdown": mdd,
                "Calmar": calmar
            })

        strat_stats = _perf_stats(strat_returns, strat_equity)
        bench_stats = _perf_stats(bench_returns, bench_equity)

        stats_df = pd.concat([strat_stats, bench_stats], axis=1)
        stats_df.columns = ["Strategy", "Benchmark"]

        return equity_df, stats_df

    # ===== Diagnostic: regime switches =====

    def regime_switch_stats(self) -> pd.Series:
        """
        Basic regime-switch diagnostics: number of switches and average duration (days).
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")
        r = self.regimes.dropna().astype(str)
        if len(r) == 0:
            return pd.Series({"Switches": 0, "Avg Duration (days)": np.nan})

        switches = (r != r.shift(1)).sum() - 1  # first element counts as a change
        # durations of consecutive runs
        runs = (r != r.shift(1)).cumsum()
        durations = r.groupby(runs).size()
        avg_dur = durations.mean() if len(durations) else np.nan

        return pd.Series({"Switches": int(max(0, switches)), "Avg Duration (days)": float(avg_dur)})
