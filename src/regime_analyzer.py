import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from .metrics import _sharpe_ratio, _max_drawdown
from .regime_labeling import compute_returns, compute_volatility, label_regimes


class RegimeAnalyzer:
    """
    Framework for volatility-based regime analysis and strategy backtesting.
    """

    def __init__(self, prices: pd.Series, log: bool = True):
        self.prices = prices.dropna()
        self.log = log
        self.returns = compute_returns(self.prices, log=log)
        self.vol = None
        self.regimes = None
        self.summary = None

    def compute_volatility(self, window: int = 20) -> pd.Series:
        self.vol = compute_volatility(self.returns, window=window)
        return self.vol

    def label_regimes(self, method: Literal["median", "quantile"] = "median", q: float = 0.33) -> pd.Series:
        if self.vol is None:
            raise ValueError("Must compute volatility first with compute_volatility().")
        self.regimes = label_regimes(self.vol, method=method, q=q)
        return self.regimes

    def compute_summary(self) -> pd.DataFrame:
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        df = pd.DataFrame({
            "price": self.prices,
            "return": self.returns,
            "vol": self.vol,
            "regime": self.regimes
        }).dropna()

        def _metrics(sub: pd.DataFrame) -> pd.Series:
            r = sub["return"]
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

        self.summary = df.groupby("regime").apply(_metrics)
        return self.summary

    def plot_regimes(self) -> None:
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.prices.index, self.prices.values, color="black", linewidth=1.5, label="Price")

        regime_colors = {"Low Vol": "green", "Mid Vol": "yellow", "High Vol": "red"}
        current_regime, start_idx = None, None

        for i in range(len(self.regimes)):
            regime = self.regimes.iloc[i]
            if regime != current_regime:
                if current_regime is not None:
                    ax.axvspan(self.prices.index[start_idx], self.prices.index[i-1],
                               color=regime_colors.get(current_regime, "gray"), alpha=0.2,
                               label=current_regime if start_idx == 0 else None)
                current_regime, start_idx = regime, i

        if current_regime is not None:
            ax.axvspan(self.prices.index[start_idx], self.prices.index[-1],
                       color=regime_colors.get(current_regime, "gray"), alpha=0.2,
                       label=current_regime)

        ax.set_title("Price with Volatility Regimes")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.show()

    def strategy_long_low_vol(self) -> pd.Series:
        """
        Simple strategy: stay long only in Low Vol regimes, flat otherwise.

        Returns
        -------
        pd.Series
            Equity curve indexed by datetime.
        """
        if self.regimes is None:
            raise ValueError("Must label regimes first with label_regimes().")

        strat_log_returns = self.returns.where(self.regimes == "Low Vol", 0.0)
        equity_curve = np.exp(strat_log_returns.cumsum())
        return equity_curve

    def compare_to_benchmark(self) -> pd.DataFrame:
        """
        Compare strategy equity curve vs. buy-and-hold benchmark.

        Returns
        -------
        pd.DataFrame
            DataFrame with equity curves and performance metrics.
        """
        # --- Strategy equity curve ---
        strat_equity = self.strategy_long_low_vol()
        strat_returns = strat_equity.pct_change().dropna()

        # --- Benchmark equity curve ---
        bench_equity = np.exp(self.returns.cumsum())
        bench_returns = bench_equity.pct_change().dropna()

        # --- Combine equity curves ---
        equity_df = pd.DataFrame({
            "Strategy": strat_equity,
            "Benchmark": bench_equity
        }, index=self.prices.index)

        # --- Plot equity curves ---
        fig, ax = plt.subplots(figsize=(12, 6))
        equity_df["Strategy"].plot(ax=ax, label="Strategy (Long Low Vol)", linewidth=2)
        equity_df["Benchmark"].plot(ax=ax, label="Benchmark (Buy & Hold)", linestyle="--", linewidth=1.5)

        ax.set_title("Strategy vs. Benchmark")
        ax.set_ylabel("Equity (Growth of $1)")
        ax.legend()
        plt.show()

        # --- Performance stats ---
        def _perf_stats(r: pd.Series, eq: pd.Series) -> pd.Series:
            total_days = len(r)
            years = total_days / 252

            # CAGR
            cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1

            # Volatility (annualized)
            vol = r.std() * np.sqrt(252)

            # Sharpe (assume rf ~ 0)
            sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

            # Max Drawdown
            mdd = (eq / eq.cummax() - 1).min()

            # Sortino
            downside = r[r < 0]
            downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
            sortino = r.mean() / downside_vol * np.sqrt(252) if downside_vol and downside_vol > 0 else np.nan

            # Hit Rate
            hit_rate = (r > 0).mean()

            # Avg Win / Loss
            avg_win = r[r > 0].mean()
            avg_loss = r[r < 0].mean()

            # Calmar Ratio
            calmar = cagr / abs(mdd) if mdd != 0 else np.nan

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

        print("\nPerformance Comparison:\n", stats_df)

        return equity_df, stats_df


    
    

