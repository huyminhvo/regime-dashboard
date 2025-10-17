"""
Demo script for RegimeAnalyzer on TSLA daily data.
"""

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

from regime.regime_analyzer import RegimeAnalyzer  # adjust import path if needed

# === 1. Download sample data ===
df = yf.download("TSLA", start="2015-01-01", end="2025-01-01")
close = df["Close"]

# === 2. Initialize analyzer ===
an = RegimeAnalyzer(close, log=True)

# Compute rolling volatility & regimes
an.compute_volatility(window=20)
an.label_regimes(method="quantile", q=0.33)

# === 3. Compute per-regime metrics ===
summary = an.compute_summary()
print("\n=== Regime Metrics ===")
print(summary.round(4))

# === 4. Visualize regimes on price ===
an.plot_regimes()

# === 5. Equity curves if holding only in each regime ===
eq_by_regime = an.plot_equity_by_regime()
print("\nFinal equity values if holding only in regime:")
print(eq_by_regime.iloc[-1].round(3))

# === 6. Strategy vs. Benchmark ===
equity_df, stats_df = an.compare_to_benchmark()
print("\n=== Strategy vs Benchmark Stats ===")
print(stats_df.round(4))

# === 7. Diagnostics ===
print("\n=== Regime Switch Diagnostics ===")
print(an.regime_switch_stats())
