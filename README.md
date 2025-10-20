# Regime Dashboard

A lightweight Python toolkit for **volatility-based regime analysis** and **market diagnostics**.

This project identifies distinct volatility regimes (Low / Mid / High), computes performance metrics for each, and visualizes how different market conditions affect returns.

Built for quick exploratory research, strategy testing, and clean presentation, showcasing quantitative analysis and Python engineering skills.



## Features

- **Automated Regime Labeling**  
  Quantile or median-based volatility segmentation of any asset.
  
- **Performance Metrics**  
  Sharpe, Sortino, CAGR, Calmar, Drawdown, Hit Rate, and more per regime.

- **Strategy Simulation**  
  Example “Long in Low-Volatility” regime strategy vs. Buy-and-Hold benchmark.

- **Visual Diagnostics**  
  Regime overlays on price charts and equity curve comparisons.

- **Modular Design**  
  Everything is cleanly separated into:
  - `metrics.py` – performance statistics  
  - `regime_labeling.py` – volatility & labeling logic  
  - `regime_analyzer.py` – main analysis class  
  - `strategies.py` – simple backtests  
  - `visualization.py` – plotting helpers  



## Example Workflow

```python
import yfinance as yf
from regime.regime_analyzer import RegimeAnalyzer

# 1. Download historical data
df = yf.download("TSLA", start="2015-01-01", end="2025-01-01")
close = df["Close"]

# 2. Initialize analyzer
an = RegimeAnalyzer(close, log=True)
an.compute_volatility(window=20)
an.label_regimes(method="quantile", q=0.33)

# 3. Summary metrics per regime
summary = an.compute_summary()
print(summary.round(4))

# 4. Visualize regimes and equity curves
an.plot_regimes()
eq_df = an.plot_equity_by_regime()

# 5. Compare strategy vs benchmark
equity_df, stats_df = an.compare_to_benchmark()
print(stats_df.round(3))
```

## Example Output

Summary Metrics (sample)

| Regime   | Days | CAGR   | Sharpe | Sortino | Max Drawdown | Hit Rate |
| -------- | ---- | ------ | ------ | ------- | ------------ | -------- |
| Low Vol  | 912  | 0.162  | 1.44   | 2.01    | -0.08        | 0.56     |
| Mid Vol  | 844  | 0.089  | 0.78   | 1.12    | -0.12        | 0.52     |
| High Vol | 791  | -0.031 | -0.45  | -0.53   | -0.28        | 0.44     |



## Installation
```
git clone https://github.com/<yourusername>/regime-dashboard.git
cd regime-dashboard
pip install -r requirements.txt
```

## Quick Demo
```
python examples/run_tsla_demo.py
```

This will:

1. Download TSLA daily data
2. Compute volatility regimes
3. Display regime plots and performance tables

## Requirements
```
pandas>=2.0
numpy>=1.25
matplotlib>=3.7
yfinance>=0.2

# Optional
seaborn>=0.13
jupyterlab>=4.0
scikit-learn>=1.5
```

## Notes
* All returns are computed as daily log returns by default unless otherwise noted.
* Annualization uses 252 trading days.
* Designed for exploratory financial analysis, not production trading.

## License
MIT License © 2025 Huy Vo