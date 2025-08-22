import yfinance as yf
import pandas as pd
import numpy as np
import math

ticker = "SPY"
raw = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)

# --- Robustly get a 1-D Close series, regardless of MultiIndex columns ---
if isinstance(raw.columns, pd.MultiIndex):
    # Preferred: ('Close', 'SPY') if yfinance tagged by ticker
    if ("Close", ticker) in raw.columns:
        close = raw[("Close", ticker)]
    elif "Close" in raw.columns.get_level_values(0):
        # Take the first Close across columns if present
        close_df = raw.xs("Close", axis=1, level=0, drop_level=False)
        close = close_df.iloc[:, 0]
    else:
        raise ValueError("Could not locate 'Close' in yfinance output.")
else:
    close = raw["Close"]

# Make it a clean Series with DatetimeIndex
close = pd.to_numeric(close, errors="coerce").dropna()
close = close[~close.index.duplicated(keep="last")].sort_index()

# === 2) Daily log returns ===
log_ret = np.log(close / close.shift(1)).dropna()

# === 3) Rolling vol (21d) ===
window = 21
rolling_vol = log_ret.rolling(window=window).std()

# === 4) Threshold via median (handle Series vs scalar robustly) ===
med = rolling_vol.dropna().median()
thr = float(med.iloc[0] if isinstance(med, pd.Series) else med)

# === 5) Regime labels (1=High vol, 0=Low vol) ===
regime = (rolling_vol >= thr).astype(int)

# === 6) Align & summarize ===
aligned = pd.DataFrame({"ret": log_ret, "regime": regime}).dropna()

def _metrics(g: pd.DataFrame) -> pd.Series:
    mu = float(g["ret"].mean())
    sd = float(g["ret"].std())
    ann_mu = mu * 252.0
    ann_sd = sd * math.sqrt(252.0)
    sharpe = ann_mu / ann_sd if ann_sd > 0 else float("nan")
    hit = float((g["ret"] > 0).mean())
    return pd.Series({
        "Days": int(g.shape[0]),
        "Mean Daily Return": mu,
        "Daily Volatility": sd,
        "Annualized Return": ann_mu,
        "Annualized Volatility": ann_sd,
        "Sharpe (rf≈0)": sharpe,
        "Hit Rate": hit,
    })

summary = aligned.groupby("regime").apply(_metrics)
summary.index = summary.index.map({0: "Low Vol", 1: "High Vol"})

print(f"Rows (prices): {len(close):d}")
print(f"Threshold (daily vol): {thr:.6f}")
print(summary.round(6))
