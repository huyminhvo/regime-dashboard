import matplotlib.pyplot as plt
import pandas as pd

def plot_regimes(prices: pd.Series, regimes: pd.Series) -> None:
    """
    Plot price series with regimes highlighted via background shading.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.
    regimes : pd.Series
        Regime labels aligned with prices.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the full price series
    ax.plot(prices.index, prices.values, color="black", linewidth=1.5, label="Price")

    # Identify contiguous stretches of each regime
    current_regime = None
    start_idx = None

    for i in range(len(regimes)):
        regime = regimes.iloc[i]

        # Start of a new regime stretch
        if regime != current_regime:
            # If we were in a regime before, shade its span
            if current_regime is not None:
                ax.axvspan(prices.index[start_idx], prices.index[i-1],
                           color=regime_colors.get(current_regime, "gray"),
                           alpha=0.2, label=current_regime if start_idx == 0 else None)
            # Update tracking
            current_regime = regime
            start_idx = i

    # Shade the last stretch
    if current_regime is not None:
        ax.axvspan(prices.index[start_idx], prices.index[-1],
                   color=regime_colors.get(current_regime, "gray"),
                   alpha=0.2, label=current_regime)

    # Cosmetics
    ax.set_title("Price with Volatility Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.show()


# Define consistent regime colors
regime_colors = {
    "Low Vol": "green",
    "Mid Vol": "yellow",
    "High Vol": "red"
}
