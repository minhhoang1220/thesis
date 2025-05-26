# /.ndmh/marketml/analysis/plot_utils.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
import logging

# Import configs to get plot save path
try:
    from marketml.configs import configs
    DEFAULT_PLOT_SAVE_DIR = getattr(configs, 'PLOTS_OUTPUT_DIR', Path("plots_output_fallback"))
except ImportError:
    print("Warning: Could not import configs for plot_utils. Using fallback plot directory.")
    DEFAULT_PLOT_SAVE_DIR = Path("plots_output_fallback")

logger = logging.getLogger(__name__)


def _ensure_save_dir(save_dir: Path):
    """Helper to create directory if it doesn't exist."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create plot save directory {save_dir}: {e}")
        return False
    return True


def plot_all_closing_prices(df_price: pd.DataFrame, save: bool = True, save_dir: Path = None):
    if df_price.empty:
        logger.warning("Cannot plot closing prices: DataFrame is empty.")
        return

    df_price = df_price.copy()
    if 'date' not in df_price.columns or 'close' not in df_price.columns or 'ticker' not in df_price.columns:
        logger.error("DataFrame must contain 'date', 'close', and 'ticker' columns for plotting.")
        return

    df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
    df_price = df_price.dropna(subset=["date", "close", "ticker"])
    df_price = df_price.sort_values("date")

    if df_price.empty:
        logger.warning("No valid data left after cleaning for plotting all closing prices.")
        return

    plt.figure(figsize=(16, 9))
    for ticker in df_price["ticker"].unique():
        df_ticker = df_price[df_price["ticker"] == ticker]
        plt.plot(df_ticker["date"], df_ticker["close"], label=ticker, linewidth=1)

    plt.title("Closing Price of All Tickers")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(fontsize="small", ncol=min(3, len(df_price["ticker"].unique())), loc='upper left')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save:
        current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
        if _ensure_save_dir(current_save_dir):
            try:
                filepath = current_save_dir / "all_closing_prices.png"
                plt.savefig(filepath)
                logger.info(f"Plot saved: {filepath.resolve()}")
            except Exception as e:
                logger.error(f"Error saving plot all_closing_prices.png: {e}")
    try:
        plt.show()
    except Exception as e:
        logger.warning(f"Could not display plot (e.g. no GUI backend): {e}")
    plt.close()


def plot_normalized_prices_by_group(df_price: pd.DataFrame, market_group: str, save: bool = True, save_dir: Path = None):
    if df_price.empty:
        logger.warning(f"Cannot plot normalized prices for group '{market_group}': DataFrame is empty.")
        return

    df_price_grouped = df_price.copy()
    if 'Market' not in df_price_grouped.columns:
        logger.error(f"Column 'Market' not found in DataFrame for group plotting.")
        return

    df_price_grouped = df_price_grouped[df_price_grouped["Market"] == market_group]
    if df_price_grouped.empty:
        logger.warning(f"No data found for market group '{market_group}'.")
        return

    if 'date' not in df_price_grouped.columns or 'close' not in df_price_grouped.columns or 'ticker' not in df_price_grouped.columns:
        logger.error("DataFrame for group must contain 'date', 'close', and 'ticker' columns.")
        return

    df_price_grouped["date"] = pd.to_datetime(df_price_grouped["date"], errors="coerce")
    df_price_grouped = df_price_grouped.dropna(subset=["date", "close", "ticker"])
    df_price_grouped = df_price_grouped.sort_values("date")

    if df_price_grouped.empty:
        logger.warning(f"No valid data left after cleaning for plotting normalized prices for group '{market_group}'.")
        return

    plt.figure(figsize=(16, 9))
    plot_success_count = 0
    for ticker in df_price_grouped["ticker"].unique():
        df_ticker = df_price_grouped[df_price_grouped["ticker"] == ticker].sort_values("date")
        if df_ticker.empty or df_ticker["close"].iloc[0] == 0 or pd.isna(df_ticker["close"].iloc[0]):
            logger.debug(f"Skipping ticker {ticker} in group {market_group} due to no data or zero/NaN starting price.")
            continue
        norm_price = df_ticker["close"] / df_ticker["close"].iloc[0]
        plt.plot(df_ticker["date"], norm_price, label=ticker, linewidth=1)
        plot_success_count +=1
    
    if plot_success_count == 0:
        logger.warning(f"No tickers plotted for group {market_group}. Skipping plot generation.")
        plt.close()
        return

    plt.title(f"Normalized Closing Prices ({market_group})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base 100 on first day)")
    plt.legend(fontsize="small", ncol=min(3, plot_success_count), loc='upper left')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save:
        current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
        if _ensure_save_dir(current_save_dir):
            try:
                filename = f"normalized_closing_prices_{market_group.lower().replace(' ', '_')}.png"
                filepath = current_save_dir / filename
                plt.savefig(filepath)
                logger.info(f"Plot saved: {filepath.resolve()}")
            except Exception as e:
                logger.error(f"Error saving plot {filename}: {e}")
    try:
        plt.show()
    except Exception as e:
        logger.warning(f"Could not display plot (e.g. no GUI backend): {e}")
    plt.close()
