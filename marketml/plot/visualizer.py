import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

PLOT_DIR = "/home/hoang/Documents/KhoaLuan/.ndmh/marketml/plot/img"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_all_closing_prices(df_price: pd.DataFrame, save=True):
    df_price = df_price.copy()
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
    df_price = df_price.dropna(subset=["Date", "Close", "Ticker"])
    df_price = df_price.sort_values("Date")

    plt.figure(figsize=(16, 9))
    for ticker in df_price["Ticker"].unique():
        df_ticker = df_price[df_price["Ticker"] == ticker]
        plt.plot(df_ticker["Date"], df_ticker["Close"], label=ticker, linewidth=1)

    plt.title("Close Price of All Tickers")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(fontsize="small", ncol=3, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if save:
        plt.savefig(os.path.join(PLOT_DIR, "all_closing_prices.png"))

    plt.show()
    plt.close()


def plot_normalized_prices_by_group(df_price: pd.DataFrame, group: str, save=True):
    df_price = df_price.copy()
    df_price = df_price[df_price["Market"] == group]
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
    df_price = df_price.dropna(subset=["Date", "Close", "Ticker"])
    df_price = df_price.sort_values("Date")

    plt.figure(figsize=(16, 9))
    for ticker in df_price["Ticker"].unique():
        df_ticker = df_price[df_price["Ticker"] == ticker].sort_values("Date")
        if df_ticker["Close"].iloc[0] == 0:
            continue
        norm_price = df_ticker["Close"] / df_ticker["Close"].iloc[0]
        plt.plot(df_ticker["Date"], norm_price, label=ticker, linewidth=1)

    plt.title(f"Normalized Close Prices ({group})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(fontsize="small", ncol=3, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if save:
        filename = f"normalized_closing_prices_{group.lower()}.png"
        plt.savefig(os.path.join(PLOT_DIR, filename))

    plt.show()
    plt.close()
