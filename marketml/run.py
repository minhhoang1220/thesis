from marketml.data.loader import load_price_data, load_financial_data
from marketml.plot.visualizer import plot_all_closing_prices, plot_normalized_prices_by_group


def main():
    print("✅ PRICE DATA")
    df_price = load_price_data(nrows=5)
    print(df_price)

    print("\n✅ FINANCIAL DATA")
    df_fin = load_financial_data(nrows=5)
    print(df_fin)

    # Full data for plotting
    df_price_full = load_price_data()

    # Plot and save figures
    plot_all_closing_prices(df_price_full)
    plot_normalized_prices_by_group(df_price_full, group="Global")


if __name__ == "__main__":
    main()
