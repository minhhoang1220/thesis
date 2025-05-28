from marketml.data_handling.loader import load_price_data, load_financial_data
from marketml.data_handling.preprocess import standardize_data

def main():
    # === PRICE DATA ===
    print("✅ PRICE DATA")
    df_price = load_price_data(nrows=5)
    df_price_standardized = standardize_data(df_price)
    print(df_price_standardized)

    # === FINANCIAL DATA ===
    print("\n✅ FINANCIAL DATA")
    df_fin = load_financial_data(nrows=5)
    df_fin_standardized = standardize_data(df_fin)
    print(df_fin_standardized)

    # # === PLOTTING SECTION ===
    # df_price_full = load_price_data()
    # df_price_full_standardized = standardize_data(df_price_full)
    # plot_all_closing_prices(df_price_full_standardized)

if __name__ == "__main__":
    main()
