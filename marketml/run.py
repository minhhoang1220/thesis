from marketml.data_handling.loader import load_price_data, load_financial_data
from marketml.data_handling.preprocess import standardize_data  # Import hàm chuẩn hóa
# from marketml.plot.visualizer import plot_all_closing_prices

def main():
    # Load and standardize price data
    print("✅ PRICE DATA")
    df_price = load_price_data(nrows=5)
    df_price_standardized = standardize_data(df_price)  # Chuẩn hóa dữ liệu giá
    print(df_price_standardized)

    # Load and standardize financial data
    print("\n✅ FINANCIAL DATA")
    df_fin = load_financial_data(nrows=5)
    df_fin_standardized = standardize_data(df_fin)  # Chuẩn hóa dữ liệu tài chính
    print(df_fin_standardized)

    # # Full data for plotting
    # df_price_full = load_price_data()  # Load full price data for plotting
    # df_price_full_standardized = standardize_data(df_price_full)  # Chuẩn hóa dữ liệu giá đầy đủ

    # # Plot and save figures
    # plot_all_closing_prices(df_price_full_standardized)

if __name__ == "__main__":
    main()
