# File: marketml/indicators/create_enriched_data.py
import pandas as pd
from pathlib import Path
import sys

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import các module cần thiết TỪ marketml
try:
    from marketml.data.loader import price_loader
    from marketml.data.loader import preprocess # Hàm add_technical_indicators nằm ở đây
except ImportError as e:
    print(f"Error importing marketml modules: {e}"); exit()

# Đường dẫn file output
OUTPUT_DIR = project_root / "marketml" / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "price_data_with_indicators.csv"

def create_and_save_enriched_data():
    """
    Loads raw price data, standardizes it, adds technical indicators,
    and saves the enriched data to a CSV file.
    """
    # --- Bước 1: Load Dữ liệu ---
    try:
        print("Loading real data...")
        raw_real_df = price_loader.load_price_data()
        print("Real data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Raw data file not found. Check path in price_loader.py.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Bước 2: Chuẩn hóa Dữ liệu ---
    print("\nStandardizing data...")
    try:
        standardized_df = preprocess.standardize_data(raw_real_df)
        print("Data standardized.")
    except Exception as e:
        print(f"Error standardizing data: {e}")
        return

    # --- Kiểm tra các cột quan trọng ---
    required_cols = ['date', 'ticker', 'close']
    missing_cols = [col for col in required_cols if col not in standardized_df.columns]
    if missing_cols:
        print(f"\nError: Missing required columns after standardization: {missing_cols}")
        return

    # --- Bước 3: Thêm các chỉ báo kỹ thuật (Grouped by Ticker) ---
    print("\nAdding technical indicators (grouped by ticker)...")
    try:
        all_indicators_df_list = []
        processed_tickers = 0
        for ticker, group_df in standardized_df.groupby('ticker'):
            print(f"  Processing ticker: {ticker} ({len(group_df)} rows)")
            group_df_sorted = group_df.sort_values('date').copy()
            group_with_indicators = preprocess.add_technical_indicators(
                group_df_sorted,
                price_col='close'
            )
            all_indicators_df_list.append(group_with_indicators)
            processed_tickers += 1

        if not all_indicators_df_list:
            print("\nWarning: No ticker groups found or processed. No data to save.")
            return

        df_with_indicators = pd.concat(all_indicators_df_list, ignore_index=True)
        print(f"\nTechnical indicators added successfully for {processed_tickers} tickers.")

    except KeyError as e:
        print(f"\nError: Column not found during indicator calculation - {e}.")
        return
    except Exception as e:
        print(f"\nError adding technical indicators: {e}")
        return

    # --- Bước 4: Lưu Dữ liệu đã làm giàu ---
    try:
        print(f"\nSaving enriched data to: {OUTPUT_FILE.resolve()}") # In đường dẫn tuyệt đối
        # Tạo thư mục nếu chưa tồn tại
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_with_indicators.to_csv(OUTPUT_FILE, index=False)
        print("Enriched data saved successfully.")

        # === KHÔI PHỤC LỆNH PRINT ĐỂ XEM DỮ LIỆU ===
        print("\nFinal Data Info:")
        df_with_indicators.info()
        print("\nFinal Data Sample (last 15 rows overall, for a sample ticker):")
        if not df_with_indicators.empty:
            # Lấy một ticker mẫu để hiển thị cho gọn
            sample_ticker_display = df_with_indicators['ticker'].unique()[0]
            print(f"--- Displaying last 15 rows for ticker: {sample_ticker_display} ---")
            print(df_with_indicators[df_with_indicators['ticker'] == sample_ticker_display].tail(15).to_string())
        else:
            print("DataFrame is empty, no sample to display.")
        # =========================================

    except Exception as e:
        print(f"\nError saving enriched data: {e}")

if __name__ == "__main__":
    create_and_save_enriched_data()