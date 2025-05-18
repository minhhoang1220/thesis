# File: marketml/indicators/create_enriched_data.py
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import các module cần thiết TỪ marketml
try:
    from marketml.data.loader import price_loader
    from marketml.data.loader import preprocess
    from marketml.configs import configs
except ImportError as e:
    print(f"Error importing marketml modules: {e}"); exit()

# === Import Arch cho GARCH ===
try:
    from arch import arch_model
    ARCH_INSTALLED = True
except ImportError:
    print("Warning: arch library not installed. GARCH volatility feature will not be calculated.")
    ARCH_INSTALLED = False
    arch_model = None
# ==============================

# Đường dẫn file output
OUTPUT_DIR = configs.ENRICHED_DATA_DIR 
OUTPUT_FILE = configs.ENRICHED_DATA_FILE #

# --- Tham số cho GARCH feature ---
GARCH_WINDOW = configs.GARCH_WINDOW 
GARCH_FORECAST_HORIZON = configs.GARCH_FORECAST_HORIZON 
# ---------------------------------

def calculate_garch_volatility_feature(series_returns: pd.Series, window: int, horizon: int):
    """
    Tính toán biến động dự báo 1 bước từ GARCH(1,1) trên một cửa sổ trượt.
    Args:
        series_returns (pd.Series): Chuỗi daily returns (đã nhân 100).
        window (int): Kích thước cửa sổ để fit GARCH.
        horizon (int): Số bước dự báo (thường là 1).
    Returns:
        pd.Series: Chuỗi chứa giá trị biến động dự báo (đã lấy căn bậc hai của variance).
                   Index giống với series_returns, có NaN ở đầu.
    """
    if not ARCH_INSTALLED or len(series_returns) < window + 10: # Cần đủ điểm để fit và rolling
        return pd.Series(np.nan, index=series_returns.index)

    volatility_forecasts = pd.Series(np.nan, index=series_returns.index)

    # Dùng rolling window để fit GARCH và dự báo
    # Lưu ý: việc này có thể rất chậm
    for i in range(window, len(series_returns)):
        current_window_returns = series_returns.iloc[i-window:i]
        if len(current_window_returns.dropna()) < window * 0.8: # Cần đủ dữ liệu không NaN trong cửa sổ
            continue
        try:
            gm = arch_model(current_window_returns.dropna(), vol='Garch', p=1, q=1, rescale=False)
            res = gm.fit(disp='off', show_summary=False)
            # Dự báo h-step ahead variance
            forecast = res.forecast(horizon=horizon, reindex=False) # reindex=False để lấy dự báo cho điểm tiếp theo của cửa sổ
            # Lấy variance dự báo cho bước đầu tiên (h=1)
            predicted_variance = forecast.variance.iloc[-1, 0] # Lấy giá trị cuối cùng của hàng cuối cùng, cột đầu tiên (h.1)
            if i < len(volatility_forecasts):
                volatility_forecasts.iloc[i] = np.sqrt(predicted_variance) / 100 # Chia lại cho 100 để về thang đo gốc của return
        except Exception as e:
            # print(f"GARCH fit/forecast error at index {i}: {e}") # Bỏ comment để debug
            pass # Bỏ qua nếu GARCH không hội tụ
    return volatility_forecasts

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
    print("\nStandardizing data..."); standardized_df = preprocess.standardize_data(raw_real_df); print("Data standardized.")
    required_cols = ['date', 'ticker', 'close']; missing_cols = [col for col in required_cols if col not in standardized_df.columns]
    if missing_cols: print(f"\nError: Missing required cols: {missing_cols}"); return

    all_enriched_df_list = []
    processed_tickers_count = 0

    for ticker, group_df in standardized_df.groupby('ticker'):
        print(f"  Processing ticker: {ticker} ({len(group_df)} rows)")
        group_df_sorted = group_df.sort_values('date').copy()

        if ARCH_INSTALLED:
            print(f"    Calculating GARCH(1,1) volatility feature for {ticker} (window={GARCH_WINDOW})... This may take a while.")
            daily_returns = group_df_sorted['close'].pct_change().dropna() * 100
            garch_vol = calculate_garch_volatility_feature(daily_returns, GARCH_WINDOW, GARCH_FORECAST_HORIZON)
            group_df_sorted['garch_vol_forecast'] = garch_vol.shift(1) # Shift(1) để là feature của ngày hiện tại
            print(f"    GARCH volatility feature calculated for {ticker}.")
        else:
            group_df_sorted['garch_vol_forecast'] = np.nan

        group_with_ta = preprocess.add_technical_indicators(
            group_df_sorted,
            price_col='close',
            volume_col='volume',
            # Truyền các tham số TA từ configs
            rsi_window=configs.RSI_WINDOW,
            macd_fast=configs.MACD_FAST, macd_slow=configs.MACD_SLOW, macd_signal=configs.MACD_SIGNAL,
            bb_window=configs.BB_WINDOW,
            sma_window=configs.SMA_WINDOW, ema_window=configs.EMA_WINDOW,
            rolling_stat_windows=configs.ROLLING_STAT_WINDOWS,
            price_zscore_window=configs.PRICE_ZSCORE_WINDOW
        )
        all_enriched_df_list.append(group_with_ta)
        processed_tickers_count += 1
    # ... (concat và save data, sử dụng OUTPUT_FILE, OUTPUT_DIR từ configs) ...
    if not all_enriched_df_list: print("\nWarning: No data to save."); return
    df_final_enriched = pd.concat(all_enriched_df_list, ignore_index=True)
    print(f"\nAll features (TA & GARCH) added successfully for {processed_tickers_count} tickers.")

    try:
        print(f"\nSaving enriched data to: {OUTPUT_FILE.resolve()}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_final_enriched.to_csv(OUTPUT_FILE, index=False)
        print("Enriched data saved successfully.")
        # df_final_enriched.info() # Bỏ comment nếu muốn xem info
    except Exception as e: print(f"\nError saving enriched data: {e}")


if __name__ == "__main__":
    create_and_save_enriched_data()