import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import warnings
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import sys

# Thêm thư mục gốc vào sys.path
project_root_cli_fc = Path(__file__).resolve().parents[1] # .ndmh/
sys.path.insert(0, str(project_root_cli_fc))

try:
    from marketml.configs import configs # <--- IMPORT CONFIGS
except ImportError:
    print("Error: Could not import configs.py. Make sure it's in marketml/configs/ and __init__.py exists.")
    exit()

# ----- Cấu hình -----
FORECAST_YEAR = configs.FORECAST_YEAR_TARGET
TRAINING_YEARS = configs.FORECAST_TRAINING_YEARS
TREND_THRESHOLD = configs.FORECAST_TREND_THRESHOLD
APPROX_TRADING_DAYS_PER_YEAR = configs.APPROX_TRADING_DAYS_PER_YEAR
# Đường dẫn đến file enriched data đã được tạo bởi create_enriched_data.py
ENRICHED_DATA_FILE_FOR_FORECAST = configs.ENRICHED_DATA_FOR_FORECAST # Hoặc configs.ENRICHED_DATA_FILE
# Đường dẫn output cho forecast
FORECAST_OUTPUT_DIR = configs.FORECASTS_DIR
# --------------------

# ===== BỎ QUA WARNINGS =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ============================

def generate_future_forecast(forecast_year=FORECAST_YEAR, training_years=TRAINING_YEARS):
    print(f"--- Generating Forecast for Year {forecast_year} using last {training_years} years of data ---")

    # --- Bước 1: Load Dữ liệu ĐÃ LÀM GIÀU ---
    print(f"DEBUG: Attempting to load from: {ENRICHED_DATA_FILE_FOR_FORECAST.resolve()}")
    try:
        df_hist = pd.read_csv(ENRICHED_DATA_FILE_FOR_FORECAST, parse_dates=['date'])
        print("Enriched data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Enriched data file not found at '{ENRICHED_DATA_FILE_FOR_FORECAST.resolve()}'.")
        print(f"Ensure '{configs.ENRICHED_DATA_FILE.name}' was created by 'create_enriched_data.py'.") # Thông báo file chính
        return None
    except Exception as e:
        print(f"Error loading enriched data: {e}"); return None

    # --- Các bước 2, 3, 4, 5 giữ nguyên như trước ---
    # ... (Xác định khoảng train, tạo ngày tương lai, lặp qua ticker, huấn luyện/dự báo ARIMA, lưu kết quả) ...

    # --- Bước 2: Xác định khoảng thời gian huấn luyện cuối cùng ---
    last_hist_date = df_hist['date'].max()
    training_end_date = last_hist_date
    training_start_date = training_end_date - pd.Timedelta(days=365 * training_years)
    print(f"Final training period: {training_start_date.date()} to {training_end_date.date()}")
    final_train_df = df_hist[(df_hist['date'] >= training_start_date) & (df_hist['date'] <= training_end_date)].copy()
    if final_train_df.empty: print("Error: No data found in the final training period."); return None

    # --- Bước 3: Ước tính số kỳ dự báo và tạo ngày tương lai ---
    n_periods_forecast = APPROX_TRADING_DAYS_PER_YEAR
    forecast_start_date = last_hist_date + pd.Timedelta(days=1)
    forecast_end_date = pd.Timestamp(f"{forecast_year}-12-31")
    future_dates = pd.bdate_range(start=forecast_start_date, end=forecast_end_date)
    n_periods_forecast = len(future_dates)
    print(f"Forecasting for {n_periods_forecast} business days from {future_dates[0].date()} to {future_dates[-1].date()}")

    # --- Bước 4: Huấn luyện và Dự báo cho từng Ticker ---
    all_forecasts = []
    unique_tickers = final_train_df['ticker'].unique()
    for ticker in unique_tickers:
        print(f"  Processing forecast for ticker: {ticker}")
        ticker_train_data = final_train_df[final_train_df['ticker'] == ticker]
        y_train_final_input = ticker_train_data['close'].pct_change().dropna()
        if y_train_final_input.empty: print(f"    Skipping {ticker}: Not enough training data."); continue
        try:
            adf_result = adfuller(y_train_final_input); p_value = adf_result[1]; d = 0
            if p_value > 0.05: d = 1
            final_arima_model = pm.auto_arima(y_train_final_input, d=d, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False)
            forecast_pct_change = final_arima_model.predict(n_periods=n_periods_forecast)
            forecast_trend = np.select([(forecast_pct_change > TREND_THRESHOLD), (forecast_pct_change < -TREND_THRESHOLD)],[1, -1], default=0)
            ticker_forecast_df = pd.DataFrame({'forecast_date': future_dates,'ticker': ticker,'forecast_pct_change': forecast_pct_change,'forecast_trend': forecast_trend})
            all_forecasts.append(ticker_forecast_df)
        except Exception as e: print(f"    Error processing forecast for {ticker}: {e}")

    # --- Bước 5: Ghép và Lưu Kết quả Dự báo ---
    if not all_forecasts: print("\nNo forecasts were generated."); return None
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    print(f"\nForecast generated successfully for {len(unique_tickers)} tickers.")
    try:
        FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Sử dụng biến từ configs
        forecast_file = FORECAST_OUTPUT_DIR / f"arima_forecast_{forecast_year}.csv"
        final_forecast_df.to_csv(forecast_file, index=False)
        print(f"Forecast saved to: {forecast_file.resolve()}")
    except Exception as e: print(f"Error saving forecast file: {e}")
    print("\nSample Forecast (last 10 rows overall):")
    print(final_forecast_df.tail(10).to_string())
    return final_forecast_df

if __name__ == "__main__":
    forecast_df = generate_future_forecast()
    if forecast_df is not None: print("\nForecast generation script finished.")
    else: print("\nForecast generation script failed.")