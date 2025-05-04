import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import warnings
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

# ----- Cấu hình -----
FORECAST_YEAR = 2025
TRAINING_YEARS = 3
TREND_THRESHOLD = 0.002
APPROX_TRADING_DAYS_PER_YEAR = 252
# --------------------

# ===== BỎ QUA WARNINGS =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ============================

def generate_future_forecast(forecast_year=FORECAST_YEAR, training_years=TRAINING_YEARS):
    print(f"--- Generating Forecast for Year {forecast_year} using last {training_years} years of data ---")

    # --- Bước 1: Load Dữ liệu ĐÃ LÀM GIÀU ---
    # *** SỬA Ở ĐÂY: Xác định đúng thư mục chứa script này (marketml) ***
    current_script_dir = Path(__file__).resolve().parent
    # => current_script_dir sẽ là .../.ndmh/marketml/

    # Đường dẫn bây giờ sẽ đúng
    ENRICHED_DATA_DIR = current_script_dir / "data" / "processed"
    ENRICHED_DATA_FILE = ENRICHED_DATA_DIR / "price_data_with_indicators.csv"
    # => ENRICHED_DATA_FILE sẽ là .../.ndmh/marketml/data/processed/price_data_with_indicators.csv

    # Thêm debug để chắc chắn
    print(f"DEBUG: Attempting to load from: {ENRICHED_DATA_FILE.resolve()}")

    try:
        # print(f"Loading enriched data from: {ENRICHED_DATA_FILE}") # Đã có debug ở trên
        df_hist = pd.read_csv(ENRICHED_DATA_FILE, parse_dates=['date'])
        print("Enriched data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Enriched data file not found at '{ENRICHED_DATA_FILE.resolve()}'.")
        print("Ensure 'marketml/indicators/create_enriched_data.py' ran successfully and saved the file to the correct location.")
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
        # *** Sửa ở đây: Xác định output_dir dựa trên current_script_dir ***
        output_dir = current_script_dir.parent / "forecasts" # Lưu vào thư mục forecasts ngang hàng với marketml
        output_dir.mkdir(parents=True, exist_ok=True)
        forecast_file = output_dir / f"arima_forecast_{forecast_year}.csv"
        final_forecast_df.to_csv(forecast_file, index=False)
        print(f"Forecast saved to: {forecast_file.resolve()}") # In đường dẫn tuyệt đối
    except Exception as e: print(f"Error saving forecast file: {e}")
    print("\nSample Forecast (last 10 rows overall):")
    print(final_forecast_df.tail(10).to_string())
    return final_forecast_df

if __name__ == "__main__":
    forecast_df = generate_future_forecast()
    if forecast_df is not None: print("\nForecast generation script finished.")
    else: print("\nForecast generation script failed.")