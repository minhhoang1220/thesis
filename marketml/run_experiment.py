import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

# ===== BỎ QUA WARNINGS =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Bỏ comment dòng dưới nếu muốn tắt cả ValueWarning của statsmodels
# from statsmodels.tools.sm_exceptions import ValueWarning
# warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels.tsa.base.tsa_model')
# ============================

# Import các thư viện cho ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

# === Import hàm metrics từ utils ===
try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    print("Error: Could not import 'marketml.utils.metrics'.")
    print("Ensure 'marketml/utils/metrics.py' exists and the script is run from the project root.")
    exit()
# ====================================


# ==============================================================================
# HÀM TẠO CỬA SỔ ROLLING/EXPANDING (Giữ nguyên)
# ==============================================================================
def create_time_series_cv_splits(
    df: pd.DataFrame, date_col: str, ticker_col: str, initial_train_period: pd.Timedelta,
    test_period: pd.Timedelta, step_period: pd.Timedelta, expanding: bool = False
):
    print(f"\nGenerating CV splits ({'Expanding' if expanding else 'Rolling'} Window):")
    print(f"  Initial Train: {initial_train_period}, Test Period: {test_period}, Step Period: {step_period}")
    min_date = df[date_col].min(); max_date = df[date_col].max()
    if min_date + initial_train_period + test_period > max_date:
        print("Warning: Not enough data for even one train-test split.")
        return
    current_train_start_date = min_date; current_train_end_date = min_date + initial_train_period
    split_index = 0
    while True:
        current_test_start_date = current_train_end_date; current_test_end_date = current_test_start_date + test_period
        if current_test_end_date > max_date: print(f"Stopping: Test end date {current_test_end_date.date()} exceeds max data date {max_date.date()}."); break
        train_mask = (df[date_col] >= current_train_start_date) & (df[date_col] < current_train_end_date)
        test_mask = (df[date_col] >= current_test_start_date) & (df[date_col] < current_test_end_date)
        train_split_df = df.loc[train_mask].copy(); test_split_df = df.loc[test_mask].copy()
        if not train_split_df.empty and not test_split_df.empty:
            print(f"  Split {split_index}: Train Range [{train_split_df[date_col].min().date()}:{train_split_df[date_col].max().date()}], Test Range [{test_split_df[date_col].min().date()}:{test_split_df[date_col].max().date()}]")
            yield split_index, train_split_df, test_split_df; split_index += 1
        if not expanding: current_train_start_date += step_period
        current_train_end_date += step_period
    print(f"Total splits generated: {split_index}")

# ==============================================================================
# SCRIPT CHÍNH (run_experiment.py)
# ==============================================================================

# --- Bước 1: Load Dữ liệu ĐÃ LÀM GIÀU ---
project_root = Path(__file__).resolve().parent
ENRICHED_DATA_DIR = project_root / "data" / "processed"
ENRICHED_DATA_FILE = ENRICHED_DATA_DIR / "price_data_with_indicators.csv"

try:
    print(f"Loading enriched data from: {ENRICHED_DATA_FILE}")
    df_with_indicators = pd.read_csv(ENRICHED_DATA_FILE, parse_dates=['date'])
    print("Enriched data loaded successfully.")
except FileNotFoundError: print(f"Error: Enriched data file not found at '{ENRICHED_DATA_FILE}'. Run create_enriched_data.py first."); exit()
except Exception as e: print(f"Error loading enriched data: {e}"); exit()

# --- Kiểm tra nhanh các cột ---
required_cols_final = ['date', 'ticker', 'close', 'RSI', 'MACD']
missing_cols_final = [col for col in required_cols_final if col not in df_with_indicators.columns]
if missing_cols_final: print(f"\nError: Missing expected columns in enriched data: {missing_cols_final}"); exit()
if not pd.api.types.is_datetime64_any_dtype(df_with_indicators['date']): print("\nError: 'date' column is not datetime type."); exit()


# --- Bước 2: Chia Train-Test và Chuẩn bị Dữ liệu ---
if not df_with_indicators.empty:
    print("\nProceeding to Train-Test Split and Data Preparation...")
    # --- Định nghĩa tham số cửa sổ, target, features ---
    INITIAL_TRAIN_YEARS = 3; TEST_YEARS = 1; STEP_YEARS = 1; USE_EXPANDING_WINDOW = False
    initial_train_td = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS); test_td = pd.Timedelta(days=365 * TEST_YEARS); step_td = pd.Timedelta(days=365 * STEP_YEARS)
    TREND_THRESHOLD = 0.002; LAG_PERIODS = [1, 3, 5]
    FEATURE_COLS = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'EMA_20', 'volume']
    for p in LAG_PERIODS: FEATURE_COLS.append(f'pct_change_lag_{p}')
    TARGET_COL_PCT = 'target_pct_change'; TARGET_COL_TREND = 'target_trend'

    # --- Tạo các split ---
    cv_splitter = create_time_series_cv_splits(df=df_with_indicators, date_col='date', ticker_col='ticker', initial_train_period=initial_train_td, test_period=test_td, step_period=step_td, expanding=USE_EXPANDING_WINDOW)
    # --- Lặp qua các split ---
    all_split_results = {} # Dictionary để lưu kết quả của TẤT CẢ các split
    for split_idx, train_df_orig, test_df_orig in cv_splitter:
        print(f"\n===== Processing CV Split {split_idx} =====")
        train_df = train_df_orig.copy(); test_df = test_df_orig.copy()
        results_this_split = {} # Dictionary để lưu kết quả của split NÀY

        # ---- 1. Tạo Target Variables (y) ----
        # print("  Creating target variables...") # Tắt bớt print
        for df_split in [train_df, test_df]:
            # Tính % thay đổi cho ngày hôm sau (dùng để tạo trend và input cho ARIMA nếu cần)
            df_split[TARGET_COL_PCT] = df_split.groupby('ticker')['close'].pct_change().shift(-1)
            # Tạo target trend (1, -1, 0)
            conditions = [(df_split[TARGET_COL_PCT]>TREND_THRESHOLD), (df_split[TARGET_COL_PCT]<-TREND_THRESHOLD)]
            choices = [1, -1]; df_split[TARGET_COL_TREND] = np.select(conditions, choices, default=0)

        # ---- 2. Tạo Features Lags (X) ----
        # print("  Creating lag features...") # Tắt bớt print
        for df_split in [train_df, test_df]:
            # Tính pct_change của ngày hiện tại (để làm lag)
            pct_change_col = df_split.groupby('ticker')['close'].pct_change()
            for p in LAG_PERIODS: df_split[f'pct_change_lag_{p}'] = pct_change_col.shift(p)

        # ---- 3. Chọn X, y và Xử lý NaN trong X bằng Imputation (Cho ML) ----
        # Vẫn thực hiện để có y_test_ml cho việc đánh giá ARIMA trend
        # print(f"  Imputing NaNs in ML features (X)...") # Tắt bớt print
        X_train_raw = train_df[FEATURE_COLS]; X_test_raw = test_df[FEATURE_COLS]
        y_train_trend_ml = train_df[TARGET_COL_TREND]; y_test_trend_ml = test_df[TARGET_COL_TREND]
        imputer = SimpleImputer(strategy='mean'); X_train_imputed_np = imputer.fit_transform(X_train_raw); X_test_imputed_np = imputer.transform(X_test_raw)
        X_train_imputed = pd.DataFrame(X_train_imputed_np, index=X_train_raw.index, columns=X_train_raw.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed_np, index=X_test_raw.index, columns=X_test_raw.columns)

        # ---- 4. Align X và y (Lấy y_test_ml cuối cùng) ----
        valid_y_train_idx = y_train_trend_ml.dropna().index; valid_y_test_idx = y_test_trend_ml.dropna().index
        # Chỉ cần lấy y_test_ml cuối cùng để đánh giá ARIMA
        # X_train_ml = X_train_imputed.loc[valid_y_train_idx]; y_train_ml = y_train_trend_ml.loc[valid_y_train_idx] # Không cần cho ARIMA
        # X_test_ml = X_test_imputed.loc[valid_y_test_idx] # Không cần cho ARIMA
        y_test_final_trend = y_test_trend_ml.loc[valid_y_test_idx] # Đây là target trend thực tế cuối cùng

        # Kiểm tra xem có đủ dữ liệu target không
        if y_test_final_trend.empty:
             print("  Skipping split: Empty final target data after alignment.")
             continue

        # ---- 5. Scaling Features (X_ml) ----
        # Vẫn thực hiện để giữ cấu trúc, dù ARIMA không dùng
        # print("  Scaling ML features...") # Tắt bớt print
        # scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_ml); X_test_scaled = scaler.transform(X_test_ml)
        # X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_ml.index, columns=X_train_ml.columns)
        # X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_ml.index, columns=X_test_ml.columns)


        # --------------------------------------------------------------
        # Phần 6: Chạy và đánh giá mô hình ARIMA
        # --------------------------------------------------------------
        print("\n--- Training and Evaluating ARIMA Model ---")
        arima_predictions_all = []; arima_actual_trends_all = []
        # Lặp qua từng ticker
        for ticker in train_df['ticker'].unique():
            # print(f"  Processing ARIMA for ticker: {ticker}") # Tắt bớt print
            train_ticker_df = train_df[train_df['ticker'] == ticker]; test_ticker_df = test_df[test_df['ticker'] == ticker]

            # Input cho ARIMA là pct_change() gốc, drop NaN đầu tiên
            y_train_arima_input = train_ticker_df['close'].pct_change().dropna()
            y_test_arima_input = test_ticker_df['close'].pct_change().dropna() # Dùng để lấy độ dài

            # Lấy target trend thực tế tương ứng với index của y_test_arima_input
            potential_trends = test_df.loc[test_df['ticker'] == ticker, TARGET_COL_TREND]
            y_test_trend_ticker = potential_trends.reindex(y_test_arima_input.index).dropna()

            if y_train_arima_input.empty or y_test_arima_input.empty or y_test_trend_ticker.empty:
                 # print(f"    Skipping ARIMA for {ticker}: Not enough data.") # Tắt bớt print
                 continue

            try:
                # ADF Test và auto_arima
                adf_result = adfuller(y_train_arima_input); p_value = adf_result[1]; d = 0
                if p_value > 0.05: d = 1
                auto_model = pm.auto_arima(y_train_arima_input, d=d, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False)

                # Dự đoán
                n_periods = len(y_test_arima_input)
                arima_preds_pct = auto_model.predict(n_periods=n_periods)

                # Align kết quả
                if len(arima_preds_pct) == len(y_test_trend_ticker): y_test_trend_ticker_aligned = y_test_trend_ticker
                elif len(y_test_trend_ticker) > len(arima_preds_pct): y_test_trend_ticker_aligned = y_test_trend_ticker.iloc[-len(arima_preds_pct):]
                else: arima_preds_pct = arima_preds_pct[-len(y_test_trend_ticker):]; y_test_trend_ticker_aligned = y_test_trend_ticker

                # Lưu kết quả đã align
                arima_predictions_all.extend(arima_preds_pct); arima_actual_trends_all.extend(y_test_trend_ticker_aligned)

            except Exception as e:
                print(f"    Error processing ARIMA for {ticker} in split {split_idx}: {e}")

        # ---- Đánh giá ARIMA sử dụng hàm metrics ----
        if arima_predictions_all:
            # Chuyển dự đoán pct_change thành trend
            arima_preds_trend = np.select([(np.array(arima_predictions_all) > TREND_THRESHOLD), (np.array(arima_predictions_all) < -TREND_THRESHOLD)], [1, -1], default=0)
            # Gọi hàm tính metrics
            arima_metrics_results = metrics.calculate_classification_metrics(
                y_true=arima_actual_trends_all, # Sử dụng trend thực tế đã align
                y_pred=arima_preds_trend,
                model_name="ARIMA"
            )
            results_this_split.update(arima_metrics_results) # Cập nhật dict kết quả
        else:
            print("    ARIMA: No predictions were generated for this split.")
            results_this_split.update({"ARIMA_Accuracy": np.nan, "ARIMA_F1_Macro": np.nan, "ARIMA_F1_Weighted": np.nan, "ARIMA_Precision_Macro": np.nan, "ARIMA_Recall_Macro": np.nan})


        # --------------------------------------------------------------
        # Phần 7: Placeholder cho Mô hình Machine Learning (Tạm thời bỏ qua)
        # --------------------------------------------------------------
        # print("\n--- Skipping ML Models for now ---")
        # Thêm kết quả NaN hoặc mặc định cho ML để bảng tổng hợp có cột
        results_this_split.update({
            "RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan, "RandomForest_F1_Weighted": np.nan,
            "XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan, "XGBoost_F1_Weighted": np.nan,
            # Thêm cho LSTM, Transformer nếu muốn
        })


        # Lưu kết quả của split này vào dictionary tổng
        all_split_results[split_idx] = results_this_split
        print(f"===== Finished CV Split {split_idx} =====")
        # break # Bỏ comment nếu chỉ muốn chạy thử 1 split

    # --- Bước 5: Tổng hợp và so sánh kết quả từ tất cả các split ---
    print("\n===== Aggregating Results Across All Splits =====")
    if all_split_results:
        final_results_df = pd.DataFrame.from_dict(all_split_results, orient='index')
        print("\n--- Performance Summary (Mean +/- Std Dev) ---")
        summary = final_results_df.agg(['mean', 'std']).T
        # Chỉ hiển thị các cột có giá trị (không phải NaN)
        summary_valid = summary.dropna(subset=['mean'])
        if not summary_valid.empty:
             summary_valid['mean_std'] = summary_valid.apply(lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1)
             print(summary_valid[['mean_std']])
        else:
             print("No valid performance metrics to display.")
        # print(final_results_df) # In toàn bộ kết quả nếu muốn xem chi tiết từng split
    else:
        print("No results to aggregate.")

else:
    print("\nEnriched data is empty or could not be loaded.")

print("\nScript finished.")