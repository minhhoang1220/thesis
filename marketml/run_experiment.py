import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
import warnings
import sys

# ===== THÊM THƯ MỤC GỐC VÀO PATH =====
project_root_from_script = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root_from_script)) # Đã bỏ comment nếu chạy bằng python -m

# ===== BỎ QUA WARNINGS =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels.tsa.base.tsa_model')
except ImportError: pass
# ============================

# === Import các module đã tách ===
try:
    from marketml.utils import metrics
    from marketml.models import (arima_model, rf_model, lstm_model,
                                 transformer_model, keras_utils, xgboost_model)
except ModuleNotFoundError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure the project structure is correct, __init__.py files exist,")
    print("and the script is run from the project root directory (.ndmh) or using 'python -m ...'.")
    exit()
# ================================

# === Kiểm tra TensorFlow một lần ===
TF_INSTALLED = keras_utils.KERAS_AVAILABLE
if not TF_INSTALLED:
     print("Warning: TensorFlow/Keras not available. LSTM and Transformer models will be skipped.")
# =================================

# === Kiểm tra XGBoost ===
try:
    import xgboost
    XGB_INSTALLED = True
except ImportError:
    print("Warning: xgboost not installed. XGBoost model will be skipped.")
    XGB_INSTALLED = False
# ========================

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
    if min_date + initial_train_period + test_period > max_date: return
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
# HÀM CHUẨN BỊ DỮ LIỆU TRONG SPLIT (Tách ra cho gọn)
# ==============================================================================
def prepare_split_data(train_df_orig, test_df_orig, feature_cols, lag_periods,
                       target_col_pct, target_col_trend, trend_threshold, n_timesteps):
    """Chuẩn bị tất cả dữ liệu cần thiết cho các mô hình trong một split."""
    print("  Preparing data for split...")
    train_df = train_df_orig.copy(); test_df = test_df_orig.copy()
    prepared_data = {'data_valid': False} # Mặc định là không hợp lệ

    try:
        # ---- Tạo Target ----
        for df_split in [train_df, test_df]:
            df_split[target_col_pct] = df_split.groupby('ticker')['close'].pct_change().shift(-1)
            conditions = [(df_split[target_col_pct]>trend_threshold), (df_split[target_col_pct]<-trend_threshold)]
            choices = [1, -1]; df_split[target_col_trend] = np.select(conditions, choices, default=0)

        # ---- Tạo Lags ----
        for df_split in [train_df, test_df]:
            pct_change_col = df_split.groupby('ticker')['close'].pct_change()
            for p in lag_periods: df_split[f'pct_change_lag_{p}'] = pct_change_col.shift(p)

        # ---- Impute & Align cho ML ----
        X_train_raw = train_df[feature_cols]; X_test_raw = test_df[feature_cols]
        y_train_trend_ml = train_df[target_col_trend]; y_test_trend_ml = test_df[target_col_trend]

        imputer = SimpleImputer(strategy='mean'); X_train_imputed_np = imputer.fit_transform(X_train_raw); X_test_imputed_np = imputer.transform(X_test_raw)
        X_train_imputed = pd.DataFrame(X_train_imputed_np, index=X_train_raw.index, columns=X_train_raw.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed_np, index=X_test_raw.index, columns=X_test_raw.columns)

        valid_y_train_idx = y_train_trend_ml.dropna().index; valid_y_test_idx = y_test_trend_ml.dropna().index
        X_train_ml = X_train_imputed.loc[valid_y_train_idx]; y_train_ml = y_train_trend_ml.loc[valid_y_train_idx]
        X_test_ml = X_test_imputed.loc[valid_y_test_idx]; y_test_ml = y_test_trend_ml.loc[valid_y_test_idx]

        if X_train_ml.empty or X_test_ml.empty:
            print("  Warning: Empty ML data after alignment. Cannot proceed with ML/Sequence models.")
            # Vẫn trả về train/test df gốc cho ARIMA nếu cần
            prepared_data.update({'train_df': train_df, 'test_df': test_df})
            return prepared_data # data_valid vẫn là False

        prepared_data['y_test_ml'] = y_test_ml # Target gốc cho ML/ARIMA trend eval

        # ---- Scaling cho ML ----
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_ml); X_test_scaled = scaler.transform(X_test_ml)
        prepared_data['X_train_scaled'] = X_train_scaled
        prepared_data['y_train_ml'] = y_train_ml.values
        prepared_data['X_test_scaled'] = X_test_scaled
        prepared_data['feature_names'] = X_train_ml.columns.tolist()

        # ---- Chuẩn bị cho Keras ----
        y_train_keras = y_train_ml + 1; y_test_keras = y_test_ml + 1
        prepared_data['n_classes'] = 3

        X_train_seq, y_train_seq = keras_utils.create_sequences(X_train_scaled, y_train_keras.values, n_timesteps)
        X_test_seq, y_test_seq = keras_utils.create_sequences(X_test_scaled, y_test_keras.values, n_timesteps)
        # Lấy đúng target trend gốc tương ứng với chuỗi test
        if len(y_test_ml) >= n_timesteps and X_test_seq.shape[0] > 0:
             prepared_data['y_test_seq_original_trend'] = y_test_ml.iloc[n_timesteps:].values[:len(y_test_seq)] # Cắt theo độ dài y_test_seq
        else:
             prepared_data['y_test_seq_original_trend'] = np.array([])


        prepared_data.update({
            'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq,
            'X_test_seq': X_test_seq
        })

        if X_train_seq.shape[0] > 0:
             class_weights_keras = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
             prepared_data['class_weight_dict'] = dict(enumerate(class_weights_keras))
             print(f"  Sequence shapes: X_train={X_train_seq.shape}, X_test={X_test_seq.shape}")
             print(f"  Class weights: {prepared_data['class_weight_dict']}")
        else:
             prepared_data['class_weight_dict'] = {}
             print("  Warning: Not enough data to create sequences.")

        prepared_data['data_valid'] = True # Đánh dấu dữ liệu hợp lệ
        prepared_data['train_df'] = train_df # Cho ARIMA
        prepared_data['test_df'] = test_df   # Cho ARIMA

    except Exception as e:
        print(f"Error during data preparation for split: {e}")
        # Đảm bảo trả về data_valid = False
        prepared_data['data_valid'] = False
        if 'train_df' not in prepared_data: prepared_data['train_df'] = train_df_orig.copy()
        if 'test_df' not in prepared_data: prepared_data['test_df'] = test_df_orig.copy()

    print("  Data preparation finished for split.")
    return prepared_data

# ==============================================================================
# SCRIPT CHÍNH (run_experiment.py)
# ==============================================================================

def run_all_experiments():
    # --- Bước 1: Load Dữ liệu ĐÃ LÀM GIÀU ---
    project_root = Path(__file__).resolve().parent
    ENRICHED_DATA_DIR = project_root / "data" / "processed"
    ENRICHED_DATA_FILE = ENRICHED_DATA_DIR / "price_data_with_indicators.csv"

    try:
        print(f"Loading enriched data from: {ENRICHED_DATA_FILE}")
        df_with_indicators = pd.read_csv(ENRICHED_DATA_FILE, parse_dates=['date'])
        print("Enriched data loaded successfully.")
    except FileNotFoundError: print(f"Error: Enriched data file not found at '{ENRICHED_DATA_FILE}'. Run 'marketml/indicators/create_enriched_data.py' first."); exit()
    except Exception as e: print(f"Error loading enriched data: {e}"); exit()

    # --- Kiểm tra nhanh các cột ---
    required_cols_final = ['date', 'ticker', 'close', 'RSI', 'MACD'] # Chỉ kiểm tra vài cột cơ bản
    missing_cols_final = [col for col in required_cols_final if col not in df_with_indicators.columns]
    if missing_cols_final: print(f"\nError: Missing expected columns: {missing_cols_final}"); exit()
    if not pd.api.types.is_datetime64_any_dtype(df_with_indicators['date']): print("\nError: 'date' column is not datetime type."); exit()


    # --- Bước 2: Chia Train-Test và Chạy Mô hình ---
    if not df_with_indicators.empty:
        print("\nStarting Cross-Validation Runs...")
        # --- Định nghĩa tham số ---
        INITIAL_TRAIN_YEARS = 3; TEST_YEARS = 1; STEP_YEARS = 1; USE_EXPANDING_WINDOW = False
        initial_train_td = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS); test_td = pd.Timedelta(days=365 * TEST_YEARS); step_td = pd.Timedelta(days=365 * STEP_YEARS)
        TREND_THRESHOLD = 0.002; LAG_PERIODS = [1, 3, 5]
        FEATURE_COLS_BASE = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'EMA_20', 'volume']
        FEATURE_COLS = FEATURE_COLS_BASE + [f'pct_change_lag_{p}' for p in LAG_PERIODS]
        TARGET_COL_PCT = 'target_pct_change'; TARGET_COL_TREND = 'target_trend'
        N_TIMESTEPS = 10; LSTM_UNITS = 50; TRANSFORMER_HEAD_SIZE = 64; TRANSFORMER_NUM_HEADS = 4
        TRANSFORMER_FF_DIM = 64; DROPOUT_RATE = 0.1; EPOCHS = 15; BATCH_SIZE = 64

        cv_splitter = create_time_series_cv_splits(df=df_with_indicators, date_col='date', ticker_col='ticker', initial_train_period=initial_train_td, test_period=test_td, step_period=step_td, expanding=USE_EXPANDING_WINDOW)
        all_split_results = {} # Lưu kết quả tất cả splits

        for split_idx, train_df_orig, test_df_orig in cv_splitter:
            print(f"\n===== Processing CV Split {split_idx} =====")
            results_this_split = {} # Lưu kết quả split này

            # ---- Chuẩn bị dữ liệu ----
            prep_data = prepare_split_data(
                train_df_orig, test_df_orig, FEATURE_COLS, LAG_PERIODS,
                TARGET_COL_PCT, TARGET_COL_TREND, TREND_THRESHOLD, N_TIMESTEPS
            )

            # ---- Chạy Mô hình & Đánh giá ----
            if prep_data.get('data_valid', False):
                # ARIMA
                arima_results = arima_model.run_arima_evaluation(
                    train_df=prep_data['train_df'], test_df=prep_data['test_df'],
                    target_col_trend=TARGET_COL_TREND, trend_threshold=TREND_THRESHOLD)
                results_this_split.update(arima_results)

                # RandomForest
                rf_results = rf_model.run_rf_evaluation(
                    X_train_scaled=prep_data['X_train_scaled'], y_train=prep_data['y_train_ml'],
                    X_test_scaled=prep_data['X_test_scaled'], y_test=prep_data['y_test_ml'])
                results_this_split.update(rf_results)

                # ===== GỌI XGBOOST =====
                if XGB_INSTALLED:
                     xgb_results = xgboost_model.run_xgboost_evaluation(
                         X_train_scaled=prep_data['X_train_scaled'], y_train=prep_data['y_train_ml'],
                         X_test_scaled=prep_data['X_test_scaled'], y_test=prep_data['y_test_ml'])
                     results_this_split.update(xgb_results)
                else:
                     # Cập nhật NaN nếu XGBoost không được cài đặt
                     results_this_split.update({"XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan, "XGBoost_F1_Weighted": np.nan,
                                                "XGBoost_Precision_Macro": np.nan, "XGBoost_Recall_Macro": np.nan})
                # =======================

                # LSTM
                lstm_results = lstm_model.run_lstm_evaluation(
                    X_train_seq=prep_data['X_train_seq'], y_train_seq=prep_data['y_train_seq'],
                    X_test_seq=prep_data['X_test_seq'], y_test_original_trend=prep_data['y_test_seq_original_trend'],
                    class_weight_dict=prep_data['class_weight_dict'], n_classes=prep_data['n_classes'],
                    n_timesteps=N_TIMESTEPS, n_features=len(FEATURE_COLS), lstm_units=LSTM_UNITS,
                    dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
                results_this_split.update(lstm_results)

                # Transformer
                transformer_results = transformer_model.run_transformer_evaluation(
                    X_train_seq=prep_data['X_train_seq'], y_train_seq=prep_data['y_train_seq'],
                    X_test_seq=prep_data['X_test_seq'], y_test_original_trend=prep_data['y_test_seq_original_trend'],
                    class_weight_dict=prep_data['class_weight_dict'], n_classes=prep_data['n_classes'],
                    n_timesteps=N_TIMESTEPS, n_features=len(FEATURE_COLS), head_size=TRANSFORMER_HEAD_SIZE,
                    num_heads=TRANSFORMER_NUM_HEADS, ff_dim=TRANSFORMER_FF_DIM, dropout_rate=DROPOUT_RATE,
                    epochs=EPOCHS, batch_size=BATCH_SIZE)
                results_this_split.update(transformer_results)

            else:
                 print(f"Skipping model evaluation for Split {split_idx} due to invalid prepared data.")
                 # Gán NaN cho tất cả kết quả nếu dữ liệu không hợp lệ
                 results_this_split = {metric + "_" + model: np.nan
                                       for model in ["ARIMA", "RandomForest", "XGBoost", "LSTM", "Transformer"]
                                       for metric in ["Accuracy", "F1_Macro", "F1_Weighted", "Precision_Macro", "Recall_Macro"]}

            # Lưu kết quả của split này
            all_split_results[split_idx] = results_this_split
            print(f"===== Finished CV Split {split_idx} =====")
            # break # Bỏ comment để chạy thử 1 split

        # --- Bước 5: Tổng hợp và so sánh kết quả ---
        print("\n===== Aggregating Results Across All Splits =====")
        if all_split_results:
            final_results_df = pd.DataFrame.from_dict(all_split_results, orient='index')
            # Loại bỏ các cột toàn NaN trước khi tính toán thống kê
            final_results_df.dropna(axis=1, how='all', inplace=True)

            if not final_results_df.empty:
                print("\n--- Performance Summary (Mean +/- Std Dev) ---")
                summary = final_results_df.agg(['mean', 'std']).T
                summary_valid = summary.dropna(subset=['mean']).copy() # Tránh SettingWithCopyWarning
                summary_valid_for_display = summary.dropna(subset=['mean']).copy()
                
                if not summary_valid.empty:
                    summary_valid['mean_std'] = summary_valid.apply(lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1)
                    print(summary_valid[['mean_std']])
                else:
                    print("No valid performance metrics to display after dropping NaNs.")
                    print("\nFull Results per Split:") # In chi tiết nếu cần debug
                    print(final_results_df)
            
            # ===== LƯU KẾT QUẢ TỔNG HỢP RA CSV =====
            try:
                results_output_dir = project_root / "results"
                results_output_dir.mkdir(parents=True, exist_ok=True)
                results_file_path = results_output_dir / "model_performance_summary.csv"

                # Lưu DataFrame summary gốc (chứa mean và std riêng biệt)
                # Hoặc summary_valid nếu bạn muốn bỏ các metric không có mean
                # Ưu tiên lưu summary_valid nếu nó không rỗng, ngược lại lưu summary
                df_to_save = summary_valid_for_display.drop(columns=['mean_std_display']) if not summary_valid_for_display.empty and 'mean_std_display' in summary_valid_for_display else summary.copy()

                if not df_to_save.empty:
                    df_to_save.to_csv(results_file_path)
                    print(f"\nPerformance summary (mean, std) saved to: {results_file_path.resolve()}")
                else:
                    print("No summary data to save.")


                # Lưu kết quả chi tiết từng split
                detailed_results_file_path = results_output_dir / "model_performance_detailed.csv"
                final_results_df.to_csv(detailed_results_file_path)
                print(f"Detailed results per split saved to: {detailed_results_file_path.resolve()}")

            except Exception as e:
                print(f"Error saving results to CSV: {e}")
            # =======================================
            
            else:
                print("No results to aggregate after dropping NaN columns.")
        else:
            print("No results to aggregate.")

    else:
        print("\nEnriched data is empty or could not be loaded.")

    print("\nScript finished.")

if __name__ == "__main__":
    run_all_experiments()