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

# === Import các module đã tách và configs ===
try:
    from marketml.utils import metrics
    from marketml.models import (arima_model, rf_model, lstm_model,
                                 transformer_model, keras_utils, xgboost_model, svm_model)
    from marketml.configs import configs # <--- THÊM IMPORT CONFIGS
except ModuleNotFoundError as e:
    print(f"Error importing necessary modules or configs: {e}")
    # ... (thông báo lỗi giữ nguyên) ...
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

# === SKLearn đã được kiểm tra gián tiếp qua các model khác ===
# nhưng có thể thêm kiểm tra riêng nếu muốn
try:
    import sklearn
    SKLEARN_INSTALLED = True
except ImportError:
    print("Warning: scikit-learn not installed. RF and SVM models might be skipped.")
    SKLEARN_INSTALLED = False # Mặc dù rf_model.py và svm_model.py có kiểm tra riêng

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
        X_train_raw = train_df[feature_cols].copy(); X_test_raw = test_df[feature_cols].copy()
        y_train_trend_ml = train_df[target_col_trend]; y_test_trend_ml = test_df[target_col_trend]

        # Xử lý các cột toàn NaN TRƯỚC KHI impute
        # Ví dụ: điền 0 cho garch_vol_forecast nếu nó toàn NaN trong train
        if 'garch_vol_forecast' in X_train_raw.columns and X_train_raw['garch_vol_forecast'].isnull().all():
            print("    Warning: 'garch_vol_forecast' is all NaN in X_train_raw. Filling with 0 before imputation.")
            X_train_raw['garch_vol_forecast'] = X_train_raw['garch_vol_forecast'].fillna(0)
            # Áp dụng tương tự cho X_test_raw (dùng giá trị 0 hoặc một chiến lược khác)
            if 'garch_vol_forecast' in X_test_raw.columns:
                X_test_raw['garch_vol_forecast'] = X_test_raw['garch_vol_forecast'].fillna(0)

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
    ENRICHED_DATA_FILE = configs.ENRICHED_DATA_FILE

    try:
        print(f"Loading enriched data from: {ENRICHED_DATA_FILE}")
        df_with_indicators = pd.read_csv(ENRICHED_DATA_FILE, parse_dates=['date'])
        print("Enriched data loaded successfully.")
    except FileNotFoundError: print(f"Error: Enriched data file not found at '{ENRICHED_DATA_FILE}'. Run 'marketml/indicators/create_enriched_data.py' first."); exit()
    except Exception as e: print(f"Error loading enriched data: {e}"); exit()
    
    # --- Kiểm tra nhanh các cột ---
    # (Sử dụng configs.BASE_FEATURE_COLS để kiểm tra một phần, hoặc tạo danh sách đầy đủ hơn)
    # Ví dụ một cách kiểm tra đơn giản:
    required_cols_check = ['date', 'ticker', 'close'] + configs.BASE_FEATURE_COLS[:3] # Kiểm tra vài cột cơ bản
    missing_cols_final = [col for col in required_cols_check if col not in df_with_indicators.columns and col != 'garch_vol_forecast' or (col == 'garch_vol_forecast' and not ARCH_INSTALLED and col not in df_with_indicators.columns)]
    if missing_cols_final: print(f"\nError: Missing some expected columns from enriched data: {missing_cols_final}"); # exit() # Có thể comment exit() để linh hoạt hơn

    if not pd.api.types.is_datetime64_any_dtype(df_with_indicators['date']): print("\nError: 'date' column is not datetime type."); exit()

    if not df_with_indicators.empty:
        print("\nStarting Cross-Validation Runs...")
        # --- Lấy tham số từ configs ---
        initial_train_td = configs.INITIAL_TRAIN_TIMEDELTA
        test_td = configs.TEST_TIMEDELTA
        step_td = configs.STEP_TIMEDELTA
        USE_EXPANDING_WINDOW = configs.USE_EXPANDING_WINDOW
        TREND_THRESHOLD = configs.TREND_THRESHOLD
        LAG_PERIODS = configs.LAG_PERIODS

        FEATURE_COLS_BASE = configs.BASE_FEATURE_COLS
        FEATURE_COLS = sorted(list(set(FEATURE_COLS_BASE + [f'pct_change_lag_{p}' for p in LAG_PERIODS])))
        print(f"\nUsing {len(FEATURE_COLS)} features for ML models (base + lags): {FEATURE_COLS}")

        TARGET_COL_PCT = configs.TARGET_COL_PCT
        TARGET_COL_TREND = configs.TARGET_COL_TREND
        
        N_ITER_SEARCH_SKLEARN = configs.N_ITER_SEARCH_SKLEARN
        CV_FOLDS_TUNING_SKLEARN = configs.CV_FOLDS_TUNING_SKLEARN

        N_TIMESTEPS = configs.N_TIMESTEPS_SEQUENCE
        
        # Keras params
        lstm_params_set = {
            'lstm_units': configs.LSTM_UNITS, 
            'dropout_rate': configs.LSTM_DROPOUT_RATE, 
            'learning_rate': configs.LSTM_LEARNING_RATE, 
            'epochs': configs.KERAS_EPOCHS, 
            'batch_size': configs.KERAS_BATCH_SIZE,
            # Thêm các tham số callback vào đây nếu các hàm model nhận chúng
            'validation_split': configs.KERAS_VALIDATION_SPLIT,
            'early_stopping_patience': configs.KERAS_EARLY_STOPPING_PATIENCE,
            'reduce_lr_patience': configs.KERAS_REDUCE_LR_PATIENCE,
            'reduce_lr_factor': configs.KERAS_REDUCE_LR_FACTOR,
            'min_lr': configs.KERAS_MIN_LR
        }
        transformer_params_set = {
            'num_transformer_blocks': configs.TRANSFORMER_NUM_BLOCKS, 
            'head_size': configs.TRANSFORMER_HEAD_SIZE, 
            'num_heads': configs.TRANSFORMER_NUM_HEADS,
            'ff_dim': configs.TRANSFORMER_FF_DIM, 
            'dropout_rate': configs.TRANSFORMER_DROPOUT_RATE,
            'learning_rate': configs.TRANSFORMER_LEARNING_RATE,
            'epochs': configs.KERAS_EPOCHS, 
            'batch_size': configs.KERAS_BATCH_SIZE,
            'weight_decay': configs.TRANSFORMER_WEIGHT_DECAY,
            # Thêm các tham số callback
            'validation_split': configs.KERAS_VALIDATION_SPLIT,
            'early_stopping_patience': configs.KERAS_EARLY_STOPPING_PATIENCE
        }
        # ====================================================

        cv_splitter = create_time_series_cv_splits(
            df=df_with_indicators, date_col='date', ticker_col='ticker', 
            initial_train_period=initial_train_td, 
            test_period=test_td, 
            step_period=step_td, 
            expanding=USE_EXPANDING_WINDOW
        )
        all_split_results = {}

        for split_idx, train_df_orig, test_df_orig in cv_splitter:
            print(f"\n===== Processing CV Split {split_idx} =====")
            results_this_split = {}

            prep_data = prepare_split_data(
                train_df_orig, test_df_orig, FEATURE_COLS, LAG_PERIODS,
                TARGET_COL_PCT, TARGET_COL_TREND, TREND_THRESHOLD, N_TIMESTEPS
            )

            if prep_data.get('data_valid', False):
                # ARIMA
                arima_results = arima_model.run_arima_evaluation(
                    prep_data['train_df'], prep_data['test_df'],
                    TARGET_COL_TREND, TREND_THRESHOLD) # Sử dụng biến từ configs (đã gán ở trên)
                results_this_split.update(arima_results)
                
                # GARCH placeholder
                results_this_split.update({"GARCH_Accuracy": np.nan, "GARCH_F1_Macro": np.nan, "GARCH_F1_Weighted": np.nan,
                                           "GARCH_Precision_Macro": np.nan, "GARCH_Recall_Macro": np.nan})

                # RandomForest
                rf_results = rf_model.run_rf_evaluation(
                    prep_data['X_train_scaled'], prep_data['y_train_ml'],
                    prep_data['X_test_scaled'], prep_data['y_test_ml'],
                    n_iter_search=N_ITER_SEARCH_SKLEARN, # Sử dụng biến từ configs
                    cv_folds_tuning=CV_FOLDS_TUNING_SKLEARN # Sử dụng biến từ configs
                )
                results_this_split.update(rf_results)

                # XGBoost
                if XGB_INSTALLED:
                     xgb_results = xgboost_model.run_xgboost_evaluation(
                         prep_data['X_train_scaled'], prep_data['y_train_ml'],
                         prep_data['X_test_scaled'], prep_data['y_test_ml'],
                         n_iter_search=N_ITER_SEARCH_SKLEARN, # Sử dụng biến từ configs
                         cv_folds_tuning=CV_FOLDS_TUNING_SKLEARN # Sử dụng biến từ configs
                     )
                     results_this_split.update(xgb_results)
                
                # SVM
                if SKLEARN_INSTALLED:
                    svm_results = svm_model.run_svm_evaluation(
                        prep_data['X_train_scaled'], prep_data['y_train_ml'],
                        prep_data['X_test_scaled'], prep_data['y_test_ml'],
                        n_iter_search=N_ITER_SEARCH_SKLEARN, # Sử dụng biến từ configs
                        cv_folds_tuning=CV_FOLDS_TUNING_SKLEARN # Sử dụng biến từ configs
                    )
                    results_this_split.update(svm_results)
                else: # Placeholder
                    results_this_split.update({"SVM_Accuracy": np.nan, "SVM_F1_Macro": np.nan, "SVM_F1_Weighted": np.nan,
                                               "SVM_Precision_Macro": np.nan, "SVM_Recall_Macro": np.nan})
                
                current_n_features = prep_data['X_train_scaled'].shape[1] if prep_data['X_train_scaled'].ndim == 2 else prep_data['X_train_seq'].shape[2]
                
                # LSTM
                lstm_results = lstm_model.run_lstm_evaluation(
                    prep_data['X_train_seq'], prep_data['y_train_seq'], prep_data['X_test_seq'],
                    prep_data['y_test_seq_original_trend'], prep_data['class_weight_dict'], prep_data['n_classes'],
                    N_TIMESTEPS, current_n_features,
                    **lstm_params_set # Truyền dict tham số từ configs
                )
                results_this_split.update(lstm_results)

                # Transformer
                transformer_results = transformer_model.run_transformer_evaluation(
                    prep_data['X_train_seq'], prep_data['y_train_seq'], prep_data['X_test_seq'],
                    prep_data['y_test_seq_original_trend'], prep_data['class_weight_dict'], prep_data['n_classes'],
                    N_TIMESTEPS, current_n_features,
                    **transformer_params_set # Truyền dict tham số từ configs
                )
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

        # --- Bước 5: Tổng hợp và so sánh kết quả từ tất cả các split ---
        print("\n===== Aggregating Results Across All Splits =====")
        if all_split_results:  # Chỉ tiếp tục nếu có kết quả từ ít nhất một split
            final_results_df = pd.DataFrame.from_dict(all_split_results, orient='index')
            # Loại bỏ các cột mà TẤT CẢ các giá trị trong cột đó là NaN
            final_results_df.dropna(axis=1, how='all', inplace=True)

            # Khởi tạo các DataFrame sẽ lưu
            df_to_save_summary = pd.DataFrame() # Sẽ chứa mean, std của các metric hợp lệ
            df_to_save_detailed = final_results_df.copy() # Luôn lưu final_results_df nếu nó không rỗng (sau khi drop cột toàn NaN)

            if not final_results_df.empty: # Nếu còn lại bất kỳ cột nào sau khi dropna(axis=1, how='all')
                print("\n--- Performance Summary (Mean +/- Std Dev) ---")

                # 1. Xác định các cột metrics là số để tính toán
                numeric_metric_cols = [
                    col for col in final_results_df.columns
                    if "Accuracy" in col or "F1" in col or "Precision" in col or "Recall" in col
                ]

                if numeric_metric_cols: # Nếu có cột metric số nào tồn tại
                    summary = final_results_df[numeric_metric_cols].agg(['mean', 'std']).T
                    # summary_valid chỉ lấy các hàng (metrics) mà giá trị 'mean' không phải là NaN
                    summary_valid = summary.dropna(subset=['mean']).copy()

                    if not summary_valid.empty:
                        summary_valid_for_display = summary_valid.copy()
                        summary_valid_for_display['mean_std_display_col'] = summary_valid_for_display.apply(
                            lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1
                        )
                        print(summary_valid_for_display[['mean_std_display_col']])
                        df_to_save_summary = summary_valid # DataFrame này chứa mean, std để lưu
                    else:
                        print("No valid performance metrics (with non-NaN mean) to display in summary.")
                        # df_to_save_summary vẫn là DataFrame rỗng
                else:
                    print("No numeric metric columns found in results to aggregate for summary.")
                    # df_to_save_summary vẫn là DataFrame rỗng

                # ===== LƯU KẾT QUẢ TỔNG HỢP VÀ CHI TIẾT RA CSV =====
                try:
                    results_output_dir = configs.RESULTS_DIR # <--- THAY ĐỔI
                    results_output_dir.mkdir(parents=True, exist_ok=True)

                    # Lưu file summary (chỉ chứa mean, std của các metric hợp lệ)
                    summary_file_path = results_output_dir / "marketml" / "model_performance_summary.csv"
                    if not df_to_save_summary.empty:
                        df_to_save_summary.to_csv(summary_file_path)
                        print(f"\nPerformance summary (mean, std) saved to: {summary_file_path.resolve()}")
                    else:
                        print("No summary data (mean, std) to save (empty or all NaNs after aggregation).")

                    # Lưu kết quả chi tiết của từng split (df_to_save_detailed là final_results_df đã được xử lý)
                    detailed_results_file_path = results_output_dir / "marketml" / "model_performance_detailed.csv"
                    if not df_to_save_detailed.empty: # df_to_save_detailed là final_results_df
                        df_to_save_detailed.to_csv(detailed_results_file_path)
                        print(f"Detailed results per split saved to: {detailed_results_file_path.resolve()}")
                    else:
                        print("No detailed results per split to save (final_results_df was empty or became empty after dropping all-NaN columns).")

                except Exception as e:
                    print(f"Error saving results to CSV: {e}")
                # =================================================

            else: # final_results_df rỗng sau khi drop cột toàn NaN
                print("No results to aggregate (final_results_df is empty after dropping all-NaN columns).")
        else: # all_split_results rỗng ngay từ đầu
            print("No results from any split to aggregate.")

    else: # df_with_indicators rỗng
        print("\nEnriched data is empty or could not be loaded. No experiments run.")

    print("\nScript finished.")

if __name__ == "__main__":
    run_all_experiments()