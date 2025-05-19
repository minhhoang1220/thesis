# marketml/generate_signals.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging # Sử dụng logging

# --- Thêm project root vào sys.path ---
# Giả định generate_signals.py nằm trong .ndmh/marketml/
PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

# --- Imports từ project ---
try:
    from marketml.configs import configs
    from marketml.data.loader import preprocess # Để standardize (mặc dù enriched data đã standardize rồi)
    from marketml.models import xgboost_model # Hoặc model bạn chọn (RF, SVM)
    # Nếu dùng Keras thì import lstm_model, transformer_model, keras_utils
    from marketml.log import suppress_common_warnings, setup_basic_logging, set_random_seeds
    # Các imports cần thiết cho data preparation (giống run_experiment.py)
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    # (Thêm các import khác nếu model bạn chọn yêu cầu, ví dụ class_weight cho Keras)
except ImportError as e:
    print(f"CRITICAL ERROR in generate_signals.py: Could not import necessary modules. {e}")
    print("Ensure all modules are correctly placed and __init__.py files exist.")
    sys.exit(1) # Thoát nếu không import được

# --- Thiết lập Logging ---
# (Bạn có thể dùng lại hàm setup_basic_logging từ utils nếu đã tạo)
try:
    logger = setup_basic_logging(log_level=logging.INFO, log_file_name="generate_signals.log")
    suppress_common_warnings() # Bỏ qua warnings
    if hasattr(configs, 'RANDOM_SEED'):
        set_random_seeds(configs.RANDOM_SEED)
except Exception as log_setup_e:
    print(f"Error setting up logger/environment in generate_signals: {log_setup_e}")
    # Fallback basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def prepare_data_for_signal_generation(df_full, feature_cols, lag_periods,
                                       target_col_pct, target_col_trend, trend_threshold,
                                       start_date_filter, end_date_filter,
                                       fit_imputer_scaler=True, # True cho tập train, False cho tập test
                                       existing_imputer=None, existing_scaler=None): # Dùng cho tập test
    """
    Chuẩn bị dữ liệu X và y (nếu là tập train) cho một khoảng thời gian cụ thể.
    Tương tự prepare_split_data từ run_experiment.py nhưng đơn giản hóa cho một split.
    """
    logger.info(f"Preparing data from {start_date_filter} to {end_date_filter}...")
    df_filtered = df_full[(df_full['date'] >= pd.to_datetime(start_date_filter)) & \
                          (df_full['date'] <= pd.to_datetime(end_date_filter))].copy()

    if df_filtered.empty:
        logger.warning(f"No data found in the specified date range: {start_date_filter} to {end_date_filter}")
        return None, None, None, None, None # Trả về None cho tất cả

    # ---- Tạo Target (chỉ cần cho tập train) ----
    if fit_imputer_scaler: # Chỉ tạo target cho tập train
        df_filtered[target_col_pct] = df_filtered.groupby('ticker')['close'].pct_change().shift(-1)
        conditions = [
            (df_filtered[target_col_pct] > trend_threshold),
            (df_filtered[target_col_pct] < -trend_threshold)
        ]
        choices = [1, -1] # Lớp Tăng (1), Giảm (-1)
        df_filtered[target_col_trend] = np.select(conditions, choices, default=0) # Đi ngang (0)

    # ---- Tạo Lags ----
    # (Đảm bảo rằng pct_change được tính trên dữ liệu gốc trước khi lọc để tránh lỗi ở biên)
    # Hoặc, chấp nhận NaN ở đầu sau khi tạo lag trên df_filtered
    df_full_copy_for_lag = df_full.copy() # Tạo bản sao để không thay đổi df_full gốc
    df_full_copy_for_lag['pct_change_temp'] = df_full_copy_for_lag.groupby('ticker')['close'].pct_change()
    
    lag_features_dict = {}
    for p in lag_periods:
        lag_col_name = f'pct_change_lag_{p}'
        # Shift pct_change_temp và gán vào lag_features_dict[lag_col_name]
        # Sau đó merge các cột này vào df_filtered dựa trên index
        df_full_copy_for_lag[lag_col_name] = df_full_copy_for_lag.groupby('ticker')['pct_change_temp'].shift(p)
        lag_features_dict[lag_col_name] = df_full_copy_for_lag[[lag_col_name]] # Giữ lại index gốc

    # Merge các cột lag vào df_filtered
    for lag_col, lag_series in lag_features_dict.items():
        df_filtered = df_filtered.join(lag_series, how='left')


    # ---- Impute & Align ----
    X_raw = df_filtered[feature_cols].copy()
    
    # Xử lý cột garch_vol_forecast nếu nó toàn NaN
    if 'garch_vol_forecast' in X_raw.columns and X_raw['garch_vol_forecast'].isnull().all():
        logger.warning("'garch_vol_forecast' is all NaN. Filling with 0 before imputation.")
        X_raw['garch_vol_forecast'] = X_raw['garch_vol_forecast'].fillna(0)

    if fit_imputer_scaler: # Tập train
        imputer = SimpleImputer(strategy='mean')
        X_imputed_np = imputer.fit_transform(X_raw)
        
        scaler = StandardScaler()
        X_scaled_np = scaler.fit_transform(X_imputed_np)
        
        y_ml = df_filtered[target_col_trend].copy()
        # Loại bỏ NaN trong target VÀ các hàng tương ứng trong X
        valid_y_idx = y_ml.dropna().index
        X_scaled_final = pd.DataFrame(X_scaled_np, index=X_raw.index, columns=X_raw.columns).loc[valid_y_idx]
        y_ml_final = y_ml.loc[valid_y_idx]

        if X_scaled_final.empty or y_ml_final.empty:
            logger.error("X_scaled_final or y_ml_final is empty after NaN drop in training data. Aborting.")
            return None, None, None, None, None
        
        # Map target sang 0, 1, 2 cho XGBoost/RF/SVM (nếu model yêu cầu)
        # Giả sử -1 (Giảm) -> 0, 0 (Đi ngang) -> 1, 1 (Tăng) -> 2
        y_ml_mapped = y_ml_final.replace({-1: 0, 0: 1, 1: 2})

        return X_scaled_final.values, y_ml_mapped.values, imputer, scaler, df_filtered.loc[X_scaled_final.index][['date', 'ticker'] + feature_cols]

    else: # Tập test/predict
        if existing_imputer is None or existing_scaler is None:
            logger.error("Imputer and Scaler must be provided for test/prediction data.")
            return None, None, None, None, None
        
        X_imputed_np = existing_imputer.transform(X_raw)
        X_scaled_np = existing_scaler.transform(X_imputed_np)
        
        # Giữ lại tất cả các hàng sau khi scale cho dự đoán, NaN trong features sẽ được xử lý bởi imputer
        # Nếu có NaN còn sót lại sau transform (ít khả năng với mean imputer), cần xử lý thêm.
        X_scaled_final = pd.DataFrame(X_scaled_np, index=X_raw.index, columns=X_raw.columns)
        
        # Chỉ trả về các cột cần thiết để merge lại với xác suất
        # Lấy index gốc của X_scaled_final (trước khi có thể dropna nếu có)
        # để đảm bảo số hàng khớp với xác suất
        df_for_output = df_filtered.loc[X_scaled_final.index][['date', 'ticker'] + feature_cols].copy()
        return X_scaled_final.values, None, None, None, df_for_output


def train_soft_signal_model(X_train, y_train_mapped):
    """
    Huấn luyện mô hình được chọn trong configs.SOFT_SIGNAL_MODEL_NAME.
    """
    model_name = configs.SOFT_SIGNAL_MODEL_NAME
    logger.info(f"Training soft signal model: {model_name}...")

    if model_name == 'XGBoost':
        # Sử dụng tham số tốt nhất đã tìm được hoặc một bộ tham số mặc định tốt
        # Ví dụ: lấy từ configs nếu bạn đã thêm chúng
        xgb_params = {
            'n_estimators': configs.N_ESTIMATORS_XGB_SOFT_SIGNAL if hasattr(configs, 'N_ESTIMATORS_XGB_SOFT_SIGNAL') else 200,
            'learning_rate': configs.LR_XGB_SOFT_SIGNAL if hasattr(configs, 'LR_XGB_SOFT_SIGNAL') else 0.05,
            'max_depth': configs.MAX_DEPTH_XGB_SOFT_SIGNAL if hasattr(configs, 'MAX_DEPTH_XGB_SOFT_SIGNAL') else 5,
            # Thêm các tham số khác nếu cần
            'objective': 'multi:softmax', # Vì y_train_mapped là 0, 1, 2
            'num_class': 3,
            'use_label_encoder': False, # Quan trọng từ XGBoost 1.0
            'eval_metric': 'mlogloss',
            'random_state': configs.RANDOM_SEED
        }
        model = xgboost_model.xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train_mapped)
        logger.info(f"{model_name} trained successfully.")
        return model
    # elif model_name == 'RandomForest':
    #     # Tương tự cho RandomForest
    #     pass
    else:
        logger.error(f"Soft signal model '{model_name}' is not supported in this script.")
        raise ValueError(f"Unsupported model: {model_name}")

def generate_and_save_signals():
    logger.info("Starting soft signal generation process...")

    # --- 1. Load Enriched Data ---
    try:
        df_enriched = pd.read_csv(configs.ENRICHED_DATA_FILE, parse_dates=['date'])
        logger.info(f"Loaded enriched data from: {configs.ENRICHED_DATA_FILE}")
    except FileNotFoundError:
        logger.error(f"Enriched data file not found at {configs.ENRICHED_DATA_FILE}. Aborting.")
        return
    except Exception as e:
        logger.error(f"Error loading enriched data: {e}. Aborting.")
        return

    # --- 2. Xác định Feature Columns ---
    # (Đảm bảo các cột lag được thêm vào BASE_FEATURE_COLS một cách đúng đắn)
    feature_cols_with_lags = configs.BASE_FEATURE_COLS + [f'pct_change_lag_{p}' for p in configs.LAG_PERIODS]
    feature_cols_with_lags = sorted(list(set(feature_cols_with_lags))) # Loại bỏ trùng lặp và sắp xếp
    logger.info(f"Using features: {feature_cols_with_lags}")


    # --- 3. Chuẩn bị Dữ liệu Huấn luyện cho Soft Signal Model ---
    X_train, y_train_mapped, imputer, scaler, _ = prepare_data_for_signal_generation(
        df_full=df_enriched,
        feature_cols=feature_cols_with_lags,
        lag_periods=configs.LAG_PERIODS,
        target_col_pct=configs.TARGET_COL_PCT,
        target_col_trend=configs.TARGET_COL_TREND,
        trend_threshold=configs.TREND_THRESHOLD,
        start_date_filter=configs.TIME_RANGE_START, # Huấn luyện trên toàn bộ dữ liệu đến SOFT_SIGNAL_TRAIN_END_DATE
        end_date_filter=configs.SOFT_SIGNAL_TRAIN_END_DATE,
        fit_imputer_scaler=True
    )
    if X_train is None or y_train_mapped is None:
        logger.error("Failed to prepare training data for soft signal model. Aborting.")
        return

    # --- 4. Huấn luyện Soft Signal Model ---
    try:
        signal_model = train_soft_signal_model(X_train, y_train_mapped)
    except Exception as e:
        logger.error(f"Error training soft signal model: {e}. Aborting.")
        return

    # --- 5. Chuẩn bị Dữ liệu Dự đoán ---
    X_predict, _, _, _, df_predict_info = prepare_data_for_signal_generation(
        df_full=df_enriched,
        feature_cols=feature_cols_with_lags,
        lag_periods=configs.LAG_PERIODS,
        target_col_pct=configs.TARGET_COL_PCT, # Không thực sự dùng để tạo y_test, nhưng cần để nhất quán
        target_col_trend=configs.TARGET_COL_TREND,
        trend_threshold=configs.TREND_THRESHOLD,
        start_date_filter=configs.SOFT_SIGNAL_PREDICT_START_DATE,
        end_date_filter=configs.SOFT_SIGNAL_PREDICT_END_DATE,
        fit_imputer_scaler=False, # Quan trọng: dùng imputer/scaler đã fit
        existing_imputer=imputer,
        existing_scaler=scaler
    )
    if X_predict is None or df_predict_info is None:
        logger.error("Failed to prepare prediction data for soft signal model. Aborting.")
        return

    # --- 6. Dự đoán Xác suất ---
    logger.info(f"Predicting probabilities for period {configs.SOFT_SIGNAL_PREDICT_START_DATE} to {configs.SOFT_SIGNAL_PREDICT_END_DATE}...")
    try:
        probabilities = signal_model.predict_proba(X_predict)
    except Exception as e:
        logger.error(f"Error predicting probabilities: {e}. Aborting.")
        return

    # --- 7. Tạo DataFrame Output và Lưu ---
    # probabilities sẽ có shape (n_samples, n_classes)
    # Giả sử mapping: 0: Giảm (-1), 1: Đi ngang (0), 2: Tăng (1)
    # df_predict_info chứa 'date', 'ticker' và các features tương ứng với X_predict

    # Đảm bảo số hàng khớp
    if len(df_predict_info) != probabilities.shape[0]:
        logger.error(f"Mismatch in rows between df_predict_info ({len(df_predict_info)}) and probabilities ({probabilities.shape[0]}). Aborting.")
        # Điều này có thể xảy ra nếu việc xử lý NaN trong X_predict khác với df_predict_info
        return

    output_df = df_predict_info[['date', 'ticker']].copy() # Lấy date, ticker
    
    # Lấy tên cột cho xác suất từ tên model (ví dụ)
    prob_col_decrease = f"prob_decrease_{configs.SOFT_SIGNAL_MODEL_NAME}"
    prob_col_neutral = f"prob_neutral_{configs.SOFT_SIGNAL_MODEL_NAME}"
    prob_col_increase = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}"

    output_df[prob_col_decrease] = probabilities[:, 0]
    output_df[prob_col_neutral] = probabilities[:, 1]
    output_df[prob_col_increase] = probabilities[:, 2]

    # (Tùy chọn) Thêm các features kỹ thuật quan trọng vào output_df nếu muốn
    # for col in ['RSI', 'MACD', 'garch_vol_forecast']: # Ví dụ
    #     if col in df_predict_info.columns:
    #         output_df[col] = df_predict_info[col]

    try:
        output_path = configs.CLASSIFICATION_PROBS_FILE_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True) # Tạo thư mục results nếu chưa có
        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully generated and saved soft signals to: {output_path}")
        logger.info(f"Output DataFrame head:\n{output_df.head().to_string()}")
    except Exception as e:
        logger.error(f"Error saving soft signals: {e}")

if __name__ == "__main__":
    generate_and_save_signals()
    logger.info("generate_signals.py script finished.")