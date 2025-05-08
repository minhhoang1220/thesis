import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight 
import warnings

# ===== BỎ QUA WARNINGS =====
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module='statsmodels.tsa.base.tsa_model')
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

# === Import TensorFlow/Keras ===
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Flatten
    from keras.utils import to_categorical # Có thể cần nếu dùng categorical_crossentropy
    # Giới hạn sử dụng GPU memory nếu cần thiết (tùy chọn)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    TF_INSTALLED = True
except ImportError:
    print("Warning: TensorFlow is not installed. LSTM and Transformer models will be skipped.")
    TF_INSTALLED = False
# ==============================

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
# HÀM TẠO CHUỖI CHO LSTM/TRANSFORMER
# ==============================================================================
def create_sequences(X_data, y_data, time_steps=1):
    """
    Chuyển đổi dữ liệu 2D thành chuỗi 3D cho mô hình sequence.
    """
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        v = X_data[i:(i + time_steps)] # Lấy chuỗi features
        Xs.append(v)
        ys.append(y_data[i + time_steps]) # Lấy target tương ứng sau chuỗi
    return np.array(Xs), np.array(ys)

# ==============================================================================

# ==============================================================================
# HÀM XÂY DỰNG TRANSFORMER ENCODER ĐƠN GIẢN
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs]) # Skip connection

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x) # Project back to input dimension
    return Add()([x, res]) # Skip connection


# ===============================================================================

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
    # === Tham số cho Sequence Models ===
    N_TIMESTEPS = 10  # Số ngày trong quá khứ để nhìn lại cho mỗi sequence
    LSTM_UNITS = 50
    TRANSFORMER_HEAD_SIZE = 64
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_FF_DIM = 64
    DROPOUT_RATE = 0.1
    EPOCHS = 15 # Số epochs huấn luyện (có thể cần tăng/giảm)
    BATCH_SIZE = 64 # Kích thước batch (có thể cần tăng/giảm)

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
        X_train_ml = X_train_imputed.loc[valid_y_train_idx]; y_train_ml = y_train_trend_ml.loc[valid_y_train_idx]
        X_test_ml = X_test_imputed.loc[valid_y_test_idx]; y_test_ml = y_test_trend_ml.loc[valid_y_test_idx]
        if X_train_ml.empty or X_test_ml.empty: 
            print("  Skipping split: Empty ML data after alignment."); 
            continue

        # ---- 5. Scaling Features (X_ml) ----
        # Vẫn thực hiện để giữ cấu trúc, dù ARIMA không dùng
        print("  Scaling ML features...") # Tắt bớt print
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train_ml); X_test_scaled = scaler.transform(X_test_ml)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_ml.index, columns=X_train_ml.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test_ml.index, columns=X_test_ml.columns)

        # ---- 6. CHUẨN BỊ DỮ LIỆU CHO KERAS ----
        # Chuyển đổi nhãn trend từ (-1, 0, 1) thành (0, 1, 2)
        y_train_keras = y_train_ml + 1
        y_test_keras = y_test_ml + 1
        n_classes = 3 # Số lượng lớp (Giảm, Đi ngang, Tăng)

        # Tạo sequences
        print(f"  Creating sequences with time_steps={N_TIMESTEPS}...")
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_keras.values, N_TIMESTEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_keras.values, N_TIMESTEPS)

        # Lấy lại nhãn gốc (-1, 0, 1) tương ứng với y_test_seq để đánh giá cuối cùng
        # Index của y_test_seq sẽ tương ứng với N_TIMESTEPS dòng cuối của y_test_ml gốc
        y_test_seq_original_trend = y_test_ml.iloc[N_TIMESTEPS:].values


        print(f"  Sequence shapes: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}, X_test={X_test_seq.shape}, y_test={y_test_seq.shape}")

        if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
            print("  Skipping sequence models: Not enough data to create sequences.")
            TF_INSTALLED = False # Tạm thời coi như không cài TF để bỏ qua phần sau
        else:
             # Tính class weights cho dữ liệu train sequence (nhãn 0, 1, 2)
             class_weights_keras = class_weight.compute_class_weight(
                 'balanced',
                 classes=np.unique(y_train_seq),
                 y=y_train_seq
             )
             class_weight_dict = dict(enumerate(class_weights_keras))
             print(f"  Class weights for Keras models: {class_weight_dict}")

        # --------------------------------------------------------------
        # Phần 7: Chạy và đánh giá mô hình ARIMA
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
        # Phần 8: Huấn luyện và đánh giá Mô hình Machine Learning
        # --------------------------------------------------------------
        print("\n--- Training and Evaluating ML Models ---")

        # ===== CODE RANDOM FOREST =====
        try:
            from sklearn.ensemble import RandomForestClassifier
            print("  Training RandomForest...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=10, min_samples_leaf=5)
            rf_model.fit(X_train_scaled_df, y_train_ml)
            print("  Predicting with RandomForest...")
            rf_preds = rf_model.predict(X_test_scaled_df)
            # ---- Đánh giá RF ----
            rf_metrics_results = metrics.calculate_classification_metrics(y_test_ml, rf_preds, model_name="RandomForest")
            results_this_split.update(rf_metrics_results)
        except ImportError:
             print("Error: scikit-learn not installed. Cannot run RandomForest.")
             results_this_split.update({"RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan, "RandomForest_F1_Weighted": np.nan, "RandomForest_Precision_Macro": np.nan, "RandomForest_Recall_Macro": np.nan})
        except Exception as e:
             print(f"Error during RandomForest execution: {e}")
             results_this_split.update({"RandomForest_Accuracy": np.nan, "RandomForest_F1_Macro": np.nan, "RandomForest_F1_Weighted": np.nan, "RandomForest_Precision_Macro": np.nan, "RandomForest_Recall_Macro": np.nan})
        # ===========================

        # ---- Thêm code cho XGBoost, LSTM, Transformer tại đây ----
        # ===== LSTM =====
        if TF_INSTALLED and X_train_seq.shape[0] > 0: # Chỉ chạy nếu TF được cài và có sequence data
            print("\n  --- Training LSTM Model ---")
            try:
                keras.backend.clear_session() # Xóa session cũ nếu có
                n_features = X_train_seq.shape[2]

                lstm_model = Sequential([
                    Input(shape=(N_TIMESTEPS, n_features)),
                    LSTM(LSTM_UNITS, return_sequences=False), # Chỉ cần output cuối cùng
                    Dropout(DROPOUT_RATE),
                    Dense(n_classes, activation='softmax') # Output layer cho 3 lớp
                ])

                lstm_model.compile(loss='sparse_categorical_crossentropy', # Vì y là số nguyên (0, 1, 2)
                                   optimizer='adam',
                                   metrics=['accuracy'])
                # print(lstm_model.summary()) # In cấu trúc model nếu cần

                print(f"    Training LSTM for {EPOCHS} epochs...")
                history_lstm = lstm_model.fit(
                    X_train_seq,
                    y_train_seq,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    class_weight=class_weight_dict, # Sử dụng class weights
                    validation_split=0.1, # Dùng 10% train data để validation trong quá trình train
                    shuffle=True, # Xáo trộn dữ liệu mỗi epoch
                    verbose=0 # Đặt verbose=1 hoặc 2 để xem chi tiết quá trình train
                )
                print("    Training finished.")

                print("    Predicting with LSTM...")
                lstm_pred_probs = lstm_model.predict(X_test_seq)
                lstm_pred_keras = np.argmax(lstm_pred_probs, axis=1) # Lấy index của lớp có xác suất cao nhất (0, 1, 2)
                lstm_pred_trend = lstm_pred_keras - 1 # Chuyển về nhãn gốc (-1, 0, 1)

                # ---- Đánh giá LSTM ----
                lstm_metrics_results = metrics.calculate_classification_metrics(
                    y_true=y_test_seq_original_trend, # So sánh với nhãn gốc
                    y_pred=lstm_pred_trend,
                    model_name="LSTM"
                )
                results_this_split.update(lstm_metrics_results)

            except Exception as e:
                print(f"Error during LSTM execution: {e}")
                results_this_split.update({"LSTM_Accuracy": np.nan, "LSTM_F1_Macro": np.nan, "LSTM_F1_Weighted": np.nan}) # Cập nhật NaN
        else:
            print("  Skipping LSTM Model (TensorFlow not installed or no sequence data).")
            results_this_split.update({"LSTM_Accuracy": np.nan, "LSTM_F1_Macro": np.nan, "LSTM_F1_Weighted": np.nan}) # Cập nhật NaN

        # ===== Transformer =====
        if TF_INSTALLED and X_train_seq.shape[0] > 0:
            print("\n  --- Training Transformer Model ---")
            try:
                keras.backend.clear_session()
                n_features = X_train_seq.shape[2]
                input_shape = (N_TIMESTEPS, n_features)

                inputs = Input(shape=input_shape)
                # Xây dựng nhiều lớp encoder nếu muốn, ở đây dùng 1 lớp
                x = transformer_encoder(inputs, TRANSFORMER_HEAD_SIZE, TRANSFORMER_NUM_HEADS, TRANSFORMER_FF_DIM, DROPOUT_RATE)
                x = GlobalAveragePooling1D(data_format="channels_last")(x) # Hoặc Flatten()
                x = Dropout(DROPOUT_RATE)(x)
                outputs = Dense(n_classes, activation="softmax")(x)

                transformer_model = keras.Model(inputs, outputs)

                transformer_model.compile(loss="sparse_categorical_crossentropy",
                                          optimizer="adam",
                                          metrics=["accuracy"])
                # print(transformer_model.summary())

                print(f"    Training Transformer for {EPOCHS} epochs...")
                history_transformer = transformer_model.fit(
                    X_train_seq,
                    y_train_seq,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    class_weight=class_weight_dict,
                    validation_split=0.1,
                    shuffle=True,
                    verbose=0
                )
                print("    Training finished.")

                print("    Predicting with Transformer...")
                transformer_pred_probs = transformer_model.predict(X_test_seq)
                transformer_pred_keras = np.argmax(transformer_pred_probs, axis=1)
                transformer_pred_trend = transformer_pred_keras - 1

                # ---- Đánh giá Transformer ----
                transformer_metrics_results = metrics.calculate_classification_metrics(
                    y_true=y_test_seq_original_trend,
                    y_pred=transformer_pred_trend,
                    model_name="Transformer"
                )
                results_this_split.update(transformer_metrics_results)

            except Exception as e:
                print(f"Error during Transformer execution: {e}")
                results_this_split.update({"Transformer_Accuracy": np.nan, "Transformer_F1_Macro": np.nan, "Transformer_F1_Weighted": np.nan})
        else:
            print("  Skipping Transformer Model (TensorFlow not installed or no sequence data).")
            results_this_split.update({"Transformer_Accuracy": np.nan, "Transformer_F1_Macro": np.nan, "Transformer_F1_Weighted": np.nan})


        # Ví dụ placeholder cho XGBoost (để giữ cột trong kết quả cuối)
        results_this_split.update({"XGBoost_Accuracy": np.nan, "XGBoost_F1_Macro": np.nan, "XGBoost_F1_Weighted": np.nan})
        # Thêm placeholder tương tự cho LSTM, Transformer nếu bạn dự định chạy chúng


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
        summary_valid = summary.dropna(subset=['mean'])
        if not summary_valid.empty:
             summary_valid['mean_std'] = summary_valid.apply(lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1)
             print(summary_valid[['mean_std']])
        else: print("No valid performance metrics to display.")
    else: print("No results to aggregate.")

else:
    print("\nEnriched data is empty or could not be loaded.")

print("\nScript finished.")