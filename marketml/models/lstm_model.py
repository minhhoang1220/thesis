# File: marketml/models/lstm_model.py
import numpy as np
# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
    from .keras_utils import KERAS_AVAILABLE # Kiểm tra Keras từ utils
except ModuleNotFoundError:
     try:
        from ..utils import metrics
        from .keras_utils import KERAS_AVAILABLE
     except ImportError:
        print("Error: Cannot import metrics or keras_utils module in lstm_model.py")
        metrics = None; KERAS_AVAILABLE = False

if KERAS_AVAILABLE:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
else:
    Sequential = None; LSTM = None; Dense = None; Dropout = None; Input = None; keras = None

def run_lstm_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                        class_weight_dict, n_classes, n_timesteps, n_features, **kwargs):
    """
    Huấn luyện, dự đoán và đánh giá mô hình LSTM.
    """
    print("\n--- Training and Evaluating LSTM Model ---")
    results = {}
    default_metrics = {"LSTM_Accuracy": np.nan, "LSTM_F1_Macro": np.nan, "LSTM_F1_Weighted": np.nan,
                       "LSTM_Precision_Macro": np.nan, "LSTM_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not KERAS_AVAILABLE or X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print("  Skipping LSTM Model (TensorFlow not installed or no sequence data).")
        return results
    if metrics is None:
        print("    Skipping LSTM evaluation: Metrics module not imported.")
        return results

    try:
        keras.backend.clear_session()

        # Lấy tham số tuning từ kwargs hoặc dùng giá trị mặc định
        lstm_units = kwargs.get('lstm_units', 64) # Thử 32, 64, 128
        dropout_rate = kwargs.get('dropout_rate', 0.3) # Thử 0.3 - 0.5
        learning_rate = kwargs.get('learning_rate', 1e-4)
        epochs = kwargs.get('epochs', 50) # Tăng epochs, dùng EarlyStopping
        batch_size = kwargs.get('batch_size', 64) # Thử 32, 64, 128

        print(f"  Training LSTM with params: units={lstm_units}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

        lstm_model = Sequential([
            Input(shape=(n_timesteps, n_features)),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(n_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=learning_rate)
        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)

        print(f"    Training LSTM for up to {epochs} epochs with EarlyStopping...")
        history_lstm = lstm_model.fit(
            X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict, validation_split=0.1, shuffle=True,
            callbacks=[early_stopping, reduce_lr], verbose=0 # Thêm callbacks
        )
        print(f"    Training finished after {len(history_lstm.history['loss'])} epochs.")


        print("    Predicting with LSTM...")
        lstm_pred_probs = lstm_model.predict(X_test_seq)
        lstm_pred_keras = np.argmax(lstm_pred_probs, axis=1)
        lstm_pred_trend = lstm_pred_keras - 1

        lstm_metrics = metrics.calculate_classification_metrics(y_test_original_trend, lstm_pred_trend, model_name="LSTM")
        results.update(lstm_metrics)
        # Lưu lại tham số đã dùng
        results["LSTM_ParamsUsed"] = f"units={lstm_units},dropout={dropout_rate},lr={learning_rate},batch={batch_size},epochs_run={len(history_lstm.history['loss'])}"


    except Exception as e:
        print(f"Error during LSTM execution: {e}")
        # Giữ nguyên NaN

    return results