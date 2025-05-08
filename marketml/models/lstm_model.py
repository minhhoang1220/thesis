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
else:
    Sequential = None; LSTM = None; Dense = None; Dropout = None; Input = None; keras = None

def run_lstm_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                        class_weight_dict, n_classes, n_timesteps, n_features,
                        lstm_units, dropout_rate, epochs, batch_size):
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
        lstm_model = Sequential([
            Input(shape=(n_timesteps, n_features)),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(n_classes, activation='softmax')
        ])
        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(f"    Training LSTM for {epochs} epochs...")
        lstm_model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
                       class_weight=class_weight_dict, validation_split=0.1, shuffle=True, verbose=0)
        print("    Training finished.")

        print("    Predicting with LSTM...")
        lstm_pred_probs = lstm_model.predict(X_test_seq)
        lstm_pred_keras = np.argmax(lstm_pred_probs, axis=1)
        lstm_pred_trend = lstm_pred_keras - 1 # Về nhãn -1, 0, 1

        lstm_metrics = metrics.calculate_classification_metrics(y_test_original_trend, lstm_pred_trend, model_name="LSTM")
        results.update(lstm_metrics) # Ghi đè NaN

    except Exception as e:
        print(f"Error during LSTM execution: {e}")
        # Giữ nguyên kết quả NaN

    return results