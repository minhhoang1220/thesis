# File: marketml/models/transformer_model.py
import numpy as np
# Đảm bảo có thể import từ thư mục cha hoặc utils đã trong sys.path
try:
    from marketml.utils import metrics
    # Import utils và encoder từ keras_utils
    from .keras_utils import KERAS_AVAILABLE, transformer_encoder
except ModuleNotFoundError:
     try:
        from ..utils import metrics
        from .keras_utils import KERAS_AVAILABLE, transformer_encoder
     except ImportError:
        print("Error: Cannot import metrics or keras_utils module in transformer_model.py")
        metrics = None; KERAS_AVAILABLE = False; transformer_encoder = None

if KERAS_AVAILABLE:
    from tensorflow import keras
    from keras.models import Model
    from keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense
else:
    Model = None; Input = None; GlobalAveragePooling1D = None; Dropout = None; Dense = None; keras = None

def run_transformer_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                               class_weight_dict, n_classes, n_timesteps, n_features,
                               head_size, num_heads, ff_dim, dropout_rate, epochs, batch_size):
    """
    Huấn luyện, dự đoán và đánh giá mô hình Transformer đơn giản.
    """
    print("\n--- Training and Evaluating Transformer Model ---")
    results = {}
    default_metrics = {"Transformer_Accuracy": np.nan, "Transformer_F1_Macro": np.nan, "Transformer_F1_Weighted": np.nan,
                       "Transformer_Precision_Macro": np.nan, "Transformer_Recall_Macro": np.nan}
    results.update(default_metrics) # Khởi tạo

    if not KERAS_AVAILABLE or transformer_encoder is None or X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print("  Skipping Transformer Model (TensorFlow not installed, encoder error or no sequence data).")
        return results
    if metrics is None:
        print("    Skipping Transformer evaluation: Metrics module not imported.")
        return results

    try:
        keras.backend.clear_session()
        input_shape = (n_timesteps, n_features)
        inputs = Input(shape=input_shape)
        x = transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate)
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(n_classes, activation="softmax")(x)
        transformer_model = Model(inputs, outputs)

        transformer_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(f"    Training Transformer for {epochs} epochs...")
        transformer_model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
                              class_weight=class_weight_dict, validation_split=0.1, shuffle=True, verbose=0)
        print("    Training finished.")

        print("    Predicting with Transformer...")
        transformer_pred_probs = transformer_model.predict(X_test_seq)
        transformer_pred_keras = np.argmax(transformer_pred_probs, axis=1)
        transformer_pred_trend = transformer_pred_keras - 1

        transformer_metrics = metrics.calculate_classification_metrics(y_test_original_trend, transformer_pred_trend, model_name="Transformer")
        results.update(transformer_metrics) # Ghi đè NaN

    except Exception as e:
        print(f"Error during Transformer execution: {e}")
        # Giữ nguyên kết quả NaN

    return results