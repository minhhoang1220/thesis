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
    from keras.callbacks import EarlyStopping
    try:
        _AdamW = keras.optimizers.AdamW # Gán vào biến tạm có dấu gạch dưới
    except AttributeError: # Nếu keras.optimizers không có AdamW (phiên bản TF/Keras cũ)
        print("Warning: keras.optimizers.AdamW not found. Falling back to keras.optimizers.Adam for Transformer.")
        _AdamW = keras.optimizers.Adam # Fallback
    try:
        _CosineDecay = keras.optimizers.schedules.CosineDecay
    except AttributeError:
        _CosineDecay = None
        print("Warning: keras.optimizers.schedules.CosineDecay not available. Will use fixed learning rate for Transformer.")
else:
    Model = None; Input = None; GlobalAveragePooling1D = None; Dropout = None; Dense = None; keras = None

def run_transformer_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                               class_weight_dict, n_classes, n_timesteps, n_features, **kwargs):
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

        # Lấy tham số tuning từ kwargs
        num_transformer_blocks = kwargs.get('num_transformer_blocks', 2) # Thử 2-3
        head_size = kwargs.get('head_size', 64) # Thử 64, 128
        num_heads = kwargs.get('num_heads', 2)   # Thử 2, 4
        ff_dim = kwargs.get('ff_dim', 64)       # Thử 64, 128
        dropout_rate = kwargs.get('dropout_rate', 0.25) # Tăng lên
        learning_rate = kwargs.get('learning_rate', 1e-4) # AdamW + scheduler
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 64)

        print(f"  Training Transformer with params: blocks={num_transformer_blocks}, head_size={head_size}, num_heads={num_heads}, ff_dim={ff_dim}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")


        input_shape = (n_timesteps, n_features)
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks): # Xếp chồng nhiều khối encoder
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)

        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(n_classes, activation="softmax")(x)
        transformer_model = Model(inputs, outputs)

        # Optimizer và Scheduler
        if _CosineDecay is not None: # Kiểm tra biến đã import
            decay_steps = epochs * (X_train_seq.shape[0] // batch_size + 1)
            lr_schedule = _CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.0)
            optimizer = _AdamW(learning_rate=lr_schedule, weight_decay=1e-4) # Sử dụng _AdamW đã import/fallback
        else:
            optimizer = _AdamW(learning_rate=learning_rate, weight_decay=1e-4) # Sử dụng _AdamW đã import/fallback

        transformer_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

        print(f"    Training Transformer for up to {epochs} epochs with EarlyStopping...")
        history_transformer = transformer_model.fit(
            X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict, validation_split=0.1, shuffle=True,
            callbacks=[early_stopping], verbose=0
        )
        epochs_run = len(history_transformer.history['loss'])
        print(f"    Training finished after {epochs_run} epochs.")

        print("    Predicting with Transformer...")
        transformer_pred_probs = transformer_model.predict(X_test_seq)
        transformer_pred_keras = np.argmax(transformer_pred_probs, axis=1)
        transformer_pred_trend = transformer_pred_keras - 1

        transformer_metrics_results = metrics.calculate_classification_metrics(y_test_original_trend, transformer_pred_trend, model_name="Transformer")
        results.update(transformer_metrics_results)
        results["Transformer_ParamsUsed"] = f"blocks={num_transformer_blocks},head_size={head_size},num_heads={num_heads},ff_dim={ff_dim},dropout={dropout_rate},lr_used,epochs_run={epochs_run}"

    except Exception as e:
        print(f"Error during Transformer execution: {e}")
        # Giữ nguyên NaN

    return results