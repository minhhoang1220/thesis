# File: marketml/models/keras_utils.py
import numpy as np
try:
    # Chỉ import các thành phần cần thiết ở đây
    from tensorflow import keras
    from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Add, Input, GlobalAveragePooling1D
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    # Định nghĩa dummy nếu cần để tránh lỗi import ở nơi khác
    LayerNormalization, MultiHeadAttention, Dropout, Dense, Add, Input, GlobalAveragePooling1D = (None,) * 7


def create_sequences(X_data, y_data, time_steps=1):
    """
    Chuyển đổi dữ liệu 2D thành chuỗi 3D cho mô hình sequence.
    """
    if y_data is None: # Chỉ tạo sequence cho X (ví dụ: khi dự báo thực tế)
        Xs = []
        # Lặp đến hết để bao gồm chuỗi cuối cùng
        for i in range(len(X_data) - time_steps + 1):
             v = X_data[i:(i + time_steps)]
             Xs.append(v)
        return np.array(Xs), None # Return None for ys
    else:
        Xs, ys = [], []
        for i in range(len(X_data) - time_steps):
            v = X_data[i:(i + time_steps)]
            Xs.append(v)
            ys.append(y_data[i + time_steps]) # Lấy target sau chuỗi
        return np.array(Xs), np.array(ys)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Một khối mã hóa Transformer đơn giản."""
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for transformer_encoder.")

    # Attention and Normalization
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs]) # Skip connection 1

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x) # Project back to input dimension
    return Add()([x, res]) # Skip connection 2