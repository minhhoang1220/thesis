# File: marketml/models/keras_utils.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
    from keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Add, Input, GlobalAveragePooling1D
    KERAS_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow/Keras not found. Keras-dependent models and utilities will not be available.")
    KERAS_AVAILABLE = False
    LayerNormalization, MultiHeadAttention, Dropout, Dense, Add, Input, GlobalAveragePooling1D = (None,) * 7


def create_sequences(X_data, y_data, time_steps=1):
    """
    Convert 2D data into 3D sequences for sequence models.
    """
    if not isinstance(X_data, np.ndarray): X_data = np.array(X_data)
    if y_data is not None and not isinstance(y_data, np.ndarray): y_data = np.array(y_data)

    if len(X_data) < time_steps:
        logger.warning(f"Not enough data (len: {len(X_data)}) to create sequences with time_steps={time_steps}. Returning empty arrays.")
        if y_data is not None:
            return np.array([]).reshape(0, time_steps, X_data.shape[1] if X_data.ndim > 1 else 1), np.array([])
        else:
            return np.array([]).reshape(0, time_steps, X_data.shape[1] if X_data.ndim > 1 else 1), None

    Xs = []
    if y_data is None:
        for i in range(len(X_data) - time_steps + 1):
             v = X_data[i:(i + time_steps)]
             Xs.append(v)
        return np.array(Xs), None
    else:
        ys = []
        for i in range(len(X_data) - time_steps):
            v = X_data[i:(i + time_steps)]
            Xs.append(v)
            ys.append(y_data[i + time_steps])
        return np.array(Xs), np.array(ys)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """A simple Transformer encoder block."""
    if not KERAS_AVAILABLE:
        logger.error("TensorFlow/Keras is required for transformer_encoder but not available.")
        raise ImportError("TensorFlow/Keras is required for transformer_encoder.")

    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])
