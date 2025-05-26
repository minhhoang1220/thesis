# File: marketml/models/lstm_model.py
import numpy as np
import logging

try:
    from marketml.utils import metrics
    from .keras_utils import KERAS_AVAILABLE
except ModuleNotFoundError:
    print("CRITICAL ERROR in lstm_model.py: Could not import 'marketml.utils.metrics' or '.keras_utils'.")
    raise

if KERAS_AVAILABLE:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout, Input
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
else:
    Sequential, LSTM, Dense, Dropout, Input, keras, Adam = (None,) * 7


def run_lstm_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                        class_weight_dict, n_classes, n_timesteps, n_features,
                        logger: logging.Logger, **kwargs):
    logger.info("--- Training and Evaluating LSTM Model ---")
    results = {}
    default_metrics = {
        "LSTM_Accuracy": np.nan, "LSTM_F1_Macro": np.nan, "LSTM_F1_Weighted": np.nan,
        "LSTM_Precision_Macro": np.nan, "LSTM_Recall_Macro": np.nan, "LSTM_ParamsUsed": "Skipped"
    }
    results.update(default_metrics)

    if not KERAS_AVAILABLE:
        logger.warning("  Skipping LSTM Model: TensorFlow/Keras not available.")
        return results
    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0 or y_test_original_trend.shape[0] == 0:
        logger.warning(f"  Skipping LSTM Model: Empty sequence data provided (X_train_seq: {X_train_seq.shape}, X_test_seq: {X_test_seq.shape}, y_test_original_trend: {y_test_original_trend.shape}).")
        return results
    
    try:
        keras.backend.clear_session()

        lstm_units = kwargs.get('lstm_units', 64)
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 64)
        validation_split = kwargs.get('validation_split', 0.1)
        early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        reduce_lr_patience = kwargs.get('reduce_lr_patience', 5)
        reduce_lr_factor = kwargs.get('reduce_lr_factor', 0.2)
        min_lr = kwargs.get('min_lr', 1e-6)

        logger.info(f"  Training LSTM with params: units={lstm_units}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

        lstm_model = Sequential([
            Input(shape=(n_timesteps, n_features)),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(n_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=learning_rate)
        lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        callbacks_list = []
        if early_stopping_patience > 0:
            callbacks_list.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0))
        if reduce_lr_patience > 0:
            callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=min_lr, verbose=0))

        logger.info(f"    Training LSTM for up to {epochs} epochs...")
        history_lstm = lstm_model.fit(
            X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict if class_weight_dict else None,
            validation_split=validation_split if validation_split > 0 else None,
            shuffle=True,
            callbacks=callbacks_list if callbacks_list else None,
            verbose=0
        )
        epochs_run = len(history_lstm.history['loss'])
        logger.info(f"    LSTM training finished after {epochs_run} epochs.")

        logger.info("    Predicting with LSTM...")
        if X_test_seq.shape[0] > 0:
            lstm_pred_probs = lstm_model.predict(X_test_seq)
            lstm_pred_keras = np.argmax(lstm_pred_probs, axis=1)
            lstm_pred_trend = lstm_pred_keras - 1

            if len(lstm_pred_trend) == len(y_test_original_trend):
                lstm_metrics = metrics.calculate_classification_metrics(
                    y_test_original_trend, lstm_pred_trend, model_name="LSTM", logger=logger
                )
                results.update(lstm_metrics)
            else:
                logger.warning(f"    LSTM: Length mismatch between predictions ({len(lstm_pred_trend)}) and original trend targets ({len(y_test_original_trend)}). Metrics not calculated.")
        else:
            logger.warning("    LSTM: X_test_seq is empty, no predictions to make.")

        results["LSTM_ParamsUsed"] = f"units={lstm_units},dropout={dropout_rate},lr={learning_rate},batch={batch_size},epochs_run={epochs_run}"

    except Exception as e:
        logger.error(f"Error during LSTM execution: {e}", exc_info=True)

    return results
