# File: marketml/models/transformer_model.py
import numpy as np
import logging

try:
    from marketml.utils import metrics
    from .keras_utils import KERAS_AVAILABLE, transformer_encoder
except ModuleNotFoundError:
    print("CRITICAL ERROR in transformer_model.py: Could not import 'marketml.utils.metrics' or '.keras_utils'.")
    raise

if KERAS_AVAILABLE:
    from tensorflow import keras
    from keras.models import Model
    from keras.layers import Input, GlobalAveragePooling1D, Dropout, Dense
    from keras.callbacks import EarlyStopping
    try:
        _AdamW = keras.optimizers.AdamW
    except AttributeError:
        _AdamW = keras.optimizers.Adam
    try:
        _CosineDecay = keras.optimizers.schedules.CosineDecay
    except AttributeError:
        _CosineDecay = None
else:
    Model, Input, GlobalAveragePooling1D, Dropout, Dense, keras, _AdamW, _CosineDecay = (None,) * 8


def run_transformer_evaluation(X_train_seq, y_train_seq, X_test_seq, y_test_original_trend,
                               class_weight_dict, n_classes, n_timesteps, n_features,
                               logger: logging.Logger, **kwargs):
    logger.info("--- Training and Evaluating Transformer Model ---")
    results = {}
    default_metrics = {
        "Transformer_Accuracy": np.nan, "Transformer_F1_Macro": np.nan,
        "Transformer_F1_Weighted": np.nan, "Transformer_Precision_Macro": np.nan,
        "Transformer_Recall_Macro": np.nan, "Transformer_ParamsUsed": "Skipped"
    }
    results.update(default_metrics)

    if not KERAS_AVAILABLE or transformer_encoder is None:
        logger.warning("  Skipping Transformer Model: TensorFlow/Keras or transformer_encoder not available.")
        return results
    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0 or y_test_original_trend.shape[0] == 0 :
        logger.warning(f"  Skipping Transformer Model: Empty sequence data provided (X_train_seq: {X_train_seq.shape}, X_test_seq: {X_test_seq.shape}, y_test_original_trend: {y_test_original_trend.shape}).")
        return results

    try:
        keras.backend.clear_session()

        num_transformer_blocks = kwargs.get('num_transformer_blocks', 2)
        head_size = kwargs.get('head_size', 64)
        num_heads = kwargs.get('num_heads', 2)
        ff_dim = kwargs.get('ff_dim', 64)
        dropout_rate = kwargs.get('dropout_rate', 0.25)
        learning_rate = kwargs.get('learning_rate', 1e-4)
        epochs = kwargs.get('epochs', 50)
        batch_size = kwargs.get('batch_size', 64)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        validation_split = kwargs.get('validation_split', 0.1)
        early_stopping_patience = kwargs.get('early_stopping_patience', 10)

        logger.info(f"  Training Transformer with params: blocks={num_transformer_blocks}, head_size={head_size}, num_heads={num_heads}, ff_dim={ff_dim}, dropout={dropout_rate}, lr={learning_rate}, epochs={epochs}, batch_size={batch_size}, wd={weight_decay}")

        input_shape = (n_timesteps, n_features)
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
        x = GlobalAveragePooling1D(data_format="channels_last")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(n_classes, activation="softmax")(x)
        transformer_model = Model(inputs, outputs)

        optimizer_to_use = _AdamW
        if _CosineDecay:
            num_train_steps_per_epoch = (X_train_seq.shape[0] + batch_size - 1) // batch_size
            decay_steps = epochs * num_train_steps_per_epoch
            lr_schedule = _CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.0)
            optimizer = optimizer_to_use(learning_rate=lr_schedule, weight_decay=weight_decay)
            logger.info("    Using AdamW with CosineDecay learning rate schedule.")
        else:
            optimizer = optimizer_to_use(learning_rate=learning_rate, weight_decay=weight_decay)
            logger.warning("    CosineDecay not available. Using AdamW with fixed learning rate.")

        transformer_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        callbacks_list = []
        if early_stopping_patience > 0:
             callbacks_list.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=0))

        logger.info(f"    Training Transformer for up to {epochs} epochs...")
        history_transformer = transformer_model.fit(
            X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict if class_weight_dict else None,
            validation_split=validation_split if validation_split > 0 else None,
            shuffle=True, callbacks=callbacks_list if callbacks_list else None, verbose=0
        )
        epochs_run = len(history_transformer.history['loss'])
        logger.info(f"    Transformer training finished after {epochs_run} epochs.")

        logger.info("    Predicting with Transformer...")
        if X_test_seq.shape[0] > 0:
            transformer_pred_probs = transformer_model.predict(X_test_seq)
            transformer_pred_keras = np.argmax(transformer_pred_probs, axis=1)
            transformer_pred_trend = transformer_pred_keras - 1

            if len(transformer_pred_trend) == len(y_test_original_trend):
                transformer_metrics_results = metrics.calculate_classification_metrics(
                    y_test_original_trend, transformer_pred_trend, model_name="Transformer", logger=logger
                )
                results.update(transformer_metrics_results)
            else:
                logger.warning(f"    Transformer: Length mismatch between predictions ({len(transformer_pred_trend)}) and original trend targets ({len(y_test_original_trend)}). Metrics not calculated.")
        else:
            logger.warning("    Transformer: X_test_seq is empty, no predictions to make.")

        results["Transformer_ParamsUsed"] = f"blocks={num_transformer_blocks},head_size={head_size},num_heads={num_heads},ff_dim={ff_dim},dropout={dropout_rate},lr_used,epochs_run={epochs_run}"

    except Exception as e:
        logger.error(f"Error during Transformer execution: {e}", exc_info=True)
    return results
