# /.ndmh/marketml/pipelines/02_train_forecasting_models.py

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ===== Import modules from marketml and configs =====
try:
    from marketml.configs import configs
    from marketml.utils import logger_setup, metrics
    from marketml.models import (
        arima_model,
        rf_model,
        lstm_model,
        transformer_model,
        keras_utils,
        xgboost_model,
        svm_model
    )
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR in 02_train_forecasting_models.py: Could not import necessary marketml modules. {e}")
    print("Ensure the marketml package is installed correctly (e.g., `pip install -e .` from .ndmh) or PYTHONPATH is set.")
    raise

# ===== Logger and environment setup =====
logger = logger_setup.setup_basic_logging(log_file_name="train_forecasting_models.log")
logger_setup.suppress_common_warnings()
logger_setup.set_random_seeds(configs.RANDOM_SEED)

# ===== Check optional libraries =====
TF_INSTALLED = keras_utils.KERAS_AVAILABLE
if not TF_INSTALLED:
     logger.warning("TensorFlow/Keras not available. LSTM and Transformer models will be skipped.")

try:
    import xgboost
    XGB_INSTALLED = True
except ImportError:
    logger.warning("xgboost not installed. XGBoost model will be skipped.")
    XGB_INSTALLED = False

try:
    import sklearn
    SKLEARN_INSTALLED = True
except ImportError:
    logger.warning("scikit-learn not installed. RF, SVM, StandardScaler, SimpleImputer, class_weight might fail.")
    SKLEARN_INSTALLED = False

try:
    from arch import arch_model
    ARCH_INSTALLED = True
except ImportError:
    logger.info("arch library not installed. If 'garch_vol_forecast' is used, it might be all NaNs.")
    ARCH_INSTALLED = False

# ==============================================================================
# FUNCTION TO CREATE ROLLING/EXPANDING WINDOWS
# ==============================================================================
def create_time_series_cv_splits(
    df: pd.DataFrame, date_col: str, ticker_col: str, initial_train_period: pd.Timedelta,
    test_period: pd.Timedelta, step_period: pd.Timedelta, expanding: bool = False
):
    logger.info(f"Generating CV splits ({'Expanding' if expanding else 'Rolling'} Window):")
    logger.info(f"  Initial Train: {initial_train_period}, Test Period: {test_period}, Step Period: {step_period}")
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    if min_date + initial_train_period + test_period > max_date:
        logger.warning("Not enough data for even one CV split based on current settings.")
        return

    current_train_start_date = min_date
    current_train_end_date = min_date + initial_train_period
    split_index = 0

    while True:
        current_test_start_date = current_train_end_date
        current_test_end_date = current_test_start_date + test_period

        if current_test_end_date > max_date:
            logger.info(f"Stopping CV split generation: Test end date {current_test_end_date.date()} would exceed max data date {max_date.date()}.")
            break

        train_mask = (df[date_col] >= current_train_start_date) & (df[date_col] < current_train_end_date)
        test_mask = (df[date_col] >= current_test_start_date) & (df[date_col] < current_test_end_date)

        train_split_df = df.loc[train_mask].copy()
        test_split_df = df.loc[test_mask].copy()

        if not train_split_df.empty and not test_split_df.empty:
            logger.info(f"  Split {split_index}: Train [{train_split_df[date_col].min().date()}:{train_split_df[date_col].max().date()}], "
                        f"Test [{test_split_df[date_col].min().date()}:{test_split_df[date_col].max().date()}]")
            yield split_index, train_split_df, test_split_df
            split_index += 1
        else:
            logger.warning(f"  Skipping split generation for train end {current_train_end_date.date()}: train or test data is empty.")

        if not expanding:
            current_train_start_date += step_period
        current_train_end_date += step_period

    logger.info(f"Total CV splits generated: {split_index}")

# ==============================================================================
# FUNCTION TO PREPARE DATA IN EACH SPLIT
# ==============================================================================
def prepare_split_data(train_df_orig, test_df_orig, feature_cols, lag_periods,
                       target_col_pct, target_col_trend, trend_threshold, n_timesteps):
    logger.info("  Preparing data for the current CV split...")
    train_df = train_df_orig.copy()
    test_df = test_df_orig.copy()
    prepared_data = {'data_valid': False}

    try:
        for df_split in [train_df, test_df]:
            df_split[target_col_pct] = df_split.groupby('ticker')['close'].pct_change().shift(-1)
            conditions = [
                (df_split[target_col_pct] > trend_threshold),
                (df_split[target_col_pct] < -trend_threshold)
            ]
            choices = [1, -1]
            df_split[target_col_trend] = np.select(conditions, choices, default=0)

        for df_split in [train_df, test_df]:
            pct_change_series = df_split.groupby('ticker')['close'].pct_change()
            for p in lag_periods:
                df_split[f'pct_change_lag_{p}'] = pct_change_series.groupby(df_split['ticker']).shift(p)

        X_train_raw = train_df[feature_cols].copy()
        X_test_raw = test_df[feature_cols].copy()
        y_train_trend_ml = train_df[target_col_trend]
        y_test_trend_ml = test_df[target_col_trend]

        for col_to_check in feature_cols:
            if col_to_check in X_train_raw.columns and X_train_raw[col_to_check].isnull().all():
                logger.warning(f"    Feature '{col_to_check}' is all NaN in training data for this split. Filling with 0 before imputation.")
                X_train_raw[col_to_check] = X_train_raw[col_to_check].fillna(0)
                if col_to_check in X_test_raw.columns:
                    X_test_raw[col_to_check] = X_test_raw[col_to_check].fillna(0)

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed_np = imputer.fit_transform(X_train_raw)
        X_test_imputed_np = imputer.transform(X_test_raw)

        X_train_imputed = pd.DataFrame(X_train_imputed_np, index=X_train_raw.index, columns=X_train_raw.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed_np, index=X_test_raw.index, columns=X_test_raw.columns)

        valid_y_train_idx = y_train_trend_ml.dropna().index
        X_train_ml = X_train_imputed.loc[X_train_imputed.index.isin(valid_y_train_idx)]
        y_train_ml_aligned = y_train_trend_ml.loc[y_train_trend_ml.index.isin(valid_y_train_idx)]

        valid_y_test_idx = y_test_trend_ml.dropna().index
        X_test_ml = X_test_imputed.loc[X_test_imputed.index.isin(valid_y_test_idx)]
        y_test_ml_aligned = y_test_trend_ml.loc[y_test_trend_ml.index.isin(valid_y_test_idx)]

        if X_train_ml.empty or y_train_ml_aligned.empty or X_test_ml.empty or y_test_ml_aligned.empty:
            logger.warning("  ML data (X or y) is empty after alignment and NaN drop. Cannot proceed with ML/Sequence models for this split.")
            prepared_data.update({'train_df': train_df_orig, 'test_df': test_df_orig})
            return prepared_data

        prepared_data['y_test_ml'] = y_test_ml_aligned.values

        scaler = StandardScaler()
        X_train_scaled_np = scaler.fit_transform(X_train_ml)
        X_test_scaled_np = scaler.transform(X_test_ml)

        prepared_data['X_train_scaled'] = X_train_scaled_np
        prepared_data['y_train_ml'] = y_train_ml_aligned.values
        prepared_data['X_test_scaled'] = X_test_scaled_np
        prepared_data['feature_names'] = X_train_ml.columns.tolist()

        y_train_keras = y_train_ml_aligned.replace({-1: 0, 0: 1, 1: 2}).values
        y_test_keras_for_seq_target = y_test_ml_aligned.replace({-1: 0, 0: 1, 1: 2}).values

        prepared_data['n_classes'] = 3

        X_train_seq, y_train_seq = keras_utils.create_sequences(X_train_scaled_np, y_train_keras, n_timesteps)
        X_test_seq, _ = keras_utils.create_sequences(X_test_scaled_np, None, n_timesteps)

        if len(y_test_ml_aligned) >= n_timesteps and X_test_seq.shape[0] > 0:
            prepared_data['y_test_seq_original_trend'] = y_test_ml_aligned.iloc[n_timesteps -1 :].values[:len(X_test_seq)]
        else:
            prepared_data['y_test_seq_original_trend'] = np.array([])

        prepared_data.update({
            'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq,
            'X_test_seq': X_test_seq
        })

        if X_train_seq.shape[0] > 0 and y_train_seq.shape[0] > 0:
            unique_classes_in_seq = np.unique(y_train_seq)
            if len(unique_classes_in_seq) > 1:
                class_weights_keras = class_weight.compute_class_weight(
                    'balanced',
                    classes=unique_classes_in_seq,
                    y=y_train_seq
                )
                prepared_data['class_weight_dict'] = dict(zip(unique_classes_in_seq, class_weights_keras))
            else:
                prepared_data['class_weight_dict'] = {unique_classes_in_seq[0]: 1.0}
            logger.info(f"  Sequence shapes: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}, X_test={X_test_seq.shape}")
            logger.info(f"  Class weights for Keras: {prepared_data.get('class_weight_dict', 'Not computed')}")
        else:
            prepared_data['class_weight_dict'] = {}
            logger.warning("  Not enough data to create sequences for Keras models for this split.")

        prepared_data['data_valid'] = True
        prepared_data['train_df_with_target_for_arima'] = train_df
        prepared_data['test_df_with_target_for_arima'] = test_df 

    except Exception as e:
        logger.error(f"Error during data preparation for split: {e}", exc_info=True)
        prepared_data['data_valid'] = False
        if 'train_df' not in prepared_data: prepared_data['train_df'] = train_df_orig.copy()
        if 'test_df' not in prepared_data: prepared_data['test_df'] = test_df_orig.copy()

    logger.info("  Data preparation finished for the current CV split.")
    return prepared_data

# ==============================================================================
# MAIN FUNCTION TO RUN EXPERIMENTS
# ==============================================================================
def main():
    logger.info("Starting: 02_train_forecasting_models pipeline")

    try:
        logger.info(f"Loading enriched data from: {configs.ENRICHED_DATA_FILE}")
        df_with_indicators = pd.read_csv(configs.ENRICHED_DATA_FILE, parse_dates=['date'])
        logger.info("Enriched data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Enriched data file not found at '{configs.ENRICHED_DATA_FILE}'. "
                     f"Run '01_build_features.py' first. Exiting.")
        return
    except Exception as e:
        logger.error(f"CRITICAL: Error loading enriched data: {e}. Exiting.", exc_info=True)
        return

    required_cols_check = ['date', 'ticker', 'close'] + configs.BASE_FEATURE_COLS[:2]
    missing_cols = []
    for col in required_cols_check:
        if col not in df_with_indicators.columns:
            if col == 'garch_vol_forecast' and not ARCH_INSTALLED:
                logger.info(f"Feature 'garch_vol_forecast' not found, but ARCH library is not installed. This is expected.")
            else:
                missing_cols.append(col)
    
    if missing_cols:
        logger.error(f"CRITICAL: Missing some expected columns from enriched data: {missing_cols}. Exiting.")
        return

    if not pd.api.types.is_datetime64_any_dtype(df_with_indicators['date']):
        logger.error("CRITICAL: 'date' column in enriched data is not datetime type. Exiting.")
        return

    if df_with_indicators.empty:
        logger.error("CRITICAL: Enriched data is empty. No experiments can be run. Exiting.")
        return

    logger.info("Starting Cross-Validation Runs for forecasting models...")
    initial_train_td = configs.INITIAL_TRAIN_TIMEDELTA
    test_td = configs.TEST_TIMEDELTA
    step_td = configs.STEP_TIMEDELTA
    use_expanding_window = configs.USE_EXPANDING_WINDOW
    trend_threshold = configs.TREND_THRESHOLD
    lag_periods = configs.LAG_PERIODS

    feature_cols_base = configs.BASE_FEATURE_COLS
    actual_base_features = [f for f in feature_cols_base if f in df_with_indicators.columns]
    if len(actual_base_features) < len(feature_cols_base):
        missing_base = set(feature_cols_base) - set(actual_base_features)
        logger.warning(f"Some BASE_FEATURE_COLS from config are missing in the loaded data: {missing_base}. They will be excluded.")

    feature_cols_for_ml = sorted(list(set(actual_base_features + [f'pct_change_lag_{p}' for p in lag_periods])))
    logger.info(f"Using {len(feature_cols_for_ml)} features for ML models: {feature_cols_for_ml}")

    target_col_pct = configs.TARGET_COL_PCT
    target_col_trend = configs.TARGET_COL_TREND
    
    n_iter_search_sklearn = configs.N_ITER_SEARCH_SKLEARN
    cv_folds_tuning_sklearn = configs.CV_FOLDS_TUNING_SKLEARN
    n_timesteps_sequence = configs.N_TIMESTEPS_SEQUENCE
    
    lstm_params_set = {
        'lstm_units': configs.LSTM_UNITS, 'dropout_rate': configs.LSTM_DROPOUT_RATE,
        'learning_rate': configs.LSTM_LEARNING_RATE, 'epochs': configs.KERAS_EPOCHS,
        'batch_size': configs.KERAS_BATCH_SIZE, 'validation_split': configs.KERAS_VALIDATION_SPLIT,
        'early_stopping_patience': configs.KERAS_EARLY_STOPPING_PATIENCE,
        'reduce_lr_patience': configs.KERAS_REDUCE_LR_PATIENCE,
        'reduce_lr_factor': configs.KERAS_REDUCE_LR_FACTOR, 'min_lr': configs.KERAS_MIN_LR
    }
    transformer_params_set = {
        'num_transformer_blocks': configs.TRANSFORMER_NUM_BLOCKS, 'head_size': configs.TRANSFORMER_HEAD_SIZE,
        'num_heads': configs.TRANSFORMER_NUM_HEADS, 'ff_dim': configs.TRANSFORMER_FF_DIM,
        'dropout_rate': configs.TRANSFORMER_DROPOUT_RATE, 'learning_rate': configs.TRANSFORMER_LEARNING_RATE,
        'epochs': configs.KERAS_EPOCHS, 'batch_size': configs.KERAS_BATCH_SIZE,
        'weight_decay': configs.TRANSFORMER_WEIGHT_DECAY, 'validation_split': configs.KERAS_VALIDATION_SPLIT,
        'early_stopping_patience': configs.KERAS_EARLY_STOPPING_PATIENCE
    }
    
    # --- Step 2: Create Cross-Validation Splits ---
    cv_splitter = create_time_series_cv_splits(
        df=df_with_indicators, date_col='date', ticker_col='ticker',
        initial_train_period=initial_train_td, test_period=test_td,
        step_period=step_td, expanding=use_expanding_window
    )
    all_split_results = {}

    for split_idx, train_df_orig, test_df_orig in cv_splitter:
        logger.info(f"===== Processing CV Split {split_idx} =====")
        results_this_split = {}

        prep_data = prepare_split_data(
            train_df_orig, test_df_orig, feature_cols_for_ml, lag_periods,
            target_col_pct, target_col_trend, trend_threshold, n_timesteps_sequence
        )

        if prep_data.get('data_valid', False):
            logger.info("  Running ARIMA model evaluation...")
            arima_results = arima_model.run_arima_evaluation(
                prep_data['train_df_with_target_for_arima'],
                prep_data['test_df_with_target_for_arima'],
                target_col_trend,
                trend_threshold,
                logger=logger
            )
            results_this_split.update(arima_results)
            
            if SKLEARN_INSTALLED:
                logger.info("  Running RandomForest model evaluation...")
                rf_results = rf_model.run_rf_evaluation(
                    prep_data['X_train_scaled'], prep_data['y_train_ml'],
                    prep_data['X_test_scaled'], prep_data['y_test_ml'],
                    n_iter_search=n_iter_search_sklearn,
                    cv_folds_tuning=cv_folds_tuning_sklearn,
                    logger=logger
                )
                results_this_split.update(rf_results)

            if XGB_INSTALLED:
                logger.info("  Running XGBoost model evaluation...")
                xgb_results = xgboost_model.run_xgboost_evaluation(
                    prep_data['X_train_scaled'], prep_data['y_train_ml'],
                    prep_data['X_test_scaled'], prep_data['y_test_ml'],
                    n_iter_search=n_iter_search_sklearn,
                    cv_folds_tuning=cv_folds_tuning_sklearn,
                    logger=logger
                )
                results_this_split.update(xgb_results)
            
            if SKLEARN_INSTALLED:
                logger.info("  Running SVM model evaluation...")
                svm_results = svm_model.run_svm_evaluation(
                    prep_data['X_train_scaled'], prep_data['y_train_ml'],
                    prep_data['X_test_scaled'], prep_data['y_test_ml'],
                    n_iter_search=n_iter_search_sklearn,
                    cv_folds_tuning=cv_folds_tuning_sklearn,
                    logger=logger
                )
                results_this_split.update(svm_results)
            
            if prep_data['X_train_scaled'].ndim == 2:
                current_n_features_for_seq = prep_data['X_train_scaled'].shape[1]
            elif prep_data['X_train_seq'].ndim == 3:
                current_n_features_for_seq = prep_data['X_train_seq'].shape[2]
            else:
                logger.error("  Could not determine number of features for sequence models. Skipping LSTM/Transformer.")
                current_n_features_for_seq = 0

            if TF_INSTALLED and prep_data['X_train_seq'].shape[0] > 0 and current_n_features_for_seq > 0:
                logger.info("  Running LSTM model evaluation...")
                lstm_results = lstm_model.run_lstm_evaluation(
                    prep_data['X_train_seq'], prep_data['y_train_seq'],
                    prep_data['X_test_seq'], prep_data['y_test_seq_original_trend'],
                    prep_data['class_weight_dict'], prep_data['n_classes'],
                    n_timesteps_sequence, current_n_features_for_seq,
                    **lstm_params_set,
                    logger=logger
                )
                results_this_split.update(lstm_results)
            elif TF_INSTALLED:
                logger.warning("  Skipping LSTM for this split: No valid sequence data or features.")

            if TF_INSTALLED and prep_data['X_train_seq'].shape[0] > 0 and current_n_features_for_seq > 0:
                logger.info("  Running Transformer model evaluation...")
                transformer_results = transformer_model.run_transformer_evaluation(
                    prep_data['X_train_seq'], prep_data['y_train_seq'],
                    prep_data['X_test_seq'], prep_data['y_test_seq_original_trend'],
                    prep_data['class_weight_dict'], prep_data['n_classes'],
                    n_timesteps_sequence, current_n_features_for_seq,
                    **transformer_params_set,
                    logger=logger
                )
                results_this_split.update(transformer_results)
            elif TF_INSTALLED:
                logger.warning("  Skipping Transformer for this split: No valid sequence data or features.")

        else:
            logger.warning(f"Skipping model evaluation for Split {split_idx} due to invalid prepared data.")
            model_names_for_nan = ["ARIMA", "RandomForest", "XGBoost", "SVM", "LSTM", "Transformer"]
            metric_suffixes_for_nan = ["Accuracy", "F1_Macro", "F1_Weighted", "Precision_Macro", "Recall_Macro"]
            for model_name_nan in model_names_for_nan:
                for metric_suffix_nan in metric_suffixes_for_nan:
                    results_this_split[f"{model_name_nan}_{metric_suffix_nan}"] = np.nan

        all_split_results[split_idx] = results_this_split
        logger.info(f"===== Finished CV Split {split_idx} =====")

    # --- Step 5: Aggregate and Save Results ---
    logger.info("===== Aggregating Results Across All CV Splits =====")
    if not all_split_results:
        logger.warning("No results from any CV split to aggregate. Exiting.")
        return

    final_results_df = pd.DataFrame.from_dict(all_split_results, orient='index')
    final_results_df.dropna(axis=1, how='all', inplace=True)

    if final_results_df.empty:
        logger.warning("Final results DataFrame is empty after dropping all-NaN columns. Nothing to save or summarize.")
        return

    try:
        configs.RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        detailed_results_file_path = configs.MODEL_PERF_DETAILED_FILE
        final_results_df.to_csv(detailed_results_file_path)
        logger.info(f"Detailed results per CV split saved to: {detailed_results_file_path.resolve()}")
    except Exception as e:
        logger.error(f"Error saving detailed results to CSV: {e}", exc_info=True)

    logger.info("--- Performance Summary (Mean +/- Std Dev across CV Splits) ---")
    numeric_metric_cols = [col for col in final_results_df.columns if any(m_name in col for m_name in ["Accuracy", "F1", "Precision", "Recall"])]

    if not numeric_metric_cols:
        logger.warning("No numeric metric columns found in aggregated results to create a summary.")
    else:
        summary_stats = final_results_df[numeric_metric_cols].agg(['mean', 'std']).T
        summary_stats_valid = summary_stats.dropna(subset=['mean']).copy()

        if not summary_stats_valid.empty:
            summary_display = summary_stats_valid.copy()
            summary_display['Mean +/- Std'] = summary_display.apply(
                lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1
            )
            logger.info(f"\n{summary_display[['Mean +/- Std']].to_string()}")

            try:
                summary_file_path = configs.MODEL_PERF_SUMMARY_FILE
                summary_stats_valid.to_csv(summary_file_path)
                logger.info(f"Performance summary (mean, std) saved to: {summary_file_path.resolve()}")
            except Exception as e:
                logger.error(f"Error saving performance summary to CSV: {e}", exc_info=True)
        else:
            logger.warning("No valid performance metrics (with non-NaN mean) to display or save in summary.")

    logger.info("Finished: 02_train_forecasting_models pipeline")

if __name__ == "__main__":
    main()