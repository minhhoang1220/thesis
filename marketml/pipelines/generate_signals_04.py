# /.ndmh/marketml/pipelines/04_generate_signals.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ===== Imports from marketml and configs =====
try:
    from marketml.configs import configs
    from marketml.data_handling import preprocess
    from marketml.models import xgboost_model
    from marketml.utils import logger_setup
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except ImportError as e:
    print(f"CRITICAL ERROR in 04_generate_signals.py: Could not import necessary marketml modules. {e}")
    print("Ensure the marketml package is installed correctly or PYTHONPATH is set.")
    raise

# ===== Logger and environment setup =====
logger = logger_setup.setup_basic_logging(log_file_name="generate_signals.log")
logger_setup.suppress_common_warnings()
if hasattr(configs, 'RANDOM_SEED'):
    logger_setup.set_random_seeds(configs.RANDOM_SEED)

# ===== XGBoost check (if selected model) =====
XGB_INSTALLED = False
if configs.SOFT_SIGNAL_MODEL_NAME == 'XGBoost':
    try:
        import xgboost
        XGB_INSTALLED = True
    except ImportError:
        logger.error("XGBoost selected as SOFT_SIGNAL_MODEL_NAME, but xgboost library is not installed. This pipeline will likely fail.")


def prepare_data_for_signal_generation(df_full, feature_cols, lag_periods,
                                       target_col_pct, target_col_trend, trend_threshold,
                                       start_date_filter, end_date_filter,
                                       fit_imputer_scaler=True,
                                       existing_imputer=None, existing_scaler=None):
    logger.info(f"Preparing data for signal generation from {start_date_filter} to {end_date_filter}...")
    df_filtered = df_full[(df_full['date'] >= pd.to_datetime(start_date_filter)) & \
                          (df_full['date'] <= pd.to_datetime(end_date_filter))].copy()

    if df_filtered.empty:
        logger.warning(f"No data found in the specified date range: {start_date_filter} to {end_date_filter}")
        return None, None, None, None, None

    # ---- Target creation (only for training set) ----
    if fit_imputer_scaler:
        df_filtered[target_col_pct] = df_filtered.groupby('ticker')['close'].pct_change().shift(-1)
        conditions = [
            (df_filtered[target_col_pct] > trend_threshold),
            (df_filtered[target_col_pct] < -trend_threshold)
        ]
        choices = [1, -1]
        df_filtered[target_col_trend] = np.select(conditions, choices, default=0)

    # ---- Lagged features creation ----
    df_with_lags_temp = df_full.copy()
    df_with_lags_temp['pct_change_temp'] = df_with_lags_temp.groupby('ticker')['close'].pct_change()

    for p in lag_periods:
        lag_col_name = f'pct_change_lag_{p}'
        df_with_lags_temp[lag_col_name] = df_with_lags_temp.groupby('ticker')['pct_change_temp'].shift(p)
        if df_filtered.index.equals(df_with_lags_temp.index):
             df_filtered[lag_col_name] = df_with_lags_temp[lag_col_name]
        else:
            df_filtered = df_filtered.join(df_with_lags_temp[[lag_col_name]], how='left')

    # ---- Impute & align features ----
    actual_feature_cols = [col for col in feature_cols if col in df_filtered.columns]
    missing_from_filtered = set(feature_cols) - set(actual_feature_cols)
    if missing_from_filtered:
        logger.warning(f"Some feature columns were not found in the filtered dataframe after lag creation: {missing_from_filtered}. These will be excluded.")
    
    if not actual_feature_cols:
        logger.error("No feature columns available after lag creation and filtering. Cannot proceed.")
        return None, None, None, None, None

    X_raw = df_filtered[actual_feature_cols].copy()

    for col_to_check in actual_feature_cols:
        if X_raw[col_to_check].isnull().all():
            logger.warning(f"Feature '{col_to_check}' is all NaN in the current data slice. Filling with 0 before imputation.")
            X_raw[col_to_check] = X_raw[col_to_check].fillna(0)

    if fit_imputer_scaler:
        imputer = SimpleImputer(strategy='mean')
        X_imputed_np = imputer.fit_transform(X_raw)
        scaler = StandardScaler()
        X_scaled_np = scaler.fit_transform(X_imputed_np)
        
        y_ml_series = df_filtered[target_col_trend].copy()
        valid_y_idx = y_ml_series.dropna().index
        valid_idx_in_X = X_raw.index.intersection(valid_y_idx)

        if valid_idx_in_X.empty:
            logger.error("No valid (non-NaN target) data points remain after target alignment. Aborting.")
            return None, None, None, None, None

        X_scaled_final = pd.DataFrame(X_scaled_np, index=X_raw.index, columns=actual_feature_cols).loc[valid_idx_in_X]
        y_ml_final = y_ml_series.loc[valid_idx_in_X]

        if X_scaled_final.empty or y_ml_final.empty:
            logger.error("X_scaled_final or y_ml_final is empty after NaN drop in training data. Aborting.")
            return None, None, None, None, None
        
        y_ml_mapped = y_ml_final.replace({-1: 0, 0: 1, 1: 2}).values
        
        df_info_columns = ['date', 'ticker'] + actual_feature_cols
        df_info = df_filtered.loc[X_scaled_final.index, [col for col in df_info_columns if col in df_filtered.columns]].copy()
        return X_scaled_final.values, y_ml_mapped, imputer, scaler, df_info
    else:
        if existing_imputer is None or existing_scaler is None:
            logger.error("Imputer and Scaler must be provided for prediction data preparation.")
            return None, None, None, None, None
        
        X_imputed_np = existing_imputer.transform(X_raw)
        X_scaled_np = existing_scaler.transform(X_imputed_np)
        X_scaled_final_df = pd.DataFrame(X_scaled_np, index=X_raw.index, columns=actual_feature_cols)
        
        df_info_columns = ['date', 'ticker'] + actual_feature_cols
        df_info = df_filtered.loc[X_scaled_final_df.index, [col for col in df_info_columns if col in df_filtered.columns]].copy()
        return X_scaled_final_df.values, None, None, None, df_info


def train_soft_signal_model(X_train, y_train_mapped):
    model_name = configs.SOFT_SIGNAL_MODEL_NAME
    logger.info(f"Training soft signal model: {model_name}...")

    if model_name == 'XGBoost':
        if not XGB_INSTALLED:
            logger.error("XGBoost is selected but not installed. Cannot train model.")
            raise ImportError("XGBoost library not found.")
        
        xgb_params = {
            'n_estimators': getattr(configs, 'N_ESTIMATORS_XGB_SOFT_SIGNAL', 200),
            'learning_rate': getattr(configs, 'LR_XGB_SOFT_SIGNAL', 0.05),
            'max_depth': getattr(configs, 'MAX_DEPTH_XGB_SOFT_SIGNAL', 5),
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': configs.RANDOM_SEED
        }
        model = xgboost_model.xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train_mapped)
        logger.info(f"{model_name} trained successfully.")
        return model
    else:
        logger.error(f"Soft signal model '{model_name}' is not currently supported in this script.")
        raise ValueError(f"Unsupported model for soft signal generation: {model_name}")

def main():
    logger.info("Starting: 04_generate_signals pipeline")

    # --- 1. Load Enriched Data ---
    try:
        df_enriched = pd.read_csv(configs.ENRICHED_DATA_FILE, parse_dates=['date'])
        if df_enriched.empty:
            logger.error(f"Loaded enriched data from {configs.ENRICHED_DATA_FILE} is empty. Aborting.")
            return
        logger.info(f"Loaded enriched data from: {configs.ENRICHED_DATA_FILE}. Shape: {df_enriched.shape}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: Enriched data file not found at {configs.ENRICHED_DATA_FILE}. "
                     f"Run '01_build_features.py' first. Aborting.")
        return
    except Exception as e:
        logger.error(f"CRITICAL: Error loading enriched data: {e}. Aborting.", exc_info=True)
        return

    # --- 2. Determine Feature Columns ---
    feature_cols_base = configs.BASE_FEATURE_COLS
    actual_base_features = [f for f in feature_cols_base if f in df_enriched.columns]
    if len(actual_base_features) < len(feature_cols_base):
        missing_base = set(feature_cols_base) - set(actual_base_features)
        logger.warning(f"Some BASE_FEATURE_COLS from config are missing in the loaded data: {missing_base}. They will be excluded.")
    
    feature_cols_with_lags = sorted(list(set(actual_base_features + [f'pct_change_lag_{p}' for p in configs.LAG_PERIODS])))
    logger.info(f"Using features for signal model: {feature_cols_with_lags}")

    # --- 3. Prepare Training Data for Soft Signal Model ---
    logger.info("Preparing training data for the soft signal model...")
    X_train, y_train_mapped, imputer, scaler, _ = prepare_data_for_signal_generation(
        df_full=df_enriched,
        feature_cols=feature_cols_with_lags,
        lag_periods=configs.LAG_PERIODS,
        target_col_pct=configs.TARGET_COL_PCT,
        target_col_trend=configs.TARGET_COL_TREND,
        trend_threshold=configs.TREND_THRESHOLD,
        start_date_filter=configs.TIME_RANGE_START,
        end_date_filter=configs.SOFT_SIGNAL_TRAIN_END_DATE,
        fit_imputer_scaler=True
    )
    if X_train is None or y_train_mapped is None:
        logger.error("Failed to prepare training data for soft signal model. Aborting.")
        return

    # --- 4. Train Soft Signal Model ---
    try:
        signal_model = train_soft_signal_model(X_train, y_train_mapped)
    except Exception as e:
        logger.error(f"Error training soft signal model: {e}. Aborting.", exc_info=True)
        return

    # --- 5. Prepare Prediction Data ---
    logger.info("Preparing prediction data for the soft signal model...")
    X_predict, _, _, _, df_predict_info = prepare_data_for_signal_generation(
        df_full=df_enriched,
        feature_cols=feature_cols_with_lags,
        lag_periods=configs.LAG_PERIODS,
        target_col_pct=configs.TARGET_COL_PCT,
        target_col_trend=configs.TARGET_COL_TREND,
        trend_threshold=configs.TREND_THRESHOLD,
        start_date_filter=configs.SOFT_SIGNAL_PREDICT_START_DATE,
        end_date_filter=configs.SOFT_SIGNAL_PREDICT_END_DATE,
        fit_imputer_scaler=False,
        existing_imputer=imputer,
        existing_scaler=scaler
    )
    if X_predict is None or df_predict_info is None or df_predict_info.empty:
        logger.error("Failed to prepare prediction data or no data in prediction range. Aborting.")
        return

    # --- 6. Predict Probabilities ---
    logger.info(f"Predicting probabilities for period {configs.SOFT_SIGNAL_PREDICT_START_DATE} to {configs.SOFT_SIGNAL_PREDICT_END_DATE}...")
    try:
        probabilities = signal_model.predict_proba(X_predict)
    except Exception as e:
        logger.error(f"Error predicting probabilities: {e}. Aborting.", exc_info=True)
        return

    # --- 7. Create Output DataFrame and Save ---
    if len(df_predict_info) != probabilities.shape[0]:
        logger.error(f"CRITICAL: Mismatch in rows between df_predict_info ({len(df_predict_info)}) "
                     f"and predicted probabilities ({probabilities.shape[0]}). This indicates an issue in "
                     f"data preparation alignment. Aborting save.")
        return

    output_df = df_predict_info[['date', 'ticker']].copy()
    
    model_name_suffix = configs.SOFT_SIGNAL_MODEL_NAME
    output_df[f"prob_decrease_{model_name_suffix}"] = probabilities[:, 0]
    output_df[f"prob_neutral_{model_name_suffix}"] = probabilities[:, 1]
    output_df[f"prob_increase_{model_name_suffix}"] = probabilities[:, 2]

    try:
        output_path = configs.CLASSIFICATION_PROBS_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully generated and saved soft signals to: {output_path}")
        if not output_df.empty:
             logger.info(f"Output DataFrame head:\n{output_df.head().to_string()}")
        else:
            logger.warning("Generated output_df for signals is empty.")
    except Exception as e:
        logger.error(f"Error saving soft signals: {e}", exc_info=True)

    logger.info("Finished: 04_generate_signals pipeline")

if __name__ == "__main__":
    main()

