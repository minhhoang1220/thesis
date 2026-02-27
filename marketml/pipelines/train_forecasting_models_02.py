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
    raise

# ===== Logger and environment setup =====
logger = logger_setup.setup_basic_logging(log_file_name="train_forecasting_models.log")
logger_setup.suppress_common_warnings()
logger_setup.set_random_seeds(configs.RANDOM_SEED)

# ===== Check libraries status =====
TF_INSTALLED = keras_utils.KERAS_AVAILABLE
XGB_INSTALLED = True
try: import xgboost
except ImportError: XGB_INSTALLED = False
SKLEARN_INSTALLED = True
try: import sklearn
except ImportError: SKLEARN_INSTALLED = False
ARCH_INSTALLED = True
try: from arch import arch_model
except ImportError: ARCH_INSTALLED = False

def create_time_series_cv_splits(df, date_col, ticker_col, initial_train_period, test_period, step_period, expanding=False):
    logger.info(f"Generating CV splits ({'Expanding' if expanding else 'Rolling'} Window):")
    min_date, max_date = df[date_col].min(), df[date_col].max()
    if min_date + initial_train_period + test_period > max_date: return
    current_train_start_date = min_date
    current_train_end_date = min_date + initial_train_period
    split_index = 0
    while True:
        current_test_start_date = current_train_end_date
        current_test_end_date = current_test_start_date + test_period
        if current_test_end_date > max_date: break
        train_mask = (df[date_col] >= current_train_start_date) & (df[date_col] < current_train_end_date)
        test_mask = (df[date_col] >= current_test_start_date) & (df[date_col] < current_test_end_date)
        train_split_df, test_split_df = df.loc[train_mask].copy(), df.loc[test_mask].copy()
        if not train_split_df.empty and not test_split_df.empty:
            yield split_index, train_split_df, test_split_df
            split_index += 1
        if not expanding: current_train_start_date += step_period
        current_train_end_date += step_period

def prepare_split_data(train_df_orig, test_df_orig, feature_cols, lag_periods, target_col_pct, target_col_trend, trend_threshold, n_timesteps):
    train_df, test_df = train_df_orig.copy(), test_df_orig.copy()
    prepared_data = {'data_valid': False}
    try:
        for df_split in [train_df, test_df]:
            df_split[target_col_pct] = df_split.groupby('ticker')['close'].pct_change().shift(-1)
            conditions = [(df_split[target_col_pct] > trend_threshold), (df_split[target_col_pct] < -trend_threshold)]
            df_split[target_col_trend] = np.select(conditions, [1, -1], default=0)
            pct_series = df_split.groupby('ticker')['close'].pct_change()
            for p in lag_periods:
                df_split[f'pct_change_lag_{p}'] = pct_series.groupby(df_split['ticker']).shift(p)
        X_train_raw, X_test_raw = train_df[feature_cols].copy(), test_df[feature_cols].copy()
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), index=X_train_raw.index, columns=X_train_raw.columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), index=X_test_raw.index, columns=X_test_raw.columns)
        y_train, y_test = train_df[target_col_trend], test_df[target_col_trend]
        scaler = StandardScaler()
        prepared_data.update({
            'X_train_scaled': scaler.fit_transform(X_train_imp), 'y_train_ml': y_train.values,
            'X_test_scaled': scaler.transform(X_test_imp), 'y_test_ml': y_test.values,
            'n_classes': 3, 'data_valid': True,
            'train_df_with_target_for_arima': train_df, 'test_df_with_target_for_arima': test_df
        })
        if TF_INSTALLED:
            y_train_k = y_train.replace({-1: 0, 0: 1, 1: 2}).values
            X_train_seq, y_train_seq = keras_utils.create_sequences(prepared_data['X_train_scaled'], y_train_k, n_timesteps)
            X_test_seq, _ = keras_utils.create_sequences(prepared_data['X_test_scaled'], None, n_timesteps)
            prepared_data.update({'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq, 'X_test_seq': X_test_seq,
                                 'y_test_seq_original_trend': y_test.iloc[n_timesteps-1:].values[:len(X_test_seq)],
                                 'class_weight_dict': dict(zip(*np.unique(y_train_seq, return_counts=True))) if y_train_seq.size else {}})
    except Exception as e: logger.error(f"Error preparing split: {e}")
    return prepared_data

def main():
    logger.info("Starting Pipeline...")
    try: df_with_indicators = pd.read_csv(configs.ENRICHED_DATA_FILE, parse_dates=['date'])
    except Exception: return
    
    cv_splitter = create_time_series_cv_splits(df_with_indicators, 'date', 'ticker', configs.INITIAL_TRAIN_TIMEDELTA, configs.TEST_TIMEDELTA, configs.STEP_TIMEDELTA, configs.USE_EXPANDING_WINDOW)
    all_split_results, all_probs_to_save = {}, []
    
    # Params from config
    feature_cols = sorted(list(set(configs.BASE_FEATURE_COLS + [f'pct_change_lag_{p}' for p in configs.LAG_PERIODS])))
    lstm_params = {'lstm_units': configs.LSTM_UNITS, 'dropout_rate': configs.LSTM_DROPOUT_RATE, 'learning_rate': configs.LSTM_LEARNING_RATE, 'epochs': configs.KERAS_EPOCHS, 'batch_size': configs.KERAS_BATCH_SIZE}

    for split_idx, train_orig, test_orig in cv_splitter:
        logger.info(f"===== Processing Split {split_idx} =====")
        results_this_split = {}
        prep = prepare_split_data(train_orig, test_orig, feature_cols, configs.LAG_PERIODS, configs.TARGET_COL_PCT, configs.TARGET_COL_TREND, configs.TREND_THRESHOLD, configs.N_TIMESTEPS_SEQUENCE)
        
        if prep.get('data_valid'):
            # Run Models
            arima_res = arima_model.run_arima_evaluation(prep['train_df_with_target_for_arima'], prep['test_df_with_target_for_arima'], configs.TARGET_COL_TREND, configs.TREND_THRESHOLD, logger)
            rf_res = rf_model.run_rf_evaluation(prep['X_train_scaled'], prep['y_train_ml'], prep['X_test_scaled'], prep['y_test_ml'], logger=logger)
            xgb_res = xgboost_model.run_xgboost_evaluation(prep['X_train_scaled'], prep['y_train_ml'], prep['X_test_scaled'], prep['y_test_ml'], logger=logger)
            svm_res = svm_model.run_svm_evaluation(prep['X_train_scaled'], prep['y_train_ml'], prep['X_test_scaled'], prep['y_test_ml'], logger=logger)
            lstm_res = lstm_model.run_lstm_evaluation(prep['X_train_seq'], prep['y_train_seq'], prep['X_test_seq'], prep['y_test_seq_original_trend'], prep['class_weight_dict'], prep['n_classes'], configs.N_TIMESTEPS_SEQUENCE, prep['X_train_scaled'].shape[1], logger, **lstm_params)
            tf_res = transformer_model.run_transformer_evaluation(prep['X_train_seq'], prep['y_train_seq'], prep['X_test_seq'], prep['y_test_seq_original_trend'], prep['class_weight_dict'], prep['n_classes'], configs.N_TIMESTEPS_SEQUENCE, prep['X_train_scaled'].shape[1], logger)

            # Aggregate Probs
            split_df = prep['test_df_with_target_for_arima'][['date', 'ticker']].copy()
            def collect_probs(df_target, res_dict, model_name):
                prob_key = f"{model_name}_Probs"
                if prob_key in res_dict:
                    probs = res_dict[prob_key]
                    if model_name in ["LSTM", "Transformer"]:
                        n_skip = configs.N_TIMESTEPS_SEQUENCE - 1
                        sub = df_target.iloc[n_skip : n_skip + len(probs)].copy()
                        for i, col in enumerate(['decrease', 'neutral', 'increase']): sub[f'prob_{col}_{model_name}'] = probs[:, i]
                        return pd.merge(df_target, sub, on=['date', 'ticker'], how='left')
                    else:
                        for i, col in enumerate(['decrease', 'neutral', 'increase']): df_target[f'prob_{col}_{model_name}'] = probs[:, i]
                return df_target

            models = [("ARIMA", arima_res), ("RandomForest", rf_res), ("XGBoost", xgb_res), ("SVM", svm_res), ("LSTM", lstm_res), ("Transformer", tf_res)]
            for name, res in models:
                split_df = collect_probs(split_df, res, name)
                results_this_split.update({k: v for k, v in res.items() if "_Probs" not in k})
            all_probs_to_save.append(split_df)
        all_split_results[split_idx] = results_this_split

    if all_probs_to_save:
        pd.concat(all_probs_to_save).drop_duplicates(['date', 'ticker']).to_csv(configs.CLASSIFICATION_PROBS_FILE, index=False)
    
    final_df = pd.DataFrame.from_dict(all_split_results, orient='index')
    final_df.dropna(axis=1, how='all', inplace=True)
    final_df.to_csv(configs.MODEL_PERF_DETAILED_FILE)
    
    summary = final_df.agg(['mean', 'std']).T
    summary['Mean +/- Std'] = summary.apply(lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1)
    summary.transpose().to_csv(configs.MODEL_PERF_SUMMARY_FILE)
    logger.info("Finished.")

if __name__ == "__main__": main()
