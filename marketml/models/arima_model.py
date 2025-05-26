# File: marketml/models/arima_model.py
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
import logging

try:
    from marketml.utils import metrics
except ModuleNotFoundError:
    print("CRITICAL ERROR in arima_model.py: Could not import 'marketml.utils.metrics'. Ensure PYTHONPATH is set or run as module.")
    raise

def run_arima_evaluation(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                         target_col_trend: str, trend_threshold: float, 
                         logger: logging.Logger):
    """
    Train, predict, and evaluate the ARIMA model for all tickers in the split.
    """
    logger.info("--- Training and Evaluating ARIMA Model ---")
    results_this_split = {}
    arima_predictions_all = []
    arima_actual_trends_all = []

    default_metrics = {
        "ARIMA_Accuracy": np.nan, "ARIMA_F1_Macro": np.nan, "ARIMA_F1_Weighted": np.nan,
        "ARIMA_Precision_Macro": np.nan, "ARIMA_Recall_Macro": np.nan
    }
    results_this_split.update(default_metrics)

    for ticker in train_df['ticker'].unique():
        logger.debug(f"  Processing ARIMA for ticker: {ticker}")
        train_ticker_df = train_df[train_df['ticker'] == ticker]
        test_ticker_df = test_df[test_df['ticker'] == ticker]

        y_train_arima_input = train_ticker_df['close'].pct_change().dropna()
        y_test_arima_input = test_ticker_df['close'].pct_change().dropna()
        
        # Align y_test_trend_ticker with y_test_arima_input (which has NaNs dropped)
        potential_trends_for_ticker = test_df.loc[test_df['ticker'] == ticker, target_col_trend]
        y_test_trend_ticker = potential_trends_for_ticker.reindex(y_test_arima_input.index).dropna()

        if y_train_arima_input.empty or y_test_arima_input.empty or y_test_trend_ticker.empty:
            logger.warning(f"    Skipping ARIMA for {ticker}: Insufficient data after processing "
                           f"(train_len={len(y_train_arima_input)}, test_len={len(y_test_arima_input)}, trend_len={len(y_test_trend_ticker)}).")
            continue

        try:
            adf_result = adfuller(y_train_arima_input)
            p_value = adf_result[1]
            d_order = 1 if p_value > 0.05 else 0
            logger.debug(f"    ADF p-value for {ticker}: {p_value:.4f}, d_order: {d_order}")

            auto_model = pm.auto_arima(
                y_train_arima_input, d=d_order,
                start_p=1, start_q=1, max_p=3, max_q=3,
                seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore', trace=False
            )
            
            n_periods = len(y_test_arima_input)
            if n_periods == 0:
                logger.warning(f"    Skipping ARIMA prediction for {ticker}: n_periods is 0.")
                continue
                
            arima_preds_pct = auto_model.predict(n_periods=n_periods)

            # Ensure alignment between predictions and actuals if lengths differ
            if len(arima_preds_pct) != len(y_test_trend_ticker):
                logger.warning(f"    Length mismatch for {ticker}: arima_preds_pct ({len(arima_preds_pct)}) "
                               f"vs y_test_trend_ticker ({len(y_test_trend_ticker)}). Attempting alignment.")
                min_len = min(len(arima_preds_pct), len(y_test_trend_ticker))
                arima_preds_pct = arima_preds_pct[:min_len]
                y_test_trend_ticker_aligned = y_test_trend_ticker.iloc[:min_len]
            else:
                y_test_trend_ticker_aligned = y_test_trend_ticker

            arima_predictions_all.extend(arima_preds_pct)
            arima_actual_trends_all.extend(y_test_trend_ticker_aligned.values)

        except Exception as e:
            logger.error(f"    Error processing ARIMA for {ticker}: {e}", exc_info=True)

    if arima_predictions_all and arima_actual_trends_all:
        arima_preds_trend_np = np.select(
            [(np.array(arima_predictions_all) > trend_threshold),
             (np.array(arima_predictions_all) < -trend_threshold)],
            [1, -1], default=0
        )
        try:
            arima_metrics_results = metrics.calculate_classification_metrics(
                np.array(arima_actual_trends_all), arima_preds_trend_np, model_name="ARIMA", logger=logger
            )
            results_this_split.update(arima_metrics_results)
        except Exception as e_metrics:
            logger.error(f"    Error calculating ARIMA metrics: {e_metrics}", exc_info=True)
    else:
        logger.info("    ARIMA: No valid predictions were generated for this split to calculate aggregate metrics.")

    return results_this_split
