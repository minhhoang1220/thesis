# /.ndmh/marketml/data_handling/preprocess.py
import pandas as pd
import numpy as np
import logging

try:
    from marketml.features import ta_indicators
except ModuleNotFoundError:
    # Log the initial error to stdout if logger is not yet set up
    print("CRITICAL ERROR in preprocess.py: Could not import 'marketml.features.ta_indicators'.")
    raise 

logger = logging.getLogger(__name__)

def standardize_data(df: pd.DataFrame, remove_columns: list[str] = None) -> pd.DataFrame:
    """
    Standardizes column names and attempts to convert relevant columns to numeric.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input to standardize_data is not a Pandas DataFrame.")
        return pd.DataFrame()
    if df.empty:
        logger.warning("Input DataFrame to standardize_data is empty.")
        return df.copy()

    df_std = df.copy()
    try:
        df_std.columns = (df_std.columns.str.strip().str.lower()
                      .str.replace(' ', '_', regex=False)
                      .str.replace('-', '_', regex=False)
                      .str.replace('(', '', regex=False)
                      .str.replace(')', '', regex=False)
                      .str.replace('%', 'percent', regex=False)
                      .str.replace('/', '_', regex=False))
    except AttributeError:
        logger.warning("Could not standardize all column names (possibly non-string column names).")

    if remove_columns:
        cols_to_remove_standardized = [
            str(col).strip().lower()
               .replace(' ', '_').replace('-', '_')
               .replace('(', '').replace(')', '')
               .replace('%', 'percent').replace('/', '_')
            for col in remove_columns
        ]
        cols_to_drop_present = [col for col in cols_to_remove_standardized if col in df_std.columns]
        if cols_to_drop_present:
            df_std.drop(columns=cols_to_drop_present, inplace=True)
            logger.debug(f"Removed columns: {cols_to_drop_present}")

    # Columns to exclude from numeric conversion (case-insensitive after standardization)
    exclude_cols_from_numeric = {"symbol", "year", "ticker", "date", "market"}

    for col in df_std.columns:
        if col not in exclude_cols_from_numeric:
            if not pd.api.types.is_numeric_dtype(df_std[col]):
                df_std[col] = pd.to_numeric(df_std[col], errors="coerce")
                if df_std[col].isnull().any():
                    logger.debug(f"NaNs introduced in column '{col}' after to_numeric conversion.")
    return df_std

def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
    indicators_to_add: list[str] = None,
    rsi_window: int = 14,
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
    bb_window: int = 20, bb_num_std: int = 2,
    sma_window: int = 20, ema_window: int = 20,
    rolling_stat_windows: list = None, 
    price_zscore_window: int = 20,
    logger_instance: logging.Logger = None
) -> pd.DataFrame:
    current_logger = logger_instance if logger_instance else logger

    if ta_indicators is None:
        current_logger.error("ta_indicators module not loaded. Cannot add technical indicators.")
        return df

    if df.empty:
        current_logger.warning("Input DataFrame for add_technical_indicators is empty.")
        return df.copy()

    df_out = df.copy()

    if price_col not in df_out.columns:
        current_logger.error(f"Price column '{price_col}' not found in DataFrame for TA calculation.")
        raise ValueError(f"Price column '{price_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df_out[price_col]):
        current_logger.error(f"Price column '{price_col}' must be numeric for TA calculation.")
        raise TypeError(f"Price column '{price_col}' must be numeric.")

    # Default list of indicators if none provided
    if indicators_to_add is None:
        indicators_to_add = [
            'rsi', 'macd', 'bollinger', 'sma', 'ema',
            'obv', 'rolling_close', 'rolling_rsi', 'price_zscore'
        ]
    current_logger.info(f"Adding technical indicators: {', '.join(indicators_to_add)} for price_col='{price_col}'")

    if 'rsi' in indicators_to_add:
        current_logger.debug(f"  Computing RSI (window={rsi_window})...")
        df_out = ta_indicators.compute_rsi(df_out, column=price_col, window=rsi_window)
    if 'macd' in indicators_to_add:
        current_logger.debug(f"  Computing MACD (fast={macd_fast}, slow={macd_slow}, signal={macd_signal})...")
        df_out = ta_indicators.compute_macd(df_out, column=price_col, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if 'bollinger' in indicators_to_add:
        current_logger.debug(f"  Computing Bollinger Bands (window={bb_window}, std={bb_num_std})...")
        df_out = ta_indicators.compute_bollinger_bands(df_out, column=price_col, window=bb_window, num_std=bb_num_std)
    if 'sma' in indicators_to_add:
        current_logger.debug(f"  Computing SMA_{sma_window}...")
        df_out = ta_indicators.compute_sma(df_out, column=price_col, window=sma_window)
    if 'ema' in indicators_to_add:
        current_logger.debug(f"  Computing EMA_{ema_window}...")
        df_out = ta_indicators.compute_ema(df_out, column=price_col, window=ema_window)

    if 'obv' in indicators_to_add:
        if volume_col not in df_out.columns:
            current_logger.warning(f"  Volume column '{volume_col}' not found. Skipping OBV.")
        elif not pd.api.types.is_numeric_dtype(df_out[volume_col]):
            current_logger.warning(f"  Volume column '{volume_col}' is not numeric. Skipping OBV.")
        else:
            current_logger.debug(f"  Computing OBV (volume_col='{volume_col}')...")
            df_out = ta_indicators.compute_obv(df_out, close_col=price_col, volume_col=volume_col)

    current_rolling_windows = rolling_stat_windows if rolling_stat_windows is not None else [5, 10]
    if 'rolling_close' in indicators_to_add:
        current_logger.debug(f"  Computing Rolling Stats for '{price_col}' (windows={current_rolling_windows})...")
        df_out = ta_indicators.compute_rolling_stats(df_out, column=price_col, windows=current_rolling_windows, stats=['mean', 'std'])
    if 'rolling_rsi' in indicators_to_add:
        if 'RSI' in df_out.columns:
            current_logger.debug(f"  Computing Rolling Stats for 'RSI' (windows={current_rolling_windows})...")
            df_out = ta_indicators.compute_rolling_stats(df_out, column='RSI', windows=current_rolling_windows, stats=['mean', 'std'])
        else:
            current_logger.warning("  Cannot compute rolling RSI because RSI column was not found (possibly not calculated or added).")

    if 'price_zscore' in indicators_to_add:
        current_logger.debug(f"  Computing Price Z-score (window={price_zscore_window})...")
        df_out = ta_indicators.compute_price_zscore(df_out, close_col=price_col, window=price_zscore_window)

    return df_out
