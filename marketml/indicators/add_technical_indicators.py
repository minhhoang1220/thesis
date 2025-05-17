import pandas as pd
import numpy as np
import ta_indicators

def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume', # Thêm volume_col
    indicators_to_add: list[str] = None,
    rsi_window: int = 14,
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
    bb_window: int = 20, bb_num_std: int = 2,
    sma_window: int = 20, ema_window: int = 20,
    # Thêm tham số cho rolling stats
    rolling_stat_windows: list = None, # Ví dụ [5, 10]
    price_zscore_window: int = 20 # Cho Z-score
) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật và features mới vào DataFrame.
    """
    if ta_indicators is None: # ... (như cũ)
    df_out = df.copy()
    if price_col not in df_out.columns: # ... (như cũ)
    if not pd.api.types.is_numeric_dtype(df_out[price_col]): # ... (như cũ)

    if indicators_to_add is None:
        indicators_to_add = ['rsi', 'macd', 'bollinger', 'sma', 'ema', 'obv', 'rolling_close', 'rolling_rsi', 'price_zscore'] # Thêm chỉ báo mới

    # Các chỉ báo TA cơ bản
    if 'rsi' in indicators_to_add:
        df_out = ta_indicators.compute_rsi(df_out, column=price_col, window=rsi_window)
    if 'macd' in indicators_to_add:
        df_out = ta_indicators.compute_macd(df_out, column=price_col, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if 'bollinger' in indicators_to_add:
        df_out = ta_indicators.compute_bollinger_bands(df_out, column=price_col, window=bb_window, num_std=bb_num_std)
    if 'sma' in indicators_to_add:
        df_out = ta_indicators.compute_sma(df_out, column=price_col, window=sma_window)
    if 'ema' in indicators_to_add:
        df_out = ta_indicators.compute_ema(df_out, column=price_col, window=ema_window)

    # On-Balance Volume
    if 'obv' in indicators_to_add:
        if volume_col not in df_out.columns:
            print(f"Warning: Volume column '{volume_col}' not found for OBV calculation. Skipping OBV.")
        elif not pd.api.types.is_numeric_dtype(df_out[volume_col]):
            print(f"Warning: Volume column '{volume_col}' is not numeric. Skipping OBV.")
        else:
            df_out = ta_indicators.compute_obv(df_out, close_col=price_col, volume_col=volume_col)

    # Rolling stats cho giá đóng cửa
    if 'rolling_close' in indicators_to_add:
        if rolling_stat_windows is None: rolling_stat_windows = [5, 10] # Mặc định
        df_out = ta_indicators.compute_rolling_stats(df_out, column=price_col, windows=rolling_stat_windows, stats=['mean', 'std'])

    # Rolling stats cho RSI (chỉ tính nếu RSI đã được tính)
    if 'rolling_rsi' in indicators_to_add and 'RSI' in df_out.columns:
        if rolling_stat_windows is None: rolling_stat_windows = [5, 10] # Mặc định
        df_out = ta_indicators.compute_rolling_stats(df_out, column='RSI', windows=rolling_stat_windows, stats=['mean', 'std'])
    elif 'rolling_rsi' in indicators_to_add and 'RSI' not in df_out.columns:
        print("Warning: Cannot compute rolling RSI because RSI was not calculated. Skipping.")

    # Z-score của giá
    if 'price_zscore' in indicators_to_add:
        df_out = ta_indicators.compute_price_zscore(df_out, close_col=price_col, window=price_zscore_window)

    return df_out