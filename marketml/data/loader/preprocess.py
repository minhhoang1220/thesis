import pandas as pd
import numpy as np # Cần numpy cho np.select

try:
    from marketml.indicators import ta_indicators
except ModuleNotFoundError:
    print("CRITICAL ERROR in preprocess.py: Could not import 'marketml.indicators.ta_indicators'.")
    print("Ensure the project structure allows this import and ta_indicators.py exists.")
    ta_indicators = None
    raise

def standardize_data(df: pd.DataFrame, remove_columns: list[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_', regex=False)
                  .str.replace('-', '_', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('%', 'percent', regex=False)
                  .str.replace('/', '_', regex=False))
    standardized_df_columns = df.columns.tolist()
    if remove_columns:
        cols_to_remove_standardized = [
            col.strip().lower()
               .replace(' ', '_').replace('-', '_')
               .replace('(', '').replace(')', '')
               .replace('%', 'percent').replace('/', '_')
            for col in remove_columns
        ]
        cols_to_drop = [col for col in cols_to_remove_standardized if col in standardized_df_columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
    exclude_cols = {"symbol", "year", "ticker", "date"}
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

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
    price_zscore_window: int = 20
) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật được chọn vào DataFrame.

    Parameters:
        df (pd.DataFrame): Dữ liệu đã chuẩn hóa, yêu cầu có cột 'price_col'.
        price_col (str): Tên cột giá dùng để tính chỉ báo (mặc định là 'close').
        indicators_to_add (list): Danh sách các chỉ báo cần tính (viết thường).
                                  Nếu None, mặc định tính tất cả: ['rsi', 'macd', 'bollinger', 'sma', 'ema'].
        rsi_window (int): Window cho RSI.
        macd_fast (int): Fast period cho MACD.
        macd_slow (int): Slow period cho MACD.
        macd_signal (int): Signal period cho MACD.
        bb_window (int): Window cho Bollinger Bands.
        bb_num_std (int): Số độ lệch chuẩn cho Bollinger Bands.
        sma_window (int): Window cho SMA.
        ema_window (int): Window cho EMA.

    Returns:
        pd.DataFrame: DataFrame có thêm các cột chỉ báo.
    """
    if ta_indicators is None:
        print("Error: ta_indicators module not loaded. Cannot add technical indicators.")
        return df # Trả về df gốc nếu không load được module chỉ báo

    df_out = df.copy()

    if price_col not in df_out.columns:
        raise ValueError(f"DataFrame must contain the price column '{price_col}' for TA.")
    if not pd.api.types.is_numeric_dtype(df_out[price_col]):
        raise ValueError(f"Price column '{price_col}' must be numeric for TA.")

    if indicators_to_add is None:
        indicators_to_add = [
            'rsi', 'macd', 'bollinger', 'sma', 'ema', # Chỉ báo cũ
            'obv', 'rolling_close', 'rolling_rsi', 'price_zscore' # Chỉ báo mới
        ]
        print(f"  Adding indicators: {', '.join(indicators_to_add)}")

    if 'rsi' in indicators_to_add:
        print(f"    Computing RSI (window={rsi_window})...")
        df_out = ta_indicators.compute_rsi(df_out, column=price_col, window=rsi_window)
    if 'macd' in indicators_to_add:
        print(f"    Computing MACD (fast={macd_fast}, slow={macd_slow}, signal={macd_signal})...")
        df_out = ta_indicators.compute_macd(df_out, column=price_col, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if 'bollinger' in indicators_to_add:
        print(f"    Computing Bollinger Bands (window={bb_window}, std={bb_num_std})...")
        df_out = ta_indicators.compute_bollinger_bands(df_out, column=price_col, window=bb_window, num_std=bb_num_std)
    if 'sma' in indicators_to_add:
        print(f"    Computing SMA_{sma_window}...")
        df_out = ta_indicators.compute_sma(df_out, column=price_col, window=sma_window)
    if 'ema' in indicators_to_add:
        print(f"    Computing EMA_{ema_window}...")
        df_out = ta_indicators.compute_ema(df_out, column=price_col, window=ema_window)
    # On-Balance Volume
    if 'obv' in indicators_to_add:
        if volume_col not in df_out.columns:
            print(f"    Warning: Volume column '{volume_col}' not found for OBV. Skipping OBV.")
        elif not pd.api.types.is_numeric_dtype(df_out[volume_col]):
            print(f"    Warning: Volume column '{volume_col}' is not numeric. Skipping OBV.")
        else:
            print(f"    Computing OBV...")
            df_out = ta_indicators.compute_obv(df_out, close_col=price_col, volume_col=volume_col)

    # Rolling stats cho giá đóng cửa
    if 'rolling_close' in indicators_to_add:
        # Đặt giá trị mặc định cho rolling_stat_windows nếu nó là None
        current_rolling_windows = rolling_stat_windows if rolling_stat_windows is not None else [5, 10]
        print(f"    Computing Rolling Stats for '{price_col}' (windows={current_rolling_windows})...")
        df_out = ta_indicators.compute_rolling_stats(df_out, column=price_col, windows=current_rolling_windows, stats=['mean', 'std'])

    # Rolling stats cho RSI (chỉ tính nếu RSI đã được tính)
    if 'rolling_rsi' in indicators_to_add:
        if 'RSI' in df_out.columns:
            current_rolling_windows_rsi = rolling_stat_windows if rolling_stat_windows is not None else [5, 10]
            print(f"    Computing Rolling Stats for 'RSI' (windows={current_rolling_windows_rsi})...")
            df_out = ta_indicators.compute_rolling_stats(df_out, column='RSI', windows=current_rolling_windows_rsi, stats=['mean', 'std'])
        else:
            print("    Warning: Cannot compute rolling RSI because RSI was not calculated or added. Skipping.")

    # Z-score của giá
    if 'price_zscore' in indicators_to_add:
        print(f"    Computing Price Z-score (window={price_zscore_window})...")
        df_out = ta_indicators.compute_price_zscore(df_out, close_col=price_col, window=price_zscore_window)

    return df_out