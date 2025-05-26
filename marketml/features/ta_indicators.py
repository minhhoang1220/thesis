import pandas as pd
import numpy as np

# Relative Strength Index (RSI)
def compute_rsi(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI) using Exponential Moving Average (EMA)
    for average gain and average loss (Wilder's smoothing).
    """
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use EMA (com = window - 1 to match span=window for standard EMA)
    # adjust=False for compatibility with most charting platforms.
    # min_periods=window to ensure enough initial data for EMA.
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9) # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df['RSI'] = rsi
    return df

# Moving Average Convergence Divergence (MACD)
def compute_macd(df: pd.DataFrame, column: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = df[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - signal_line

    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = macd_hist
    return df

# Bollinger Bands
def compute_bollinger_bands(df: pd.DataFrame, column: str = 'close', window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    rolling_mean = df[column].rolling(window=window, min_periods=window).mean()
    rolling_std = df[column].rolling(window=window, min_periods=window).std()

    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    return df

# Simple Moving Average (SMA)
def compute_sma(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Calculate Simple Moving Average (SMA)."""
    df[f'SMA_{window}'] = df[column].rolling(window=window, min_periods=window).mean()
    return df

# Exponential Moving Average (EMA)
def compute_ema(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Calculate Exponential Moving Average (EMA)."""
    df[f'EMA_{window}'] = df[column].ewm(span=window, adjust=False, min_periods=window).mean()
    return df

# On-Balance Volume (OBV)
def compute_obv(df: pd.DataFrame, close_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """Calculate On-Balance Volume (OBV)."""
    diff = df[close_col].diff()
    direction = np.sign(diff).fillna(0) # 1 if up, -1 if down, 0 if unchanged or first NaN
    if len(direction) > 1 and direction.iloc[0] == 0 and direction.iloc[1] != 0:
        direction.iloc[0] = direction.iloc[1]
    elif len(direction) == 1:
        direction.iloc[0] = 0

    obv_values = (direction * df[volume_col]).cumsum()
    df['OBV'] = obv_values
    return df

# Rolling Statistics
def compute_rolling_stats(df: pd.DataFrame, column: str, windows: list, stats: list = None) -> pd.DataFrame:
    """
    Calculate rolling statistics for a column.
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to calculate statistics.
        windows (list): List of window sizes (e.g., [5, 10]).
        stats (list): List of statistics to calculate (e.g., ['mean', 'std']).
                      If None, defaults to ['mean', 'std'].
    Returns:
        pd.DataFrame: DataFrame with added rolling statistics columns.
    """
    if stats is None:
        stats = ['mean', 'std']

    df_out = df.copy()
    if column not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[column]):
        print(f"Warning: Column '{column}' not found or not numeric for rolling stats. Skipping.")
        return df_out

    for window in windows:
        if len(df_out) < window:
            print(f"Warning: Not enough data for window {window} on column {column}. Skipping.")
            continue
        for stat_func_name in stats:
            new_col_name = f'{column}_roll_{stat_func_name}_{window}'
            try:
                if stat_func_name == 'mean':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).mean().shift(1)
                elif stat_func_name == 'std':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).std().shift(1)
                elif stat_func_name == 'min':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).min().shift(1)
                elif stat_func_name == 'max':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).max().shift(1)
            except Exception as e:
                print(f"Error calculating rolling {stat_func_name} for {column} with window {window}: {e}")
                df_out[new_col_name] = np.nan
    return df_out

# Z-score of price compared to rolling mean
def compute_price_zscore(df: pd.DataFrame, close_col: str = 'close', window: int = 20) -> pd.DataFrame:
    """Calculate Z-score of closing price compared to rolling mean."""
    if len(df) < window:
        df[f'{close_col}_zscore_{window}'] = np.nan
        return df

    rolling_mean = df[close_col].rolling(window=window, min_periods=window).mean()
    rolling_std = df[close_col].rolling(window=window, min_periods=window).std()

    z_score = (df[close_col] - rolling_mean.shift(1)) / (rolling_std.shift(1) + 1e-9)
    df[f'{close_col}_zscore_{window}'] = z_score
    return df
