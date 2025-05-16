import pandas as pd
import numpy as np

# Relative Strength Index (RSI)
def compute_rsi(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.DataFrame:
    """
    Tính toán Relative Strength Index (RSI) sử dụng Exponential Moving Average (EMA)
    cho average gain và average loss (theo Wilder's smoothing).
    """
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)  # Tương đương delta.clip(lower=0)
    loss = -delta.where(delta < 0, 0.0) # Tương đương -delta.clip(upper=0)

    # Sử dụng EMA (com = window - 1 để tương đương span=window cho EMA chuẩn)
    # adjust=False để khớp với nhiều nền tảng charting hơn.
    # min_periods=window để đảm bảo có đủ dữ liệu ban đầu cho EMA.
    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9) #Make sure not to divide by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df['RSI'] = rsi
    return df


# Moving Average Convergence Divergence (MACD)
def compute_macd(df: pd.DataFrame, column: str = 'close', fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Tính toán Moving Average Convergence Divergence (MACD)."""
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
    """Tính toán Bollinger Bands."""
    rolling_mean = df[column].rolling(window=window, min_periods=window).mean()
    rolling_std = df[column].rolling(window=window, min_periods=window).std()

    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    return df

# Simple Moving Average (SMA)
def compute_sma(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Tính toán Simple Moving Average (SMA)."""
    df[f'SMA_{window}'] = df[column].rolling(window=window, min_periods=window).mean()
    return df

# Exponential Moving Average (EMA)
def compute_ema(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Tính toán Exponential Moving Average (EMA)."""
    df[f'EMA_{window}'] = df[column].ewm(span=window, adjust=False, min_periods=window).mean()
    return df
