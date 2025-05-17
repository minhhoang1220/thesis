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

# On-Balance Volume (OBV)
def compute_obv(df: pd.DataFrame, close_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """Tính toán On-Balance Volume (OBV)."""
    # OBV được tính cho từng ticker, nên hàm này sẽ được áp dụng sau groupby
    # Tuy nhiên, logic tính toán cơ bản là trên một chuỗi
    diff = df[close_col].diff()
    direction = np.sign(diff).fillna(0) # 1 nếu tăng, -1 nếu giảm, 0 nếu không đổi hoặc NaN đầu tiên
    # Đặt hướng của ngày đầu tiên dựa trên ngày thứ hai (nếu có) hoặc là 0
    if len(direction) > 1 and direction.iloc[0] == 0 and direction.iloc[1] != 0:
        direction.iloc[0] = direction.iloc[1] # Gán hướng của ngày đầu bằng ngày sau nếu ngày đầu là 0
    elif len(direction) == 1: # Trường hợp chỉ có 1 dòng
        direction.iloc[0] = 0

    obv_values = (direction * df[volume_col]).cumsum()
    df['OBV'] = obv_values
    return df

# Rolling Statistics
def compute_rolling_stats(df: pd.DataFrame, column: str, windows: list, stats: list = None) -> pd.DataFrame:
    """
    Tính toán các thống kê trượt (rolling statistics) cho một cột.
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Tên cột để tính thống kê.
        windows (list): Danh sách các kích thước cửa sổ (ví dụ: [5, 10]).
        stats (list): Danh sách các thống kê cần tính (ví dụ: ['mean', 'std']).
                      Nếu None, mặc định là ['mean', 'std'].
    Returns:
        pd.DataFrame: DataFrame với các cột thống kê trượt đã được thêm vào.
    """
    if stats is None:
        stats = ['mean', 'std']

    df_out = df.copy()
    # Đảm bảo cột tồn tại và là số
    if column not in df_out.columns or not pd.api.types.is_numeric_dtype(df_out[column]):
        print(f"Warning: Column '{column}' not found or not numeric for rolling stats. Skipping.")
        return df_out # Trả về df gốc

    for window in windows:
        if len(df_out) < window: # Bỏ qua nếu không đủ dữ liệu cho cửa sổ
            print(f"Warning: Not enough data for window {window} on column {column}. Skipping.")
            continue
        for stat_func_name in stats:
            new_col_name = f'{column}_roll_{stat_func_name}_{window}'
            try:
                # Sử dụng min_periods=window để chỉ tính khi đủ dữ liệu
                # shift(1) để tránh data leakage (dùng giá trị của t-1 để dự đoán t)
                if stat_func_name == 'mean':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).mean().shift(1)
                elif stat_func_name == 'std':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).std().shift(1)
                elif stat_func_name == 'min':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).min().shift(1)
                elif stat_func_name == 'max':
                    df_out[new_col_name] = df_out[column].rolling(window=window, min_periods=window).max().shift(1)
                # Có thể thêm các stat khác nếu cần
            except Exception as e:
                print(f"Error calculating rolling {stat_func_name} for {column} with window {window}: {e}")
                df_out[new_col_name] = np.nan # Gán NaN nếu có lỗi
    return df_out

# Z-score của giá so với rolling mean
def compute_price_zscore(df: pd.DataFrame, close_col: str = 'close', window: int = 20) -> pd.DataFrame:
    """Tính Z-score của giá đóng cửa so với rolling mean."""
    if len(df) < window:
        df[f'{close_col}_zscore_{window}'] = np.nan
        return df

    rolling_mean = df[close_col].rolling(window=window, min_periods=window).mean()
    rolling_std = df[close_col].rolling(window=window, min_periods=window).std()

    # shift(1) để dùng rolling mean/std của ngày hôm trước
    z_score = (df[close_col] - rolling_mean.shift(1)) / (rolling_std.shift(1) + 1e-9) # Thêm epsilon tránh chia 0
    df[f'{close_col}_zscore_{window}'] = z_score
    return df