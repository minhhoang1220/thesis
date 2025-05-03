import pandas as pd
import numpy as np

print("--- Loading ta_indicators.py ---")

# Relative Strength Index (RSI)
def compute_rsi(df, column='close', window=14):
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

print("--- Defined compute_rsi ---")

# Moving Average Convergence Divergence (MACD)
def compute_macd(df, column='close', fast=12, slow=26, signal=9):
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = macd - signal_line
    return df

# Bollinger Bands
def compute_bollinger_bands(df, column='close', window=20, num_std=2):
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    return df

# Simple Moving Average (SMA)
def compute_sma(df, column='close', window=20):
    df[f'SMA_{window}'] = df[column].rolling(window=window).mean()
    return df

# Exponential Moving Average (EMA)
def compute_ema(df, column='close', window=20):
    df[f'EMA_{window}'] = df[column].ewm(span=window, adjust=False).mean()
    return df

print("--- Finished loading ta_indicators.py ---")

# Apply all indicators
def add_ta_indicators(df):
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_sma(df, window=20)
    df = compute_ema(df, window=20)
    return df
