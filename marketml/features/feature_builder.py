# marketml/features/feature_builder.py
import pandas as pd
import numpy as np
from marketml.configs import configs
from marketml.features import ta_indicators
from marketml.data_handling import preprocess
from marketml.features import ta_indicators 

# === Import Arch for GARCH ===
try:
    from arch import arch_model
    ARCH_INSTALLED = True
except ImportError:
    ARCH_INSTALLED = False
    arch_model = None
# ==============================

def calculate_garch_volatility_feature(series_returns: pd.Series, window: int, horizon: int):
    """
    Calculate 1-step ahead forecasted volatility from GARCH(1,1) using a rolling window.
    Args:
        series_returns (pd.Series): Series of daily returns (already multiplied by 100).
        window (int): Window size for fitting GARCH.
        horizon (int): Forecast horizon (usually 1).
    Returns:
        pd.Series: Series containing forecasted volatility values (square root of variance).
                   Index matches series_returns, with NaN at the beginning.
    """
    if not ARCH_INSTALLED or len(series_returns) < window + 10:
        return pd.Series(np.nan, index=series_returns.index)

    volatility_forecasts = pd.Series(np.nan, index=series_returns.index)

    # Use rolling window to fit GARCH and forecast
    # Note: This can be very slow
    for i in range(window, len(series_returns)):
        current_window_returns = series_returns.iloc[i-window:i]
        if len(current_window_returns.dropna()) < window * 0.8:
            continue
        try:
            gm = arch_model(current_window_returns.dropna(), vol='Garch', p=1, q=1, rescale=False)
            res = gm.fit(disp='off', show_summary=False)
            forecast = res.forecast(horizon=horizon, reindex=False)
            predicted_variance = forecast.variance.iloc[-1, 0]
            if i < len(volatility_forecasts):
                volatility_forecasts.iloc[i] = np.sqrt(predicted_variance) / 100
        except Exception as e:
            pass
    return volatility_forecasts

def build_enriched_features(raw_real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes raw price data, adds technical indicators, and GARCH volatility.
    Returns the enriched DataFrame.
    """
    print("\nStandardizing data...")
    standardized_df = preprocess.standardize_data(raw_real_df)
    print("Data standardized.")

    all_enriched_df_list = []
    processed_tickers_count = 0

    for ticker, group_df in standardized_df.groupby('ticker'):
        print(f"  Processing ticker: {ticker} ({len(group_df)} rows)")
        group_df_sorted = group_df.sort_values('date').copy()
        if ARCH_INSTALLED:
            daily_returns = group_df_sorted['close'].pct_change().dropna() * 100
            garch_vol = calculate_garch_volatility_feature(daily_returns, configs.GARCH_WINDOW, configs.GARCH_FORECAST_HORIZON)
            group_df_sorted['garch_vol_forecast'] = garch_vol.shift(1)
        else:
            group_df_sorted['garch_vol_forecast'] = np.nan

        group_with_ta = preprocess.add_technical_indicators(
            group_df_sorted,
            price_col='close',
            volume_col='volume',
            rsi_window=configs.RSI_WINDOW,
            macd_fast=configs.MACD_FAST, macd_slow=configs.MACD_SLOW, macd_signal=configs.MACD_SIGNAL,
            bb_window=configs.BB_WINDOW,
            sma_window=configs.SMA_WINDOW, ema_window=configs.EMA_WINDOW,
            rolling_stat_windows=configs.ROLLING_STAT_WINDOWS,
            price_zscore_window=configs.PRICE_ZSCORE_WINDOW
        )
        all_enriched_df_list.append(group_with_ta)
        processed_tickers_count += 1
    if not all_enriched_df_list:
        print("\nWarning: No data to enrich.")
        return pd.DataFrame()

    df_final_enriched = pd.concat(all_enriched_df_list, ignore_index=True)
    print(f"\nAll features (TA & GARCH) added successfully for {processed_tickers_count} tickers.")
    return df_final_enriched
