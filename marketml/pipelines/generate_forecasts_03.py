# /.ndmh/marketml/pipelines/03_generate_forecasts.py

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import warnings
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

# ===== Import modules from marketml and configs =====
try:
    from marketml.configs import configs
    from marketml.utils import logger_setup
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR in 03_generate_forecasts.py: Could not import necessary marketml modules. {e}")
    print("Ensure the marketml package is installed correctly or PYTHONPATH is set.")
    raise

# ===== Setup Logger and environment =====
logger = logger_setup.setup_basic_logging(log_file_name="generate_forecasts.log")
logger_setup.suppress_common_warnings()


def generate_forecasts_for_year(forecast_year: int, training_years: int):
    """
    Generates future price trend forecasts for a specific year using ARIMA.
    """
    logger.info(f"--- Generating Forecast for Year {forecast_year} using last {training_years} years of data ---")

    # --- Step 1: Load Enriched Data ---
    enriched_data_file = configs.ENRICHED_DATA_FOR_FORECAST
    logger.info(f"Attempting to load enriched data from: {enriched_data_file.resolve()}")
    try:
        df_hist = pd.read_csv(enriched_data_file, parse_dates=['date'])
        if df_hist.empty:
            logger.error("Loaded enriched data is empty. Cannot proceed.")
            return None
        logger.info(f"Enriched data loaded successfully. Shape: {df_hist.shape}")
    except FileNotFoundError:
        logger.error(f"Enriched data file not found at '{enriched_data_file.resolve()}'.")
        logger.error(f"Ensure '{configs.ENRICHED_DATA_FILE.name}' was created by '01_build_features.py' pipeline.")
        return None
    except Exception as e:
        logger.error(f"Error loading enriched data: {e}", exc_info=True)
        return None

    # --- Step 2: Determine Final Training Period ---
    if 'date' not in df_hist.columns or df_hist['date'].isnull().all():
        logger.error("'date' column is missing or all null in historical data.")
        return None
    
    last_hist_date = df_hist['date'].max()
    if pd.isna(last_hist_date):
        logger.error("Could not determine the last historical date from the data.")
        return None

    training_end_date = last_hist_date
    training_start_date = training_end_date - pd.Timedelta(days=365 * training_years)
    
    logger.info(f"Final training period defined: {training_start_date.date()} to {training_end_date.date()}")
    final_train_df = df_hist[(df_hist['date'] >= training_start_date) & (df_hist['date'] <= training_end_date)].copy()

    if final_train_df.empty:
        logger.error("No data found in the defined final training period. Check data range or training_years setting.")
        return None

    # --- Step 3: Estimate Forecast Periods and Future Dates ---
    forecast_start_date = last_hist_date + pd.Timedelta(days=1)
    forecast_end_date = pd.Timestamp(f"{forecast_year}-12-31")

    if forecast_start_date > forecast_end_date:
        logger.error(f"Forecast start date ({forecast_start_date.date()}) is after forecast end date ({forecast_end_date.date()}). Cannot generate forecasts.")
        return None

    future_dates = pd.bdate_range(start=forecast_start_date, end=forecast_end_date)
    n_periods_forecast = len(future_dates)

    if n_periods_forecast <= 0:
        logger.error(f"No business days found for forecasting period ({forecast_start_date.date()} to {forecast_end_date.date()}).")
        return None
    logger.info(f"Forecasting for {n_periods_forecast} business days from {future_dates[0].date()} to {future_dates[-1].date()}")

    # --- Step 4: Train ARIMA and Forecast for Each Ticker ---
    all_forecasts_list = []
    unique_tickers = final_train_df['ticker'].unique()
    logger.info(f"Found {len(unique_tickers)} unique tickers in the training data.")

    for ticker in unique_tickers:
        logger.info(f"  Processing forecast for ticker: {ticker}")
        ticker_train_data = final_train_df[final_train_df['ticker'] == ticker].copy()
        
        if 'close' not in ticker_train_data.columns or not pd.api.types.is_numeric_dtype(ticker_train_data['close']):
            logger.warning(f"    Skipping {ticker}: 'close' column missing or not numeric.")
            continue

        y_train_pct_change = ticker_train_data['close'].pct_change().dropna()

        if y_train_pct_change.empty or len(y_train_pct_change) < 10:
            logger.warning(f"    Skipping {ticker}: Not enough historical percentage change data (found {len(y_train_pct_change)}).")
            continue
        
        try:
            adf_result = adfuller(y_train_pct_change)
            p_value = adf_result[1]
            d_order = 1 if p_value > 0.05 else 0
            logger.debug(f"    ADF P-value for {ticker}: {p_value:.4f}, selected d_order: {d_order}")

            final_arima_model = pm.auto_arima(
                y_train_pct_change,
                d=d_order,
                start_p=1, start_q=1, max_p=3, max_q=3,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            logger.debug(f"    AutoARIMA model for {ticker}: {final_arima_model.summary().tables[0].as_text()}")

            forecast_pct_change_values = final_arima_model.predict(n_periods=n_periods_forecast)
            
            forecast_trend_values = np.select(
                [(forecast_pct_change_values > configs.FORECAST_TREND_THRESHOLD),
                 (forecast_pct_change_values < -configs.FORECAST_TREND_THRESHOLD)],
                [1, -1], default=0
            )

            ticker_forecast_df = pd.DataFrame({
                'forecast_date': future_dates,
                'ticker': ticker,
                'forecast_pct_change': forecast_pct_change_values,
                'forecast_trend': forecast_trend_values
            })
            all_forecasts_list.append(ticker_forecast_df)
            logger.info(f"    Successfully generated forecast for {ticker}.")

        except Exception as e:
            logger.error(f"    Error processing ARIMA forecast for {ticker}: {e}", exc_info=False)

    # --- Step 5: Combine and Save Forecast Results ---
    if not all_forecasts_list:
        logger.warning("No forecasts were generated for any ticker.")
        return None

    final_forecast_df = pd.concat(all_forecasts_list, ignore_index=True)
    logger.info(f"Forecasts generated successfully for {len(final_forecast_df['ticker'].unique())} tickers.")
    logger.debug(f"Sample of final forecast (last 5 rows):\n{final_forecast_df.tail().to_string()}")

    try:
        forecast_output_dir = configs.FORECASTS_OUTPUT_DIR
        forecast_output_dir.mkdir(parents=True, exist_ok=True)
        forecast_file_path = forecast_output_dir / f"arima_forecast_{forecast_year}.csv"
        final_forecast_df.to_csv(forecast_file_path, index=False)
        logger.info(f"Forecasts saved to: {forecast_file_path.resolve()}")
    except Exception as e:
        logger.error(f"Error saving forecast file: {e}", exc_info=True)

    return final_forecast_df

def main():
    """
    Main function to orchestrate the forecast generation process.
    """
    logger.info("Starting: 03_generate_forecasts pipeline")
    
    forecast_target_year = configs.FORECAST_YEAR_TARGET
    forecast_training_yrs = configs.FORECAST_TRAINING_YEARS

    forecast_df = generate_forecasts_for_year(
        forecast_year=forecast_target_year,
        training_years=forecast_training_yrs
    )

    if forecast_df is not None and not forecast_df.empty:
        logger.info("Forecast generation script finished successfully.")
    else:
        logger.warning("Forecast generation script completed, but no forecast data was produced or saved.")
    
    logger.info("Finished: 03_generate_forecasts pipeline")

if __name__ == "__main__":
    main()
