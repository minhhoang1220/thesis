# marketml/portfolio_opt/data_preparation.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from pypfopt import risk_models, expected_returns

try:
    from marketml.configs import configs
    from marketml.data_handling import preprocess
    from marketml.utils import logger_setup
except ImportError:
    print("CRITICAL ERROR in data_preparation.py: Could not import 'marketml.configs', 'marketml.data_handling.preprocess', or 'marketml.logger_setup'.")
    raise

logger = logging.getLogger(__name__)

def load_price_data_for_portfolio(
    custom_start_date_str: str = None,
    custom_end_date_str: str = None,
    include_buffer_for_rolling: bool = True,
    logger_instance: logging.Logger = None
) -> pd.DataFrame:
    # --- Load price data for portfolio ---
    current_logger = logger_instance if logger_instance else logger
    source_file = configs.PRICE_DATA_FOR_PORTFOLIO_PATH

    log_message_prefix = "Loading price data"
    if custom_start_date_str and custom_end_date_str:
        log_message_prefix = f"Loading custom price data ({custom_start_date_str} - {custom_end_date_str})"
    current_logger.info(f"{log_message_prefix} from: {source_file}")

    try:
        df_price_full = pd.read_csv(source_file, parse_dates=['date'])
        if df_price_full.empty:
            current_logger.error(f"Price data file loaded from {source_file} is empty.")
            return pd.DataFrame()
    except FileNotFoundError:
        current_logger.error(f"Price data file not found: {source_file}")
        return pd.DataFrame()
    except Exception as e:
        current_logger.error(f"Error loading price data from {source_file}: {e}", exc_info=True)
        return pd.DataFrame()

    if not all(col in df_price_full.columns for col in ['date', 'ticker', 'close']):
        current_logger.error("Price data CSV must contain 'date', 'ticker', and 'close' columns.")
        return pd.DataFrame()
    df_price = df_price_full[['date', 'ticker', 'close']].copy()

    if configs.PORTFOLIO_ASSETS and isinstance(configs.PORTFOLIO_ASSETS, list) and len(configs.PORTFOLIO_ASSETS) > 0:
        df_price = df_price[df_price['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()
        if df_price.empty:
            current_logger.warning(f"No price data found for specified PORTFOLIO_ASSETS: {configs.PORTFOLIO_ASSETS}")
            return pd.DataFrame()
        missing_assets = [asset for asset in configs.PORTFOLIO_ASSETS if asset not in df_price['ticker'].unique()]
        if missing_assets:
            current_logger.warning(f"Specified PORTFOLIO_ASSETS not found in price data: {missing_assets}")

    if custom_start_date_str and custom_end_date_str:
        load_start_dt_cfg = pd.to_datetime(custom_start_date_str)
        load_end_dt_cfg = pd.to_datetime(custom_end_date_str)
        buffer_days = configs.RL_LOOKBACK_WINDOW_SIZE + 30 if include_buffer_for_rolling else 0
    else:
        load_start_dt_cfg = pd.to_datetime(configs.PORTFOLIO_START_DATE)
        load_end_dt_cfg = pd.to_datetime(configs.PORTFOLIO_END_DATE)
        buffer_days = max(configs.ROLLING_WINDOW_COVARIANCE, configs.ROLLING_WINDOW_RETURNS) + 60 if include_buffer_for_rolling else 0

    effective_load_start_dt = load_start_dt_cfg - pd.DateOffset(days=buffer_days * 1.5)
    effective_load_end_dt = load_end_dt_cfg

    min_overall_date = pd.to_datetime(configs.TIME_RANGE_START)
    effective_load_start_dt = max(effective_load_start_dt, min_overall_date)

    df_price_filtered = df_price[
        (df_price['date'] >= effective_load_start_dt) &
        (df_price['date'] <= effective_load_end_dt)
    ].copy()

    if df_price_filtered.empty:
        current_logger.error(f"No price data after filtering by date ({effective_load_start_dt.date()} - {effective_load_end_dt.date()}) and assets.")
        return pd.DataFrame()

    df_price_pivot = df_price_filtered.pivot(index='date', columns='ticker', values='close')
    df_price_pivot.sort_index(inplace=True)

    df_price_pivot = df_price_pivot.ffill()
    df_price_pivot.dropna(axis=1, how='all', inplace=True)

    if df_price_pivot.empty:
        current_logger.error(f"Price data pivot is empty after processing for range {effective_load_start_dt.date()} - {effective_load_end_dt.date()}.")
        return pd.DataFrame()

    current_logger.info(f"Price data for '{'custom range' if custom_start_date_str else 'portfolio backtest'}' loaded and pivoted. "
                        f"Shape: {df_price_pivot.shape}. "
                        f"Date range in pivot: {df_price_pivot.index.min().date()} to {df_price_pivot.index.max().date()}")
    return df_price_pivot

def load_financial_data_for_portfolio(logger_instance: logging.Logger = None) -> pd.DataFrame:
    # --- Load financial data for portfolio ---
    current_logger = logger_instance if logger_instance else logger
    file_path = configs.RAW_GLOBAL_FINANCIAL_FILE
    current_logger.info(f"Loading financial data from: {file_path}")
    try:
        df_fin = pd.read_csv(file_path)
        if df_fin.empty:
            current_logger.warning(f"Financial data file loaded from {file_path} is empty.")
            return pd.DataFrame()
    except FileNotFoundError:
        current_logger.error(f"Financial data file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        current_logger.error(f"Error loading financial data from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

    df_fin = preprocess.standardize_data(df_fin)
    if 'year' not in df_fin.columns:
        current_logger.error("Financial data CSV must contain a 'year' column.")
        return pd.DataFrame()
    try:
        if pd.api.types.is_datetime64_any_dtype(df_fin['year']):
            df_fin['year_dt'] = df_fin['year'].dt.year
        else:
            df_fin['year_dt'] = pd.to_numeric(df_fin['year'], errors='coerce').fillna(0).astype(int)
            if (df_fin['year_dt'] < 1900).any() or (df_fin['year_dt'] > 2100).any():
                 current_logger.warning("Unusual year values found in financial data after conversion.")
    except Exception as e:
        current_logger.error(f"Error processing 'year' column in financial data: {e}", exc_info=True)
        return pd.DataFrame()

    if configs.PORTFOLIO_ASSETS and isinstance(configs.PORTFOLIO_ASSETS, list) and len(configs.PORTFOLIO_ASSETS) > 0:
        df_fin = df_fin[df_fin['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()

    current_logger.info(f"Financial data loaded and standardized. Shape: {df_fin.shape}")
    return df_fin

def load_classification_probabilities(logger_instance: logging.Logger = None) -> pd.DataFrame:
    # --- Load classification probabilities ---
    current_logger = logger_instance if logger_instance else logger
    file_path = configs.CLASSIFICATION_PROBS_FILE
    current_logger.info(f"Loading classification probabilities from: {file_path}")
    try:
        df_probs = pd.read_csv(file_path, parse_dates=['date'])
        if df_probs.empty:
            current_logger.warning(f"Classification probabilities file loaded from {file_path} is empty.")
            return pd.DataFrame()
    except FileNotFoundError:
        current_logger.warning(f"Classification probabilities file not found at {file_path}. Proceeding without it.")
        return pd.DataFrame()
    except Exception as e:
        current_logger.error(f"Error loading classification probabilities from {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

    if configs.PORTFOLIO_ASSETS and isinstance(configs.PORTFOLIO_ASSETS, list) and len(configs.PORTFOLIO_ASSETS) > 0:
        df_probs = df_probs[df_probs['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()

    prob_col_name = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}"
    if prob_col_name not in df_probs.columns:
        current_logger.warning(f"Expected probability column '{prob_col_name}' not found in {file_path}. "
                               f"Checking for other 'prob_increase' columns...")
        found_prob_cols = [col for col in df_probs.columns if 'prob_increase' in col.lower()]
        if found_prob_cols:
            prob_col_name = found_prob_cols[0]
            current_logger.info(f"Using auto-detected probability column: '{prob_col_name}'")
        else:
            current_logger.error("No 'prob_increase' column found in classification probabilities. Cannot use this data.")
            return pd.DataFrame()

    current_logger.info(f"Classification probabilities loaded. Shape: {df_probs.shape}. Using prob col: '{prob_col_name}'")
    return df_probs

def calculate_returns(price_df_pivot: pd.DataFrame, logger_instance: logging.Logger = None) -> pd.DataFrame:
    # --- Calculate daily returns ---
    current_logger = logger_instance if logger_instance else logger
    if price_df_pivot.empty:
        current_logger.warning("Price data pivot is empty, cannot calculate returns.")
        return pd.DataFrame()
    daily_returns = price_df_pivot.pct_change()
    return daily_returns.iloc[1:]

def get_prepared_data_for_rebalance_date(
    rebalance_date: pd.Timestamp,
    all_prices_pivot: pd.DataFrame,
    all_daily_returns: pd.DataFrame,
    financial_data_filtered: pd.DataFrame = None,
    classification_probs_filtered: pd.DataFrame = None,
    logger_instance: logging.Logger = None
):
    # --- Prepare data for a rebalance date ---
    current_logger = logger_instance if logger_instance else logger
    current_logger.debug(f"Preparing data for rebalance date: {rebalance_date.date()}")

    asset_tickers = all_prices_pivot.columns.tolist()
    if not asset_tickers:
        current_logger.error("No asset tickers in all_prices_pivot. Cannot prepare data.")
        return pd.Series(dtype=float), pd.DataFrame(), {}, {}

    # 1. Expected Returns (mu) - ANNUALIZED
    returns_for_mu_calc = all_daily_returns[all_daily_returns.index < rebalance_date]
    
    mu_annualized = pd.Series(0.0, index=asset_tickers, dtype=float)
    if not returns_for_mu_calc.empty:
        window_returns = configs.ROLLING_WINDOW_RETURNS
        if len(returns_for_mu_calc) >= window_returns:
            daily_mu = returns_for_mu_calc.iloc[-window_returns:].mean()
        else:
            daily_mu = returns_for_mu_calc.mean()
            current_logger.debug(f"Mu calc: Using {len(returns_for_mu_calc)} days (less than window {window_returns}) for {rebalance_date.date()}.")
        mu_annualized = daily_mu * 252
        mu_annualized = mu_annualized.reindex(asset_tickers).fillna(0.0)
    current_logger.debug(f"Annualized mu for {rebalance_date.date()} (sample): {mu_annualized.head().to_dict()}")

    # 2. Covariance Matrix (S) - ANNUALIZED
    S_annualized = pd.DataFrame(np.diag(np.full(len(asset_tickers), 1e-9)), index=asset_tickers, columns=asset_tickers)
    prices_for_cov_calc = all_prices_pivot[all_prices_pivot.index < rebalance_date]
    
    if not prices_for_cov_calc.empty:
        window_cov = configs.ROLLING_WINDOW_COVARIANCE
        if len(prices_for_cov_calc) >= window_cov:
            price_slice_for_cov = prices_for_cov_calc.iloc[-window_cov:]
        elif len(prices_for_cov_calc) >= len(asset_tickers) and len(prices_for_cov_calc) >= 2:
            price_slice_for_cov = prices_for_cov_calc
            current_logger.debug(f"Cov calc: Using {len(price_slice_for_cov)} price points (less than window {window_cov}) for {rebalance_date.date()}.")
        else:
            price_slice_for_cov = pd.DataFrame()

        if not price_slice_for_cov.empty:
            try:
                price_slice_for_cov_cleaned = price_slice_for_cov.dropna(axis=1, how='all')
                if not price_slice_for_cov_cleaned.empty and price_slice_for_cov_cleaned.shape[0] >= price_slice_for_cov_cleaned.shape[1]:
                    S_calc_annual = risk_models.CovarianceShrinkage(price_slice_for_cov_cleaned, frequency=252).ledoit_wolf()
                    S_annualized = S_calc_annual.reindex(index=asset_tickers, columns=asset_tickers).fillna(0)
                else:
                    current_logger.warning(f"Covariance Shrinkage condition not met for {rebalance_date.date()} (data shape {price_slice_for_cov_cleaned.shape}). Using sample covariance or diagonal.")
                    daily_returns_slice_for_cov = price_slice_for_cov.pct_change().iloc[1:]
                    if len(daily_returns_slice_for_cov) >= 2:
                        S_daily_sample = daily_returns_slice_for_cov.cov()
                        S_annualized = (S_daily_sample * 252).reindex(index=asset_tickers, columns=asset_tickers).fillna(0)
                    
            except Exception as e_cov:
                current_logger.error(f"Error calculating annualized covariance matrix for {rebalance_date.date()}: {e_cov}", exc_info=True)
        
        for ticker_idx in S_annualized.index:
            if S_annualized.loc[ticker_idx, ticker_idx] <= 1e-9:
                S_annualized.loc[ticker_idx, ticker_idx] = 1e-9
    current_logger.debug(f"Annualized S for {rebalance_date.date()} (sample of diagonal): {np.diag(S_annualized)[:5]}")

    # 3. Financial Data for the rebalance_date
    current_financial_data_dict = {}
    if financial_data_filtered is not None and not financial_data_filtered.empty:
        target_financial_year = rebalance_date.year - 1
        current_logger.debug(f"Fetching financial data for year: {target_financial_year} (rebalance date: {rebalance_date.date()})")
        if 'year_dt' in financial_data_filtered.columns:
            relevant_fin_data_for_year = financial_data_filtered[financial_data_filtered['year_dt'] == target_financial_year]
            if not relevant_fin_data_for_year.empty:
                current_financial_data_dict = relevant_fin_data_for_year.set_index('ticker').to_dict('index')
            else:
                current_logger.debug(f"No financial data found for target year {target_financial_year} for rebalance date {rebalance_date.date()}.")
        else:
            current_logger.warning("'year_dt' column not found in financial_data_filtered. Cannot fetch financial data by year.")
    else:
        current_logger.debug(f"No financial data provided or it's empty for {rebalance_date.date()}.")

    # 4. Classification Probabilities for the rebalance_date
    current_classification_probs_dict = {ticker: 0.5 for ticker in asset_tickers}
    if classification_probs_filtered is not None and not classification_probs_filtered.empty:
        relevant_probs_before_rebal = classification_probs_filtered[classification_probs_filtered['date'] <= rebalance_date]
        if not relevant_probs_before_rebal.empty:
            latest_probs_records = relevant_probs_before_rebal.loc[relevant_probs_before_rebal.groupby('ticker')['date'].idxmax()]
            
            prob_col_to_use = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}"
            if prob_col_to_use not in latest_probs_records.columns:
                found_cols = [col for col in latest_probs_records.columns if 'prob_increase' in col.lower()]
                if found_cols: prob_col_to_use = found_cols[0]
                else: prob_col_to_use = None

            if prob_col_to_use:
                temp_probs_dict = latest_probs_records.set_index('ticker')[prob_col_to_use].to_dict()
                for ticker in asset_tickers:
                    current_classification_probs_dict[ticker] = temp_probs_dict.get(ticker, 0.5)
            else:
                current_logger.warning(f"Could not find a suitable 'prob_increase' column in classification_probs for {rebalance_date.date()}. Using defaults.")
    else:
        current_logger.debug(f"No classification probabilities provided or empty for {rebalance_date.date()}. Using defaults.")
    
    current_logger.debug(f"Data prepared for {rebalance_date.date()}: "
                         f"mu_annualized shape {mu_annualized.shape}, S_annualized shape {S_annualized.shape}, "
                         f"{len(current_financial_data_dict)} financial records, {len(current_classification_probs_dict)} probability records.")
    
    return mu_annualized, S_annualized, current_financial_data_dict, current_classification_probs_dict

if __name__ == '__main__':
    # --- Main test section ---
    main_logger = logger_setup.setup_basic_logging(log_file_name="data_preparation_test.log")
    main_logger.info("--- Testing data_preparation.py functions ---")
    
    test_rebalance_date_main = pd.to_datetime(configs.PORTFOLIO_START_DATE) + pd.DateOffset(months=1)

    prices_pivot_main = load_price_data_for_portfolio(logger_instance=main_logger)
    if not prices_pivot_main.empty:
        daily_returns_main = calculate_returns(prices_pivot_main, logger_instance=main_logger)
        financial_data_main = load_financial_data_for_portfolio(logger_instance=main_logger)
        
        if not configs.CLASSIFICATION_PROBS_FILE.exists():
            main_logger.warning(f"File {configs.CLASSIFICATION_PROBS_FILE} not found. Creating a dummy for testing.")
            dummy_dates_main = pd.date_range(start=configs.PORTFOLIO_START_DATE, end=configs.PORTFOLIO_END_DATE, freq='B')
            dummy_tickers_main = configs.PORTFOLIO_ASSETS if configs.PORTFOLIO_ASSETS else prices_pivot_main.columns.tolist()[:2]
            if not dummy_tickers_main: dummy_tickers_main = ['DUMMY1', 'DUMMY2']
            
            dummy_data_list = []
            prob_col_name_main = f'prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}'
            for date_entry_main in dummy_dates_main:
                for ticker_entry_main in dummy_tickers_main:
                    dummy_data_list.append({'date': date_entry_main, 'ticker': ticker_entry_main, prob_col_name_main: np.random.rand()})
            if dummy_data_list:
                 pd.DataFrame(dummy_data_list).to_csv(configs.CLASSIFICATION_PROBS_FILE, index=False)
                 main_logger.info(f"Created DUMMY classification probabilities at: {configs.CLASSIFICATION_PROBS_FILE}")
            else:
                main_logger.error("Could not create dummy probability file (no tickers).")

        classification_probs_main = load_classification_probabilities(logger_instance=main_logger)

        main_logger.info(f"\n--- Testing get_prepared_data_for_rebalance_date for {test_rebalance_date_main.date()} ---")
        mu_test, S_test, fin_data_test, class_probs_test = get_prepared_data_for_rebalance_date(
            test_rebalance_date_main,
            prices_pivot_main,
            daily_returns_main,
            financial_data_main,
            classification_probs_main,
            logger_instance=main_logger
        )

        if mu_test is not None and not mu_test.empty:
            main_logger.info(f"\nExpected Returns (mu) for {test_rebalance_date_main.date()} (sample):\n{mu_test.head().to_string()}")
        if S_test is not None and not S_test.empty:
            main_logger.info(f"\nCovariance Matrix (S) for {test_rebalance_date_main.date()} (sample):\n{S_test.iloc[:min(5, S_test.shape[0]), :min(5, S_test.shape[1])].to_string()}")
        if fin_data_test:
            main_logger.info(f"\nFinancial Data sample for {test_rebalance_date_main.date()}: First 2 items: {list(fin_data_test.items())[:2]}")
        if class_probs_test:
            main_logger.info(f"\nClassification Probs sample for {test_rebalance_date_main.date()}: First 5 items: {list(class_probs_test.items())[:5]}")
    else:
        main_logger.error("Could not load price data for testing `data_preparation.py`.")
    main_logger.info("--- Finished testing data_preparation.py functions ---")
