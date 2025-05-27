# /.ndmh/marketml/pipelines/05_run_portfolio_backtests.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib 

# ===== Import các module từ marketml và configs =====
try:
    from marketml.configs import configs
    from marketml.utils import logger_setup
    from marketml.portfolio_opt import (
        data_preparation,
        markowitz,
        black_litterman,
        backtesting,
        rl_environment,
        rl_optimizer,
        rl_scaler_handler # For FinancialFeatureScaler class
    )
    from pypfopt import risk_models, expected_returns # Used in data_preparation and directly
    from stable_baselines3 import PPO, A2C
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR in 05_run_portfolio_backtests.py: Could not import necessary marketml modules. {e}")
    print("Ensure the marketml package is installed correctly or PYTHONPATH is set.")
    raise

# ===== Thiết lập Logger và môi trường =====
logger = logger_setup.setup_basic_logging(log_file_name="run_portfolio_backtests.log")
logger_setup.suppress_common_warnings()
if hasattr(configs, 'RANDOM_SEED'):
    logger_setup.set_random_seeds(configs.RANDOM_SEED)
# ==========================================

def update_portfolio_values_daily(rebalance_history_df: pd.DataFrame, all_prices_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily portfolio values, cash, returns, and weights based on rebalance history
    and daily asset prices.
    """
    if not isinstance(rebalance_history_df, pd.DataFrame) or rebalance_history_df.empty:
        logger.warning("Rebalance history is empty or not a DataFrame, cannot update daily values.")
        return pd.DataFrame(columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})
    if not isinstance(all_prices_pivot, pd.DataFrame) or all_prices_pivot.empty:
        logger.error("All prices pivot is empty or not a DataFrame, cannot update daily values.")
        return pd.DataFrame(columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})

    if not isinstance(rebalance_history_df.index, pd.DatetimeIndex):
        logger.error("Rebalance history DataFrame index must be a DatetimeIndex.")
        return pd.DataFrame(columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})
    if not isinstance(all_prices_pivot.index, pd.DatetimeIndex):
        logger.error("All prices pivot DataFrame index must be a DatetimeIndex.")
        return pd.DataFrame(columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})


    min_date_hist = rebalance_history_df.index.min()
    max_date_rebal = rebalance_history_df.index.max()
    
    portfolio_end_date_dt = pd.to_datetime(configs.PORTFOLIO_END_DATE)
    
    # Filter all_prices_pivot to relevant date range for performance calculation
    # Ensure we have prices up to the portfolio_end_date or the last rebalance date.
    relevant_price_data_end = max(max_date_rebal, portfolio_end_date_dt)
    # And starts from the first rebalance date.
    all_prices_for_daily_calc = all_prices_pivot[
        (all_prices_pivot.index >= min_date_hist) &
        (all_prices_pivot.index <= relevant_price_data_end)
    ].copy()

    if all_prices_for_daily_calc.empty:
        logger.warning(f"No price data found in the period {min_date_hist} to {relevant_price_data_end} for daily update.")
        return pd.DataFrame(index=pd.to_datetime([]), columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})

    all_trading_days_in_period = all_prices_for_daily_calc.index
    
    daily_portfolio_df = pd.DataFrame(index=all_trading_days_in_period)
    daily_portfolio_df['value'] = np.nan
    daily_portfolio_df['cash'] = np.nan
    daily_portfolio_df['returns'] = np.nan
    daily_portfolio_df['weights'] = None

    daily_portfolio_df = daily_portfolio_df.astype({
        'value': float,
        'cash': float,
        'returns': float,
        'weights': object # Quan trọng để gán dict
    })

    last_known_portfolio_value = configs.INITIAL_CAPITAL
    last_known_cash = configs.INITIAL_CAPITAL
    last_known_holdings_shares = {} # Stores {ticker: num_shares}

    # Initialize based on the first rebalance entry if it's on or before the first trading day
    first_rebal_entry_on_or_before_start = rebalance_history_df[rebalance_history_df.index <= all_trading_days_in_period[0]]
    if not first_rebal_entry_on_or_before_start.empty:
        initial_entry = first_rebal_entry_on_or_before_start.iloc[-1] # Use the latest one if multiple
        last_known_portfolio_value = float(initial_entry['value'])
        last_known_cash = float(initial_entry['cash'])
        initial_weights = initial_entry['weights'] if isinstance(initial_entry['weights'], dict) else {}
        
        prices_at_initial_date = all_prices_for_daily_calc.loc[initial_entry.name]
        value_of_assets_at_initial = last_known_portfolio_value - last_known_cash
        
        for ticker, weight in initial_weights.items():
            price = prices_at_initial_date.get(ticker)
            if pd.notna(price) and price > 0:
                last_known_holdings_shares[ticker] = (value_of_assets_at_initial * weight) / price
            else:
                last_known_holdings_shares[ticker] = 0
    
    logger.debug(f"Initial daily update state: Value={last_known_portfolio_value:.2f}, Cash={last_known_cash:.2f}, Shares={len(last_known_holdings_shares)}")


    for current_date in all_trading_days_in_period:
        if current_date in rebalance_history_df.index:
            rebal_entry_series = rebalance_history_df.loc[current_date]
            # If multiple entries on the same day (should not happen with .iloc[-1] above, but good check)
            if isinstance(rebal_entry_series, pd.DataFrame):
                logger.warning(f"Multiple rebalance entries found for date {current_date}. Using the first one.")
                rebal_entry = rebal_entry_series.iloc[0]
            else:
                rebal_entry = rebal_entry_series

            daily_portfolio_df.loc[current_date, 'value'] = float(rebal_entry['value'])
            daily_portfolio_df.loc[current_date, 'cash'] = float(rebal_entry['cash'])
            daily_portfolio_df.loc[current_date, 'returns'] = float(rebal_entry['returns']) # This is rebal-to-rebal return
            
            current_weights_at_rebal = rebal_entry['weights'] if isinstance(rebal_entry['weights'], dict) else {}
            daily_portfolio_df.at[current_date, 'weights'] = current_weights_at_rebal
            
            last_known_portfolio_value = float(rebal_entry['value'])
            last_known_cash = float(rebal_entry['cash'])
            
            # Update shares based on new weights and prices at rebalance
            last_known_holdings_shares = {}
            prices_on_this_rebal_date = all_prices_for_daily_calc.loc[current_date]
            value_of_assets_on_rebal = last_known_portfolio_value - last_known_cash
            for ticker, weight in current_weights_at_rebal.items():
                price = prices_on_this_rebal_date.get(ticker)
                if pd.notna(price) and price > 0:
                    last_known_holdings_shares[ticker] = (value_of_assets_on_rebal * weight) / price
                else:
                    last_known_holdings_shares[ticker] = 0
        else: # Not a rebalance day, calculate value based on previous day's holdings and current day's prices
            if not last_known_holdings_shares and current_date > all_trading_days_in_period[0]:
                # If no holdings and not the very first day, means it's all cash
                daily_portfolio_df.loc[current_date, 'value'] = last_known_cash
                daily_portfolio_df.loc[current_date, 'cash'] = last_known_cash
                daily_portfolio_df.loc[current_date, 'returns'] = 0.0
                daily_portfolio_df.loc[current_date, 'weights'] = {}
                last_known_portfolio_value = last_known_cash # Update for next day's calculation
                continue

            current_prices_for_day = all_prices_for_daily_calc.loc[current_date]
            current_value_of_assets_eod = 0
            current_day_actual_weights = {}

            for ticker, shares in last_known_holdings_shares.items():
                price_for_day = current_prices_for_day.get(ticker)
                if pd.notna(price_for_day) and price_for_day > 0:
                    current_value_of_assets_eod += shares * price_for_day
                # If price is NaN or invalid, that asset's value is 0 for the day
            
            new_portfolio_value_eod = last_known_cash + current_value_of_assets_eod
            daily_portfolio_df.loc[current_date, 'value'] = new_portfolio_value_eod
            daily_portfolio_df.loc[current_date, 'cash'] = last_known_cash # Cash unchanged between rebalances
            
            if last_known_portfolio_value > 1e-9 : # Avoid division by zero or near-zero
                 daily_portfolio_df.loc[current_date, 'returns'] = (new_portfolio_value_eod - last_known_portfolio_value) / last_known_portfolio_value
            else:
                 daily_portfolio_df.loc[current_date, 'returns'] = 0.0 if new_portfolio_value_eod == 0 else np.nan # Or some large number if new_value is positive
            
            if new_portfolio_value_eod > 1e-9:
                for ticker, shares in last_known_holdings_shares.items():
                    price_for_day = current_prices_for_day.get(ticker)
                    if pd.notna(price_for_day) and price_for_day > 0:
                        current_day_actual_weights[ticker] = (shares * price_for_day) / new_portfolio_value_eod
            daily_portfolio_df.at[current_date, 'weights'] = current_day_actual_weights
            
            last_known_portfolio_value = new_portfolio_value_eod # Update for the next day

    return daily_portfolio_df.sort_index()

def main():
    logger.info("Starting: 05_run_portfolio_backtests pipeline")

    # --- 1. Load and Prepare Master Data ---
    logger.info("Loading and preparing master data for portfolio optimization run...")
    # Load all required data ONCE.
    # Price data for the entire backtest period + buffer for rolling calculations
    all_prices_pivot_raw = data_preparation.load_price_data_for_portfolio(
        custom_start_date_str=None, # Uses PORTFOLIO_START_DATE with buffer from configs
        custom_end_date_str=None,   # Uses PORTFOLIO_END_DATE from configs
        include_buffer_for_rolling=True
    )
    if all_prices_pivot_raw.empty:
        logger.error("CRITICAL: Price data for backtest is empty. Exiting.")
        return

    financial_data_master = data_preparation.load_financial_data_for_portfolio()
    # classification_probs_master will be loaded if the file exists
    classification_probs_master = data_preparation.load_classification_probabilities()
    if classification_probs_master is None: # If file not found, data_preparation returns None
        logger.warning(f"File {configs.CLASSIFICATION_PROBS_FILE} not found. Black-Litterman views relying on it may be affected.")
        # Create an empty DataFrame to avoid errors later if it's expected
        classification_probs_master = pd.DataFrame()


    # --- 1C. Filter data by valid tickers within the actual portfolio period ---
    # Define the exact portfolio period based on configs
    portfolio_start_dt = pd.to_datetime(configs.PORTFOLIO_START_DATE)
    portfolio_end_dt = pd.to_datetime(configs.PORTFOLIO_END_DATE)

    # Get prices strictly within the portfolio period to determine valid tickers
    prices_in_portfolio_period_strict = all_prices_pivot_raw[
        (all_prices_pivot_raw.index >= portfolio_start_dt) &
        (all_prices_pivot_raw.index <= portfolio_end_dt)
    ]
    # Valid tickers are those that have at least one non-NaN price during the portfolio period
    valid_tickers_for_backtest = prices_in_portfolio_period_strict.dropna(axis=1, how='all').columns.tolist()

    if not valid_tickers_for_backtest:
        logger.error(f"CRITICAL: No valid tickers found with price data within the portfolio period "
                     f"({portfolio_start_dt.date()} - {portfolio_end_dt.date()}). Exiting.")
        return
    logger.info(f"Valid tickers for backtest ({len(valid_tickers_for_backtest)}): {valid_tickers_for_backtest}")

    # Filter all loaded data to these valid tickers.
    # all_prices_pivot_for_backtest will contain data from [buffer_start, portfolio_end_date]
    # but only for valid_tickers_for_backtest.
    all_prices_pivot_for_backtest = all_prices_pivot_raw[valid_tickers_for_backtest].copy()
    all_prices_pivot_for_backtest.dropna(axis=1, how='all', inplace=True) # Drop any ticker that became all NaN after slicing by date (unlikely if valid_tickers logic is correct)

    if financial_data_master is not None and not financial_data_master.empty:
        financial_data_filtered = financial_data_master[financial_data_master['ticker'].isin(valid_tickers_for_backtest)].copy()
    else:
        financial_data_filtered = pd.DataFrame()

    if not classification_probs_master.empty:
        classification_probs_filtered = classification_probs_master[classification_probs_master['ticker'].isin(valid_tickers_for_backtest)].copy()
    else:
        classification_probs_filtered = pd.DataFrame()


    all_daily_returns_for_backtest = data_preparation.calculate_returns(all_prices_pivot_for_backtest)
    if all_daily_returns_for_backtest.empty:
        logger.error("CRITICAL: Daily returns for backtest period are empty. Exiting.")
        return

    asset_tickers_ordered = all_prices_pivot_for_backtest.columns.tolist() # Use the actual order from the price data

    # --- 2. Determine Rebalance Dates ---
    # Trading days within the strict portfolio period
    trading_days_in_strict_pf_period = prices_in_portfolio_period_strict.index
    if trading_days_in_strict_pf_period.empty:
        logger.error(f"CRITICAL: No trading days found in the strict portfolio period "
                     f"({portfolio_start_dt.date()} - {portfolio_end_dt.date()}). Exiting.")
        return

    # Actual first trading day for portfolio operations
    actual_first_trading_day_for_pf = trading_days_in_strict_pf_period.min()
    logger.info(f"Actual first trading day for portfolio operations: {actual_first_trading_day_for_pf.date()}")

    # Generate rebalance dates
    if isinstance(configs.REBALANCE_FREQUENCY, str):
        # Ensure date_range does not go beyond the last available trading day in the strict period
        potential_rebal_dates = pd.date_range(
            start=portfolio_start_dt,
            end=trading_days_in_strict_pf_period.max(), # Use actual last trading day
            freq=configs.REBALANCE_FREQUENCY
        )
        rebalance_dates_actual = []
        for r_date in potential_rebal_dates:
            # Find the first trading day on or after the potential rebalance date
            actual_day_candidates = trading_days_in_strict_pf_period[trading_days_in_strict_pf_period >= r_date]
            if not actual_day_candidates.empty:
                rebalance_dates_actual.append(actual_day_candidates[0])
        rebalance_dates = pd.DatetimeIndex(rebalance_dates_actual).unique()
    elif isinstance(configs.REBALANCE_FREQUENCY, int):
        rebalance_dates = trading_days_in_strict_pf_period[::configs.REBALANCE_FREQUENCY]
    else:
        logger.error(f"CRITICAL: Invalid REBALANCE_FREQUENCY: {configs.REBALANCE_FREQUENCY}. Exiting.")
        return

    if rebalance_dates.empty:
        logger.warning("No rebalance dates generated based on the REBALANCE_FREQUENCY and portfolio period. "
                       "The portfolio will be 'buy and hold' based on the initial allocation.")
    else:
        logger.info(f"Rebalance dates ({len(rebalance_dates)}): {rebalance_dates.strftime('%Y-%m-%d').tolist()}")


    # Dictionary to store results
    all_portfolio_metrics = {}
    all_daily_performance_dfs = {}


    # === Strategy 1: Markowitz ===
    logger.info("\n--- Running Markowitz Strategy ---")
    markowitz_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)

    # Initial allocation on the actual_first_trading_day_for_pf
    logger.info(f"Markowitz: Initial processing for {actual_first_trading_day_for_pf.date()}...")
    mu_initial_mw, S_initial_mw, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
        actual_first_trading_day_for_pf, all_prices_pivot_for_backtest, all_daily_returns_for_backtest,
        None, None # Markowitz does not use financial data or classification probs for views
    )
    if not mu_initial_mw.empty and not S_initial_mw.empty:
        target_weights_initial_mw = markowitz.optimize_markowitz_portfolio(mu_initial_mw, S_initial_mw)
        prices_initial_mw = all_prices_pivot_for_backtest.loc[actual_first_trading_day_for_pf]
        if not prices_initial_mw.isnull().all():
            markowitz_backtester.rebalance(actual_first_trading_day_for_pf, target_weights_initial_mw, prices_initial_mw)
        else:
            logger.warning(f"Markowitz: All prices NaN on initial allocation date {actual_first_trading_day_for_pf.date()}. Starting with cash.")
            markowitz_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
    else:
        logger.warning(f"Markowitz: Empty mu/S on initial allocation date {actual_first_trading_day_for_pf.date()}. Starting with cash.")
        markowitz_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})

    # Subsequent rebalances
    for rebal_date in rebalance_dates:
        if rebal_date <= actual_first_trading_day_for_pf: # Skip if rebal_date is the same or before initial allocation
            continue
        logger.info(f"Markowitz: Processing rebalance for {rebal_date.date()}...")
        mu_rebal_mw, S_rebal_mw, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
            rebal_date, all_prices_pivot_for_backtest, all_daily_returns_for_backtest, None, None
        )
        if mu_rebal_mw.empty or S_rebal_mw.empty:
            logger.warning(f"Markowitz: Empty mu/S for rebalance on {rebal_date.date()}. Holding previous weights.")
            if markowitz_backtester.portfolio_history: # Add a no-trade entry
                last_entry = markowitz_backtester.portfolio_history[-1].copy()
                last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                markowitz_backtester.portfolio_history.append(last_entry)
            continue
        target_weights_mw = markowitz.optimize_markowitz_portfolio(mu_rebal_mw, S_rebal_mw)
        prices_on_rebal_mw = all_prices_pivot_for_backtest.loc[rebal_date]
        if prices_on_rebal_mw.isnull().all():
            logger.warning(f"Markowitz: All prices NaN on rebalance date {rebal_date.date()}. Holding previous weights.")
            if markowitz_backtester.portfolio_history:
                last_entry = markowitz_backtester.portfolio_history[-1].copy()
                last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                markowitz_backtester.portfolio_history.append(last_entry)
            continue
        markowitz_backtester.rebalance(rebal_date, target_weights_mw, prices_on_rebal_mw)

    if markowitz_backtester.portfolio_history:
        # Convert portfolio history to DataFrame
        markowitz_rebal_history_df = pd.DataFrame(markowitz_backtester.portfolio_history).set_index('date')
        
        # Generate daily performance DataFrame
        daily_markowitz_perf_df = update_portfolio_values_daily(
            markowitz_rebal_history_df, 
            all_prices_pivot_for_backtest
        )
        
        # Validate the performance DataFrame
        if daily_markowitz_perf_df is not None and not daily_markowitz_perf_df.empty and 'returns' in daily_markowitz_perf_df.columns:
            logger.info(
                f"Markowitz daily performance DataFrame generated. Shape: {daily_markowitz_perf_df.shape}. "
                f"Columns: {daily_markowitz_perf_df.columns.tolist()}"
            )
            
            # Process returns series
            portfolio_returns_for_metrics = daily_markowitz_perf_df['returns'].fillna(0.0).astype(float)
            
            if not portfolio_returns_for_metrics.empty:
                try:
                    # Calculate metrics with enhanced calculator
                    metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL)
                    markowitz_metrics = metrics_calculator.calculate_metrics(
                        portfolio_returns_series=portfolio_returns_for_metrics,
                        logger_instance=logger  # Pass logger for consistent logging
                    )
                    
                    if not markowitz_metrics.empty:
                        all_portfolio_metrics['Markowitz'] = markowitz_metrics
                        all_daily_performance_dfs['Markowitz'] = daily_markowitz_perf_df
                        
                        # Save performance data
                        try:
                            output_path = configs.RESULTS_OUTPUT_DIR / getattr(
                                configs, 
                                'MARKOWITZ_PERF_DAILY_FILE_NAME', 
                                "markowitz_performance_daily.csv"
                            )
                            daily_markowitz_perf_df.to_csv(output_path)
                            logger.info(f"Successfully saved Markowitz daily performance to {output_path}")
                        except Exception as e_save:
                            logger.error(f"Failed to save Markowitz performance: {e_save}", exc_info=True)
                    else:
                        logger.warning("Markowitz metrics calculation returned empty Series")
                except Exception as e_metrics:
                    logger.error(f"Error calculating Markowitz metrics: {e_metrics}", exc_info=True)
            else:
                logger.error("Processed Markowitz returns series is empty")
        else:
            error_details = []
            if daily_markowitz_perf_df is None:
                error_details.append("is None")
            elif daily_markowitz_perf_df.empty:
                error_details.append("is empty")
            if 'returns' not in daily_markowitz_perf_df.columns:
                error_details.append("missing 'returns' column")
                
            logger.error(
                f"Invalid Markowitz performance DataFrame: {'; '.join(error_details)}. "
                f"Available columns: {daily_markowitz_perf_df.columns.tolist() if daily_markowitz_perf_df is not None else 'None'}"
            )
    else:
        logger.warning("Markowitz strategy resulted in no rebalance history - no metrics calculated")

    # === Strategy 2: Black-Litterman ===
    logger.info("\n--- Running Black-Litterman Strategy ---")
    bl_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)

    # Initial allocation for Black-Litterman
    logger.info(f"Black-Litterman: Initial processing for {actual_first_trading_day_for_pf.date()}...")
    _, S_prior_initial_bl, fin_data_initial_bl, probs_initial_bl = data_preparation.get_prepared_data_for_rebalance_date(
        actual_first_trading_day_for_pf, all_prices_pivot_for_backtest, all_daily_returns_for_backtest,
        financial_data_filtered, classification_probs_filtered
    )
    # For pi_prior and S_prior, BL model needs historical prices *before* the rebalance_date
    historical_prices_for_bl_initial_prior = all_prices_pivot_for_backtest[all_prices_pivot_for_backtest.index < actual_first_trading_day_for_pf]

    if not historical_prices_for_bl_initial_prior.empty and len(historical_prices_for_bl_initial_prior) >= 2:
        try:
            # Calculate pi_prior for views generation (can be market implied or historical)
            pi_prior_for_views_initial = expected_returns.mean_historical_return(historical_prices_for_bl_initial_prior, frequency=252)
            pi_prior_for_views_initial = pi_prior_for_views_initial.reindex(asset_tickers_ordered).fillna(0)

            P_initial, Q_initial, Omega_initial = black_litterman.generate_views_from_signals(
                asset_tickers_ordered, fin_data_initial_bl, probs_initial_bl, pi_prior_for_views_initial, actual_first_trading_day_for_pf
            )
            posterior_mu_initial_bl, posterior_S_initial_bl = black_litterman.get_black_litterman_posterior_estimates(
                historical_prices_for_bl_initial_prior, asset_tickers_ordered, None, P_initial, Q_initial, Omega_initial
            )
            if posterior_mu_initial_bl is not None and not posterior_mu_initial_bl.empty:
                target_weights_initial_bl = markowitz.optimize_markowitz_portfolio(posterior_mu_initial_bl, posterior_S_initial_bl)
                prices_initial_bl_rebal = all_prices_pivot_for_backtest.loc[actual_first_trading_day_for_pf]
                if not prices_initial_bl_rebal.isnull().all():
                    bl_backtester.rebalance(actual_first_trading_day_for_pf, target_weights_initial_bl, prices_initial_bl_rebal)
                else: # Fallback
                    logger.warning(f"BL: All prices NaN on initial allocation {actual_first_trading_day_for_pf.date()}. Starting with cash.")
                    bl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            else: # Fallback if posterior mu/S rỗng
                logger.warning(f"BL: Posterior estimates empty on initial allocation {actual_first_trading_day_for_pf.date()}. Starting with Markowitz historical.")
                # Fallback to Markowitz if BL fails
                if not mu_initial_mw.empty and not S_initial_mw.empty: # From previous Markowitz calc
                    target_weights_initial_bl = markowitz.optimize_markowitz_portfolio(mu_initial_mw, S_initial_mw)
                    prices_initial_bl_rebal = all_prices_pivot_for_backtest.loc[actual_first_trading_day_for_pf]
                    bl_backtester.rebalance(actual_first_trading_day_for_pf, target_weights_initial_bl, prices_initial_bl_rebal)
                else:
                    bl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})

        except Exception as e_init_bl:
            logger.error(f"Error during initial BL processing for {actual_first_trading_day_for_pf.date()}: {e_init_bl}", exc_info=True)
            bl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
    else:
        logger.warning(f"BL: Insufficient historical price data for prior on initial allocation {actual_first_trading_day_for_pf.date()}. Starting with cash.")
        bl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})

    # Subsequent rebalances for Black-Litterman
    for rebal_date in rebalance_dates:
        if rebal_date <= actual_first_trading_day_for_pf:
            continue
        logger.info(f"Black-Litterman: Processing rebalance for {rebal_date.date()}...")
        _, _, fin_data_rebal_bl, probs_rebal_bl = data_preparation.get_prepared_data_for_rebalance_date(
            rebal_date, all_prices_pivot_for_backtest, all_daily_returns_for_backtest,
            financial_data_filtered, classification_probs_filtered
        )
        historical_prices_for_bl_prior = all_prices_pivot_for_backtest[all_prices_pivot_for_backtest.index < rebal_date]

        if historical_prices_for_bl_prior.empty or len(historical_prices_for_bl_prior) < 2:
            logger.warning(f"BL: Insufficient historical prices for prior on {rebal_date.date()}. Holding previous weights.")
            if bl_backtester.portfolio_history:
                last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                bl_backtester.portfolio_history.append(last_entry)
            continue
        try:
            pi_prior_for_views = expected_returns.mean_historical_return(historical_prices_for_bl_prior, frequency=252)
            pi_prior_for_views = pi_prior_for_views.reindex(asset_tickers_ordered).fillna(0)

            P_rebal, Q_rebal, Omega_rebal = black_litterman.generate_views_from_signals(
                asset_tickers_ordered, fin_data_rebal_bl, probs_rebal_bl, pi_prior_for_views, rebal_date
            )
            posterior_mu_rebal_bl, posterior_S_rebal_bl = black_litterman.get_black_litterman_posterior_estimates(
                historical_prices_for_bl_prior, asset_tickers_ordered, None, P_rebal, Q_rebal, Omega_rebal
            )

            if posterior_mu_rebal_bl is not None and not posterior_mu_rebal_bl.empty:
                target_weights_bl = markowitz.optimize_markowitz_portfolio(posterior_mu_rebal_bl, posterior_S_rebal_bl)
            else: # Fallback to Markowitz historical if BL posterior fails
                logger.warning(f"BL: Posterior estimates empty for {rebal_date.date()}. Falling back to Markowitz historical.")
                mu_fallback_bl, S_fallback_bl, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
                    rebal_date, all_prices_pivot_for_backtest, all_daily_returns_for_backtest, None, None
                )
                if not mu_fallback_bl.empty and not S_fallback_bl.empty:
                    target_weights_bl = markowitz.optimize_markowitz_portfolio(mu_fallback_bl, S_fallback_bl)
                else: # Ultimate fallback: equal weights
                    logger.error(f"BL: Fallback Markowitz also failed for {rebal_date.date()}. Using equal weights.")
                    target_weights_bl = {ticker: 1.0/len(asset_tickers_ordered) for ticker in asset_tickers_ordered}

            prices_on_rebal_bl = all_prices_pivot_for_backtest.loc[rebal_date]
            if prices_on_rebal_bl.isnull().all():
                logger.warning(f"BL: All prices NaN on rebalance {rebal_date.date()}. Holding previous weights.")
                if bl_backtester.portfolio_history:
                    last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                    bl_backtester.portfolio_history.append(last_entry)
                continue
            bl_backtester.rebalance(rebal_date, target_weights_bl, prices_on_rebal_bl)
        except Exception as e_bl_rebal:
            logger.error(f"Error during BL rebalance for {rebal_date.date()}: {e_bl_rebal}", exc_info=True)
            if bl_backtester.portfolio_history:
                last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                bl_backtester.portfolio_history.append(last_entry)


    if bl_backtester.portfolio_history:
        bl_rebal_history_df = pd.DataFrame(bl_backtester.portfolio_history).set_index('date')
        daily_bl_perf_df = update_portfolio_values_daily(bl_rebal_history_df, all_prices_pivot_for_backtest)
        if daily_bl_perf_df is not None and not daily_bl_perf_df.empty:
            metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL)
            all_portfolio_metrics['BlackLitterman'] = metrics_calculator.calculate_metrics(portfolio_returns_series=daily_bl_perf_df['returns'].fillna(0))
            all_daily_performance_dfs['BlackLitterman'] = daily_bl_perf_df
            try:
                daily_bl_perf_df.to_csv(configs.RESULTS_OUTPUT_DIR / "blacklitterman_performance_daily.csv")
                logger.info("BlackLitterman daily performance saved.")
            except Exception as e_save_bl: logger.error(f"Error saving BlackLitterman performance: {e_save_bl}", exc_info=True)
        else: logger.error("BlackLitterman daily performance DataFrame is empty or None.")
    else: logger.warning("Black-Litterman strategy resulted in no rebalance history.")

    
    # === Strategy 3: Reinforcement Learning ===
    if configs.RL_STRATEGY_ENABLED:
        logger.info("\n--- Running Reinforcement Learning Strategy ---")
        configs.RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # configs.RL_LOG_DIR_FOR_SB3.mkdir(parents=True, exist_ok=True) # SB3 logs dir

        trained_rl_model = None
        # Load or initialize financial feature scaler for RL
        rl_fin_scaler = rl_scaler_handler.FinancialFeatureScaler.load(configs.RL_MODEL_DIR)


        # --- 1. Load or Train RL Model ---
        model_path_to_load = None
        best_model_path = configs.RL_MODEL_DIR / "best_model" / "best_model.zip"
        final_model_path_from_config = configs.RL_MODEL_SAVE_PATH # e.g., ppo_portfolio_agent.zip

        if best_model_path.exists(): model_path_to_load = best_model_path
        elif final_model_path_from_config.exists(): model_path_to_load = final_model_path_from_config

        if model_path_to_load:
            logger.info(f"Attempting to load existing RL model from {model_path_to_load}")
            try:
                if configs.RL_ALGORITHM.upper() == "PPO": trained_rl_model = PPO.load(model_path_to_load, device='auto')
                elif configs.RL_ALGORITHM.upper() == "A2C": trained_rl_model = A2C.load(model_path_to_load, device='auto')
                if trained_rl_model:
                    logger.info(f"Successfully loaded RL model from {model_path_to_load}.")
                    if not rl_fin_scaler.feature_names and configs.RL_FINANCIAL_FEATURES: # Check if scaler is valid
                        logger.warning(f"RL model loaded, but financial scaler from {configs.RL_MODEL_DIR} is invalid or empty. Financial features might not be scaled correctly during inference if not retrained.")
                        # Potentially force retrain if scaler is crucial and invalid
                        # trained_rl_model = None # Uncomment to force retrain if scaler is bad
            except Exception as e_load_rl:
                logger.error(f"Error loading RL model from {model_path_to_load}: {e_load_rl}. Will attempt to train a new model.", exc_info=True)
                trained_rl_model = None
        
        # --- Retrain if needed ---
        condition_to_retrain_rl = trained_rl_model is None
        if configs.RL_FINANCIAL_FEATURES and (not rl_fin_scaler.feature_names):
            if trained_rl_model: logger.warning("RL model was loaded, but financial scaler is invalid. Retraining RL model.")
            condition_to_retrain_rl = True
            trained_rl_model = None


        if condition_to_retrain_rl: 
            logger.info("Training a new RL model as no valid pre-trained model/scaler was found or forced retrain.")
            
            # Prepare training data for RL
            try:
                prices_for_rl_training = data_preparation.load_price_data_for_portfolio(
                    custom_start_date_str=configs.RL_TRAIN_DATA_START_DATE,
                    custom_end_date_str=configs.RL_TRAIN_DATA_END_DATE,
                    include_buffer_for_rolling=True
                )
                
                # Data validation checks
                if prices_for_rl_training.empty:
                    logger.error("Empty price data loaded for RL training. Skipping RL strategy.")
                elif len(prices_for_rl_training) < configs.RL_LOOKBACK_WINDOW_SIZE + 10:
                    logger.error(f"Insufficient RL training data ({len(prices_for_rl_training)} rows). Need at least {configs.RL_LOOKBACK_WINDOW_SIZE + 10} rows.")
                else:
                    # Filter and validate tickers
                    prices_for_rl_training_filtered = prices_for_rl_training[valid_tickers_for_backtest].copy()
                    prices_for_rl_training_filtered.dropna(axis=1, how='all', inplace=True)
                    
                    if prices_for_rl_training_filtered.empty:
                        logger.error("No valid tickers remaining after filtering RL training data.")
                    else:
                        logger.info(
                            f"RL Training Data: Prices shape {prices_for_rl_training_filtered.shape}, "
                            f"Tickers: {len(prices_for_rl_training_filtered.columns)}"
                        )
                        
                        # Prepare output paths
                        rl_model_main_save_path = configs.RL_MODEL_SAVE_PATH
                        rl_eval_cb_path = configs.RL_MODEL_DIR / "sb3_eval_logs"
                        rl_tb_log_path = configs.RL_MODEL_DIR / "sb3_tb_logs"
                        
                        # Create directories if they don't exist
                        for path in [rl_eval_cb_path, rl_tb_log_path]:
                            path.mkdir(parents=True, exist_ok=True)
                        
                        # Train RL agent with comprehensive logging
                        logger.info("Starting RL model training...")
                        trained_rl_model, fitted_scaler_from_train = rl_optimizer.train_rl_agent(
                            prices_df_train=prices_for_rl_training_filtered,
                            financial_data_train=financial_data_filtered,
                            classification_probs_train=classification_probs_filtered,
                            financial_features_list=configs.RL_FINANCIAL_FEATURES,
                            prob_features_list=configs.RL_PROB_FEATURES,
                            model_save_path=rl_model_main_save_path,
                            eval_callback_log_path=rl_eval_cb_path,
                            tensorboard_log_path=rl_tb_log_path,
                            logger_instance=logger,
                            initial_capital=configs.INITIAL_CAPITAL,
                            transaction_cost_bps=configs.RL_TRANSACTION_COST_BPS,
                            lookback_window_size=configs.RL_LOOKBACK_WINDOW_SIZE,
                            rebalance_frequency_days=configs.RL_REBALANCE_FREQUENCY_DAYS,
                            total_timesteps=configs.RL_TOTAL_TIMESTEPS,
                            rl_algorithm=configs.RL_ALGORITHM,
                            ppo_n_steps=configs.RL_PPO_N_STEPS,
                            ppo_batch_size=configs.RL_PPO_BATCH_SIZE,
                            ppo_n_epochs=configs.RL_PPO_N_EPOCHS,
                            ppo_gamma=configs.RL_PPO_GAMMA,
                            ppo_gae_lambda=configs.RL_PPO_GAE_LAMBDA,
                            ppo_clip_range=configs.RL_PPO_CLIP_RANGE,
                            ppo_ent_coef=configs.RL_PPO_ENT_COEF,
                            ppo_vf_coef=configs.RL_PPO_VF_COEF,
                            ppo_max_grad_norm=configs.RL_PPO_MAX_GRAD_NORM,
                            ppo_learning_rate=configs.RL_PPO_LEARNING_RATE,
                            ppo_policy_kwargs=configs.RL_PPO_POLICY_KWARGS,
                            reward_use_log_return=configs.RL_REWARD_USE_LOG_RETURN,
                            reward_turnover_penalty_factor=configs.RL_REWARD_TURNOVER_PENALTY_FACTOR
                        )
                    if trained_rl_model and fitted_scaler_from_train and fitted_scaler_from_train.feature_names:
                        rl_fin_scaler = fitted_scaler_from_train # Update scaler to the one from this training run
                        logger.info("RL model training complete. Using newly trained model and scaler.")
                    elif trained_rl_model:
                        logger.warning("RL model trained, but scaler returned was invalid. Financial features might not be scaled correctly in inference.")
                    else:
                        logger.error("RL model training failed.")
            except Exception as e:
                logger.error(f"Error during RL model training preparation: {str(e)}", exc_info=True)
            
        # --- 2. Backtest RL Strategy ---
        if trained_rl_model is None:
            logger.error("RL model not available (neither loaded nor trained). Skipping RL backtest.")
        elif configs.RL_FINANCIAL_FEATURES and not rl_fin_scaler.feature_names:
             logger.error("RL backtest: Financial features are enabled, but the financial scaler is invalid. Skipping RL backtest.")
        else:
            logger.info("Starting RL backtesting...")
            rl_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS) # Use general backtest transaction cost

            # Inference environment uses the full backtest price data and the loaded/fitted scaler
            # asset_tickers_ordered is already defined from all_prices_pivot_for_backtest
            inference_env_kwargs = {
                'prices_df': all_prices_pivot_for_backtest, # Data for the actual backtest period
                'financial_data': financial_data_filtered,
                'classification_probs': classification_probs_filtered,
                'initial_capital': configs.INITIAL_CAPITAL,
                'transaction_cost_bps': configs.TRANSACTION_COST_BPS, # Env's internal cost for reward, not for backtester
                'lookback_window_size': configs.RL_LOOKBACK_WINDOW_SIZE,
                'rebalance_frequency_days': 1, # Env steps one day at a time for state, backtester handles rebal_freq
                'financial_features': rl_fin_scaler.feature_names, # From the scaler
                'prob_features': configs.RL_PROB_FEATURES,
                'financial_feature_means': rl_fin_scaler.means,
                'financial_feature_stds': rl_fin_scaler.stds
            }
            rl_inference_env = rl_environment.PortfolioEnv(**inference_env_kwargs)
            current_rl_obs, _ = rl_inference_env.reset() # Initial observation at the start of backtest data

            # Initial allocation for RL
            logger.info(f"RL Strategy: Initial processing for {actual_first_trading_day_for_pf.date()}...")
            # Get observation for the actual_first_trading_day_for_pf
            # This requires setting the inference_env to that specific date.
            # rl_inference_env.current_step and rl_inference_env._current_prices_idx need to be set.
            try:
                # Find index for actual_first_trading_day_for_pf in the env's prices_df
                initial_obs_idx_in_env = rl_inference_env.prices_df.index.get_loc(actual_first_trading_day_for_pf)
                if initial_obs_idx_in_env < rl_inference_env.lookback_window_size:
                    logger.error(f"RL: Not enough history for initial state on {actual_first_trading_day_for_pf.date()}. "
                                 f"Required lookback {rl_inference_env.lookback_window_size}, available from index {initial_obs_idx_in_env}. Starting with cash.")
                    rl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
                else:
                    rl_inference_env._current_prices_idx = initial_obs_idx_in_env
                    current_rl_obs = rl_inference_env._get_state() # Get state for this specific day

                    predicted_weights_initial_rl = rl_optimizer.predict_rl_weights(trained_rl_model, current_rl_obs)
                    target_weights_initial_rl_assets = {
                        ticker: predicted_weights_initial_rl[i] for i, ticker in enumerate(rl_inference_env.tickers)
                    }
                    prices_initial_rl_backtest = all_prices_pivot_for_backtest.loc[actual_first_trading_day_for_pf]
                    if not prices_initial_rl_backtest.isnull().all():
                        rl_backtester.rebalance(actual_first_trading_day_for_pf, target_weights_initial_rl_assets, prices_initial_rl_backtest)
                    else:
                        logger.warning(f"RL: All prices NaN on initial allocation {actual_first_trading_day_for_pf.date()}. Starting with cash.")
                        rl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            except KeyError:
                logger.error(f"RL: Date {actual_first_trading_day_for_pf.date()} not found in RL inference env price index. Starting with cash.", exc_info=True)
                rl_backtester.portfolio_history.append({'date': actual_first_trading_day_for_pf, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})


            # Subsequent rebalances for RL
            for rebal_date in rebalance_dates:
                if rebal_date <= actual_first_trading_day_for_pf:
                    continue
                logger.info(f"RL Strategy: Processing rebalance for target date {rebal_date.date()}...")
                try:
                    # Find index for rebal_date in the env's prices_df
                    # The state should be observed at the rebalance date (or just before if action is for next period)
                    rebal_obs_idx_in_env = rl_inference_env.prices_df.index.get_loc(rebal_date)
                    if rebal_obs_idx_in_env < rl_inference_env.lookback_window_size:
                         logger.warning(f"RL: Not enough history for state on {rebal_date.date()}. Holding previous portfolio.")
                         if rl_backtester.portfolio_history:
                             last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                             rl_backtester.portfolio_history.append(last_entry)
                         continue

                    rl_inference_env._current_prices_idx = rebal_obs_idx_in_env
                    # Update env's portfolio_value and current_weights based on backtester's last state
                    # This is important if the env's internal state affects _get_state() rewards/values
                    if rl_backtester.portfolio_history:
                        rl_inference_env.portfolio_value = rl_backtester.portfolio_history[-1]['value']
                        # Convert backtester weights (dict) to env's numpy array format
                        last_bt_weights_dict = rl_backtester.portfolio_history[-1]['weights']
                        env_weights_array = np.zeros(rl_inference_env.num_assets + 1)
                        for i_ticker, ticker_name in enumerate(rl_inference_env.tickers):
                            env_weights_array[i_ticker] = last_bt_weights_dict.get(ticker_name, 0.0)
                        env_weights_array[-1] = 1.0 - np.sum(env_weights_array[:-1]) # Cash
                        rl_inference_env.current_weights = env_weights_array


                    current_rl_obs = rl_inference_env._get_state()
                    predicted_weights_rl = rl_optimizer.predict_rl_weights(trained_rl_model, current_rl_obs)
                    target_weights_rl_assets = {
                        ticker: predicted_weights_rl[i] for i, ticker in enumerate(rl_inference_env.tickers)
                    }
                    prices_on_rebal_rl = all_prices_pivot_for_backtest.loc[rebal_date]
                    if prices_on_rebal_rl.isnull().all():
                        logger.warning(f"RL: All prices NaN on rebalance {rebal_date.date()}. Holding previous.")
                        if rl_backtester.portfolio_history:
                            last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                            rl_backtester.portfolio_history.append(last_entry)
                        continue
                    rl_backtester.rebalance(rebal_date, target_weights_rl_assets, prices_on_rebal_rl)
                except KeyError:
                    logger.error(f"RL: Date {rebal_date.date()} not found in RL inference env price index. Holding previous.", exc_info=True)
                    if rl_backtester.portfolio_history:
                        last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                        rl_backtester.portfolio_history.append(last_entry)
                except Exception as e_rl_rebal_loop:
                    logger.error(f"Error during RL rebalance loop for {rebal_date.date()}: {e_rl_rebal_loop}", exc_info=True)
                    if rl_backtester.portfolio_history: # Try to keep going by holding
                        last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0
                        rl_backtester.portfolio_history.append(last_entry)


            if rl_backtester.portfolio_history:
                rl_rebal_history_df = pd.DataFrame(rl_backtester.portfolio_history).set_index('date')
                daily_rl_perf_df = update_portfolio_values_daily(rl_rebal_history_df, all_prices_pivot_for_backtest)
                if daily_rl_perf_df is not None and not daily_rl_perf_df.empty:
                    metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL)
                    all_portfolio_metrics['RL_Strategy'] = metrics_calculator.calculate_metrics(portfolio_returns_series=daily_rl_perf_df['returns'].fillna(0))
                    all_daily_performance_dfs['RL_Strategy'] = daily_rl_perf_df
                    try:
                        daily_rl_perf_df.to_csv(configs.RESULTS_OUTPUT_DIR / f"{configs.RL_ALGORITHM.lower()}_strategy_performance_daily.csv")
                        logger.info("RL Strategy daily performance saved.")
                    except Exception as e_save_rl_perf: logger.error(f"Error saving RL Strategy performance: {e_save_rl_perf}", exc_info=True)
                else: logger.error("RL Strategy daily performance DataFrame is empty or None.")
            else: logger.warning("RL strategy resulted in no rebalance history.")


    # --- 4. Consolidate and Log/Save Results ---
    if all_portfolio_metrics:
        logger.info("\n--- Portfolio Performance Summary ---")
        summary_df = pd.DataFrame.from_dict(all_portfolio_metrics, orient='index')
        if not summary_df.empty:
            logger.info(f"\n{summary_df.round(4).to_string()}") # Round for cleaner log
            try:
                summary_df.to_csv(configs.RESULTS_OUTPUT_DIR / "portfolio_strategies_summary.csv")
                logger.info(f"Portfolio strategies summary saved to {configs.RESULTS_OUTPUT_DIR / 'portfolio_strategies_summary.csv'}")
            except Exception as e_save_summary:
                logger.error(f"Error saving portfolio summary: {e_save_summary}", exc_info=True)
        else:
            logger.info("Portfolio summary DataFrame is empty.")
    else:
        logger.warning("No portfolio strategies were successfully backtested to generate a summary.")

    logger.info("Finished: 05_run_portfolio_backtests pipeline")


if __name__ == "__main__":
    main()