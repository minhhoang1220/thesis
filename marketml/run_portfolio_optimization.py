# marketml/run_portfolio_optimization.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import joblib

# ===== THÊM THƯ MỤC GỐC VÀO PATH =====
# Giả sử script này nằm trong .ndmh/marketml/
PROJECT_ROOT_SCRIPT = Path(__file__).resolve().parents[1] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_SCRIPT))

# === IMPORT MODULES ===
try:
    from marketml.configs import configs
    from marketml.log import setup
    from marketml.portfolio_opt import data_preparation, markowitz, black_litterman, backtesting
    from marketml.portfolio_opt import rl_environment, rl_optimizer # THÊM IMPORT RL
    from pypfopt import risk_models, expected_returns
    from stable_baselines3 import PPO, A2C # THÊM IMPORT CHO TẢI MODEL RL
except ImportError as e:
    print(f"CRITICAL ERROR in run_portfolio_optimization.py: Could not import necessary modules. {e}")
    sys.exit(1)

# ===== THIẾT LẬP MÔI TRƯỜNG =====
print("DEBUG: About to call setup_basic_logging from run_portfolio_optimization.py")
logger = setup.setup_basic_logging(log_level=logging.INFO, log_file_name="portfolio_opt.log")
setup.suppress_common_warnings()
if hasattr(configs, 'RANDOM_SEED'):
    setup.set_random_seeds(configs.RANDOM_SEED)


def generate_classification_signals_if_needed():
    """
    Kiểm tra nếu file soft signals tồn tại, nếu không thì gọi script generate_signals.
    Hoặc, bạn có thể tích hợp trực tiếp logic của generate_signals.py vào đây.
    Để đơn giản, hiện tại chúng ta giả định file đã được tạo hoặc tạo thủ công.
    """
    if not configs.CLASSIFICATION_PROBS_FILE_PATH.exists():
        logger.warning(f"File {configs.CLASSIFICATION_PROBS_FILE_PATH} not found.")
        logger.info("Please run 'python -m marketml.generate_signals' first to create it, or implement automatic generation here.")
        # For now, to proceed with testing, we might create a dummy file or exit
        # For this example, we'll assume it must exist or a dummy was made by data_preparation's test part
        # sys.exit("Exiting: Classification probabilities file is required.")
        # Nếu data_preparation.py tạo dummy thì có thể không cần exit
        if not configs.CLASSIFICATION_PROBS_FILE_PATH.exists(): # Kiểm tra lại sau khi data_prep có thể đã tạo dummy
             logger.error("Dummy probability file also not created. Exiting.")
             sys.exit("Exiting: Classification probabilities file is required and dummy was not created.")

# Hàm update_portfolio_values_daily (copy từ câu trả lời trước vào đây hoặc import nếu bạn tách ra utils)
def update_portfolio_values_daily(rebalance_history_df, all_prices_pivot):
    if rebalance_history_df.empty:
        logger.warning("Rebalance history is empty, cannot update daily values.")
        return pd.DataFrame(columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})

    min_date_hist = rebalance_history_df.index.min()
    max_date_rebal = rebalance_history_df.index.max()
    
    portfolio_end_date_dt = pd.to_datetime(configs.PORTFOLIO_END_DATE)
    last_available_price_date_in_range = all_prices_pivot[all_prices_pivot.index <= portfolio_end_date_dt].index.max()
    
    effective_end_date = min(
        max(max_date_rebal, last_available_price_date_in_range if pd.notna(last_available_price_date_in_range) else max_date_rebal),
        portfolio_end_date_dt
    )

    all_trading_days_in_period = all_prices_pivot.loc[min_date_hist:effective_end_date].index
    
    if all_trading_days_in_period.empty:
        logger.warning(f"No trading days found in period {min_date_hist} to {effective_end_date} for daily update.")
        return pd.DataFrame(index=pd.to_datetime([]), columns=['value', 'cash', 'returns', 'weights']).astype({'weights': object})

    # Khởi tạo DataFrame với các cột và kiểu dữ liệu
    daily_portfolio_df = pd.DataFrame(
        index=all_trading_days_in_period,
        columns=['value', 'cash', 'returns', 'weights']
    )
    # Ép kiểu cho các cột
    daily_portfolio_df['value'] = pd.to_numeric(daily_portfolio_df['value'], errors='coerce')
    daily_portfolio_df['cash'] = pd.to_numeric(daily_portfolio_df['cash'], errors='coerce')
    daily_portfolio_df['returns'] = pd.to_numeric(daily_portfolio_df['returns'], errors='coerce')
    daily_portfolio_df['weights'] = daily_portfolio_df['weights'].astype(object) # QUAN TRỌNG

    # ... (logic khởi tạo last_known_portfolio_value, last_known_cash, last_known_holdings_shares như cũ) ...
    last_known_portfolio_value = configs.INITIAL_CAPITAL
    last_known_cash = configs.INITIAL_CAPITAL
    last_known_holdings_shares = {} 

    initial_rebal_candidates = rebalance_history_df[rebalance_history_df.index <= all_trading_days_in_period[0]]
    if not initial_rebal_candidates.empty:
        last_rebal_before_start_entry = initial_rebal_candidates.iloc[-1]
        value_at_last_rebal = last_rebal_before_start_entry['value']
        cash_at_last_rebal = last_rebal_before_start_entry['cash']
        weights_at_last_rebal = last_rebal_before_start_entry['weights'] if isinstance(last_rebal_before_start_entry['weights'], dict) else {}
        prices_at_last_rebal_date_series = all_prices_pivot.loc[last_rebal_before_start_entry.name]

        value_of_assets_at_rebal = value_at_last_rebal - cash_at_last_rebal
        for ticker, weight in weights_at_last_rebal.items():
            price_at_last_rebal = prices_at_last_rebal_date_series.get(ticker)
            if pd.notna(price_at_last_rebal) and price_at_last_rebal > 0:
                last_known_holdings_shares[ticker] = (value_of_assets_at_rebal * weight) / price_at_last_rebal
            else:
                last_known_holdings_shares[ticker] = 0
        last_known_cash = cash_at_last_rebal
        last_known_portfolio_value = value_at_last_rebal


    for current_date in all_trading_days_in_period:
        if current_date in rebalance_history_df.index:
            rebal_entry = rebalance_history_df.loc[current_date]
            if isinstance(rebal_entry, pd.DataFrame):
                logger.warning(f"Multiple rebalance entries found for date {current_date}. Using the first one.")
                rebal_entry = rebal_entry.iloc[0] # Lấy hàng đầu tiên

            # Bây giờ rebal_entry nên là một Series (một hàng)
            daily_portfolio_df.at[current_date, 'value'] = float(rebal_entry['value'])
            daily_portfolio_df.at[current_date, 'cash'] = float(rebal_entry['cash'])
            daily_portfolio_df.at[current_date, 'returns'] = float(rebal_entry['returns'])
            current_weights = rebal_entry['weights'] if isinstance(rebal_entry['weights'], dict) else {}
            daily_portfolio_df.at[current_date, 'weights'] = current_weights 
            
            last_known_portfolio_value = float(rebal_entry['value']) # Ép kiểu float
            last_known_cash = float(rebal_entry['cash'])  
            
            last_known_holdings_shares = {}
            current_prices_on_rebal = all_prices_pivot.loc[current_date]
            value_of_assets_on_rebal = last_known_portfolio_value - last_known_cash
            for ticker, weight in current_weights.items():
                price_on_rebal = current_prices_on_rebal.get(ticker)
                if pd.notna(price_on_rebal) and price_on_rebal > 0:
                    last_known_holdings_shares[ticker] = (value_of_assets_on_rebal * weight) / price_on_rebal
                else:
                    last_known_holdings_shares[ticker] = 0
        else: 
            if not last_known_holdings_shares and current_date > all_trading_days_in_period[0]:
                daily_portfolio_df.at[current_date, 'value'] = last_known_portfolio_value
                daily_portfolio_df.at[current_date, 'cash'] = last_known_cash
                daily_portfolio_df.at[current_date, 'returns'] = 0.0
                daily_portfolio_df.at[current_date, 'weights'] = {} # SỬ DỤNG .at[]
                continue

            current_prices_for_day = all_prices_pivot.loc[current_date]
            current_value_of_assets_eod = 0
            current_day_actual_weights = {}

            for ticker, shares in last_known_holdings_shares.items():
                price_for_day = current_prices_for_day.get(ticker)
                if pd.notna(price_for_day) and price_for_day > 0:
                    current_value_of_assets_eod += shares * price_for_day
            
            new_portfolio_value_eod = last_known_cash + current_value_of_assets_eod
            daily_portfolio_df.at[current_date, 'value'] = new_portfolio_value_eod
            daily_portfolio_df.at[current_date, 'cash'] = last_known_cash
            
            if last_known_portfolio_value > 0 :
                 daily_portfolio_df.at[current_date, 'returns'] = (new_portfolio_value_eod - last_known_portfolio_value) / last_known_portfolio_value
            else: 
                 daily_portfolio_df.at[current_date, 'returns'] = 0.0 if new_portfolio_value_eod == 0 else np.nan
            
            if new_portfolio_value_eod > 0:
                for ticker, shares in last_known_holdings_shares.items():
                    price_for_day = current_prices_for_day.get(ticker)
                    if pd.notna(price_for_day) and price_for_day > 0:
                        current_day_actual_weights[ticker] = (shares * price_for_day) / new_portfolio_value_eod
            daily_portfolio_df.at[current_date, 'weights'] = current_day_actual_weights # SỬ DỤNG .at[]
            
            last_known_portfolio_value = new_portfolio_value_eod
            
    return daily_portfolio_df.sort_index()

def run_portfolio_strategies():
    logger.info("Starting portfolio optimization backtest...")
    generate_classification_signals_if_needed()

    # --- 1A. TẢI DỮ LIỆU GIÁ BAN ĐẦU CHO HUẤN LUYỆN RL (NẾU CẦN) ---
    prices_df_for_rl_training_loaded_at_start = pd.DataFrame()
    if configs.RL_STRATEGY_ENABLED:
        logger.info("Đang tải dữ liệu giá ban đầu cho huấn luyện RL...")
        prices_df_for_rl_training_loaded_at_start = data_preparation.load_price_data_for_portfolio(
            custom_start_date_str=configs.RL_TRAIN_DATA_START_DATE,
            custom_end_date_str=configs.RL_TRAIN_DATA_END_DATE,
            include_buffer_for_rolling=True
        )

    # --- 1B. TẢI VÀ CHUẨN BỊ DỮ LIỆU CHUNG CHO BACKTEST VÀ CÁC CHIẾN LƯỢC KHÁC ---
    logger.info("Đang tải và chuẩn bị dữ liệu chung cho backtest...")
    # all_prices_pivot này sẽ được dùng cho Markowitz, BL, và làm cơ sở cho inference env của RL
    all_prices_pivot_for_backtest_raw = data_preparation.load_price_data_for_portfolio()
    if all_prices_pivot_for_backtest_raw.empty:
        logger.error("Dữ liệu giá cho backtest rỗng. Đang thoát."); return

    # Dữ liệu đầy đủ, KHÔNG LỌC TICKERS ở đây cho Markowitz/BL
    financial_data_full_master_unfiltered = data_preparation.load_financial_data_for_portfolio()
    classification_probs_full_master_unfiltered = data_preparation.load_classification_probabilities()

    # Tải financial và classification data đầy đủ một lần
    financial_data_full = data_preparation.load_financial_data_for_portfolio()
    classification_probs_full = data_preparation.load_classification_probabilities()

    # --- 1C. XÁC ĐỊNH VALID TICKERS VÀ LỌC DỮ LIỆU CHUNG ---
    portfolio_period_prices = all_prices_pivot_for_backtest_raw.loc[
        pd.to_datetime(configs.PORTFOLIO_START_DATE):pd.to_datetime(configs.PORTFOLIO_END_DATE)
    ]
    valid_tickers_in_period = portfolio_period_prices.dropna(axis=1, how='all').columns.tolist()

    if not valid_tickers_in_period:
        logger.error(f"Không có tickers hợp lệ trong giai đoạn portfolio. Đang thoát."); return

    logger.info(f"Tickers hợp lệ cho giai đoạn backtest ({len(valid_tickers_in_period)}): {valid_tickers_in_period}")

    # Lọc all_prices_pivot_for_backtest_raw
    all_prices_pivot = all_prices_pivot_for_backtest_raw[valid_tickers_in_period].copy()
    if all_prices_pivot.empty:
        logger.error("Dữ liệu giá cho backtest rỗng sau khi lọc tickers. Đang thoát."); return

    # Lọc prices_df_for_rl_training_loaded_at_start (nếu nó không rỗng)
    if not prices_df_for_rl_training_loaded_at_start.empty:
        common_tickers_for_rl_train = [t for t in valid_tickers_in_period if t in prices_df_for_rl_training_loaded_at_start.columns]
        if not common_tickers_for_rl_train:
            logger.warning("Không có tickers chung giữa dữ liệu huấn luyện RL và giai đoạn backtest. Dữ liệu huấn luyện RL sẽ rỗng.")
            prices_df_for_rl_training_filtered = pd.DataFrame()
        else:
            prices_df_for_rl_training_filtered = prices_df_for_rl_training_loaded_at_start[common_tickers_for_rl_train].copy()
        logger.info(f"Dữ liệu giá huấn luyện RL sau khi lọc theo valid tickers: {len(prices_df_for_rl_training_filtered)} hàng, {prices_df_for_rl_training_filtered.shape[1]} tickers.")
    else:
        prices_df_for_rl_training_filtered = pd.DataFrame() # Đảm bảo nó được định nghĩa

    # Lọc financial_data_full và classification_probs_full theo valid_tickers_in_period
    # Điều này làm cho các DataFrame này nhỏ hơn và chỉ chứa dữ liệu liên quan.
    if not financial_data_full.empty and 'Ticker' in financial_data_full.columns: # Giả sử tên cột là 'Ticker'
        financial_data_full_filtered = financial_data_full[financial_data_full['Ticker'].isin(valid_tickers_in_period)].copy()
    else:
        financial_data_full_filtered = financial_data_full # Hoặc pd.DataFrame() nếu rỗng

    if not classification_probs_full.empty and 'ticker' in classification_probs_full.columns:
        classification_probs_full_filtered = classification_probs_full[classification_probs_full['ticker'].isin(valid_tickers_in_period)].copy()
    else:
        classification_probs_full_filtered = classification_probs_full # Hoặc pd.DataFrame()

    all_daily_returns = data_preparation.calculate_returns(all_prices_pivot)
    if all_daily_returns.empty:
        logger.error("Lợi nhuận hàng ngày cho backtest rỗng. Đang thoát."); return

    # --- 1D. CHUẨN BỊ DỮ LIỆU GIÁ ĐÃ LỌC TICKERS CHO HUẤN LUYỆN RL ---
    prices_df_for_rl_training_final_filtered_tickers = pd.DataFrame()
    if configs.RL_STRATEGY_ENABLED and not prices_df_for_rl_training_loaded_at_start.empty:
        common_tickers_for_rl_train = [t for t in valid_tickers_in_period if t in prices_df_for_rl_training_loaded_at_start.columns]
        if not common_tickers_for_rl_train:
            logger.warning("Không có tickers chung giữa dữ liệu huấn luyện RL và backtest. Dữ liệu giá huấn luyện RL sẽ rỗng.")
        else:
            prices_df_for_rl_training_final_filtered_tickers = prices_df_for_rl_training_loaded_at_start[common_tickers_for_rl_train].copy()
        logger.info(f"Dữ liệu giá huấn luyện RL sau khi lọc theo valid tickers: {len(prices_df_for_rl_training_final_filtered_tickers)} hàng, {prices_df_for_rl_training_final_filtered_tickers.shape[1]} tickers.")
        
    # --- 2. Xác định Ngày Tái Cân Bằng ---
    asset_tickers_ordered = all_prices_pivot.columns.tolist()
    trading_days = all_prices_pivot.loc[pd.to_datetime(configs.PORTFOLIO_START_DATE):pd.to_datetime(configs.PORTFOLIO_END_DATE)].index
    if trading_days.empty: logger.error("No trading days in portfolio period. Exiting."); return

    # Xác định ngày giao dịch đầu tiên trong khoảng thời gian portfolio
    first_portfolio_day = pd.to_datetime(configs.PORTFOLIO_START_DATE)
    actual_first_trading_day_in_pf_period = trading_days[trading_days >= first_portfolio_day].min()
    logger.info(f"Actual first trading day in portfolio period: {actual_first_trading_day_in_pf_period.date()}")

    # Xác định các ngày tái cân bằng
    if isinstance(configs.REBALANCE_FREQUENCY, str):
        rebalance_dates_potential = pd.date_range(start=configs.PORTFOLIO_START_DATE, end=configs.PORTFOLIO_END_DATE, freq=configs.REBALANCE_FREQUENCY)
        temp_rebalance_dates = []
        for r_date in rebalance_dates_potential:
            actual_rebal_day_candidates = trading_days[trading_days >= r_date]
            if not actual_rebal_day_candidates.empty: 
                temp_rebalance_dates.append(actual_rebal_day_candidates[0])
        rebalance_dates = pd.DatetimeIndex(temp_rebalance_dates).unique()
    elif isinstance(configs.REBALANCE_FREQUENCY, int): 
        rebalance_dates = trading_days[::configs.REBALANCE_FREQUENCY]
    else: 
        logger.error(f"Invalid REBALANCE_FREQUENCY: {configs.REBALANCE_FREQUENCY}")
        return
    
    if len(rebalance_dates) == 0: logger.error("No rebalance dates. Check config."); return
    rebalance_dates = rebalance_dates[rebalance_dates <= pd.to_datetime(configs.PORTFOLIO_END_DATE)]
    logger.info(f"Rebalance dates ({len(rebalance_dates)}): {rebalance_dates.strftime('%Y-%m-%d').tolist()}")

    results_summary = {}
    all_portfolio_dfs = {}
    asset_tickers_ordered = all_prices_pivot.columns.tolist()

    # --- Chiến lược 1: Markowitz ---
    logger.info("\n--- Running Markowitz Strategy ---")
    markowitz_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)

    initial_rebal_date_mw = actual_first_trading_day_in_pf_period
    logger.info(f"Markowitz: Initial processing for {initial_rebal_date_mw.date()}...")
    mu_initial_mw, S_initial_mw, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
        initial_rebal_date_mw, all_prices_pivot, all_daily_returns, None, None
    )

    if not mu_initial_mw.empty and not S_initial_mw.empty and not mu_initial_mw.isnull().all() and not S_initial_mw.isnull().all().all():
        target_weights_initial_mw = markowitz.optimize_markowitz_portfolio(mu_initial_mw, S_initial_mw)
        prices_initial_mw = all_prices_pivot.loc[initial_rebal_date_mw]
        if not prices_initial_mw.isnull().all():
            markowitz_backtester.rebalance(initial_rebal_date_mw, target_weights_initial_mw, prices_initial_mw)
        else:
            logger.warning(f"All prices NaN on Markowitz initial rebalance {initial_rebal_date_mw.date()}. Starting with cash.")
            markowitz_backtester.portfolio_history.append({'date': initial_rebal_date_mw, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
    else:
        logger.warning(f"Empty/NaN mu/S on Markowitz initial rebalance {initial_rebal_date_mw.date()}. Starting with cash.")
        markowitz_backtester.portfolio_history.append({'date': initial_rebal_date_mw, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})

    if not rebalance_dates.empty and actual_first_trading_day_in_pf_period < rebalance_dates[0]:
         markowitz_backtester.portfolio_history.append({'date': actual_first_trading_day_in_pf_period, 'value': configs.INITIAL_CAPITAL,'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
    for rebal_date in rebalance_dates:
        logger.info(f"Markowitz: Processing rebalance for {rebal_date.date()}...")
        mu_annual, S_annual, _, _ = data_preparation.get_prepared_data_for_rebalance_date(rebal_date, all_prices_pivot, all_daily_returns, None, None)
        if mu_annual.empty or S_annual.empty or mu_annual.isnull().all() or S_annual.isnull().all().all():
            logger.warning(f"Skipping Markowitz rebalance on {rebal_date.date()} due to empty/NaN mu/S.") # ... (xử lý skip) ...
            if markowitz_backtester.portfolio_history: last_entry = markowitz_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; markowitz_backtester.portfolio_history.append(last_entry)
            elif not markowitz_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : markowitz_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            continue
        target_weights_mw = markowitz.optimize_markowitz_portfolio(mu_annual, S_annual)
        prices_on_rebal_date = all_prices_pivot.loc[rebal_date]
        if prices_on_rebal_date.isnull().all(): # ... (xử lý skip) ...
             logger.warning(f"All prices NaN on Markowitz rebalance {rebal_date.date()}. Skipping.")
             if markowitz_backtester.portfolio_history: last_entry = markowitz_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; markowitz_backtester.portfolio_history.append(last_entry)
             elif not markowitz_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : markowitz_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
             continue
        markowitz_backtester.rebalance(rebal_date, target_weights_mw, prices_on_rebal_date)
    if markowitz_backtester.portfolio_history: markowitz_history_df = pd.DataFrame(markowitz_backtester.portfolio_history).set_index('date')
    else: markowitz_history_df = pd.DataFrame(columns=['value', 'cash', 'returns', 'weights'])
    daily_markowitz_perf_df = update_portfolio_values_daily(markowitz_history_df, all_prices_pivot)
    if daily_markowitz_perf_df is not None and not daily_markowitz_perf_df.empty and 'returns' in daily_markowitz_perf_df:
        metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL); results_summary['Markowitz'] = metrics_calculator.calculate_metrics(portfolio_returns_series=daily_markowitz_perf_df['returns'].fillna(0)); all_portfolio_dfs['Markowitz'] = daily_markowitz_perf_df
        try: configs.RESULTS_DIR.mkdir(parents=True, exist_ok=True); daily_markowitz_perf_df.to_csv(configs.RESULTS_DIR / "markowitz_performance_daily.csv"); logger.info(f"Markowitz daily performance saved.")
        except Exception as e_save: logger.error(f"Error saving Markowitz performance: {e_save}")
    else: logger.error("Markowitz daily perf DF empty/missing returns.")

    # --- Chiến lược 2: Black-Litterman ---
    logger.info("\n--- Running Black-Litterman Strategy ---")
    bl_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)
    
    # Xử lý ngày đầu tư đầu tiên cho Black-Litterman (nếu cần)
    # actual_first_trading_day_in_pf_period đã được định nghĩa ở trên
    initial_investment_done_bl = False
    if not rebalance_dates.empty and actual_first_trading_day_in_pf_period < rebalance_dates[0]:
        initial_rebal_date_bl = actual_first_trading_day_in_pf_period
        logger.info(f"Black-Litterman: Initial processing for {initial_rebal_date_bl.date()}...")
        
        # Lấy dữ liệu cho ngày đầu tư đầu tiên
        _, _, current_fin_data_init, current_class_probs_init = data_preparation.get_prepared_data_for_rebalance_date(
            initial_rebal_date_bl, all_prices_pivot, all_daily_returns, financial_data_full, classification_probs_full
        )
        historical_prices_for_bl_prior_init = all_prices_pivot[all_prices_pivot.index < initial_rebal_date_bl]

        if not historical_prices_for_bl_prior_init.empty and len(historical_prices_for_bl_prior_init) >= 2:
            try:
                pi_prior_init_for_views = expected_returns.mean_historical_return(historical_prices_for_bl_prior_init, frequency=252)
                pi_prior_init_for_views = pi_prior_init_for_views.reindex(asset_tickers_ordered).fillna(0)
                
                P_init, Q_init, Omega_init = black_litterman.generate_views_from_signals(
                    asset_tickers_ordered, current_fin_data_init, current_class_probs_init, pi_prior_init_for_views, initial_rebal_date_bl
                )
                posterior_mu_initial, posterior_S_initial = black_litterman.get_black_litterman_posterior_estimates(
                    historical_prices_for_bl_prior_init, asset_tickers_ordered, None, P_init, Q_init, Omega_init
                )
                if posterior_mu_initial is not None and not posterior_mu_initial.empty:
                    target_weights_initial_bl = markowitz.optimize_markowitz_portfolio(posterior_mu_initial, posterior_S_initial)
                    prices_initial_bl = all_prices_pivot.loc[initial_rebal_date_bl]
                    if not prices_initial_bl.isnull().all():
                        bl_backtester.rebalance(initial_rebal_date_bl, target_weights_initial_bl, prices_initial_bl)
                        initial_investment_done_bl = True 
                    else: # Fallback nếu giá NaN
                        bl_backtester.portfolio_history.append({'date': initial_rebal_date_bl, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
                else: # Fallback nếu posterior mu/S rỗng
                    bl_backtester.portfolio_history.append({'date': initial_rebal_date_bl, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            except Exception as e_init_bl:
                logger.error(f"Error during initial BL processing for {initial_rebal_date_bl.date()}: {e_init_bl}")
                bl_backtester.portfolio_history.append({'date': initial_rebal_date_bl, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
        else:
             logger.warning(f"Skipping BL initial rebalance on {initial_rebal_date_bl.date()} due to insufficient historical price data.")
             bl_backtester.portfolio_history.append({'date': initial_rebal_date_bl, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
    elif not rebalance_dates.empty and actual_first_trading_day_in_pf_period == rebalance_dates[0]:
        # Ngày rebalance đầu tiên trùng với ngày bắt đầu danh mục, sẽ được xử lý trong vòng lặp
        pass
    elif rebalance_dates.empty : # Không có ngày rebalance nào, nhưng vẫn cần entry đầu tiên nếu pf period có ngày
        bl_backtester.portfolio_history.append({'date': actual_first_trading_day_in_pf_period, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})


    # Vòng lặp tái cân bằng định kỳ
    for rebal_date in rebalance_dates:
        # Bỏ qua ngày đầu tiên nếu nó đã được xử lý ở trên và ngày đầu tiên đó chính là ngày rebalance đầu tiên
        if initial_investment_done_bl and rebal_date == actual_first_trading_day_in_pf_period:
             logger.info(f"Black-Litterman: Skipping rebalance for {rebal_date.date()} as it was handled as initial investment.")
             continue

        logger.info(f"Black-Litterman: Processing rebalance for {rebal_date.date()}...")

        # 1. Lấy dữ liệu financial và classification probs cho views cho NGÀY REBAL_DATE HIỆN TẠI
        _, _, current_fin_data, current_class_probs = data_preparation.get_prepared_data_for_rebalance_date(
            rebal_date, # <<--- SỬA Ở ĐÂY: Dùng rebal_date hiện tại của vòng lặp
            all_prices_pivot, 
            all_daily_returns, 
            financial_data_full, 
            classification_probs_full
        )

        # 2. Lấy historical prices để tính S_prior và pi_prior cho BlackLittermanModel
        #    Cần dữ liệu giá TRƯỚC rebal_date
        historical_prices_for_bl_prior = all_prices_pivot[all_prices_pivot.index < rebal_date]
        logger.debug(f"BL Rebal {rebal_date.date()}: Shape of historical_prices_for_bl_prior: {historical_prices_for_bl_prior.shape}")
        if historical_prices_for_bl_prior.empty or len(historical_prices_for_bl_prior) < 2: # Cần ít nhất 2 dòng để tính return/cov
            logger.warning(f"Skipping BL rebalance on {rebal_date.date()} due to insufficient historical price data for prior S/pi ({len(historical_prices_for_bl_prior)} rows).")
            if bl_backtester.portfolio_history: last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; bl_backtester.portfolio_history.append(last_entry)
            elif not bl_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : bl_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            continue
        
        logger.debug(f"BL Rebal {rebal_date.date()}: historical_prices_for_bl_prior HEAD:\n{historical_prices_for_bl_prior.head().to_string()}")
        logger.debug(f"BL Rebal {rebal_date.date()}: historical_prices_for_bl_prior TAIL:\n{historical_prices_for_bl_prior.tail().to_string()}")
        logger.debug(f"BL Rebal {rebal_date.date()}: Any NaN in historical_prices_for_bl_prior? {historical_prices_for_bl_prior.isnull().values.any()}")

        # Tính pi_prior (market implied hoặc historical mean) - ĐÃ ANNUALIZE BÊN TRONG get_black_litterman_posterior_estimates
        # Nếu không có market_caps, hàm get_black_litterman_posterior_estimates sẽ dùng historical mean
        # Chúng ta cần truyền pi_prior (annualized) vào generate_views_from_signals nếu view phụ thuộc vào nó.
        # Cách 1: Tính pi_prior ở đây trước.
        # temp_S_prior_for_pi = risk_models.CovarianceShrinkage(historical_prices_for_bl_prior, frequency=252).ledoit_wolf()
        try:
            pi_prior_annual_for_views = expected_returns.mean_historical_return(historical_prices_for_bl_prior, frequency=252)
            pi_prior_annual_for_views = pi_prior_annual_for_views.reindex(asset_tickers_ordered).fillna(0)
            logger.debug(f"BL Rebal {rebal_date.date()}: pi_prior_annual_for_views (for views) HEAD:\n{pi_prior_annual_for_views.head().to_string()}")
        except Exception as e_pi_calc:
            logger.error(f"BL Rebal {rebal_date.date()}: Error calculating pi_prior_annual_for_views: {e_pi_calc}")
            if bl_backtester.portfolio_history: last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; bl_backtester.portfolio_history.append(last_entry)
            elif not bl_backtester.portfolio_history : bl_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            continue

        # 3. Tạo Views (P, Q, Omega)
        P, Q, Omega = black_litterman.generate_views_from_signals(
            asset_tickers_ordered, current_fin_data, current_class_probs, pi_prior_annual_for_views, rebal_date
        )
        
        if P is not None:
            logger.debug(f"BL Rebal {rebal_date.date()}: P matrix shape: {P.shape}, Q vector shape: {Q.shape if Q is not None else 'None'}, Omega matrix shape: {Omega.shape if Omega is not None else 'None'}")
        else:
            logger.info(f"BL Rebal {rebal_date.date()}: No views generated.")

        # 4. Lấy posterior mu và S từ BlackLittermanModel
        #    (Hàm này sẽ tự tính S_prior và pi_prior bên trong nó)
        #    Market caps có thể để None, hàm sẽ fallback sang historical mean cho pi_prior
        posterior_mu_annual, posterior_S_annual = black_litterman.get_black_litterman_posterior_estimates(
            historical_prices_for_bl_prior,
            asset_tickers_order=asset_tickers_ordered,
            market_caps=None,
            views_P=P,
            views_Q=Q,
            views_Omega=Omega
        )

        mu_fallback, S_fallback, _, _ = data_preparation.get_prepared_data_for_rebalance_date(rebal_date, all_prices_pivot, all_daily_returns, None, None)
        if posterior_mu_annual is None or posterior_S_annual is None or posterior_mu_annual.empty or posterior_S_annual.empty:
            logger.warning(f"Failed to get BL posterior estimates for {rebal_date.date()}. Attempting to use Markowitz with historical mean/cov as fallback for this period.")
            if mu_fallback.empty or S_fallback.empty or mu_fallback.isnull().all() or S_fallback.isnull().all().all():
                logger.error(f"Fallback Markowitz also failed for {rebal_date.date()} due to empty mu/S. Skipping rebalance.")
                if bl_backtester.portfolio_history: last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; bl_backtester.portfolio_history.append(last_entry)
                elif not bl_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : bl_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
                continue
            target_weights_bl = markowitz.optimize_markowitz_portfolio(mu_fallback, S_fallback)
            logger.info(f"BL Rebal {rebal_date.date()}: Used Markowitz historical as fallback.")
        else:
        # 5. Tối ưu hóa danh mục bằng Markowitz optimizer với posterior estimates
            target_weights_bl = markowitz.optimize_markowitz_portfolio(posterior_mu_annual, posterior_S_annual)
        
        prices_on_rebal_date = all_prices_pivot.loc[rebal_date]
        if prices_on_rebal_date.isnull().all(): # ... (xử lý skip) ...
            logger.warning(f"All prices NaN on BL rebalance {rebal_date.date()}. Skipping.")
            if bl_backtester.portfolio_history: last_entry = bl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date; last_entry['returns'] = 0.0; bl_backtester.portfolio_history.append(last_entry)
            elif not bl_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : bl_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            continue
        bl_backtester.rebalance(rebal_date, target_weights_bl, prices_on_rebal_date)

    if bl_backtester.portfolio_history: bl_history_df = pd.DataFrame(bl_backtester.portfolio_history).set_index('date')
    else: bl_history_df = pd.DataFrame(columns=['value', 'cash', 'returns', 'weights'])
    daily_bl_perf_df = update_portfolio_values_daily(bl_history_df, all_prices_pivot)
    if daily_bl_perf_df is not None and not daily_bl_perf_df.empty and 'returns' in daily_bl_perf_df:
        metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL)
        results_summary['BlackLitterman'] = metrics_calculator.calculate_metrics(portfolio_returns_series=daily_bl_perf_df['returns'].fillna(0))
        all_portfolio_dfs['BlackLitterman'] = daily_bl_perf_df
        try: configs.RESULTS_DIR.mkdir(parents=True, exist_ok=True); daily_bl_perf_df.to_csv(configs.RESULTS_DIR / "blacklitterman_performance_daily.csv"); logger.info(f"BlackLitterman daily performance saved.")
        except Exception as e_save_bl: logger.error(f"Error saving BlackLitterman performance: {e_save_bl}")
    else: logger.error("BlackLitterman daily perf DF empty/missing returns.")

    # --- Reinforcement Learning Strategy ---
    if configs.RL_STRATEGY_ENABLED:
        logger.info("\n--- Đang chạy Chiến lược Học Tăng Cường ---")
        configs.RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        configs.RL_LOG_DIR.mkdir(parents=True, exist_ok=True)

        trained_rl_model = None
        fin_means_for_inference = None
        fin_stds_for_inference = None
        scaler_path = configs.RL_MODEL_DIR / "financial_scalers.joblib"

        # 1. CỐ GẮNG TẢI MODEL VÀ SCALERS
        model_path_to_load_base = configs.RL_MODEL_DIR
        best_model_path = model_path_to_load_base / "best_model" / "best_model.zip"
        final_model_path_from_config = configs.RL_MODEL_SAVE_PATH

        model_to_try_load = None
        if best_model_path.exists(): model_to_try_load = best_model_path
        elif final_model_path_from_config.exists(): model_to_try_load = final_model_path_from_config

        # 2. NẾU CẦN, HUẤN LUYỆN LẠI MODEL
        # Điều kiện huấn luyện lại: model chưa được tải HOẶC financial features được dùng mà scalers không có
        condition_to_retrain = trained_rl_model is None
        if configs.RL_FINANCIAL_FEATURES and (fin_means_for_inference is None or fin_stds_for_inference is None):
            if trained_rl_model is not None: logger.warning("Model RL đã tải nhưng thiếu financial scalers. Sẽ huấn luyện lại.")
            condition_to_retrain = True
            trained_rl_model = None

        if condition_to_retrain:
            logger.info("Đang chuẩn bị dữ liệu và huấn luyện mô hình RL mới.")
            # prices_df_for_rl_training_final_filtered đã được chuẩn bị ở phần 1D
            if prices_df_for_rl_training_final_filtered_tickers.empty or \
               len(prices_df_for_rl_training_final_filtered_tickers) < configs.RL_LOOKBACK_WINDOW_SIZE + configs.RL_REBALANCE_FREQUENCY_DAYS + 1 or \
               prices_df_for_rl_training_final_filtered_tickers.shape[1] == 0:
                 logger.error(f"Không đủ dữ liệu giá cho huấn luyện RL. Bỏ qua huấn luyện.")
            else:
                # Chuẩn bị financial và classification data cho huấn luyện RL,
                # lọc từ BẢN GỐC (master_unfiltered) theo tickers và dates của prices_df_for_rl_training_final_filtered
                financial_data_for_rl_train_slice = pd.DataFrame()
                classification_probs_for_rl_train_slice = pd.DataFrame()
                
                rl_train_slice_tickers = prices_df_for_rl_training_final_filtered_tickers.columns.tolist()
                rl_train_slice_start_date = prices_df_for_rl_training_final_filtered_tickers.index.min()
                rl_train_slice_end_date = prices_df_for_rl_training_final_filtered_tickers.index.max()

                if not financial_data_full_master_unfiltered.empty:
                    if 'date' in financial_data_full_master_unfiltered.columns:
                        financial_data_for_rl_train_slice = financial_data_full_master_unfiltered[
                            (financial_data_full_master_unfiltered['ticker'].isin(rl_train_slice_tickers)) &
                            (financial_data_full_master_unfiltered['date'] >= rl_train_slice_start_date) &
                            (financial_data_full_master_unfiltered['date'] <= rl_train_slice_end_date)
                        ].copy()
                    elif 'Year' in financial_data_full_master_unfiltered.columns and 'Ticker' in financial_data_full_master_unfiltered.columns:
                        start_year_rl_s = rl_train_slice_start_date.year
                        end_year_rl_s = rl_train_slice_end_date.year
                        financial_data_for_rl_train_slice = financial_data_full_master_unfiltered[
                            (financial_data_full_master_unfiltered['Ticker'].isin(rl_train_slice_tickers)) &
                            (financial_data_full_master_unfiltered['Year'] >= start_year_rl_s - 1) &
                            (financial_data_full_master_unfiltered['Year'] <= end_year_rl_s)
                        ].copy()
                    logger.info(f"Dữ liệu tài chính cho huấn luyện RL (slice): {len(financial_data_for_rl_train_slice)} hàng.")

                if not classification_probs_full_master_unfiltered.empty:
                    classification_probs_for_rl_train_slice = classification_probs_full_master_unfiltered[
                        (classification_probs_full_master_unfiltered['ticker'].isin(rl_train_slice_tickers)) &
                        (classification_probs_full_master_unfiltered['date'] >= rl_train_slice_start_date) &
                        (classification_probs_full_master_unfiltered['date'] <= rl_train_slice_end_date)
                    ].copy()
                    logger.info(f"Dữ liệu xác suất cho huấn luyện RL (slice): {len(classification_probs_for_rl_train_slice)} hàng.")
                
                trained_rl_model = rl_optimizer.train_rl_agent(
                    prices_df_train=prices_df_for_rl_training_final_filtered_tickers,
                    financial_data_train=financial_data_for_rl_train_slice,
                    classification_probs_train=classification_probs_for_rl_train_slice,
                    financial_features_list=configs.RL_FINANCIAL_FEATURES,
                    prob_features_list=configs.RL_PROB_FEATURES,
                    model_save_path=configs.RL_MODEL_SAVE_PATH,
                    initial_capital=configs.INITIAL_CAPITAL,
                    transaction_cost_bps=configs.RL_TRANSACTION_COST_BPS,
                    lookback_window_size=configs.RL_LOOKBACK_WINDOW_SIZE,
                    rebalance_frequency_days=configs.RL_REBALANCE_FREQUENCY_DAYS,
                    total_timesteps=configs.RL_TOTAL_TIMESTEPS,
                    rl_algorithm=configs.RL_ALGORITHM,# train_rl_agent sẽ dùng thư mục cha của path này để lưu scaler
                    log_dir=configs.RL_LOG_DIR,
                    # ... (truyền các siêu tham số PPO và reward shaping từ configs) ...
                )
                # SAU KHI HUẤN LUYỆN, TẢI LẠI SCALERS VỪA ĐƯỢC LƯU ĐỂ DÙNG CHO INFERENCE
                if trained_rl_model and scaler_path.exists():
                    try:
                        scalers = joblib.load(scaler_path)
                        fin_means_for_inference = scalers['means']
                        fin_stds_for_inference = scalers['stds']
                        logger.info(f"Đã tải financial scalers (sau huấn luyện) từ {scaler_path} để dùng cho inference.")
                    except Exception as e_load_scaler_after_train:
                        logger.error(f"Lỗi khi tải financial scalers sau huấn luyện: {e_load_scaler_after_train}")
                elif trained_rl_model:
                     logger.warning(f"Mô hình RL vừa được huấn luyện nhưng không tìm thấy file scaler tại {scaler_path}.")


        if trained_rl_model is None:
            logger.error("Mô hình RL không thể được tải hoặc huấn luyện. Bỏ qua backtest RL.")
        else:
            # --- 2. Backtest với Tác nhân đã Huấn luyện ---
            rl_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)
            
            # Chuẩn bị financial_data và classification_probs cho giai đoạn BACKTEST (sử dụng all_prices_pivot)
            financial_data_for_rl_inference = pd.DataFrame()
            classification_probs_for_rl_inference = pd.DataFrame()
            # all_prices_pivot là DataFrame giá cho giai đoạn backtest
            if not financial_data_full.empty and not all_prices_pivot.empty:
                actual_inference_start_dt = all_prices_pivot.index.min()
                actual_inference_end_dt = all_prices_pivot.index.max()
                inference_tickers = all_prices_pivot.columns.tolist()

                if 'date' in financial_data_full.columns:
                    financial_data_for_rl_inference = financial_data_full[
                        (financial_data_full['ticker'].isin(inference_tickers)) & # Lọc theo ticker của all_prices_pivot
                        (financial_data_full['date'] >= actual_inference_start_dt) &
                        (financial_data_full['date'] <= actual_inference_end_dt)
                    ].copy()
                elif 'Year' in financial_data_full.columns and 'Ticker' in financial_data_full.columns:
                    start_year_bt = actual_inference_start_dt.year
                    end_year_bt = actual_inference_end_dt.year
                    financial_data_for_rl_inference = financial_data_full[
                        (financial_data_full['Ticker'].isin(inference_tickers)) &
                        (financial_data_full['Year'] >= start_year_bt - 1) &
                        (financial_data_full['Year'] <= end_year_bt)
                    ].copy()
            
            if not classification_probs_full.empty and not all_prices_pivot.empty:
                actual_inference_start_dt = all_prices_pivot.index.min()
                actual_inference_end_dt = all_prices_pivot.index.max()
                inference_tickers = all_prices_pivot.columns.tolist()
                classification_probs_for_rl_inference = classification_probs_full[
                    (classification_probs_full['ticker'].isin(inference_tickers)) &
                    (classification_probs_full['date'] >= actual_inference_start_dt) &
                    (classification_probs_full['date'] <= actual_inference_end_dt)
                ].copy()

            logger.info(f"Dữ liệu tài chính cho inference RL: {len(financial_data_for_rl_inference)} hàng.")
            logger.info(f"Dữ liệu xác suất cho inference RL: {len(classification_probs_for_rl_inference)} hàng.")
            if fin_means_for_inference is None or fin_stds_for_inference is None:
                 logger.warning("fin_means_for_inference hoặc fin_stds_for_inference là None. Đặc trưng tài chính sẽ không được chuẩn hóa đúng cách trong inference.")


            inference_env_kwargs = {
                'prices_df': all_prices_pivot, # Dữ liệu giá cho backtest
                'financial_data': financial_data_for_rl_inference,
                'classification_probs': classification_probs_for_rl_inference,
                'initial_capital': configs.INITIAL_CAPITAL,
                'transaction_cost_bps': configs.RL_TRANSACTION_COST_BPS, # Có thể dùng TRANSACTION_COST_BPS thông thường
                'lookback_window_size': configs.RL_LOOKBACK_WINDOW_SIZE,
                'rebalance_frequency_days': 1, # Không quan trọng cho việc chỉ get_state
                'financial_features': configs.RL_FINANCIAL_FEATURES,
                'prob_features': configs.RL_PROB_FEATURES,
                'financial_feature_means': fin_means_for_inference, # ĐÃ TẢI HOẶC TÍNH TỪ HUẤN LUYỆN
                'financial_feature_stds': fin_stds_for_inference   # ĐÃ TẢI HOẶC TÍNH TỪ HUẤN LUYỆN
            }
            temp_rl_env_for_state = rl_environment.PortfolioEnv(**inference_env_kwargs)
            
            current_rl_weights_for_state_input = np.zeros(temp_rl_env_for_state.num_assets + 1)
            current_rl_weights_for_state_input[-1] = 1.0

            # Xử lý initial rebalance
            target_date_for_state = actual_first_trading_day_in_pf_period
            try:
                date_series_for_lookup = temp_rl_env_for_state.prices_df.index
                # Tìm ngày giao dịch thực tế trong index, trước hoặc bằng target_date
                idx_pos = date_series_for_lookup.searchsorted(target_date_for_state, side='right') - 1
                if idx_pos < 0 : # Nếu target_date sớm hơn tất cả các ngày có trong index
                    # Thử lấy ngày đầu tiên có thể nếu lookback cho phép
                    if configs.RL_LOOKBACK_WINDOW_SIZE < len(date_series_for_lookup):
                        idx_for_state_in_temp_env = configs.RL_LOOKBACK_WINDOW_SIZE
                    else: # Không đủ dữ liệu ngay cả cho ngày đầu tiên + lookback
                        raise IndexError(f"Không đủ dữ liệu lịch sử trong all_prices_pivot để bắt đầu RL backtest tại {target_date_for_state}.")
                else:
                    idx_for_state_in_temp_env = idx_pos
                
                actual_date_used = date_series_for_lookup[idx_for_state_in_temp_env]
                logger.info(f"RL Initial State: Target date {target_date_for_state.date()}, Actual date used for state {actual_date_used.date()}")

                if idx_for_state_in_temp_env < configs.RL_LOOKBACK_WINDOW_SIZE:
                     logger.warning(f"Không đủ lịch sử nhìn lại ({configs.RL_LOOKBACK_WINDOW_SIZE} ngày) cho trạng thái RL ban đầu vào ngày {actual_date_used.date()}. "
                                    f"Có {idx_for_state_in_temp_env +1 if idx_for_state_in_temp_env >=0 else 0} điểm dữ liệu lịch sử.")
                
                temp_rl_env_for_state._current_prices_idx = idx_for_state_in_temp_env
                temp_rl_env_for_state.current_weights = current_rl_weights_for_state_input 
                temp_rl_env_for_state.portfolio_value = configs.INITIAL_CAPITAL
                
                current_obs_for_rl = temp_rl_env_for_state._get_state()
                predicted_raw_weights_rl_action = rl_optimizer.predict_rl_weights(trained_rl_model, current_obs_for_rl)

                if predicted_raw_weights_rl_action is None:
                    logger.error(f"RL: Không thể dự đoán tỷ trọng ban đầu. Bắt đầu bằng tiền mặt.")
                    target_weights_rl_assets_only = {}
                else:
                    target_weights_rl_assets_only = {
                        ticker: predicted_raw_weights_rl_action[i]
                        for i, ticker in enumerate(temp_rl_env_for_state.tickers)
                    }
                
                prices_initial_rl = all_prices_pivot.loc[actual_date_used] # Sử dụng actual_date_used để lấy giá
                if not prices_initial_rl.isnull().all():
                    rl_backtester.rebalance(actual_date_used, target_weights_rl_assets_only, prices_initial_rl) # Rebalance vào actual_date_used
                    if predicted_raw_weights_rl_action is not None:
                        current_rl_weights_for_state_input = predicted_raw_weights_rl_action.copy()
                else:
                    logger.warning(f"Tất cả giá đều NaN trong lần tái cân bằng RL ban đầu {actual_date_used.date()}. Bắt đầu bằng tiền mặt.")
                    rl_backtester.portfolio_history.append({'date': actual_date_used, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            
            except IndexError as e_idx_init: # Bắt lỗi từ searchsorted nếu idx_pos < 0 và không xử lý được
                 logger.error(f"Lỗi IndexError khi thiết lập trạng thái RL ban đầu gần {target_date_for_state}: {e_idx_init}. Bắt đầu bằng tiền mặt.")
                 rl_backtester.portfolio_history.append({'date': target_date_for_state, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            except Exception as e_init_rl:
                 logger.error(f"Lỗi không xác định khi thiết lập trạng thái RL ban đầu gần {target_date_for_state}: {e_init_rl}. Bắt đầu bằng tiền mặt.", exc_info=True)
                 rl_backtester.portfolio_history.append({'date': target_date_for_state, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})


            for rebal_date_target in rebalance_dates: # rebalance_dates là các ngày mục tiêu
                logger.info(f"Chiến lược RL: Đang xử lý tái cân bằng cho ngày mục tiêu {rebal_date_target.date()}...")
                try:
                    date_series_for_lookup = temp_rl_env_for_state.prices_df.index
                    idx_pos = date_series_for_lookup.searchsorted(rebal_date_target, side='right') - 1
                    if idx_pos < 0:
                        logger.warning(f"Không tìm thấy ngày giao dịch phù hợp cho state trước hoặc bằng {rebal_date_target.date()}. Bỏ qua rebalance này.")
                        if rl_backtester.portfolio_history: last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date_target; last_entry['returns'] = 0.0; rl_backtester.portfolio_history.append(last_entry)
                        continue
                    
                    actual_rebal_date_for_state = date_series_for_lookup[idx_pos]
                    idx_for_state_in_temp_env = idx_pos
                    logger.info(f"RL Rebalance State: Target date {rebal_date_target.date()}, Actual date used for state & rebalance {actual_rebal_date_for_state.date()}")
                    
                    if idx_for_state_in_temp_env < configs.RL_LOOKBACK_WINDOW_SIZE:
                        logger.warning(f"Không đủ lịch sử nhìn lại ({configs.RL_LOOKBACK_WINDOW_SIZE} ngày) cho trạng thái RL vào ngày {actual_rebal_date_for_state.date()}. "
                                       f"Có {idx_for_state_in_temp_env +1 if idx_for_state_in_temp_env >=0 else 0} điểm dữ liệu lịch sử.")
                    
                    temp_rl_env_for_state._current_prices_idx = idx_for_state_in_temp_env
                    last_hist_entry = rl_backtester.portfolio_history[-1] if rl_backtester.portfolio_history else None
                    if last_hist_entry: temp_rl_env_for_state.portfolio_value = last_hist_entry['value']
                    else: temp_rl_env_for_state.portfolio_value = configs.INITIAL_CAPITAL
                    temp_rl_env_for_state.current_weights = current_rl_weights_for_state_input
                    
                    current_obs_for_rl = temp_rl_env_for_state._get_state()
                    predicted_raw_weights_rl_action = rl_optimizer.predict_rl_weights(trained_rl_model, current_obs_for_rl)
                    
                    if predicted_raw_weights_rl_action is None:
                        logger.error(f"RL: Không thể dự đoán tỷ trọng cho {actual_rebal_date_for_state.date()}. Giữ nguyên danh mục.")
                        if rl_backtester.portfolio_history: last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = actual_rebal_date_for_state; last_entry['returns'] = 0.0; rl_backtester.portfolio_history.append(last_entry)
                        continue

                    target_weights_rl_assets_only = {
                        ticker: predicted_raw_weights_rl_action[i]
                        for i, ticker in enumerate(temp_rl_env_for_state.tickers)
                    }

                    prices_on_rebal_date = all_prices_pivot.loc[actual_rebal_date_for_state] # Dùng actual_rebal_date_for_state
                    if prices_on_rebal_date.isnull().all():
                        logger.warning(f"Tất cả giá đều NaN trong lần tái cân bằng RL {actual_rebal_date_for_state.date()}. Bỏ qua tái cân bằng bằng cách giữ nguyên.")
                        if rl_backtester.portfolio_history: last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = actual_rebal_date_for_state; last_entry['returns'] = 0.0; rl_backtester.portfolio_history.append(last_entry)
                        continue

                    rl_backtester.rebalance(actual_rebal_date_for_state, target_weights_rl_assets_only, prices_on_rebal_date) # Rebalance vào actual_rebal_date_for_state
                    current_rl_weights_for_state_input = predicted_raw_weights_rl_action.copy()

                except IndexError as e_idx_rebal:
                     logger.error(f"Lỗi IndexError khi lấy ngày cho trạng thái RL gần {rebal_date_target.date()}: {e_idx_rebal}. Giữ nguyên trạng thái trước đó.")
                     if rl_backtester.portfolio_history: last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date_target; last_entry['returns'] = 0.0; rl_backtester.portfolio_history.append(last_entry)
                     continue
                except Exception as e_rl_rebal:
                    logger.error(f"Lỗi không xác định trong quá trình tái cân bằng RL cho {rebal_date_target.date()}: {e_rl_rebal}", exc_info=True)
                    if rl_backtester.portfolio_history: last_entry = rl_backtester.portfolio_history[-1].copy(); last_entry['date'] = rebal_date_target; last_entry['returns'] = 0.0; rl_backtester.portfolio_history.append(last_entry)
                    continue

                if rl_backtester.portfolio_history:
                    rl_history_df = pd.DataFrame(rl_backtester.portfolio_history).set_index('date')
                    daily_rl_perf_df = update_portfolio_values_daily(rl_history_df, all_prices_pivot) # Your existing function
                    if daily_rl_perf_df is not None and not daily_rl_perf_df.empty and 'returns' in daily_rl_perf_df:
                        metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL) # Fresh instance for metrics
                        results_summary['RL_Strategy'] = metrics_calculator.calculate_metrics(portfolio_returns_series=daily_rl_perf_df['returns'].fillna(0))
                        all_portfolio_dfs['RL_Strategy'] = daily_rl_perf_df
                        try:
                            daily_rl_perf_df.to_csv(configs.RESULTS_DIR / f"{configs.RL_ALGORITHM.lower()}_performance_daily.csv")
                            logger.info(f"RL Strategy daily performance saved.")
                        except Exception as e_save_rl: logger.error(f"Error saving RL Strategy performance: {e_save_rl}")
                    else: logger.error("RL Strategy daily perf DF empty/missing returns.")
                else: logger.info("RL backtester history is empty.")

    # --- 4. Tổng hợp và In Kết quả ---
    if results_summary:
        logger.info("\n--- Portfolio Performance Summary ---")
        summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
        if not summary_df.empty:
            logger.info(f"\n{summary_df.to_string()}")
            try:
                configs.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(configs.RESULTS_DIR / "portfolio_strategies_summary.csv")
                logger.info(f"Portfolio strategies summary saved to {configs.RESULTS_DIR / 'portfolio_strategies_summary.csv'}")
            except Exception as e_save_summary:
                logger.error(f"Error saving portfolio summary: {e_save_summary}")
        else:
            logger.info("Summary DataFrame is empty.")

    else:
        logger.info("No portfolio strategies were successfully backtested to generate a summary.")

if __name__ == "__main__":
    run_portfolio_strategies()
    logger.info("run_portfolio_optimization.py script finished.")
