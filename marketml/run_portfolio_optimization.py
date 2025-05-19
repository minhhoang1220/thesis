# marketml/run_portfolio_optimization.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# ===== THÊM THƯ MỤC GỐC VÀO PATH =====
# Giả sử script này nằm trong .ndmh/marketml/
PROJECT_ROOT_SCRIPT = Path(__file__).resolve().parents[1] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_SCRIPT))

# === IMPORT MODULES ===
try:
    from marketml.configs import configs
    from marketml.log import setup # Giả sử bạn đã đặt environment_setup.py thành setup.py trong log
    from marketml.portfolio_opt import data_preparation, markowitz, backtesting
    # Thêm black_litterman sau:
    # from marketml.portfolio_opt import black_litterman
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
            daily_portfolio_df.at[current_date, 'value'] = rebal_entry['value'] # SỬ DỤNG .at[]
            daily_portfolio_df.at[current_date, 'cash'] = rebal_entry['cash']
            daily_portfolio_df.at[current_date, 'returns'] = rebal_entry['returns']
            current_weights = rebal_entry['weights'] if isinstance(rebal_entry['weights'], dict) else {}
            daily_portfolio_df.at[current_date, 'weights'] = current_weights # SỬ DỤNG .at[]
            
            last_known_portfolio_value = rebal_entry['value']
            last_known_cash = rebal_entry['cash']
            
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

    # --- 0. Kiểm tra hoặc Tạo Soft Signals ---
    generate_classification_signals_if_needed()

    # --- 1. Load và Chuẩn bị Dữ liệu Tổng Thể ---
    logger.info("Loading and preparing master data for portfolio optimization run...")
    all_prices_pivot = data_preparation.load_price_data_for_portfolio()
    if all_prices_pivot.empty: logger.error("Price data is empty. Exiting."); return
    
    # Loại bỏ các cột/tickers mà không có dữ liệu giá trong toàn bộ khoảng thời gian backtest
    # để tránh lỗi khi reindex mu, S
    portfolio_period_prices = all_prices_pivot.loc[pd.to_datetime(configs.PORTFOLIO_START_DATE):pd.to_datetime(configs.PORTFOLIO_END_DATE)]
    valid_tickers_in_period = portfolio_period_prices.dropna(axis=1, how='all').columns
    
    if len(valid_tickers_in_period) == 0:
        logger.error(f"No valid tickers found with price data in the portfolio period {configs.PORTFOLIO_START_DATE} to {configs.PORTFOLIO_END_DATE}. Exiting.")
        return
    
    if len(valid_tickers_in_period) < len(all_prices_pivot.columns):
        logger.warning(f"Original tickers count: {len(all_prices_pivot.columns)}. Valid tickers in period: {len(valid_tickers_in_period)}. Using subset: {valid_tickers_in_period.tolist()}")
        all_prices_pivot = all_prices_pivot[valid_tickers_in_period]
        if all_prices_pivot.empty: logger.error("Price data became empty after filtering for valid tickers in period. Exiting."); return

    all_daily_returns = data_preparation.calculate_returns(all_prices_pivot)
    if all_daily_returns.empty: logger.error("Daily returns are empty. Exiting."); return
    
    financial_data_full = data_preparation.load_financial_data_for_portfolio()
    classification_probs_full = data_preparation.load_classification_probabilities()

    # --- 2. Xác định Ngày Tái Cân Bằng ---
    trading_days = all_prices_pivot.loc[pd.to_datetime(configs.PORTFOLIO_START_DATE):pd.to_datetime(configs.PORTFOLIO_END_DATE)].index
    if trading_days.empty: logger.error("No trading days in portfolio period. Exiting."); return

    if isinstance(configs.REBALANCE_FREQUENCY, str):
        rebalance_dates_potential = pd.date_range(start=configs.PORTFOLIO_START_DATE, end=configs.PORTFOLIO_END_DATE, freq=configs.REBALANCE_FREQUENCY)
        rebalance_dates = trading_days[trading_days.searchsorted(rebalance_dates_potential, side='left')].unique()
        temp_rebalance_dates = []
        for r_date in rebalance_dates_potential:
            # Tìm ngày giao dịch đầu tiên >= r_date
            actual_rebal_day_candidates = trading_days[trading_days >= r_date]
            if not actual_rebal_day_candidates.empty:
                temp_rebalance_dates.append(actual_rebal_day_candidates[0])
        rebalance_dates = pd.DatetimeIndex(temp_rebalance_dates).unique()
    elif isinstance(configs.REBALANCE_FREQUENCY, int):
        rebalance_dates = trading_days[::configs.REBALANCE_FREQUENCY]
    else:
        logger.error(f"Invalid REBALANCE_FREQUENCY: {configs.REBALANCE_FREQUENCY}"); return
    
    if len(rebalance_dates) == 0: logger.error("No rebalance dates. Check config."); return
    rebalance_dates = rebalance_dates[rebalance_dates <= pd.to_datetime(configs.PORTFOLIO_END_DATE)] # Đảm bảo không vượt quá ngày cuối
    logger.info(f"Rebalance dates ({len(rebalance_dates)}): {rebalance_dates.strftime('%Y-%m-%d').tolist()}")

    results_summary = {}
    all_portfolio_dfs = {} # Để lưu daily dfs cho các chiến lược

    # --- Chiến lược 1: Markowitz ---
    logger.info("\n--- Running Markowitz Strategy ---")
    markowitz_backtester = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL, configs.TRANSACTION_COST_BPS)
    
    # Thêm một entry ban đầu tại ngày bắt đầu portfolio nếu ngày đó không phải là ngày rebalance đầu tiên
    # để `update_portfolio_values_daily` có điểm khởi đầu.
    first_portfolio_day = pd.to_datetime(configs.PORTFOLIO_START_DATE)
    # Tìm ngày giao dịch thực tế đầu tiên >= first_portfolio_day
    actual_first_trading_day_in_pf_period = trading_days[trading_days >= first_portfolio_day].min()

    if not rebalance_dates.empty and actual_first_trading_day_in_pf_period < rebalance_dates[0]:
         markowitz_backtester.portfolio_history.append({
            'date': actual_first_trading_day_in_pf_period, 
            'value': configs.INITIAL_CAPITAL,
            'weights': {}, 
            'cash': configs.INITIAL_CAPITAL, 
            'returns': 0.0
        })

    for rebal_date in rebalance_dates:
        logger.info(f"Markowitz: Processing rebalance for {rebal_date.date()}...")
        mu, S, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
            rebal_date, all_prices_pivot, all_daily_returns, None, None
        )
        mu_annual, S_annual, _, _ = data_preparation.get_prepared_data_for_rebalance_date(
            rebal_date, all_prices_pivot, all_daily_returns, None, None
        )
        if mu_annual.empty or S_annual.empty or mu_annual.isnull().all() or S_annual.isnull().all().all():
            logger.warning(f"Skipping Markowitz rebalance on {rebal_date.date()} due to empty or all-NaN annualized mu/S.")
            if markowitz_backtester.portfolio_history:
                last_entry = markowitz_backtester.portfolio_history[-1].copy()
                last_entry['date'] = rebal_date
                last_entry['returns'] = 0.0
                markowitz_backtester.portfolio_history.append(last_entry)
            elif not markowitz_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period : # Nếu là ngày đầu tiên và lỗi
                 markowitz_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            
            continue

        target_weights_mw = markowitz.optimize_markowitz_portfolio(mu_annual, S_annual) # Truyền mu và S hàng năm
        
        prices_on_rebal_date = all_prices_pivot.loc[rebal_date]
        if prices_on_rebal_date.isnull().all(): # Kiểm tra nếu tất cả giá là NaN
            logger.warning(f"All prices are NaN on rebalance date {rebal_date.date()}. Skipping rebalance.")
            if markowitz_backtester.portfolio_history:
                last_entry = markowitz_backtester.portfolio_history[-1].copy()
                last_entry['date'] = rebal_date
                last_entry['returns'] = 0.0
                markowitz_backtester.portfolio_history.append(last_entry)
            elif not markowitz_backtester.portfolio_history and rebal_date == actual_first_trading_day_in_pf_period :
                 markowitz_backtester.portfolio_history.append({'date': rebal_date, 'value': configs.INITIAL_CAPITAL, 'weights': {}, 'cash': configs.INITIAL_CAPITAL, 'returns': 0.0})
            continue

        markowitz_backtester.rebalance(rebal_date, target_weights_mw, prices_on_rebal_date)
    
    # Tạo DataFrame lịch sử từ list các dict
    if markowitz_backtester.portfolio_history:
        markowitz_history_df = pd.DataFrame(markowitz_backtester.portfolio_history).set_index('date')
    else: # Nếu không có entry nào trong history (ví dụ mọi rebalance đều skip)
        logger.error("Markowitz backtester history is empty. Cannot proceed.")
        markowitz_history_df = pd.DataFrame(columns=['value', 'cash', 'returns', 'weights'])

    # Sau vòng lặp rebalance, cập nhật giá trị hàng ngày
    daily_markowitz_perf_df = update_portfolio_values_daily(markowitz_history_df, all_prices_pivot)

    if daily_markowitz_perf_df is not None and not daily_markowitz_perf_df.empty and 'returns' in daily_markowitz_perf_df:
        # Tạo một instance Backtester MỚI chỉ để gọi calculate_metrics,
        # không làm thay đổi trạng thái của markowitz_backtester gốc dùng cho rebalancing.
        # Hoặc làm cho calculate_metrics là một static method hoặc hàm riêng.
        # Hiện tại, calculate_metrics trong class PortfolioBacktester đang dùng self.get_portfolio_performance_df()
        # nên chúng ta cần truyền returns series cho nó.
        metrics_calculator = backtesting.PortfolioBacktester(configs.INITIAL_CAPITAL) # Dummy instance
        results_summary['Markowitz'] = metrics_calculator.calculate_metrics(
            portfolio_returns_series=daily_markowitz_perf_df['returns'].fillna(0) # fillna ở đây cho chắc
        )
        all_portfolio_dfs['Markowitz'] = daily_markowitz_perf_df
        try:
            configs.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            daily_markowitz_perf_df.to_csv(configs.RESULTS_DIR / "markowitz_performance_daily.csv")
            logger.info(f"Markowitz daily performance saved to {configs.RESULTS_DIR / 'markowitz_performance_daily.csv'}")
        except Exception as e_save:
            logger.error(f"Error saving Markowitz performance: {e_save}")

    else:
        logger.error("Markowitz daily performance DataFrame is empty or missing 'returns' column after daily update.")


    # --- (Thêm Chiến lược Black-Litterman ở đây sau khi Markowitz chạy ổn) ---
    # ... Tương tự như Markowitz, nhưng gọi hàm BL ...


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
