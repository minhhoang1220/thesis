# marketml/portfolio_opt/data_preparation.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from pypfopt import risk_models

# --- Thêm project root vào sys.path ---
# Giả định data_preparation.py nằm trong .ndmh/marketml/portfolio_opt/
PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[2] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

try:
    from marketml.configs import configs
    from marketml.data.loader import preprocess # Để standardize_data
    # (Không cần import environment_setup ở đây nếu script này không chạy độc lập và không cần setup riêng)
except ImportError as e:
    print(f"CRITICAL ERROR in data_preparation.py: Could not import necessary modules. {e}")
    sys.exit(1)

# Lấy logger đã được thiết lập bởi script gọi (ví dụ: run_portfolio_optimization.py)
# Hoặc thiết lập một logger riêng nếu file này có thể chạy độc lập để test
logger = logging.getLogger(__name__)

def load_price_data_for_portfolio(
    custom_start_date_str: str = None,
    custom_end_date_str: str = None,
    include_buffer_for_rolling: bool = True
):
    """
    Load và pivot dữ liệu giá đã được enriched.
    Lọc theo ASSETS và khoảng thời gian cần thiết cho rolling calculations.
    """
    source_file = configs.PRICE_DATA_FOR_PORTFOLIO_PATH
    log_message_prefix = "Tải dữ liệu giá"
    if custom_start_date_str and custom_end_date_str:
        log_message_prefix = f"Tải dữ liệu giá tùy chỉnh ({custom_start_date_str} - {custom_end_date_str})"
    logger.info(f"{log_message_prefix} từ: {source_file}")


    try:
        df_price_full = pd.read_csv(source_file, parse_dates=['date'])
    except FileNotFoundError:
        logger.error(f"Tệp dữ liệu giá không tìm thấy: {source_file}")
        return pd.DataFrame()

    df_price = df_price_full[['date', 'ticker', 'close']].copy()

    if configs.PORTFOLIO_ASSETS:
        df_price = df_price[df_price['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()
        missing_assets = [asset for asset in configs.PORTFOLIO_ASSETS if asset not in df_price['ticker'].unique()]
        if missing_assets:
            logger.warning(f"Các tài sản không tìm thấy trong dữ liệu giá: {missing_assets}")

    # Xác định khoảng thời gian cần load
    if custom_start_date_str and custom_end_date_str:
        load_start_dt_config = pd.to_datetime(custom_start_date_str)
        load_end_dt_config = pd.to_datetime(custom_end_date_str)
        if include_buffer_for_rolling:
            # Buffer cho custom range, ví dụ, đủ cho RL_LOOKBACK_WINDOW_SIZE
            # Hoặc một buffer chung nhỏ hơn
            buffer_days_custom = configs.RL_LOOKBACK_WINDOW_SIZE + 15 # Thêm 15 ngày làm việc buffer
            load_start_dt = load_start_dt_config - pd.DateOffset(days=buffer_days_custom)
            load_end_dt = load_end_dt_config # Không thêm buffer vào ngày kết thúc của custom range
        else:
            load_start_dt = load_start_dt_config
            load_end_dt = load_end_dt_config
    else: # Sử dụng ngày từ config cho portfolio backtest (Markowitz, BL)
        load_start_dt_config = pd.to_datetime(configs.PORTFOLIO_START_DATE)
        load_end_dt_config = pd.to_datetime(configs.PORTFOLIO_END_DATE)
        if include_buffer_for_rolling:
            buffer_days = max(configs.ROLLING_WINDOW_COVARIANCE, configs.ROLLING_WINDOW_RETURNS) + 30
            load_start_dt = load_start_dt_config - pd.DateOffset(days=buffer_days * 1.5) # Nhân 1.5 để đảm bảo có đủ ngày làm việc
        else:
            load_start_dt = load_start_dt_config
        load_end_dt = load_end_dt_config

    # Đảm bảo load_start_date không sớm hơn dữ liệu có sẵn từ file gốc nếu cần
    # Điều này quan trọng nếu TIME_RANGE_START của bạn là ngày cụ thể
    # và bạn không muốn load dữ liệu trước ngày đó ngay cả khi buffer yêu cầu.
    min_available_date_overall = pd.to_datetime(configs.TIME_RANGE_START)
    load_start_dt = max(load_start_dt, min_available_date_overall)
    
    # Lọc theo ngày
    df_price_filtered = df_price[(df_price['date'] >= load_start_dt) & (df_price['date'] <= load_end_dt)].copy()

    if df_price_filtered.empty:
        logger.error(f"Không có dữ liệu giá sau khi lọc theo ngày ({load_start_dt.date()} - {load_end_dt.date()}) và tài sản.")
        return pd.DataFrame()

    df_price_pivot = df_price_filtered.pivot(index='date', columns='ticker', values='close')
    
    # Fill NaN: ffill trước để lấp các ngày nghỉ lễ, sau đó bfill để lấp các giá trị NaN ở đầu nếu có
    # Quan trọng: Nếu một mã không có dữ liệu ở đầu khoảng thời gian, bfill có thể kéo dữ liệu từ tương lai về.
    # Cân nhắc kỹ hơn về chiến lược fillna. ffill() là phổ biến.
    df_price_pivot = df_price_pivot.ffill()
    # Có thể drop các cột toàn NaN sau ffill nếu một mã không có dữ liệu nào trong toàn bộ khoảng đã lọc
    df_price_pivot.dropna(axis=1, how='all', inplace=True)
    # bfill sau đó có thể hữu ích nếu vẫn còn NaN ở đầu cho một số mã cụ thể, nhưng cần cẩn thận
    # df_price_pivot = df_price_pivot.bfill()


    logger.info(f"Dữ liệu giá cho '{'custom range' if custom_start_date_str else 'portfolio backtest'}' đã tải và pivot. "
                f"Shape: {df_price_pivot.shape}. "
                f"Từ: {df_price_pivot.index.min().date() if not df_price_pivot.empty else 'N/A'} "
                f"Đến: {df_price_pivot.index.max().date() if not df_price_pivot.empty else 'N/A'}")
    return df_price_pivot

def load_financial_data_for_portfolio():
    """
    Load và chuẩn hóa dữ liệu tài chính.
    """
    logger.info(f"Loading financial data from: {configs.FINANCIAL_DATA_FILE_PATH}")
    try:
        df_fin = pd.read_csv(configs.FINANCIAL_DATA_FILE_PATH)
    except FileNotFoundError:
        logger.error(f"Financial data file not found: {configs.FINANCIAL_DATA_FILE_PATH}")
        return None # Trả về None nếu không tìm thấy file

    df_fin = preprocess.standardize_data(df_fin) # Sử dụng hàm standardize bạn đã có
    # Đảm bảo cột 'year' là số nguyên hoặc có thể chuyển đổi
    try:
        # Chuyển đổi 'year' sang datetime, chỉ lấy năm, sau đó gán lại cho dễ so sánh
        df_fin['year_dt'] = pd.to_datetime(df_fin['year'], format='%Y').dt.year
    except Exception as e:
        logger.error(f"Error converting 'year' column in financial data to year integer: {e}")
        return None


    if configs.PORTFOLIO_ASSETS:
        df_fin = df_fin[df_fin['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()

    logger.info(f"Financial data loaded. Shape: {df_fin.shape if df_fin is not None else 'None'}")
    return df_fin

def load_classification_probabilities():
    """
    Load file xác suất dự báo từ mô hình classification.
    """
    logger.info(f"Loading classification probabilities from: {configs.CLASSIFICATION_PROBS_FILE_PATH}")
    try:
        df_probs = pd.read_csv(configs.CLASSIFICATION_PROBS_FILE_PATH, parse_dates=['date'])
    except FileNotFoundError:
        logger.warning(f"Classification probabilities file not found at {configs.CLASSIFICATION_PROBS_FILE_PATH}. Proceeding without it.")
        return None # Trả về None nếu không tìm thấy file

    if configs.PORTFOLIO_ASSETS:
        df_probs = df_probs[df_probs['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()

    prob_col_name = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}"
    if prob_col_name not in df_probs.columns:
        logger.warning(f"Expected probability column '{prob_col_name}' not found in {configs.CLASSIFICATION_PROBS_FILE_PATH}. Check SOFT_SIGNAL_MODEL_NAME.")
        # Có thể thử tìm cột prob_increase khác nếu tên không khớp chính xác
        found_prob_cols = [col for col in df_probs.columns if 'prob_increase' in col]
        if found_prob_cols:
            prob_col_name = found_prob_cols[0]
            logger.info(f"Using found probability column: '{prob_col_name}'")
        else:
            logger.error("No 'prob_increase' column found. Cannot use classification probabilities.")
            return None


    logger.info(f"Classification probabilities loaded. Shape: {df_probs.shape if df_probs is not None else 'None'}")
    return df_probs

def calculate_returns(price_df_pivot: pd.DataFrame):
    """
    Tính toán lợi nhuận hàng ngày từ DataFrame giá đã pivot.
    """
    if price_df_pivot.empty:
        logger.warning("Price data is empty, cannot calculate returns.")
        return pd.DataFrame()
    # Lợi nhuận hàng ngày
    daily_returns = price_df_pivot.pct_change()
    # Bỏ dòng NaN đầu tiên sau pct_change
    return daily_returns.iloc[1:]

def get_prepared_data_for_rebalance_date(
    rebalance_date: pd.Timestamp,
    all_prices_pivot: pd.DataFrame,
    all_daily_returns: pd.DataFrame,
    financial_data_full: pd.DataFrame = None,
    classification_probs_full: pd.DataFrame = None
):
    logger.debug(f"Preparing data for rebalance date: {rebalance_date.date()}")
    """
    Chuẩn bị dữ liệu cần thiết (mu, S, financial, probabilities) cho một ngày tái cân bằng cụ thể.
    """

    # --- 1. Expected Returns (mu) ---
    # Sử dụng historical mean returns với rolling window
    # Lấy dữ liệu returns đến NGÀY TRƯỚC rebalance_date
    returns_for_mu_calc = all_daily_returns[all_daily_returns.index < rebalance_date]
    
    mu_daily = pd.Series(dtype=float)
    if not returns_for_mu_calc.empty:
        if len(returns_for_mu_calc) >= configs.ROLLING_WINDOW_RETURNS:
            mu_daily = returns_for_mu_calc.iloc[-configs.ROLLING_WINDOW_RETURNS:].mean()
        else:
            mu_daily = returns_for_mu_calc.mean()
            logger.debug(f"Not enough data for ROLLING_WINDOW_RETURNS ({configs.ROLLING_WINDOW_RETURNS}) at {rebalance_date.date()}, using mean of {len(returns_for_mu_calc)} days for daily mu.")
    mu_daily = mu_daily.reindex(all_prices_pivot.columns, fill_value=0.0)

    # --- 2. Covariance Matrix (S) - DAILY ---
    returns_for_cov_calc = all_daily_returns[all_daily_returns.index < rebalance_date]
    S_daily = pd.DataFrame(index=all_prices_pivot.columns, columns=all_prices_pivot.columns, dtype=float)
    if not returns_for_cov_calc.empty:
        # Lấy dữ liệu returns cho cửa sổ rolling
        rolling_returns_for_cov = returns_for_cov_calc.iloc[-configs.ROLLING_WINDOW_COVARIANCE:]

        # Điều kiện để sử dụng shrinkage: số quan sát (ngày) >= số tài sản
        # PyPortfolioOpt's CovarianceShrinkage có thể xử lý trường hợp ít quan sát hơn,
        # nhưng sample_cov() truyền thống sẽ kém ổn định.
        if len(rolling_returns_for_cov) >= len(all_prices_pivot.columns) and len(rolling_returns_for_cov) >= configs.ROLLING_WINDOW_COVARIANCE * 0.8: # Cần đủ dữ liệu cho cửa sổ và ít nhất bằng số tài sản
            try:
                # Sử dụng Ledoit-Wolf shrinkage
                # `prices_data=False` vì chúng ta đang truyền returns_data
                # `frequency` ở đây là của dữ liệu input (daily returns), không phải để annualize
                # PyPortfolioOpt's risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
                # lại nhận prices_df, không phải returns.
                # Chúng ta sẽ dùng hàm trực tiếp từ pypfopt.risk_models nếu có,
                # hoặc dùng sample_cov rồi có thể áp dụng shrinkage nếu cần.
                # Cách đơn giản nhất là dùng sample_cov nếu muốn tự kiểm soát,
                # hoặc nếu muốn shrinkage, có thể dùng các hàm như `ledoit_wolf_single_factor`
                # nếu có returns.
                # Tuy nhiên, `risk_models.CovarianceShrinkage(prices_df_for_shrinkage).ledoit_wolf()`
                # là cách chuẩn của PyPortfolioOpt. Nó cần giá, không phải returns.
                # Vậy, chúng ta cần lấy slice giá tương ứng.
                
                # Lấy slice giá cho cửa sổ tính covariance
                prices_for_cov_shrinkage = all_prices_pivot.loc[rolling_returns_for_cov.index] # Lấy giá tại các ngày có returns
                # Đảm bảo không có cột nào toàn NaN trong slice giá này
                prices_for_cov_shrinkage_cleaned = prices_for_cov_shrinkage.dropna(axis=1, how='all')
                
                if not prices_for_cov_shrinkage_cleaned.empty and prices_for_cov_shrinkage_cleaned.shape[0] >= prices_for_cov_shrinkage_cleaned.shape[1]:
                    S_calculated_daily = risk_models.CovarianceShrinkage(prices_for_cov_shrinkage_cleaned, returns_data=False, frequency=252).ledoit_wolf()
                    # frequency=252 ở đây sẽ annualize S, chúng ta cần S daily trước
                    # Nên có thể dùng sample_cov() rồi sau đó tự annualize
                    # Hoặc nếu ledoit_wolf trả về S_annual, ta cần de-annualize nó.
                    # Để đơn giản và nhất quán: tính sample_cov daily, rồi annualize sau.
                    # Nếu muốn shrinkage, nên apply shrinkage lên S_daily.
                    # PyPortfolioOpt không có hàm shrinkage trực tiếp cho S_daily đã có.
                    # ---> QUAY LẠI SAMPLE_COV cho S_daily, sau đó có thể xem xét các phương pháp shrinkage phức tạp hơn nếu cần.
                    # Hiện tại, giữ sample_cov để đảm bảo logic rõ ràng.

                    logger.debug(f"Calculating sample covariance for {len(rolling_returns_for_cov)} days at {rebalance_date.date()}.")
                    S_calculated_daily = rolling_returns_for_cov.cov()

                else: # Fallback nếu không đủ dữ liệu cho shrinkage hoặc prices
                    logger.warning(f"Not enough data or valid prices for Covariance Shrinkage at {rebalance_date.date()}. Falling back to sample covariance or diagonal.")
                    if len(rolling_returns_for_cov) >= 2: # Cần ít nhất 2 điểm để tính cov
                        S_calculated_daily = rolling_returns_for_cov.cov()
                    else: # Không đủ để tính sample cov
                        S_calculated_daily = pd.DataFrame(np.diag(np.full(len(all_prices_pivot.columns), 1e-6)), index=all_prices_pivot.columns, columns=all_prices_pivot.columns)

            except Exception as e_cov:
                logger.error(f"Error calculating covariance at {rebalance_date.date()}: {e_cov}. Using diagonal matrix.")
                variances_daily = rolling_returns_for_cov.var().reindex(all_prices_pivot.columns, fill_value=1e-6)
                variances_daily[variances_daily <= 0] = 1e-6
                S_calculated_daily = pd.DataFrame(np.diag(variances_daily.values), index=all_prices_pivot.columns, columns=all_prices_pivot.columns)

            S_daily = S_calculated_daily.reindex(index=all_prices_pivot.columns, columns=all_prices_pivot.columns).fillna(0)
            for ticker in S_daily.index: # Đảm bảo đường chéo không phải là 0 và dương
                if pd.isna(S_daily.loc[ticker, ticker]) or S_daily.loc[ticker, ticker] <= 0 :
                    S_daily.loc[ticker, ticker] = 1e-6 # Giá trị dương rất nhỏ

        elif len(rolling_returns_for_cov) >= 2 : # Không đủ cho window dài nhưng đủ để tính sample cov
            logger.warning(f"Not enough data for ROLLING_WINDOW_COVARIANCE ({configs.ROLLING_WINDOW_COVARIANCE}) at {rebalance_date.date()} (have {len(rolling_returns_for_cov)} days). Using sample covariance on available data.")
            S_calculated_daily = rolling_returns_for_cov.cov()
            S_daily = S_calculated_daily.reindex(index=all_prices_pivot.columns, columns=all_prices_pivot.columns).fillna(0)
            for ticker in S_daily.index:
                if pd.isna(S_daily.loc[ticker, ticker]) or S_daily.loc[ticker, ticker] <= 0 :
                    S_daily.loc[ticker, ticker] = 1e-6
        else: # Không đủ dữ liệu để tính sample covariance (ví dụ, ít hơn 2 ngày returns)
            logger.warning(f"Not enough data (have {len(rolling_returns_for_cov)} days) to calculate any covariance at {rebalance_date.date()}. Using diagonal identity matrix.")
            S_daily = pd.DataFrame(np.diag(np.full(len(all_prices_pivot.columns), 1e-6)), index=all_prices_pivot.columns, columns=all_prices_pivot.columns)
    else: # Nếu returns_for_cov_calc rỗng (ngày đầu tiên của dữ liệu)
        logger.warning(f"No historical returns available to calculate covariance at {rebalance_date.date()}. Using diagonal identity matrix.")
        S_daily = pd.DataFrame(np.diag(np.full(len(all_prices_pivot.columns), 1e-6)), index=all_prices_pivot.columns, columns=all_prices_pivot.columns)

    # --- ANNUALIZE mu and S ---
    # PyPortfolioOpt thường kỳ vọng mu và S hàng năm nếu risk_free_rate là hàng năm.
    # Có thể để PyPortfolioOpt tự làm điều này nếu truyền frequency=252,
    # HOẶC chúng ta tự làm ở đây. Tự làm sẽ rõ ràng hơn.
    days_in_year = 252 # Số ngày giao dịch giả định trong một năm
    
    # Annualize mu: mu_annual = (1 + mu_daily_mean) ^ N - 1 (nếu mu_daily là arithmetic mean)
    # Hoặc đơn giản: mu_annual = mu_daily_mean * N
    # PyPortfolioOpt's mean_historical_return dùng compounding nếu returns_data=False
    # Chúng ta đã có daily returns, nên có thể tính annual compounded return
    # mu_annualized = (1 + mu_daily).pow(days_in_year) - 1 # Cách này không đúng nếu mu_daily là mean
    # Cách đúng hơn:
    # Nếu mu_daily là arithmetic mean:
    # mu_annualized = mu_daily * days_in_year # Arithmetic annualization
    # Nếu muốn geometric mean (thường tốt hơn cho returns dài hạn):
    # Cần tính geometric mean từ returns_for_mu_calc
    if not returns_for_mu_calc.empty:
        # (1+r1)*(1+r2)*...*(1+rN) = (1+Rg)^N => Rg = product(1+ri)^(1/N) - 1
        # For simplicity here, we'll use arithmetic annualization of the mean daily return
        mu_annualized = mu_daily * days_in_year
    else:
        mu_annualized = mu_daily # Vẫn là series rỗng hoặc toàn 0

    S_annualized = S_daily * days_in_year
    logger.debug(f"Annualized mu head for {rebalance_date.date()}: {mu_annualized.head().to_dict()}")


    # --- 3. Financial Data --- (Logic lấy năm Y-1 đã sửa ở lần trước)
    current_financial_data = {}
    if financial_data_full is not None and not financial_data_full.empty:
        target_financial_year = rebalance_date.year - 1
        logger.debug(f"Fetching financial data for year: {target_financial_year} (rebalance date: {rebalance_date.date()})")
        relevant_fin_data = financial_data_full[financial_data_full['year_dt'] == target_financial_year]
        if not relevant_fin_data.empty:
            current_financial_data = relevant_fin_data.set_index('ticker').to_dict('index')
        else:
            logger.warning(f"No financial data found for target year {target_financial_year} for rebalance date {rebalance_date.date()}. Financial data will be empty.")
    else:
        logger.debug(f"No financial data provided or it's empty for {rebalance_date.date()}.")

    # --- 4. Classification Probabilities ---
    current_classification_probs = {} 
    if classification_probs_full is not None and not classification_probs_full.empty:
        relevant_probs = classification_probs_full[classification_probs_full['date'] <= rebalance_date]
        if not relevant_probs.empty:
            latest_probs_records = relevant_probs.loc[relevant_probs.groupby('ticker')['date'].idxmax()]
            prob_col_name = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}"
            if prob_col_name not in latest_probs_records.columns:
                found_prob_cols = [col for col in latest_probs_records.columns if 'prob_increase' in col]
                if found_prob_cols: prob_col_name = found_prob_cols[0]
                else: prob_col_name = None
            if prob_col_name and prob_col_name in latest_probs_records.columns:
                current_classification_probs = latest_probs_records.set_index('ticker')[prob_col_name].to_dict()
            else:
                logger.warning(f"Could not find 'prob_increase' column in classification_probs for {rebalance_date.date()}.")
        for ticker in all_prices_pivot.columns:
            if ticker not in current_classification_probs:
                current_classification_probs[ticker] = 0.5 
    else:
        logger.debug(f"No classification probabilities provided or it's empty for {rebalance_date.date()}.")
        for ticker in all_prices_pivot.columns:
            current_classification_probs[ticker] = 0.5

    # Đảm bảo mu_annualized và S_annualized có cùng thứ tự cột/index
    common_tickers = S_annualized.index 
    
    mu_final = mu_annualized.reindex(common_tickers).fillna(0)
    S_final = S_annualized.reindex(index=common_tickers, columns=common_tickers).fillna(0)
    
    # Đảm bảo ma trận hiệp phương sai là positive semi-definite và đường chéo > 0
    # Cách đơn giản nhất là đảm bảo phương sai trên đường chéo là dương
    for ticker_idx in S_final.index:
        if S_final.loc[ticker_idx, ticker_idx] <= 0:
            S_final.loc[ticker_idx, ticker_idx] = 1e-6 # Một giá trị dương nhỏ
            # Đặt các hiệp phương sai của ticker này với các ticker khác về 0 nếu phương sai của nó là 0 (để tránh vấn đề)
            for other_ticker_idx in S_final.columns:
                if ticker_idx != other_ticker_idx:
                    S_final.loc[ticker_idx, other_ticker_idx] = 0
                    S_final.loc[other_ticker_idx, ticker_idx] = 0
    
    # Kiểm tra lại nếu mu_final có NaN sau reindex (ít khả năng nếu đã fill_value=0)
    mu_final.fillna(0, inplace=True)


    logger.debug(f"Data prepared for {rebalance_date.date()}: mu_annualized shape {mu_final.shape}, S_annualized shape {S_final.shape}, "
                 f"{len(current_financial_data)} financial records, {len(current_classification_probs)} probability records.")
    return mu_final, S_final, current_financial_data, current_classification_probs

if __name__ == '__main__':
    # --- Phần này để test nhanh các hàm trong file này ---
    logger.info("--- Testing data_preparation.py functions ---")
    
    # Thiết lập ngày giả định để test
    test_rebalance_date = pd.to_datetime(configs.PORTFOLIO_START_DATE) + pd.DateOffset(months=1)

    # Load dữ liệu
    prices_pivot = load_price_data_for_portfolio()
    if not prices_pivot.empty:
        daily_returns = calculate_returns(prices_pivot)
        financial_data = load_financial_data_for_portfolio()
        
        # Tạo file dummy classification_probabilities.csv nếu chưa có để test
        if not configs.CLASSIFICATION_PROBS_FILE_PATH.exists():
            logger.warning(f"Dummy CLASSIFICATION_PROBS_FILE_PATH: {configs.CLASSIFICATION_PROBS_FILE_PATH} not found. Creating a dummy one for testing.")
            dummy_dates = pd.date_range(start=configs.PORTFOLIO_START_DATE, end=configs.PORTFOLIO_END_DATE, freq='B')
            dummy_tickers = configs.PORTFOLIO_ASSETS if configs.PORTFOLIO_ASSETS else prices_pivot.columns.tolist()[:2]
            if not dummy_tickers: dummy_tickers = ['AAPL_dummy', 'MSFT_dummy'] # fallback
            
            dummy_data = []
            for date_entry in dummy_dates:
                for ticker_entry in dummy_tickers:
                    dummy_data.append({'date': date_entry, 'ticker': ticker_entry, f'prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}': np.random.rand()})
            if dummy_data:
                 pd.DataFrame(dummy_data).to_csv(configs.CLASSIFICATION_PROBS_FILE_PATH, index=False)
                 logger.info(f"Created a DUMMY classification_probabilities.csv for testing.")
            else:
                logger.error("Could not create dummy probability file because no tickers were defined.")

        classification_probs = load_classification_probabilities()

        logger.info(f"\n--- Testing get_prepared_data_for_rebalance_date for {test_rebalance_date.date()} ---")
        mu, S, fin_data, class_probs = get_prepared_data_for_rebalance_date(
            test_rebalance_date,
            prices_pivot,
            daily_returns,
            financial_data,
            classification_probs
        )

        if mu is not None and not mu.empty:
            logger.info(f"\nExpected Returns (mu) for {test_rebalance_date.date()}:\n{mu.head().to_string()}")
        if S is not None and not S.empty:
            logger.info(f"\nCovariance Matrix (S) for {test_rebalance_date.date()} (sample):\n{S.iloc[:5, :5].to_string()}")
        if fin_data:
            logger.info(f"\nFinancial Data sample for {test_rebalance_date.date()}: First 2 items: {list(fin_data.items())[:2]}")
        if class_probs:
            logger.info(f"\nClassification Probs sample for {test_rebalance_date.date()}: First 5 items: {list(class_probs.items())[:5]}")
    else:
        logger.error("Could not load price data for testing.")
    logger.info("--- Finished testing data_preparation.py functions ---")