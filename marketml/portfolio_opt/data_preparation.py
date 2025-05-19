# marketml/portfolio_opt/data_preparation.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

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

def load_price_data_for_portfolio():
    """
    Load và pivot dữ liệu giá đã được enriched.
    Lọc theo ASSETS và khoảng thời gian cần thiết cho rolling calculations.
    """
    logger.info(f"Loading price data for portfolio from: {configs.PRICE_DATA_FOR_PORTFOLIO_PATH}")
    try:
        df_price_full = pd.read_csv(configs.PRICE_DATA_FOR_PORTFOLIO_PATH, parse_dates=['date'])
    except FileNotFoundError:
        logger.error(f"Price data file not found: {configs.PRICE_DATA_FOR_PORTFOLIO_PATH}")
        return pd.DataFrame()

    # Chỉ giữ các cột cần thiết ban đầu
    df_price = df_price_full[['date', 'ticker', 'close']].copy() # Thêm 'volume' nếu cần sau này

    # Lọc theo danh sách ASSETS trong configs
    if configs.PORTFOLIO_ASSETS:
        df_price = df_price[df_price['ticker'].isin(configs.PORTFOLIO_ASSETS)].copy()
        # Kiểm tra xem tất cả assets có trong dữ liệu không
        missing_assets = [asset for asset in configs.PORTFOLIO_ASSETS if asset not in df_price['ticker'].unique()]
        if missing_assets:
            logger.warning(f"Assets not found in price data: {missing_assets}")

    # Xác định khoảng thời gian cần load (bao gồm cả dữ liệu lịch sử cho rolling)
    # Lấy thêm dữ liệu trước PORTFOLIO_START_DATE để tính rolling covariance/returns
    # Ví dụ: cần ít nhất ROLLING_WINDOW_COVARIANCE ngày trước đó
    buffer_days = max(configs.ROLLING_WINDOW_COVARIANCE, configs.ROLLING_WINDOW_RETURNS) + 30 # Thêm buffer
    # Chuyển đổi ngày bắt đầu và kết thúc sang datetime
    portfolio_start_dt = pd.to_datetime(configs.PORTFOLIO_START_DATE)
    portfolio_end_dt = pd.to_datetime(configs.PORTFOLIO_END_DATE)
    
    load_start_date = portfolio_start_dt - pd.DateOffset(days=buffer_days * 1.5) # Nhân 1.5 để đảm bảo đủ ngày làm việc
    # Đảm bảo load_start_date không sớm hơn dữ liệu có sẵn
    min_available_date = pd.to_datetime(configs.TIME_RANGE_START)
    load_start_date = max(load_start_date, min_available_date)

    df_price = df_price[(df_price['date'] >= load_start_date) & (df_price['date'] <= portfolio_end_dt)].copy()

    if df_price.empty:
        logger.error("No price data found after filtering by date and assets.")
        return pd.DataFrame()

    # Pivot để mỗi cột là một ticker, index là date (dùng cho giá đóng cửa)
    df_price_pivot = df_price.pivot(index='date', columns='ticker', values='close')
    
    # Fill NaN bằng ffill rồi bfill để xử lý các ngày nghỉ ngẫu nhiên của từng mã
    df_price_pivot = df_price_pivot.ffill().bfill()
    # Sau đó drop các cột (tickers) mà vẫn còn toàn NaN (nếu có mã không có dữ liệu nào)
    df_price_pivot.dropna(axis=1, how='all', inplace=True)

    logger.info(f"Price data loaded and pivoted. Shape: {df_price_pivot.shape}")
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
        if len(returns_for_cov_calc) >= configs.ROLLING_WINDOW_COVARIANCE:
            S_calculated_daily = returns_for_cov_calc.iloc[-configs.ROLLING_WINDOW_COVARIANCE:].cov()
            S_daily = S_calculated_daily.reindex(index=all_prices_pivot.columns, columns=all_prices_pivot.columns).fillna(0)
            for ticker in S_daily.index:
                if S_daily.loc[ticker, ticker] == 0 : S_daily.loc[ticker, ticker] = 1e-6
        else:
            logger.warning(f"Not enough data for ROLLING_WINDOW_COVARIANCE ({configs.ROLLING_WINDOW_COVARIANCE}) at {rebalance_date.date()} for daily S. Using diagonal matrix.")
            variances_daily = returns_for_cov_calc.var().reindex(all_prices_pivot.columns, fill_value=1e-6)
            variances_daily[variances_daily <= 0] = 1e-6
            S_daily = pd.DataFrame(np.diag(variances_daily.values), index=all_prices_pivot.columns, columns=all_prices_pivot.columns)
    else:
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
    common_tickers = all_prices_pivot.columns.intersection(mu_annualized.index).intersection(S_annualized.index)
    if len(common_tickers) < len(all_prices_pivot.columns):
        logger.warning(f"Some tickers are missing from calculated annualized mu/S for {rebalance_date.date()}. Using intersection: {len(common_tickers)} tickers.")
    
    mu_final = mu_annualized.reindex(common_tickers).fillna(0)
    S_final = S_annualized.reindex(index=common_tickers, columns=common_tickers).fillna(0)
    for ticker in S_final.index:
        if S_final.loc[ticker, ticker] == 0: S_final.loc[ticker, ticker] = 1e-6 # Cho phương sai rất nhỏ nếu = 0

    logger.debug(f"Data prepared for {rebalance_date.date()}: mu_annualized shape {mu_final.shape}, S_annualized shape {S_final.shape}, "
                 f"{len(current_financial_data)} financial records, {len(current_classification_probs)} probability records.")
    return mu_final, S_final, current_financial_data, current_classification_probs # Trả về mu và S đã được annualize

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