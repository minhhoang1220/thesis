# marketml/portfolio_opt/markowitz.py
import pandas as pd
from pypfopt import EfficientFrontier, objective_functions
# from pypfopt.expected_returns import mean_historical_return # Bạn đã tính mu ở data_preparation
# from pypfopt.risk_models import CovarianceShrinkage # Bạn đã tính S ở data_preparation
import logging

# --- Thêm project root vào sys.path ---
from pathlib import Path
import sys
PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[2] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

try:
    from marketml.configs import configs
except ImportError as e:
    print(f"CRITICAL ERROR in markowitz.py: Could not import configs. {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

def optimize_markowitz_portfolio(mu: pd.Series, S: pd.DataFrame):
    """
    Tối ưu hóa danh mục Markowitz.
    mu: Expected returns (Series, index là tickers)
    S: Covariance matrix (DataFrame, index/columns là tickers)
    """
    if mu is None or S is None or mu.empty or S.empty:
        logger.error("Expected returns (mu) or Covariance matrix (S) is missing or empty for Markowitz.")
        # Trả về trọng số đều nếu lỗi và có thông tin về số lượng tài sản từ S hoặc mu
        num_assets = 0
        asset_list = []
        if S is not None and not S.empty:
            num_assets = len(S.columns)
            asset_list = S.columns.tolist()
        elif mu is not None and not mu.empty:
            num_assets = len(mu.index)
            asset_list = mu.index.tolist()
        
        if num_assets > 0:
            logger.warning(f"Defaulting to equal weights for {num_assets} assets.")
            return {ticker: 1.0/num_assets for ticker in asset_list}
        else:
            logger.error("Cannot determine number of assets for default equal weights.")
            return {}


    # Đảm bảo mu và S có cùng index/columns và không có NaN nào ảnh hưởng đến EF
    common_tickers = mu.index.intersection(S.index)
    mu_cleaned = mu.loc[common_tickers].fillna(0) # Điền 0 cho mu nếu có NaN sau khi align
    S_cleaned = S.loc[common_tickers, common_tickers].fillna(0)
    for ticker in S_cleaned.index: # Đảm bảo đường chéo không phải là 0
        if S_cleaned.loc[ticker, ticker] == 0: S_cleaned.loc[ticker, ticker] = 1e-6


    if mu_cleaned.empty or S_cleaned.empty or len(mu_cleaned) != len(S_cleaned):
        logger.error(f"mu_cleaned (len {len(mu_cleaned)}) or S_cleaned (shape {S_cleaned.shape}) is empty or mismatched after cleaning. Cannot optimize.")
        num_assets = len(configs.PORTFOLIO_ASSETS) if configs.PORTFOLIO_ASSETS else 15 # Fallback
        return {ticker: 1.0/num_assets for ticker in (configs.PORTFOLIO_ASSETS if configs.PORTFOLIO_ASSETS else mu.index)}


    ef = EfficientFrontier(mu_cleaned, S_cleaned, weight_bounds=configs.MARKOWITZ_WEIGHT_BOUNDS)

    try:
        # Thêm L2 regularization để giúp ổn định trọng số, đặc biệt khi có nhiều tài sản
        # hoặc khi ma trận hiệp phương sai gần suy biến.
        # lambda_val = 0.1 # Giá trị lambda cho L2 reg, có thể đưa vào configs
        # ef.add_objective(objective_functions.L2_reg, gamma=lambda_val)

        if configs.MARKOWITZ_OBJECTIVE == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        elif configs.MARKOWITZ_OBJECTIVE == 'min_volatility':
            weights = ef.min_volatility()
        # Thêm các mục tiêu khác nếu cần (efficient_risk, efficient_return)
        # elif configs.MARKOWITZ_OBJECTIVE == 'efficient_risk' and configs.MARKOWITZ_TARGET_VOLATILITY is not None:
        #     weights = ef.efficient_risk(target_volatility=configs.MARKOWITZ_TARGET_VOLATILITY)
        # elif configs.MARKOWITZ_OBJECTIVE == 'efficient_return' and configs.MARKOWITZ_TARGET_RETURN is not None:
        #     weights = ef.efficient_return(target_return=configs.MARKOWITZ_TARGET_RETURN)
        else:
            logger.warning(f"Markowitz objective '{configs.MARKOWITZ_OBJECTIVE}' not recognized. Defaulting to max_sharpe.")
            weights = ef.max_sharpe(risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        
        cleaned_weights = ef.clean_weights() # Làm sạch trọng số (ví dụ: làm tròn số rất nhỏ về 0)
        logger.debug(f"Optimized Markowitz weights: {cleaned_weights}")
        return cleaned_weights
    except Exception as e:
        logger.error(f"Error during Markowitz optimization with PyPortfolioOpt: {e}")
        # Trả về trọng số đều nếu lỗi
        num_assets = len(mu_cleaned.index)
        logger.warning(f"Defaulting to equal weights for {num_assets} assets due to optimization error.")
        return {ticker: 1.0/num_assets for ticker in mu_cleaned.index}