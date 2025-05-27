# marketml/portfolio_opt/markowitz.py
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, objective_functions
import logging

try:
    from marketml.configs import configs
except ImportError:
    print("CRITICAL ERROR in markowitz.py: Could not import 'marketml.configs'.")
    raise

logger = logging.getLogger(__name__)

def optimize_markowitz_portfolio(mu: pd.Series, S: pd.DataFrame, logger_instance: logging.Logger = None) -> dict:
    """
    Optimizes a Markowitz portfolio based on expected returns and covariance matrix.

    Args:
        mu (pd.Series): Expected annualized returns, indexed by ticker.
        S (pd.DataFrame): Annualized covariance matrix, indexed/columned by ticker.
        logger_instance (logging.Logger, optional): Logger instance to use.

    Returns:
        dict: Dictionary of optimal weights {ticker: weight}, or equal weights on failure.
    """
    current_logger = logger_instance if logger_instance else logger

    if mu is None or S is None or mu.empty or S.empty:
        current_logger.error("Expected returns (mu) or Covariance matrix (S) is missing or empty for Markowitz optimization.")
        num_assets = 0
        asset_list = []
        if S is not None and not S.empty:
            asset_list = S.columns.tolist()
            num_assets = len(asset_list)
        elif mu is not None and not mu.empty:
            asset_list = mu.index.tolist()
            num_assets = len(asset_list)
        
        if num_assets > 0:
            current_logger.warning(f"Defaulting to equal weights for {num_assets} assets due to missing mu/S.")
            return {ticker: 1.0 / num_assets for ticker in asset_list}
        else:
            current_logger.error("Cannot determine number of assets for default equal weights as mu and S are unusable.")
            return {}

    # Align mu and S to common tickers and handle NaNs
    common_tickers = mu.index.intersection(S.index).unique()
    if not common_tickers.any():
        current_logger.error("No common tickers found between mu and S. Cannot optimize.")
        return {ticker: 1.0 / len(mu.index) for ticker in mu.index} if not mu.empty else {}


    mu_cleaned = mu.loc[common_tickers].fillna(0.0)
    S_cleaned = S.loc[common_tickers, common_tickers].fillna(0.0)

    # Ensure diagonal of S_cleaned is positive
    for ticker_idx in S_cleaned.index:
        if S_cleaned.loc[ticker_idx, ticker_idx] <= 1e-9: # Use a small epsilon
            S_cleaned.loc[ticker_idx, ticker_idx] = 1e-9
            # Optionally zero out off-diagonal elements for this ticker if variance was ~0
            # for other_ticker_idx in S_cleaned.columns:
            #     if ticker_idx != other_ticker_idx:
            #         S_cleaned.loc[ticker_idx, other_ticker_idx] = 0.0
            #         S_cleaned.loc[other_ticker_idx, ticker_idx] = 0.0


    if mu_cleaned.empty or S_cleaned.empty or S_cleaned.shape[0] != S_cleaned.shape[1] or len(mu_cleaned) != len(S_cleaned):
        current_logger.error(f"mu_cleaned (len {len(mu_cleaned)}) or S_cleaned (shape {S_cleaned.shape}) "
                             f"is empty or mismatched after cleaning. Cannot optimize.")
        # Fallback to equal weights based on available tickers in mu_cleaned
        num_assets_fallback = len(mu_cleaned.index)
        if num_assets_fallback > 0:
            return {ticker: 1.0 / num_assets_fallback for ticker in mu_cleaned.index}
        return {}
        
    current_logger.debug(f"Optimizing Markowitz for {len(mu_cleaned)} assets. Weight bounds: {configs.MARKOWITZ_WEIGHT_BOUNDS}")

    try:
        ef = EfficientFrontier(mu_cleaned, S_cleaned, weight_bounds=configs.MARKOWITZ_WEIGHT_BOUNDS)
        
        # Optional L2 regularization
        if hasattr(configs, 'MARKOWITZ_L2_REG_GAMMA') and configs.MARKOWITZ_L2_REG_GAMMA > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=configs.MARKOWITZ_L2_REG_GAMMA)
            current_logger.debug(f"Added L2 regularization with gamma={configs.MARKOWITZ_L2_REG_GAMMA}")

        objective_choice = configs.MARKOWITZ_OBJECTIVE.lower()
        if objective_choice == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        elif objective_choice == 'min_volatility':
            weights = ef.min_volatility()
        # Add other objectives like efficient_risk or efficient_return if needed,
        # checking for corresponding target_volatility/target_return in configs
        else:
            current_logger.warning(f"Markowitz objective '{configs.MARKOWITZ_OBJECTIVE}' not recognized or fully specified. Defaulting to 'max_sharpe'.")
            weights = ef.max_sharpe(risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        
        cleaned_weights = ef.clean_weights()
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        current_logger.info(f"Markowitz optimization successful. Portfolio performance (expected): AnnReturn={ef.portfolio_performance()[0]:.2%}, AnnVol={ef.portfolio_performance()[1]:.2%}, Sharpe={ef.portfolio_performance()[2]:.2f}")
        current_logger.debug(f"Optimized Markowitz weights: {cleaned_weights}")
        return cleaned_weights

    except Exception as e:
        current_logger.error(f"Error during Markowitz optimization with PyPortfolioOpt: {e}", exc_info=True)
        current_logger.debug(f"Mu values at time of error for Markowitz: {mu_cleaned.to_dict()}")
        current_logger.debug(f"Covariance matrix S at time of error for Markowitz (sample):\n{S_cleaned.iloc[:min(5, S_cleaned.shape[0]), :min(5, S_cleaned.shape[1])].to_string()}")
        
        num_assets_error_fallback = len(mu_cleaned.index)
        current_logger.warning(f"Defaulting to equal weights for {num_assets_error_fallback} assets due to optimization error.")
        return {ticker: 1.0 / num_assets_error_fallback for ticker in mu_cleaned.index}