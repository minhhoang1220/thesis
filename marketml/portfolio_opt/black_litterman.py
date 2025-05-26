# marketml/portfolio_opt/black_litterman.py
import pandas as pd
import numpy as np
import logging
# import sys # XÓA DÒNG NÀY
# from pathlib import Path # Không cần nếu sys.path bị xóa

try:
    from marketml.configs import configs
    from pypfopt import BlackLittermanModel, risk_models, expected_returns
except ImportError:
    print("CRITICAL ERROR in black_litterman.py: Could not import 'marketml.configs' or 'pypfopt'.")
    raise

logger = logging.getLogger(__name__)

def _get_omega_from_confidence(confidence_level: float) -> float:
    """Helper to get omega value based on confidence from configs."""
    # Ensure BL_OMEGA_CONF_MAPPING is sorted by threshold descending if using loop below
    # Default omega if no threshold is met (should be the last item with threshold 0.0)
    default_omega = 0.05 # Fallback if mapping is not configured
    if hasattr(configs, 'BL_OMEGA_CONF_MAPPING') and configs.BL_OMEGA_CONF_MAPPING:
        for threshold, omega_val in sorted(configs.BL_OMEGA_CONF_MAPPING, key=lambda x: x[0], reverse=True):
            if confidence_level >= threshold: # Use >= for thresholds
                return omega_val
        return configs.BL_OMEGA_CONF_MAPPING[-1][1] # Return omega for lowest threshold if none met
    return default_omega


def generate_views_from_signals(
    asset_tickers: list,
    current_financial_data: dict, # {ticker: {fin_metric: value}}
    current_classification_probs: dict, # {ticker: prob_increase}
    pi_prior_annualized: pd.Series, # Indexed by ticker
    rebalance_date: pd.Timestamp,
    logger_instance: logging.Logger = None # Optional logger instance
):
    """
    Generates P, Q, Omega matrices for Black-Litterman based on financial data and classification probabilities.
    """
    # Use passed logger or module logger
    current_logger = logger_instance if logger_instance else logger

    views_P_list = []
    views_Q_list = []
    omega_values_for_diag = [] # Will store variances for Omega diagonal

    num_assets = len(asset_tickers)
    if num_assets == 0:
        current_logger.warning(f"No asset tickers provided for view generation on {rebalance_date.date()}.")
        return None, None, None

    for i, ticker in enumerate(asset_tickers):
        prob_increase = current_classification_probs.get(ticker, 0.5) # Default to neutral if no prob
        fin_data_for_ticker = current_financial_data.get(ticker, {})
        roe = fin_data_for_ticker.get('roe', 0.0) # Default ROE to 0 if not found
        if isinstance(roe, str): # Try to convert ROE if it's a string
            try: roe = float(roe)
            except ValueError: roe = 0.0; current_logger.warning(f"Could not convert ROE '{fin_data_for_ticker.get('roe')}' to float for {ticker}.")


        # --- View 1: Based on Probability of Price Increase ---
        if prob_increase > configs.BL_PROB_THRESHOLD_STRONG_VIEW:
            p_vector = np.zeros(num_assets)
            p_vector[i] = 1.0 # View is on this specific asset
            
            # Q: Absolute view on expected annual return
            # Scale outperformance based on probability strength relative to neutral (0.5)
            # Max outperformance is BL_EXPECTED_OUTPERFORMANCE_STRONG when prob_increase = 1.0
            # Avoid division by zero if BL_PROB_THRESHOLD_STRONG_VIEW is close to 0.5
            denominator_scaling = (1.0 - 0.5) # Max possible deviation from neutral
            if denominator_scaling < 1e-6 : denominator_scaling = 1e-6 # prevent division by zero
            
            scaled_outperformance = configs.BL_EXPECTED_OUTPERFORMANCE_STRONG * \
                                    ((prob_increase - 0.5) / denominator_scaling)
            q_value = configs.MARKOWITZ_RISK_FREE_RATE + scaled_outperformance # View is an absolute expected return
            
            views_P_list.append(p_vector)
            views_Q_list.append(q_value)
            
            conf_prob_view = configs.BL_VIEW_CONFIDENCE_STRONG # Confidence for this type of view
            omega_val_prob = _get_omega_from_confidence(conf_prob_view) # Variance of view error
            omega_values_for_diag.append(omega_val_prob)
            current_logger.debug(f"BL View (Prob Increase) for {ticker} on {rebalance_date.date()}: "
                                 f"Prob={prob_increase:.2f}, Q={q_value:.4f}, Omega_diag_val={omega_val_prob:.6f}")

        # --- View 2: Based on High ROE (Example) ---
        # Ensure ROE threshold is in configs
        roe_threshold_config = getattr(configs, 'BL_ROE_HIGH_THRESHOLD', 20.0) # Default to 20%
        roe_outperform_config = getattr(configs, 'BL_ROE_OUTPERFORM_EXPECTATION', 0.02) # Default 2%
        roe_view_confidence = getattr(configs, 'BL_ROE_VIEW_CONFIDENCE', 0.6) # Default 60% confidence

        if roe > roe_threshold_config:
            p_vector_roe = np.zeros(num_assets)
            p_vector_roe[i] = 1.0
            
            # Q: Asset with high ROE is expected to outperform its prior by a certain amount
            prior_return_for_ticker = pi_prior_annualized.get(ticker, configs.MARKOWITZ_RISK_FREE_RATE) # Use prior or risk-free
            q_value_roe = prior_return_for_ticker + roe_outperform_config
            
            views_P_list.append(p_vector_roe)
            views_Q_list.append(q_value_roe)
            
            omega_val_roe = _get_omega_from_confidence(roe_view_confidence)
            omega_values_for_diag.append(omega_val_roe)
            current_logger.debug(f"BL View (High ROE) for {ticker} on {rebalance_date.date()}: "
                                 f"ROE={roe:.2f}, Q={q_value_roe:.4f}, Omega_diag_val={omega_val_roe:.6f}")

    if not views_P_list:
        current_logger.info(f"No specific views generated on {rebalance_date.date()} based on current signals and thresholds.")
        return None, None, None

    P_matrix = np.array(views_P_list)
    Q_vector = np.array(views_Q_list)
    # Omega is a diagonal matrix representing the uncertainty of each view (variance of view errors)
    Omega_matrix = np.diag(omega_values_for_diag)

    current_logger.debug(f"Generated P_matrix ({P_matrix.shape}) for {rebalance_date.date()}")
    current_logger.debug(f"Generated Q_vector ({Q_vector.shape}) for {rebalance_date.date()}")
    current_logger.debug(f"Generated Omega_matrix ({Omega_matrix.shape}) for {rebalance_date.date()}")

    # Dimensionality check
    if not (P_matrix.shape[0] == Q_vector.shape[0] == Omega_matrix.shape[0] == Omega_matrix.shape[1]):
        current_logger.error(f"Dimension mismatch in P, Q, or Omega for {rebalance_date.date()}. "
                             f"P:{P_matrix.shape}, Q:{Q_vector.shape}, Omega:{Omega_matrix.shape}. Cannot proceed with these views.")
        return None, None, None
        
    return P_matrix, Q_vector, Omega_matrix


def get_black_litterman_posterior_estimates(
    historical_prices_for_prior: pd.DataFrame,
    asset_tickers_order: list,
    market_caps: dict = None, # {ticker: market_cap_value}
    views_P: np.ndarray = None,
    views_Q: np.ndarray = None,
    views_Omega: np.ndarray = None, # Uncertainty matrix for views
    logger_instance: logging.Logger = None # Optional logger instance
):
    """
    Calculates Black-Litterman posterior expected returns and covariance matrix.
    """
    current_logger = logger_instance if logger_instance else logger

    if historical_prices_for_prior.empty or len(historical_prices_for_prior) < 2:
        current_logger.error("Historical prices are empty or insufficient for Black-Litterman prior calculation.")
        return None, None
    if not asset_tickers_order:
        current_logger.error("asset_tickers_order is empty. Cannot proceed.")
        return None, None

    # Ensure historical_prices_for_prior has columns in the specified order and no all-NaN columns
    try:
        historical_prices_for_prior_reordered = historical_prices_for_prior[asset_tickers_order].copy()
        historical_prices_for_prior_reordered.dropna(axis=1, how='all', inplace=True)
        if historical_prices_for_prior_reordered.shape[1] != len(asset_tickers_order):
            current_logger.warning("Some asset tickers were dropped from historical_prices_for_prior due to all NaNs. "
                                   "The Black-Litterman model will run on the reduced set of assets.")
            # Update asset_tickers_order to reflect actual columns used
            asset_tickers_order = historical_prices_for_prior_reordered.columns.tolist()
            if not asset_tickers_order:
                 current_logger.error("All asset tickers dropped from historical_prices_for_prior. Cannot calculate priors.")
                 return None, None
    except KeyError as e_key:
        current_logger.error(f"One or more tickers in asset_tickers_order not found in historical_prices_for_prior columns: {e_key}")
        return None, None

    # 1. Calculate S_prior (Annualized Covariance Matrix)
    try:
        # Use Ledoit-Wolf shrinkage for a more stable covariance matrix
        S_prior = risk_models.CovarianceShrinkage(historical_prices_for_prior_reordered, frequency=252).ledoit_wolf()
        # Ensure S_prior has the correct order and fill any potential NaNs (e.g., if a ticker had constant price)
        S_prior = S_prior.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0)
        for ticker in S_prior.index: # Ensure diagonal elements (variances) are positive
             if S_prior.loc[ticker, ticker] <= 1e-9: S_prior.loc[ticker, ticker] = 1e-9 # Small positive variance
        current_logger.debug(f"S_prior (annualized) calculated. Shape: {S_prior.shape}")
    except Exception as e_S_prior:
        current_logger.error(f"Error calculating S_prior for Black-Litterman: {e_S_prior}", exc_info=True)
        return None, None

    # 2. Calculate pi_prior (Market-implied equilibrium returns - Annualized)
    pi_prior = None
    try:
        if market_caps:
            # Ensure market_caps are ordered and available for all assets in S_prior
            ordered_market_caps_series = pd.Series({ticker: market_caps.get(ticker, 1e9) for ticker in asset_tickers_order}) # Default large cap if missing
            # PyPortfolioOpt expects market_caps as a Series or dict matching S_prior.index
            delta = expected_returns.market_implied_risk_aversion(S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
            pi_prior = expected_returns.market_implied_prior_returns(ordered_market_caps_series, delta, S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
            current_logger.debug("pi_prior calculated using market-implied risk aversion.")
        else:
            current_logger.warning("No market caps provided for Black-Litterman. Using mean historical returns as prior pi (annualized).")
            pi_prior = expected_returns.mean_historical_return(historical_prices_for_prior_reordered, frequency=252)
        
        pi_prior = pi_prior.reindex(asset_tickers_order).fillna(0) # Ensure order and fill NaNs
        current_logger.debug(f"pi_prior (annualized) calculated. Head: {pi_prior.head().to_dict()}")

    except Exception as e_pi_prior:
        current_logger.error(f"Error calculating pi_prior for Black-Litterman: {e_pi_prior}", exc_info=True)
        # Return S_prior as it might still be useful, pi_prior failed
        return None, S_prior # Or (None, None) if S_prior is also compromised

    # If no views, Black-Litterman model returns the prior estimates
    if views_P is None or views_Q is None or views_P.size == 0:
        current_logger.info("No views provided or views are empty. Black-Litterman model will return prior estimates.")
        return pi_prior, S_prior

    # 3. Instantiate and run BlackLittermanModel
    try:
        bl_model_args = {
            "cov_matrix": S_prior,
            "pi": pi_prior,
            "P": views_P,
            "Q": views_Q,
            "tau": configs.BL_TAU,
            # risk_aversion is used by PyPortfolioOpt if Omega is not explicitly provided,
            # or if using methods like Idzorek's that require it.
            # "risk_aversion": configs.BL_RISK_AVERSION # Usually not needed if Omega is well-defined
        }
        # Only add Omega if it's valid and provided.
        # PyPortfolioOpt calculates a default Omega if not given, based on P, S_prior, and tau.
        if views_Omega is not None and views_Omega.size > 0 and views_Omega.shape == (views_P.shape[0], views_P.shape[0]):
            bl_model_args["omega"] = views_Omega # Corrected key to 'omega'
            current_logger.debug("Using provided Omega matrix for Black-Litterman.")
        else:
            current_logger.info("Omega not provided, invalid, or empty. PyPortfolioOpt will calculate a default Omega.")
            # Optionally, you can compute and log the default omega:
            # default_omega_calc = BlackLittermanModel.default_omega(P=views_P, cov_matrix=S_prior, tau=configs.BL_TAU)
            # logger.debug(f"Calculated default Omega shape: {default_omega_calc.shape}")
            # bl_model_args["omega"] = default_omega_calc


        bl_model = BlackLittermanModel(**bl_model_args)
        
        # Get posterior estimates (annualized, as priors were annualized)
        posterior_mu = bl_model.bl_returns()
        posterior_S = bl_model.bl_cov()
        
        # Ensure correct order and fill NaNs for posterior estimates
        posterior_mu = posterior_mu.reindex(asset_tickers_order).fillna(0)
        posterior_S = posterior_S.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0)
        for ticker in posterior_S.index: # Ensure diagonal elements are positive
             if posterior_S.loc[ticker, ticker] <= 1e-9: posterior_S.loc[ticker, ticker] = 1e-9

        current_logger.info(f"Black-Litterman posterior estimates calculated. mu_posterior head: {posterior_mu.head().to_dict()}")
        return posterior_mu, posterior_S

    except Exception as e_bl_model:
        current_logger.error(f"Error running BlackLittermanModel: {e_bl_model}. Returning prior estimates if available, else None.", exc_info=True)
        return pi_prior if pi_prior is not None else None, S_prior if S_prior is not None else None