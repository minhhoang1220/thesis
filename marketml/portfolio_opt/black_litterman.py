# marketml/portfolio_opt/black_litterman.py
import pandas as pd
import numpy as np
import logging

try:
    from marketml.configs import configs
    from pypfopt import BlackLittermanModel, risk_models, expected_returns
except ImportError:
    print("CRITICAL ERROR in black_litterman.py: Could not import 'marketml.configs' or 'pypfopt'.")
    raise

logger = logging.getLogger(__name__)

def _get_omega_from_confidence(confidence_level: float) -> float:
    # Section: Omega value selection based on confidence
    default_omega = 0.05
    if hasattr(configs, 'BL_OMEGA_CONF_MAPPING') and configs.BL_OMEGA_CONF_MAPPING:
        for threshold, omega_val in sorted(configs.BL_OMEGA_CONF_MAPPING, key=lambda x: x[0], reverse=True):
            if confidence_level >= threshold:
                return omega_val
        return configs.BL_OMEGA_CONF_MAPPING[-1][1]
    return default_omega


def generate_views_from_signals(
    asset_tickers: list,
    current_financial_data: dict,
    current_classification_probs: dict,
    pi_prior_annualized: pd.Series,
    rebalance_date: pd.Timestamp,
    logger_instance: logging.Logger = None
):
    # Section: Generate P, Q, Omega matrices for Black-Litterman
    current_logger = logger_instance if logger_instance else logger

    views_P_list = []
    views_Q_list = []
    omega_values_for_diag = []

    num_assets = len(asset_tickers)
    if num_assets == 0:
        current_logger.warning(f"No asset tickers provided for view generation on {rebalance_date.date()}.")
        return None, None, None

    for i, ticker in enumerate(asset_tickers):
        prob_increase = current_classification_probs.get(ticker, 0.5)
        fin_data_for_ticker = current_financial_data.get(ticker, {})
        roe = fin_data_for_ticker.get('roe', 0.0)
        if isinstance(roe, str):
            try: roe = float(roe)
            except ValueError: roe = 0.0; current_logger.warning(f"Could not convert ROE '{fin_data_for_ticker.get('roe')}' to float for {ticker}.")

        # View: Based on Probability of Price Increase
        if prob_increase > configs.BL_PROB_THRESHOLD_STRONG_VIEW:
            p_vector = np.zeros(num_assets)
            p_vector[i] = 1.0
            denominator_scaling = (1.0 - 0.5)
            if denominator_scaling < 1e-6 : denominator_scaling = 1e-6
            scaled_outperformance = configs.BL_EXPECTED_OUTPERFORMANCE_STRONG * \
                                    ((prob_increase - 0.5) / denominator_scaling)
            q_value = configs.MARKOWITZ_RISK_FREE_RATE + scaled_outperformance
            views_P_list.append(p_vector)
            views_Q_list.append(q_value)
            conf_prob_view = configs.BL_VIEW_CONFIDENCE_STRONG
            omega_val_prob = _get_omega_from_confidence(conf_prob_view)
            omega_values_for_diag.append(omega_val_prob)
            current_logger.debug(f"BL View (Prob Increase) for {ticker} on {rebalance_date.date()}: "
                                 f"Prob={prob_increase:.2f}, Q={q_value:.4f}, Omega_diag_val={omega_val_prob:.6f}")

        # View: Based on High ROE
        roe_threshold_config = getattr(configs, 'BL_ROE_HIGH_THRESHOLD', 20.0)
        roe_outperform_config = getattr(configs, 'BL_ROE_OUTPERFORM_EXPECTATION', 0.02)
        roe_view_confidence = getattr(configs, 'BL_ROE_VIEW_CONFIDENCE', 0.6)

        if roe > roe_threshold_config:
            p_vector_roe = np.zeros(num_assets)
            p_vector_roe[i] = 1.0
            prior_return_for_ticker = pi_prior_annualized.get(ticker, configs.MARKOWITZ_RISK_FREE_RATE)
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
    Omega_matrix = np.diag(omega_values_for_diag)

    current_logger.debug(f"Generated P_matrix ({P_matrix.shape}) for {rebalance_date.date()}")
    current_logger.debug(f"Generated Q_vector ({Q_vector.shape}) for {rebalance_date.date()}")
    current_logger.debug(f"Generated Omega_matrix ({Omega_matrix.shape}) for {rebalance_date.date()}")

    if not (P_matrix.shape[0] == Q_vector.shape[0] == Omega_matrix.shape[0] == Omega_matrix.shape[1]):
        current_logger.error(f"Dimension mismatch in P, Q, or Omega for {rebalance_date.date()}. "
                             f"P:{P_matrix.shape}, Q:{Q_vector.shape}, Omega:{Omega_matrix.shape}. Cannot proceed with these views.")
        return None, None, None
        
    return P_matrix, Q_vector, Omega_matrix


def get_black_litterman_posterior_estimates(
    historical_prices_for_prior: pd.DataFrame,
    asset_tickers_order: list,
    market_caps: dict = None,
    views_P: np.ndarray = None,
    views_Q: np.ndarray = None,
    views_Omega: np.ndarray = None,
    logger_instance: logging.Logger = None
):
    # Section: Black-Litterman posterior calculation
    current_logger = logger_instance if logger_instance else logger

    if historical_prices_for_prior.empty or len(historical_prices_for_prior) < 2:
        current_logger.error("Historical prices are empty or insufficient for Black-Litterman prior calculation.")
        return None, None
    if not asset_tickers_order:
        current_logger.error("asset_tickers_order is empty. Cannot proceed.")
        return None, None

    try:
        historical_prices_for_prior_reordered = historical_prices_for_prior[asset_tickers_order].copy()
        historical_prices_for_prior_reordered.dropna(axis=1, how='all', inplace=True)
        if historical_prices_for_prior_reordered.shape[1] != len(asset_tickers_order):
            current_logger.warning("Some asset tickers were dropped from historical_prices_for_prior due to all NaNs. "
                                   "The Black-Litterman model will run on the reduced set of assets.")
            asset_tickers_order = historical_prices_for_prior_reordered.columns.tolist()
            if not asset_tickers_order:
                 current_logger.error("All asset tickers dropped from historical_prices_for_prior. Cannot calculate priors.")
                 return None, None
    except KeyError as e_key:
        current_logger.error(f"One or more tickers in asset_tickers_order not found in historical_prices_for_prior columns: {e_key}")
        return None, None

    # Covariance matrix (annualized)
    try:
        S_prior = risk_models.CovarianceShrinkage(historical_prices_for_prior_reordered, frequency=252).ledoit_wolf()
        S_prior = S_prior.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0)
        for ticker in S_prior.index:
             if S_prior.loc[ticker, ticker] <= 1e-9: S_prior.loc[ticker, ticker] = 1e-9
        current_logger.debug(f"S_prior (annualized) calculated. Shape: {S_prior.shape}")
    except Exception as e_S_prior:
        current_logger.error(f"Error calculating S_prior for Black-Litterman: {e_S_prior}", exc_info=True)
        return None, None

    # Market-implied equilibrium returns (annualized)
    pi_prior = None
    try:
        if market_caps:
            ordered_market_caps_series = pd.Series({ticker: market_caps.get(ticker, 1e9) for ticker in asset_tickers_order})
            delta = expected_returns.market_implied_risk_aversion(S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
            pi_prior = expected_returns.market_implied_prior_returns(ordered_market_caps_series, delta, S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
            current_logger.debug("pi_prior calculated using market-implied risk aversion.")
        else:
            current_logger.warning("No market caps provided for Black-Litterman. Using mean historical returns as prior pi (annualized).")
            pi_prior = expected_returns.mean_historical_return(historical_prices_for_prior_reordered, frequency=252)
        
        pi_prior = pi_prior.reindex(asset_tickers_order).fillna(0)
        current_logger.debug(f"pi_prior (annualized) calculated. Head: {pi_prior.head().to_dict()}")

    except Exception as e_pi_prior:
        current_logger.error(f"Error calculating pi_prior for Black-Litterman: {e_pi_prior}", exc_info=True)
        return None, S_prior

    if views_P is None or views_Q is None or views_P.size == 0:
        current_logger.info("No views provided or views are empty. Black-Litterman model will return prior estimates.")
        return pi_prior, S_prior

    # Black-Litterman model
    try:
        bl_model_args = {
            "cov_matrix": S_prior,
            "pi": pi_prior,
            "P": views_P,
            "Q": views_Q,
            "tau": configs.BL_TAU,
        }
        if views_Omega is not None and views_Omega.size > 0 and views_Omega.shape == (views_P.shape[0], views_P.shape[0]):
            bl_model_args["omega"] = views_Omega
            current_logger.debug("Using provided Omega matrix for Black-Litterman.")
        else:
            current_logger.info("Omega not provided, invalid, or empty. PyPortfolioOpt will calculate a default Omega.")

        bl_model = BlackLittermanModel(**bl_model_args)
        posterior_mu = bl_model.bl_returns()
        posterior_S = bl_model.bl_cov()
        posterior_mu = posterior_mu.reindex(asset_tickers_order).fillna(0)
        posterior_S = posterior_S.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0)
        for ticker in posterior_S.index:
             if posterior_S.loc[ticker, ticker] <= 1e-9: posterior_S.loc[ticker, ticker] = 1e-9

        current_logger.info(f"Black-Litterman posterior estimates calculated. mu_posterior head: {posterior_mu.head().to_dict()}")
        return posterior_mu, posterior_S

    except Exception as e_bl_model:
        current_logger.error(f"Error running BlackLittermanModel: {e_bl_model}. Returning prior estimates if available, else None.", exc_info=True)
        return pi_prior if pi_prior is not None else None, S_prior if S_prior is not None else None
