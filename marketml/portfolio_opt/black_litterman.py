# marketml/portfolio_opt/black_litterman.py
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

try:
    from marketml.configs import configs
    from pypfopt import BlackLittermanModel, risk_models, expected_returns
except ImportError as e:
    print(f"CRITICAL ERROR in black_litterman.py: Could not import necessary modules. {e}")
    sys.exit(1)

# Lấy logger đã được cấu hình bởi script chính
logger = logging.getLogger(__name__)
# Không cần fallback basicConfig ở đây nữa nếu script chính luôn setup logger.

def generate_views_from_signals(
    asset_tickers: list,
    current_financial_data: dict,
    current_classification_probs: dict,
    pi_prior_annualized: pd.Series,
    rebalance_date: pd.Timestamp # <<--- THÊM rebalance_date
):
    views_P_list = []
    views_Q_list = []
    views_confidences_list = [] # Hoặc trực tiếp tính omega_values
    omega_values_for_diag = [] # Sử dụng list này để xây dựng Omega

    num_assets = len(asset_tickers)

    for i, ticker in enumerate(asset_tickers):
        prob_increase = current_classification_probs.get(ticker, 0.5)
        fin_data = current_financial_data.get(ticker, {})
        roe = fin_data.get('roe', 0)

        # View 1: Dựa trên xác suất tăng giá
        if prob_increase > configs.BL_PROB_THRESHOLD_STRONG_VIEW:
            p_vector = np.zeros(num_assets)
            p_vector[i] = 1.0
            
            # Q: View tuyệt đối về lợi nhuận kỳ vọng hàng năm
            # Scale mức outperform dựa trên độ mạnh của xác suất so với 0.5 (trung lập)
            # Mức outperform tối đa là BL_EXPECTED_OUTPERFORMANCE_STRONG khi prob_increase = 1.0
            scaled_outperformance = configs.BL_EXPECTED_OUTPERFORMANCE_STRONG * \
                                    ((prob_increase - 0.5) / (1.0 - 0.5))
            q_value = configs.MARKOWITZ_RISK_FREE_RATE + scaled_outperformance
            
            views_P_list.append(p_vector)
            views_Q_list.append(q_value)
            # Xác định omega_value dựa trên confidence
            conf = configs.BL_VIEW_CONFIDENCE_STRONG
            if conf > 0.95: omega_val_current = 0.001
            elif conf > 0.7: omega_val_current = 0.005
            elif conf > 0.5: omega_val_current = 0.01
            else: omega_val_current = 0.05
            omega_values_for_diag.append(omega_val_current)
            logger.debug(f"BL View (Prob Increase) for {ticker} on {rebalance_date.date()}: Q={q_value:.4f}, Conf implied Omega={omega_val_current:.4f}")

        # View 3: Dựa trên ROE cao
        if roe > 25: # Giả sử ROE được lưu dưới dạng số (ví dụ 25 cho 25%)
            p_vector_roe = np.zeros(num_assets)
            p_vector_roe[i] = 1.0
            q_value_roe = pi_prior_annualized.get(ticker, configs.MARKOWITZ_RISK_FREE_RATE) + 0.02 # Kỳ vọng outperform prior 2% (hoặc risk-free + 2% nếu prior không có)
            
            views_P_list.append(p_vector_roe)
            views_Q_list.append(q_value_roe)
            # Xác định omega_value dựa trên confidence (ví dụ 0.6 cho view này)
            conf_roe = 0.6
            if conf_roe > 0.95: omega_val_roe = 0.001
            elif conf_roe > 0.7: omega_val_roe = 0.005
            elif conf_roe > 0.5: omega_val_roe = 0.01
            else: omega_val_roe = 0.05
            omega_values_for_diag.append(omega_val_roe)
            logger.debug(f"BL View (High ROE) for {ticker} on {rebalance_date.date()}: Q={q_value_roe:.4f}, Conf implied Omega={omega_val_roe:.4f}")

    if not views_P_list:
        logger.info(f"No views generated on {rebalance_date.date()} with current signals.") # SỬA Ở ĐÂY
        return None, None, None

    P_matrix = np.array(views_P_list)
    Q_vector = np.array(views_Q_list)
    Omega_matrix = np.diag(omega_values_for_diag) # Xây dựng Omega từ list các variance

    # Log sau khi đã tạo ma trận
    logger.debug(f"Generated P_matrix ({P_matrix.shape}) for {rebalance_date.date()}:\n{P_matrix if P_matrix.size > 0 else 'Empty'}")
    logger.debug(f"Generated Q_vector ({Q_vector.shape}) for {rebalance_date.date()}:\n{Q_vector if Q_vector.size > 0 else 'Empty'}")
    logger.debug(f"Generated Omega_matrix ({Omega_matrix.shape}) for {rebalance_date.date()}:\n{Omega_matrix if Omega_matrix.size > 0 else 'Empty'}")

    if P_matrix.shape[0] != Q_vector.shape[0] or P_matrix.shape[0] != Omega_matrix.shape[0]:
        logger.error(f"Dimension mismatch in P, Q, Omega for {rebalance_date.date()}. P:{P_matrix.shape}, Q:{Q_vector.shape}, Omega:{Omega_matrix.shape}")
        return None, None, None
        
    return P_matrix, Q_vector, Omega_matrix


def get_black_litterman_posterior_estimates(
    historical_prices_for_prior: pd.DataFrame,
    asset_tickers_order: list, # <<--- THÊM THAM SỐ NÀY để đảm bảo thứ tự
    market_caps: dict = None,
    views_P: np.ndarray = None,
    views_Q: np.ndarray = None,
    views_Omega: np.ndarray = None # <<--- Sửa: bạn có thể chọn không truyền Omega
):
    if historical_prices_for_prior.empty or len(historical_prices_for_prior) < 2: # Cần ít nhất 2 điểm giá để tính return/cov
        logger.error("Historical prices are empty or insufficient for Black-Litterman prior calculation.")
        return None, None

    # Đảm bảo historical_prices_for_prior có đúng thứ tự cột
    historical_prices_for_prior = historical_prices_for_prior[asset_tickers_order]


    # 1. Tính S_prior (Annualized)
    try:
        S_prior = risk_models.CovarianceShrinkage(historical_prices_for_prior, frequency=252).ledoit_wolf()
        S_prior = S_prior.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0) # Đảm bảo thứ tự và fillna
        for ticker in S_prior.index: # Đảm bảo đường chéo dương
             if pd.isna(S_prior.loc[ticker, ticker]) or S_prior.loc[ticker, ticker] <= 0 : S_prior.loc[ticker, ticker] = 1e-6
        logger.debug(f"S_prior (annualized) calculated. Shape: {S_prior.shape}")
    except Exception as e_S_prior:
        logger.error(f"Error calculating S_prior for Black-Litterman: {e_S_prior}")
        return None, None

    # 2. Tính pi_prior (Market-implied equilibrium returns - Annualized)
    pi_prior = None
    try:
        # Sắp xếp lại market_caps theo asset_tickers_order nếu có
        ordered_market_caps = None
        if market_caps:
            ordered_market_caps = {ticker: market_caps.get(ticker, 1e9) for ticker in asset_tickers_order} # 1e9 là fallback

        if ordered_market_caps:
            delta = expected_returns.market_implied_risk_aversion(S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
            pi_prior = expected_returns.market_implied_prior_returns(ordered_market_caps, delta, S_prior, risk_free_rate=configs.MARKOWITZ_RISK_FREE_RATE)
        else:
            logger.warning("No market caps provided for Black-Litterman. Using mean historical returns as prior pi (annualized).")
            pi_prior = expected_returns.mean_historical_return(historical_prices_for_prior, frequency=252)
        
        pi_prior = pi_prior.reindex(asset_tickers_order).fillna(0) # Đảm bảo thứ tự và fillna
        logger.debug(f"pi_prior (annualized) calculated. Head: {pi_prior.head().to_dict()}")

    except Exception as e_pi_prior:
        logger.error(f"Error calculating pi_prior for Black-Litterman: {e_pi_prior}")
        return None, S_prior

    if views_P is None or views_Q is None or views_P.size == 0: # views_Omega có thể là None
        logger.info("No views provided or views are empty. BL model returns prior estimates.")
        return pi_prior, S_prior

    # 4. Tạo và chạy BlackLittermanModel
    try:
        bl_model_args = {
            "cov_matrix": S_prior,
            "pi": pi_prior,
            "P": views_P,
            "Q": views_Q,
            "tau": configs.BL_TAU,
            "risk_aversion": configs.BL_RISK_AVERSION # Được dùng nếu Omega không được cung cấp tường minh
        }
        if views_Omega is not None and views_Omega.size > 0: # Chỉ truyền Omega nếu nó được tính toán và không rỗng
            bl_model_args["Omega"] = views_Omega
        else:
            logger.info("Omega not provided or empty, PyPortfolioOpt will calculate it based on P, S_prior, tau (and risk_aversion for some methods).")
            # Hoặc bạn có thể tính Omega mặc định ở đây:
            # default_omega = bl.default_omega(P=views_P, cov_matrix=S_prior, tau=configs.BL_TAU)
            # bl_model_args["Omega"] = default_omega


        bl_model = BlackLittermanModel(**bl_model_args)
        
        posterior_mu = bl_model.bl_returns()
        posterior_S = bl_model.bl_cov()
        
        posterior_mu = posterior_mu.reindex(asset_tickers_order).fillna(0)
        posterior_S = posterior_S.reindex(index=asset_tickers_order, columns=asset_tickers_order).fillna(0)
        for ticker in posterior_S.index:
             if pd.isna(posterior_S.loc[ticker, ticker]) or posterior_S.loc[ticker, ticker] <= 0 :
                posterior_S.loc[ticker, ticker] = 1e-6

        logger.info(f"Black-Litterman posterior estimates calculated. mu_posterior head: {posterior_mu.head().to_dict()}")
        return posterior_mu, posterior_S
    except Exception as e_bl:
        logger.error(f"Error running BlackLittermanModel: {e_bl}. Returning prior estimates.")
        logger.exception("Detailed error for BlackLittermanModel:") # In traceback của lỗi BL
        return pi_prior, S_prior