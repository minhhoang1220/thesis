# marketml/portfolio_opt/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from marketml.configs import configs
from marketml.portfolio_opt.rl_scaler_handler import FinancialFeatureScaler

logger = logging.getLogger(__name__)

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 prices_df: pd.DataFrame,
                 financial_data: pd.DataFrame = None,
                 classification_probs: pd.DataFrame = None,
                 initial_capital: float = 100000,
                 transaction_cost_bps: int = 10,
                 lookback_window_size: int = 30,
                 rebalance_frequency_days: int = 1, # How many trading days per env step
                 max_steps_per_episode: int = None,
                 reward_use_log_return: bool = True,
                 reward_turnover_penalty_factor: float = 0.0,
                 financial_features: list = None,
                 prob_features: list = None,
                 financial_feature_means: pd.Series = None, # Means fitted on training data
                 financial_feature_stds: pd.Series = None,  # Stds fitted on training data
                 logger_instance: logging.Logger = None
                 ):
        super(PortfolioEnv, self).__init__()
        self.logger = logger_instance if logger_instance else logger # Use passed logger or module logger

        if prices_df.empty:
            self.logger.error("CRITICAL: prices_df is empty. Cannot initialize PortfolioEnv.")
            raise ValueError("prices_df cannot be empty.")
        self.prices_df = prices_df.copy()
        if not isinstance(self.prices_df.index, pd.DatetimeIndex):
            try:
                self.prices_df.index = pd.to_datetime(self.prices_df.index)
            except Exception as e:
                self.logger.error(f"Failed to convert prices_df index to DatetimeIndex: {e}", exc_info=True)
                raise

        self.financial_data_full = financial_data.copy() if financial_data is not None else pd.DataFrame()
        self.classification_probs_full = classification_probs.copy() if classification_probs is not None else pd.DataFrame()

        # Standardize 'date' columns if they exist and are not index
        if not self.financial_data_full.empty and 'date' in self.financial_data_full.columns:
            try: self.financial_data_full['date'] = pd.to_datetime(self.financial_data_full['date'])
            except: self.logger.warning("Could not parse 'date' in financial_data_full.")
        if not self.classification_probs_full.empty and 'date' in self.classification_probs_full.columns:
            try: self.classification_probs_full['date'] = pd.to_datetime(self.classification_probs_full['date'])
            except: self.logger.warning("Could not parse 'date' in classification_probs_full.")

        # Ensure 'Year' in financial_data is numeric if present
        if not self.financial_data_full.empty and 'Year' in self.financial_data_full.columns:
            if not pd.api.types.is_numeric_dtype(self.financial_data_full['Year']):
                try:
                    self.financial_data_full['Year'] = pd.to_numeric(self.financial_data_full['Year'])
                except ValueError:
                    self.logger.error("Failed to convert 'Year' in financial_data_full to numeric. Financial features may fail.")
                    self.financial_data_full = pd.DataFrame() # Invalidate if crucial column is bad

        self.tickers = self.prices_df.columns.tolist()
        if not self.tickers:
            self.logger.error("CRITICAL: No tickers found in prices_df. Cannot initialize PortfolioEnv.")
            raise ValueError("prices_df must have tickers as columns.")
        self.num_assets = len(self.tickers)

        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.lookback_window_size = int(lookback_window_size)
        self.rebalance_frequency_days = int(rebalance_frequency_days)

        self.reward_use_log_return = reward_use_log_return
        self.reward_turnover_penalty_factor = float(reward_turnover_penalty_factor)

        self.financial_features_to_use = financial_features if financial_features else []
        self.prob_features_to_use = prob_features if prob_features else []

        # Initialize internal scaler for financial features
        self.internal_fin_scaler = None
        if self.financial_features_to_use:
            if financial_feature_means is not None and financial_feature_stds is not None:
                self.internal_fin_scaler = FinancialFeatureScaler(
                    feature_names=self.financial_features_to_use,
                    means=financial_feature_means,
                    stds=financial_feature_stds
                )
                self.logger.info("PortfolioEnv: Initialized internal FinancialFeatureScaler with provided means/stds.")
            else:
                self.logger.warning("PortfolioEnv: Financial features are specified, but means/stds for scaling are missing. Features will not be scaled, or an empty scaler will be used.")
                # Create an empty scaler so transform doesn't fail, but it won't scale
                self.internal_fin_scaler = FinancialFeatureScaler(feature_names=self.financial_features_to_use)


        num_fin_features_per_asset = len(self.financial_features_to_use)
        num_prob_features_per_asset = len(self.prob_features_to_use)

        min_data_len_needed = self.lookback_window_size + self.rebalance_frequency_days
        if len(self.prices_df) < min_data_len_needed:
            self.logger.error(f"Insufficient price data. Rows: {len(self.prices_df)}, Minimum needed: {min_data_len_needed}.")
            raise ValueError("Insufficient price data for lookback and rebalance frequency.")
        
        # Action: weights for assets + cash (normalized by softmax later)
        self.action_space = spaces.Box(low=-5, high=5, shape=(self.num_assets + 1,), dtype=np.float32) # Wider range for raw NN output

        # Observation space dimensions
        price_hist_dim = self.num_assets * self.lookback_window_size
        weights_dim = self.num_assets + 1 # Current weights including cash
        # cash_ratio_dim = 1 # Already included in weights_dim[-1]
        ptf_value_dim = 1  # Normalized portfolio value
        financial_data_dim = self.num_assets * num_fin_features_per_asset
        prob_data_dim = self.num_assets * num_prob_features_per_asset
        
        state_dim = price_hist_dim + weights_dim + ptf_value_dim + financial_data_dim + prob_data_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        self.logger.info(f"Observation space dimension: {state_dim} ("
                    f"PriceHist({self.lookback_window_size}d): {price_hist_dim}, Weights: {weights_dim}, PtfValue_Norm: {ptf_value_dim}, "
                    f"Financial({num_fin_features_per_asset}pA): {financial_data_dim}, Probs({num_prob_features_per_asset}pA): {prob_data_dim})")
        
        self.current_step = 0
        self._current_prices_idx = self.lookback_window_size -1 # Start so first _get_state uses prices up to this index for history
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0 # Start with all cash

        if max_steps_per_episode is None:
            if self.rebalance_frequency_days <= 0:
                self.logger.error("rebalance_frequency_days must be positive.")
                raise ValueError("rebalance_frequency_days must be positive.")
            # Max steps is number of rebalance periods possible
            self.max_steps = (len(self.prices_df) - self.lookback_window_size) // self.rebalance_frequency_days
        else:
            self.max_steps = int(max_steps_per_episode)
        
        if self.max_steps <= 0 :
            self.logger.error(f"Calculated max_steps ({self.max_steps}) is not positive. Check prices_df length, lookback, and rebalance frequency.")
            raise ValueError(f"Calculated max_steps ({self.max_steps}) is not positive.")
        
        self.logger.info(
            f"PortfolioEnv initialized: Tickers: {self.num_assets}, Lookback: {self.lookback_window_size}, "
            f"Rebal Freq (days): {self.rebalance_frequency_days}, Max episode steps: {self.max_steps}. "
            f"Price data from {self.prices_df.index.min().date()} to {self.prices_df.index.max().date()}."
        )
        self.episode_return_history = []
        self.episode_portfolio_log_returns = []

    def _get_state(self) -> np.ndarray:
        # Ensure _current_prices_idx is valid
        if not (self.lookback_window_size -1 <= self._current_prices_idx < len(self.prices_df)):
            self.logger.error(f"Invalid _current_prices_idx: {self._current_prices_idx} for prices_df length {len(self.prices_df)} and lookback {self.lookback_window_size}.")
            # This state indicates a serious issue, return a zero state
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. Price history features (log returns of lookback_window_size)
        start_idx = self._current_prices_idx - self.lookback_window_size + 1
        end_idx = self._current_prices_idx + 1 # Slice includes end_idx-1
        
        price_history_slice = self.prices_df.iloc[start_idx:end_idx][self.tickers]
        # Log returns are generally preferred for stationarity in RL states for prices
        log_returns_history = np.log(price_history_slice / price_history_slice.shift(1)).fillna(0.0)
        price_features = log_returns_history.values.flatten().astype(np.float32) # Shape: (num_assets * lookback_window_size)

        # 2. Current portfolio weights (including cash)
        weights_features = self.current_weights.astype(np.float32)

        # 3. Normalized portfolio value
        ptf_value_feature = np.array([self.portfolio_value / self.initial_capital], dtype=np.float32)
        
        # Current date for fetching financial and probability features
        current_date_for_aux_features = self.prices_df.index[self._current_prices_idx]

        # 4. Financial data features
        financial_features_all_assets = []
        if self.financial_features_to_use and self.internal_fin_scaler:
            target_fin_year = current_date_for_aux_features.year - 1 # Financials from previous year-end
            for ticker in self.tickers:
                asset_fin_data_series = pd.Series(dtype=float, index=self.financial_features_to_use) # Ensure all features present, default NaN
                if not self.financial_data_full.empty and 'Ticker' in self.financial_data_full.columns and 'Year' in self.financial_data_full.columns:
                    ticker_year_data = self.financial_data_full[
                        (self.financial_data_full['Ticker'] == ticker) &
                        (self.financial_data_full['Year'] == target_fin_year)
                    ]
                    if not ticker_year_data.empty:
                        # Populate asset_fin_data_series with values from the record
                        record = ticker_year_data.iloc[0]
                        for feat in self.financial_features_to_use:
                            asset_fin_data_series[feat] = record.get(feat) # Keeps NaN if feature not in record

                scaled_asset_fin_features = self.internal_fin_scaler.transform(asset_fin_data_series)
                financial_features_all_assets.extend(scaled_asset_fin_features)
        else: # No financial features or no scaler
            financial_features_all_assets = np.zeros(self.num_assets * len(self.financial_features_to_use), dtype=np.float32)
        financial_features_array = np.array(financial_features_all_assets, dtype=np.float32)


        # 5. Classification probabilities features
        prob_features_all_assets = []
        if self.prob_features_to_use and not self.classification_probs_full.empty:
            prob_col_name_base = f"prob_increase_{configs.SOFT_SIGNAL_MODEL_NAME}" # Assuming this exists from configs
            # Verify the actual column name exists
            actual_prob_col_names = [col for col in self.classification_probs_full.columns if prob_col_name_base in col]
            if not actual_prob_col_names and self.prob_features_to_use: # If specific prob features are listed but base name not found
                 self.logger.warning(f"Base probability column '{prob_col_name_base}' not found. Will use 0.5 for probabilities.")
            
            for ticker in self.tickers:
                asset_probs = np.full(len(self.prob_features_to_use), 0.5, dtype=np.float32) # Default to 0.5 (neutral)
                if 'date' in self.classification_probs_full.columns and 'ticker' in self.classification_probs_full.columns:
                    relevant_ticker_probs = self.classification_probs_full[
                        (self.classification_probs_full['ticker'] == ticker) &
                        (self.classification_probs_full['date'] <= current_date_for_aux_features)
                    ]
                    if not relevant_ticker_probs.empty:
                        latest_record = relevant_ticker_probs.sort_values('date', ascending=False).iloc[0]
                        for idx, feature_name_pattern in enumerate(self.prob_features_to_use): # e.g. "prob_increase_XGBoost"
                            # Try to find the exact match or a column containing the pattern
                            col_to_use = feature_name_pattern
                            if col_to_use not in latest_record.index: # If exact pattern not a column
                                found_cols = [c for c in latest_record.index if feature_name_pattern in c]
                                if found_cols: col_to_use = found_cols[0]
                                else: col_to_use = None

                            if col_to_use and pd.notna(latest_record.get(col_to_use)):
                                try: asset_probs[idx] = float(latest_record.get(col_to_use))
                                except (ValueError,TypeError): self.logger.warning(f"Could not convert prob feature {col_to_use} to float for {ticker}")
                prob_features_all_assets.extend(asset_probs)
        else: # No probability features or no probability data
             prob_features_all_assets = np.zeros(self.num_assets * len(self.prob_features_to_use), dtype=np.float32)
        prob_features_array = np.array(prob_features_all_assets, dtype=np.float32)

        # Concatenate all feature parts
        state = np.concatenate([
            price_features,
            weights_features,
            ptf_value_feature,
            financial_features_array,
            prob_features_array
        ]).astype(np.float32)

        if state.shape[0] != self.observation_space.shape[0]:
            self.logger.critical(f"STATE SHAPE MISMATCH! Expected {self.observation_space.shape[0]}, got {state.shape[0]}. "
                                 f"Price: {price_features.shape}, Wts: {weights_features.shape}, PtfVal: {ptf_value_feature.shape}, "
                                 f"Fin: {financial_features_array.shape}, Prob: {prob_features_array.shape}")
            # Pad or truncate if desperate, but this indicates a fundamental issue.
            # For now, return a zero state to avoid crashing SB3, but this needs fixing.
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding for reproducibility
        self.current_step = 0
        # Start at an index where a full lookback window is available
        self._current_prices_idx = self.lookback_window_size -1 # iloc uses 0-based index, so index for Nth item is N-1
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0 # Start with 100% cash
        
        self.episode_return_history = []
        self.episode_portfolio_log_returns = []

        observation = self._get_state()
        info = {} # Additional info to return, if any
        self.logger.debug(f"Env Reset: Start prices_idx={self._current_prices_idx}, Portfolio Value={self.portfolio_value:.2f}")
        return observation, info

    def step(self, action: np.ndarray):
        """Execute one step in the environment given the action."""
        # 1. Process action to get target weights (with numerical stability checks)
        if np.isnan(action).any() or np.isinf(action).any():
            self.logger.warning(f"NaN or Inf in raw action: {action}. Using equal weights for this step.")
            target_weights = np.ones(self.num_assets + 1, dtype=np.float32) / (self.num_assets + 1)
        else:
            exp_action = np.exp(action - np.max(action))  # Subtract max for numerical stability
            target_weights = exp_action / np.sum(exp_action)
        target_weights = target_weights.astype(np.float32)

        portfolio_value_before_trade = self.portfolio_value
        prices_at_rebalance_start = self.prices_df.iloc[self._current_prices_idx][self.tickers].values

        # Handle NaN or non-positive prices at rebalance start
        if np.isnan(prices_at_rebalance_start).any() or np.any(prices_at_rebalance_start <= 0):
            self.logger.warning(
                f"NaN or non-positive prices at rebalance start (idx {self._current_prices_idx}). "
                "No trades executed, holding previous weights."
            )
            transaction_costs = 0.0
            
            # Calculate portfolio value change from market movement of current holdings
            next_prices_idx = self._current_prices_idx + self.rebalance_frequency_days
            if next_prices_idx < len(self.prices_df):
                prices_at_rebalance_end = self.prices_df.iloc[next_prices_idx][self.tickers].values
                if not (np.isnan(prices_at_rebalance_end).any() or np.any(prices_at_rebalance_end <= 0)):
                    # Calculate asset values based on current shares and price changes
                    current_shares = np.zeros_like(self.current_weights[:-1])
                    valid_prices_mask = prices_at_rebalance_start > 0
                    current_shares[valid_prices_mask] = (
                        self.current_weights[:-1][valid_prices_mask] * portfolio_value_before_trade
                    ) / prices_at_rebalance_start[valid_prices_mask]
                    
                    value_of_assets_eod = np.sum(current_shares * prices_at_rebalance_end)
                    cash_value = self.current_weights[-1] * portfolio_value_before_trade
                    self.portfolio_value = value_of_assets_eod + cash_value
            
            # current_weights remain unchanged
        else:
            # Calculate transaction costs
            current_asset_values = self.current_weights[:-1] * portfolio_value_before_trade
            target_asset_values = target_weights[:-1] * portfolio_value_before_trade
            trades_value = np.sum(np.abs(target_asset_values - current_asset_values))
            transaction_costs = trades_value * self.transaction_cost_pct
            
            # Apply transaction costs
            portfolio_value_after_costs = portfolio_value_before_trade - transaction_costs
            
            # Handle bankruptcy case
            if portfolio_value_after_costs < 0:
                self.logger.warning(
                    f"Portfolio value became negative ({portfolio_value_after_costs:.2f}) after transaction costs. "
                    "Setting to 0."
                )
                portfolio_value_after_costs = 0.0
                self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
                self.current_weights[-1] = 1.0  # All cash (which is 0)
            
            # Calculate new cash position
            cash_after_trade = portfolio_value_after_costs * target_weights[-1]
            assets_value_after_costs = portfolio_value_after_costs - cash_after_trade
            
            # Calculate new shares for each asset
            shares = np.zeros_like(target_weights[:-1])
            valid_prices_mask = prices_at_rebalance_start > 0
            
            if np.sum(target_weights[:-1]) > 1e-9:  # Avoid division by zero
                relative_asset_weights = target_weights[:-1] / np.sum(target_weights[:-1])
                dollar_value_per_asset = relative_asset_weights * assets_value_after_costs
                shares[valid_prices_mask] = dollar_value_per_asset[valid_prices_mask] / prices_at_rebalance_start[valid_prices_mask]
            
            # Calculate portfolio value at end of period
            next_prices_idx = self._current_prices_idx + self.rebalance_frequency_days
            if next_prices_idx >= len(self.prices_df):  # End of data
                self.portfolio_value = portfolio_value_after_costs
                self.current_weights = target_weights
            else:
                prices_at_rebalance_end = self.prices_df.iloc[next_prices_idx][self.tickers].values
                if np.isnan(prices_at_rebalance_end).any() or np.any(prices_at_rebalance_end <= 0):
                    self.logger.warning(
                        f"NaN or non-positive prices at end of holding period (idx {next_prices_idx}). "
                        "Using start-of-period prices for asset valuation."
                    )
                    prices_at_rebalance_end = prices_at_rebalance_start
                
                value_of_assets_eod = np.sum(shares * prices_at_rebalance_end)
                self.portfolio_value = value_of_assets_eod + cash_after_trade
                
                # Update weights based on new values
                if self.portfolio_value > 1e-6:
                    self.current_weights[:-1] = (shares * prices_at_rebalance_end) / self.portfolio_value
                    self.current_weights[-1] = cash_after_trade / self.portfolio_value
                    self.current_weights /= np.sum(self.current_weights)  # Normalize
                else:
                    self.current_weights[:-1] = 0.0
                    self.current_weights[-1] = 1.0  # All (zero) cash

        # Calculate step return and reward
        step_portfolio_log_return = 0.0
        if portfolio_value_before_trade > 1e-9:
            current_ptf_val_for_log = max(self.portfolio_value, 1e-9)
            step_portfolio_log_return = np.log(current_ptf_val_for_log / portfolio_value_before_trade)
        
        self.episode_portfolio_log_returns.append(step_portfolio_log_return)

        # Calculate reward
        if self.reward_use_log_return:
            reward = step_portfolio_log_return
        else:
            reward = (self.portfolio_value / portfolio_value_before_trade - 1) if portfolio_value_before_trade > 1e-9 else 0.0
        
        # Apply turnover penalty if applicable
        if (not (np.isnan(prices_at_rebalance_start).any() or np.any(prices_at_rebalance_start <= 0)) and
                self.reward_turnover_penalty_factor > 0 and portfolio_value_before_trade > 1e-9):
            turnover_ratio = trades_value / portfolio_value_before_trade
            reward -= self.reward_turnover_penalty_factor * turnover_ratio
        
        self.episode_return_history.append(reward)

        # Update indices and step count
        self._current_prices_idx += self.rebalance_frequency_days
        self.current_step += 1

        # Check termination conditions
        terminated = (
            self.portfolio_value < 0.01 * self.initial_capital or  # Substantial loss
            self.current_step >= self.max_steps or  # Reached max steps
            self._current_prices_idx >= len(self.prices_df)  # End of data
        )
        
        truncated = False  # For time limits not related to task completion
        
        # Prepare observation and info
        observation = self._get_state() if not terminated else np.zeros_like(self.observation_space.sample())
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.tolist(),
            'transaction_costs': transaction_costs,
            'step_reward': reward,
            'step_log_return': step_portfolio_log_return,
            'raw_log_return': step_portfolio_log_return
        }
        
        if terminated:
            info.update({
                'cumulative_reward_shaped': np.sum(self.episode_return_history),
                'cumulative_log_return': np.sum(self.episode_portfolio_log_returns),
                'sharpe_ratio': self.calculate_sharpe(self.episode_portfolio_log_returns)
            })
            self.logger.info(
                f"Episode terminated. Final Ptf Value: {self.portfolio_value:.2f}, "
                f"Cum Rew (shaped): {info['cumulative_reward_shaped']:.4f}, "
                f"Cum LogRet: {info['cumulative_log_return']:.4f}"
            )
        
        self.logger.debug(
            f"Step: {self.current_step}/{self.max_steps}, PtfVal: {self.portfolio_value:.2f}, "
            f"Rew: {reward:.5f}, LogRet: {step_portfolio_log_return:.5f}, "
            f"TC: {transaction_costs:.2f}, Wts: {[f'{w:.2f}' for w in self.current_weights]}"
        )
        
        return observation, reward, terminated, truncated, info
        
    def calculate_sharpe(self, log_returns_history: list, risk_free_rate_annual: float = 0.0) -> float:
        if not log_returns_history or len(log_returns_history) < 20: # Need a reasonable number of returns
            return 0.0 # Or np.nan
        
        log_returns_array = np.array(log_returns_history)
        
        # Assuming log_returns_history are per rebalance_frequency_days
        # To annualize, we need to know how many such periods are in a year
        periods_per_year = 252.0 / self.rebalance_frequency_days
        
        mean_period_log_return = np.mean(log_returns_array)
        std_period_log_return = np.std(log_returns_array)

        # Annualize mean and std
        # For log returns, mean_annual = mean_period * periods_per_year
        # std_annual = std_period * sqrt(periods_per_year)
        annualized_mean_log_return = mean_period_log_return * periods_per_year
        annualized_std_log_return = std_period_log_return * np.sqrt(periods_per_year)
        
        # Daily risk-free rate (approx) if annual is given
        # risk_free_rate_period = risk_free_rate_annual / periods_per_year # This is arithmetic
        # For log returns, it's often simpler to use rf=0 or an already periodized rf rate.
        # PyPortfolioOpt typically expects annualized rf rate.
        # Let's assume risk_free_rate_annual is what we want to subtract from annualized_mean_log_return

        if annualized_std_log_return < 1e-9:
            # If no volatility, Sharpe is undefined or +/- inf
            if abs(annualized_mean_log_return - risk_free_rate_annual) < 1e-9: return 0.0
            return np.inf * np.sign(annualized_mean_log_return - risk_free_rate_annual)
        
        sharpe_ratio = (annualized_mean_log_return - risk_free_rate_annual) / annualized_std_log_return
        return sharpe_ratio

    def render(self, mode='human'):
        if mode == 'human':
            self.logger.info(f"Render - Step: {self.current_step}, Ptf Value: {self.portfolio_value:.2f}, "
                        f"Weights: {[f'{w:.3f}' for w in self.current_weights]}")

    def close(self):
        self.logger.debug("PortfolioEnv closed.")
        pass