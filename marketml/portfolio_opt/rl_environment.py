# marketml/portfolio_opt/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from marketml.configs import configs

logger = logging.getLogger(__name__)

class PortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 prices_df: pd.DataFrame,
                 financial_data: pd.DataFrame = None, # Đã có
                 classification_probs: pd.DataFrame = None, # Đã có
                 initial_capital=100000,
                 transaction_cost_bps=10,
                 lookback_window_size=30,
                 rebalance_frequency_days=1,
                 max_steps_per_episode=None,
                 reward_use_log_return=True,
                 reward_turnover_penalty_factor=0.0,
                 # THIẾU CÁC THAM SỐ NÀY TRONG PHIÊN BẢN HIỆN TẠI CỦA BẠN:
                 financial_features: list = None,
                 prob_features: list = None,
                 financial_feature_means: pd.Series = None,
                 financial_feature_stds: pd.Series = None
                 ):
 # Tùy chọn: số bước tối đa trước khi một episode kết thúc

        super(PortfolioEnv, self).__init__()

        self.prices_df = prices_df.copy()
        # Đảm bảo index là DatetimeIndex để .loc hoạt động như mong đợi với ngày tháng
        if not isinstance(self.prices_df.index, pd.DatetimeIndex):
            try:
                self.prices_df.index = pd.to_datetime(self.prices_df.index)
            except Exception as e:
                logger.error(f"Không thể chuyển index của prices_df thành DatetimeIndex: {e}")
                raise
        
        self.financial_data_full = financial_data.copy() if financial_data is not None else pd.DataFrame()
        self.classification_probs_full = classification_probs.copy() if classification_probs is not None else pd.DataFrame()

        # Đảm bảo financial_data_full và classification_probs_full có cột 'date' và 'ticker' nếu chúng không phải là index
        # Và index của chúng nên là DatetimeIndex nếu 'date' không phải là cột
        if not self.financial_data_full.empty:
            if 'date' in self.financial_data_full.columns and not isinstance(self.financial_data_full.index, pd.DatetimeIndex):
                try: self.financial_data_full['date'] = pd.to_datetime(self.financial_data_full['date'])
                except: pass # Bỏ qua nếu không phải dạng ngày tháng
            # Nếu financial_data có 'year' thay vì 'date', cần logic xử lý riêng trong _get_state
            if 'Year' in self.financial_data_full.columns and 'Ticker' in self.financial_data_full.columns:
                logger.info("Financial data có cột 'year' và 'ticker'.")
                if not pd.api.types.is_numeric_dtype(self.financial_data_full['Year']):
                    try:
                        self.financial_data_full['Year'] = pd.to_numeric(self.financial_data_full['Year'])
                    except ValueError:
                        logger.error("Không thể chuyển cột 'Year' trong financial data thành số.")
                        # Có thể raise lỗi ở đây hoặc để self.financial_data_full rỗng nếu 'Year' quan trọng
            else:
                 logger.warning("Financial data không có cột 'Year' và 'Ticker' như mong đợi. Sẽ không sử dụng financial features.")
                 self.financial_data_full = pd.DataFrame()


        if not self.classification_probs_full.empty:
            if 'date' in self.classification_probs_full.columns and not isinstance(self.classification_probs_full.index, pd.DatetimeIndex):
                try: self.classification_probs_full['date'] = pd.to_datetime(self.classification_probs_full['date'])
                except: pass

        self.tickers = self.prices_df.columns.tolist()
        self.num_assets = len(self.tickers)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.lookback_window_size = lookback_window_size
        self.rebalance_frequency_days = rebalance_frequency_days # Số ngày giao dịch mỗi bước tái cân bằng

        self.reward_use_log_return = reward_use_log_return
        self.reward_turnover_penalty_factor = reward_turnover_penalty_factor

        self.financial_features_to_use = financial_features if financial_features else []
        self.prob_features_to_use = prob_features if prob_features else []
        
        # Lưu trữ mean/std để chuẩn hóa
        self.fin_feat_means = financial_feature_means
        self.fin_feat_stds = financial_feature_stds
        if self.financial_features_to_use and (self.fin_feat_means is None or self.fin_feat_stds is None):
            logger.warning("Đang sử dụng financial features nhưng financial_feature_means/stds không được cung cấp. Các đặc trưng sẽ không được chuẩn hóa đúng cách!")

        
        num_financial_features_per_asset = len(self.financial_features_to_use)
        num_prob_features_per_asset = len(self.prob_features_to_use)

        if self.prices_df.empty or len(self.prices_df) < self.lookback_window_size + self.rebalance_frequency_days:
            logger.error(
                f"Dữ liệu giá không đủ. Số dòng: {len(self.prices_df)}, "
                f"Yêu cầu tối thiểu: {self.lookback_window_size + self.rebalance_frequency_days}"
            )
            raise ValueError("Dữ liệu giá không đủ cho cửa sổ nhìn lại và tần suất tái cân bằng đã cho.")
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets + 1,), dtype=np.float32)

        # Cập nhật state_dim
        price_hist_dim = self.num_assets * self.lookback_window_size
        weights_dim = self.num_assets + 1
        cash_ratio_dim = 1
        ptf_value_dim = 1
        financial_data_dim = self.num_assets * num_financial_features_per_asset
        prob_data_dim = self.num_assets * num_prob_features_per_asset
        state_dim = price_hist_dim + weights_dim + cash_ratio_dim + ptf_value_dim + financial_data_dim + prob_data_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        logger.info(f"Observation space dimension: {state_dim} "
                    f"(PriceHist: {price_hist_dim}, Weights: {weights_dim}, CashRatio: {cash_ratio_dim}, PtfVal: {ptf_value_dim}, "
                    f"Financial: {financial_data_dim} ({num_financial_features_per_asset} per asset), Probs: {prob_data_dim} ({num_prob_features_per_asset} per asset))")
        
        self.current_step = 0
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        self.portfolio_value = self.initial_capital
        self._current_prices_idx = self.lookback_window_size # Index số nguyên cho iloc
        
        self.max_steps_config = max_steps_per_episode
        if self.max_steps_config is None:
            if self.rebalance_frequency_days <= 0:
                logger.error("rebalance_frequency_days phải lớn hơn 0.")
                raise ValueError("rebalance_frequency_days phải lớn hơn 0.")
            min_data_points_needed = self.lookback_window_size + self.rebalance_frequency_days
            if len(self.prices_df) < min_data_points_needed: self.max_steps = 0
            else: self.max_steps = (len(self.prices_df) - self.lookback_window_size) // self.rebalance_frequency_days
        else: self.max_steps = self.max_steps_config
        
        # Thêm logging chi tiết
        logger.info(
            f"PortfolioEnv khởi tạo: "
            f"Số dòng prices_df: {len(self.prices_df)}, "
            f"Ngày bắt đầu prices_df: {self.prices_df.index.min().date() if not self.prices_df.empty and isinstance(self.prices_df.index, pd.DatetimeIndex) else 'N/A'}, "
            f"Ngày kết thúc prices_df: {self.prices_df.index.max().date() if not self.prices_df.empty and isinstance(self.prices_df.index, pd.DatetimeIndex) else 'N/A'}, "
            f"Lookback: {self.lookback_window_size}, Rebal Freq: {self.rebalance_frequency_days}. "
            f"Max_steps_config: {self.max_steps_config}, Calculated max_steps: {self.max_steps}"
        )
        if self.max_steps <= 0 :
            logger.error(f"Calculated max_steps ({self.max_steps}) không hợp lệ. Kiểm tra độ dài prices_df và các tham số.")
            raise ValueError(f"Calculated max_steps ({self.max_steps}) không hợp lệ.")
        
        self.episode_return_history = []
        self.episode_portfolio_log_returns = [] # Để tính Sharpe


    def _get_state(self):
        """Xây dựng quan sát trạng thái."""
        # 1. Thay đổi giá quá khứ (lợi nhuận)
        # Đảm bảo _current_prices_idx hợp lệ để cắt lát
        start_idx_price = self._current_prices_idx - self.lookback_window_size
        end_idx_price = self._current_prices_idx
        if start_idx_price < 0: effective_start_idx_price = 0; num_missing_rows_price = abs(start_idx_price)
        else: effective_start_idx_price = start_idx_price; num_missing_rows_price = 0

        expected_len_price = self.num_assets * self.lookback_window_size
        if end_idx_price > len(self.prices_df) or effective_start_idx_price >= end_idx_price or effective_start_idx_price < 0 : # Thêm check effective_start_idx_price < 0
             logger.warning(f"Chỉ số không hợp lệ cho price history: effective_start={effective_start_idx_price}, end={end_idx_price}. Padding zeros.")
             price_history_features = np.zeros(expected_len_price, dtype=np.float32)
        else:
            price_history = self.prices_df.iloc[effective_start_idx_price:end_idx_price][self.tickers]
            price_history_returns_raw = price_history.pct_change()
            if not price_history_returns_raw.empty: price_history_returns_raw.iloc[0] = price_history_returns_raw.iloc[0].fillna(0.0)
            price_history_returns_filled = price_history_returns_raw.fillna(0.0).values.flatten().astype(np.float32)
            
            current_len = len(price_history_returns_filled)
            if num_missing_rows_price > 0: # Prepend zeros if effective_start_idx was 0 due to negative start_idx_price
                padding_zeros_price = np.zeros(self.num_assets * num_missing_rows_price, dtype=np.float32)
                price_history_features = np.concatenate((padding_zeros_price, price_history_returns_filled))
                current_len = len(price_history_features) # Update current_len after prepending
            else:
                price_history_features = price_history_returns_filled
            
            if current_len < expected_len_price: # Append zeros if still shorter
                price_history_features = np.concatenate((price_history_features, np.zeros(expected_len_price - current_len, dtype=np.float32)))
            elif current_len > expected_len_price: # Truncate if longer
                price_history_features = price_history_features[:expected_len_price]

        # 2. Current weights, cash ratio, portfolio value
        current_weights_features = self.current_weights.astype(np.float32)
        cash_ratio_feature = np.array([self.current_weights[-1]], dtype=np.float32)
        normalized_portfolio_value_feature = np.array([self.portfolio_value / self.initial_capital], dtype=np.float32)
        
        current_date_for_features = self.prices_df.index[self._current_prices_idx]

        # 3. Financial data features
        expected_fin_len = self.num_assets * len(self.financial_features_to_use)
        financial_features_flat_list = []
        if not self.financial_data_full.empty and self.financial_features_to_use:
            target_year = current_date_for_features.year - 1
            for ticker in self.tickers:
                asset_fin_features_raw = np.zeros(len(self.financial_features_to_use), dtype=np.float32)
                # SỬA TÊN CỘT: 'Ticker' và 'Year' theo file financial_data.csv
                ticker_year_data = self.financial_data_full[
                    (self.financial_data_full['Ticker'] == ticker) &
                    (self.financial_data_full['Year'] == target_year)
                ]
                if not ticker_year_data.empty:
                    record = ticker_year_data.iloc[0]
                    for j, feature_name in enumerate(self.financial_features_to_use):
                        raw_value = record.get(feature_name, 0.0)
                        try:
                            numeric_value = float(raw_value)
                        except (ValueError, TypeError):
                            logger.warning(f"Không thể chuyển đổi financial feature '{feature_name}' của {ticker} (giá trị: {raw_value}) thành float. Dùng 0.0.")
                            numeric_value = 0.0

                        if self.fin_feat_means is not None and self.fin_feat_stds is not None and \
                           feature_name in self.fin_feat_means and feature_name in self.fin_feat_stds and \
                           pd.notna(self.fin_feat_stds[feature_name]) and self.fin_feat_stds[feature_name] > 1e-6:
                            mean_val = self.fin_feat_means[feature_name]
                            std_val = self.fin_feat_stds[feature_name]
                            asset_fin_features_raw[j] = (numeric_value - mean_val) / std_val
                        else:
                            asset_fin_features_raw[j] = numeric_value
                financial_features_flat_list.extend(asset_fin_features_raw)
        
        financial_features_final_array = np.array(financial_features_flat_list, dtype=np.float32)
        if financial_features_final_array.size != expected_fin_len: # Đảm bảo kích thước đúng ngay cả khi list rỗng
            logger.debug(f"Kích thước financial_features_final_array ({financial_features_final_array.size}) không khớp mong đợi ({expected_fin_len}). Sẽ dùng mảng zeros.")
            financial_features_final_array = np.zeros(expected_fin_len, dtype=np.float32)


        # 4. Classification probabilities features
        expected_prob_len = self.num_assets * len(self.prob_features_to_use)
        prob_features_flat_list = []
        if not self.classification_probs_full.empty and self.prob_features_to_use:
            for ticker in self.tickers:
                relevant_ticker_probs = self.classification_probs_full[
                    (self.classification_probs_full['ticker'] == ticker) &
                    (self.classification_probs_full['date'] <= current_date_for_features) # 'date' là cột trong classification_probs_full
                ]
                asset_prob_features = np.full(len(self.prob_features_to_use), 0.5, dtype=np.float32)
                if not relevant_ticker_probs.empty:
                    record = relevant_ticker_probs.sort_values('date', ascending=False).iloc[0]
                    for j, feature_name in enumerate(self.prob_features_to_use):
                        asset_prob_features[j] = float(record.get(feature_name, 0.5))
                prob_features_flat_list.extend(asset_prob_features)
        
        prob_features_final_array = np.array(prob_features_flat_list, dtype=np.float32)
        if prob_features_final_array.size != expected_prob_len:
            logger.debug(f"Kích thước prob_features_final_array ({prob_features_final_array.size}) không khớp mong đợi ({expected_prob_len}). Sẽ dùng mảng 0.5.")
            prob_features_final_array = np.full(expected_prob_len, 0.5, dtype=np.float32)


        state_parts = [
            price_history_features,
            current_weights_features,
            cash_ratio_feature,
            normalized_portfolio_value_feature,
            financial_features_final_array, # Luôn nối, ngay cả khi nó là mảng zeros
            prob_features_final_array      # Luôn nối, ngay cả khi nó là mảng 0.5s
        ]
        state = np.concatenate(state_parts)

        if state.shape[0] != self.observation_space.shape[0]:
            logger.error(f"Lỗi kích thước trạng thái CUỐI CÙNG! Mong đợi {self.observation_space.shape[0]}, nhận được {state.shape[0]}. ")
            # ... (log chi tiết hơn các thành phần nếu cần)
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._current_prices_idx = self.lookback_window_size
        self.portfolio_value = self.initial_capital
        self.current_weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        self.episode_return_history = [] # Lưu tổng phần thưởng của episode
        self.episode_portfolio_log_returns = [] # Lưu log return của danh mục ở mỗi bước

        observation = self._get_state()
        info = {}
        logger.debug(f"Env Reset: Chỉ số bắt đầu {self._current_prices_idx}, Giá trị danh mục {self.portfolio_value:.2f}")
        return observation, info

    def step(self, action: np.ndarray):
        """
        action: đầu ra thô từ mạng policy cho tỷ trọng mục tiêu mới (tài sản + tiền mặt)
        """
        # 1. Xác định tỷ trọng mục tiêu từ hành động (ví dụ: áp dụng softmax)
        target_weights_raw = action
        target_weights = np.exp(target_weights_raw) / np.sum(np.exp(target_weights_raw))
        target_weights = target_weights.astype(np.float32)

        # 2. Tính toán chi phí giao dịch
        # Giá trị tài sản cần mua/bán
        # Giá trị tài sản hiện tại (không bao gồm tiền mặt)
        portfolio_value_before_trade_and_cost = self.portfolio_value # Lưu giá trị trước khi hành động
        prices_at_rebalance_start = self.prices_df.iloc[self._current_prices_idx][self.tickers].values
        
        # Kiểm tra giá NaN
        if np.isnan(prices_at_rebalance_start).any() or np.any(prices_at_rebalance_start <= 0):
            logger.warning(f"Giá NaN hoặc <=0 tại bước {self.current_step}, idx {self._current_prices_idx}. Giữ nguyên danh mục.")
            # Không giao dịch, giá trị danh mục không đổi so với kỳ trước (ngoại trừ biến động giá nếu có)
            # Để đơn giản, coi như không có thay đổi giá trị nếu không giao dịch được
            step_portfolio_log_return = 0.0 # Không có lợi nhuận/lỗ từ giao dịch này
            transaction_costs = 0.0
            # Cần cập nhật giá trị danh mục dựa trên biến động giá nếu không giao dịch
            # For simplicity now, if cannot trade, assume portfolio value from market move is based on *current_weights*
            next_prices_idx_no_trade = self._current_prices_idx + self.rebalance_frequency_days
            if next_prices_idx_no_trade < len(self.prices_df):
                prices_at_rebalance_end_no_trade = self.prices_df.iloc[next_prices_idx_no_trade][self.tickers].values
                if not np.isnan(prices_at_rebalance_end_no_trade).any() and not np.any(prices_at_rebalance_end_no_trade <= 0):
                    # shares based on current_weights (before trying to apply new action)
                    current_shares = np.zeros_like(self.current_weights[:-1])
                    valid_prices_mask_current = prices_at_rebalance_start > 0 # Use start prices for share calculation
                    current_shares[valid_prices_mask_current] = (self.current_weights[:-1][valid_prices_mask_current] * portfolio_value_before_trade_and_cost) / prices_at_rebalance_start[valid_prices_mask_current]

                    value_of_assets_eod = np.sum(current_shares * prices_at_rebalance_end_no_trade)
                    cash_value = self.current_weights[-1] * portfolio_value_before_trade_and_cost
                    self.portfolio_value = value_of_assets_eod + cash_value
                    if portfolio_value_before_trade_and_cost > 1e-6:
                        step_portfolio_log_return = np.log(self.portfolio_value / portfolio_value_before_trade_and_cost)
                else: # Cannot determine next prices either
                    pass # self.portfolio_value remains portfolio_value_before_trade_and_cost
            # Weights không thay đổi
        else:
            current_asset_values = self.current_weights[:-1] * portfolio_value_before_trade_and_cost
            target_asset_dollar_values = target_weights[:-1] * portfolio_value_before_trade_and_cost # Mục tiêu giá trị $ cho tài sản

            trades_value = np.sum(np.abs(target_asset_dollar_values - current_asset_values))
            transaction_costs = trades_value * self.transaction_cost_pct
            portfolio_value_after_costs = portfolio_value_before_trade_and_cost - transaction_costs

            # Tiền mặt sau khi phân bổ cho tài sản mục tiêu và trừ chi phí
            cash_after_asset_alloc_and_costs = portfolio_value_after_costs * target_weights[-1]

            # Giá trị tài sản mục tiêu sau chi phí
            assets_value_after_costs = portfolio_value_after_costs * (1 - target_weights[-1])


            next_prices_idx = self._current_prices_idx + self.rebalance_frequency_days
            if next_prices_idx >= len(self.prices_df):
                self.portfolio_value = portfolio_value_after_costs # Không có biến động giá thêm
                step_portfolio_log_return = np.log(self.portfolio_value / portfolio_value_before_trade_and_cost) if portfolio_value_before_trade_and_cost > 1e-6 else 0.0
                self.current_weights = target_weights # Cập nhật về tỷ trọng mục tiêu

                terminated = True
                truncated = False
                self.episode_portfolio_log_returns.append(step_portfolio_log_return)
                reward = step_portfolio_log_return
                if self.reward_turnover_penalty_factor > 0 and portfolio_value_before_trade_and_cost > 1e-6:
                     turnover_ratio = trades_value / portfolio_value_before_trade_and_cost
                     reward -= self.reward_turnover_penalty_factor * turnover_ratio
                self.episode_return_history.append(reward)
                terminated = True; truncated = False
                info = {'portfolio_value': self.portfolio_value, 'weights': self.current_weights, 'transaction_costs': transaction_costs, 'step_reward': reward, 'raw_log_return': step_portfolio_log_return}
                if terminated: info['cumulative_return'] = np.sum(self.episode_return_history); info['sharpe_ratio'] = self.calculate_sharpe(self.episode_portfolio_log_returns)
                return self._get_state(), reward, terminated, truncated, info

            prices_at_rebalance_end = self.prices_df.iloc[next_prices_idx][self.tickers].values
            if np.isnan(prices_at_rebalance_end).any() or np.any(prices_at_rebalance_end <= 0):
                 logger.warning(f"Giá NaN hoặc <=0 ở cuối kỳ tái cân bằng (idx {next_prices_idx}). Dùng giá đầu kỳ.")
                 prices_at_rebalance_end = prices_at_rebalance_start

        # Số lượng cổ phiếu dựa trên giá trị tài sản mục tiêu sau chi phí và giá đầu kỳ
            shares = np.zeros_like(target_weights[:-1])
            valid_prices_mask_start = prices_at_rebalance_start > 0
            # assets_value_after_costs là tổng giá trị $ dự định cho các tài sản
            # target_weights[:-1] / np.sum(target_weights[:-1]) là tỷ trọng tương đối của từng tài sản trong phần tài sản
            if np.sum(target_weights[:-1]) > 1e-9: # Tránh chia cho 0 nếu không có tài sản nào
                relative_asset_weights = target_weights[:-1] / np.sum(target_weights[:-1])
                dollar_value_per_asset_target = relative_asset_weights * assets_value_after_costs
                shares[valid_prices_mask_start] = dollar_value_per_asset_target[valid_prices_mask_start] / prices_at_rebalance_start[valid_prices_mask_start]
            
            value_of_assets_at_step_end = np.sum(shares * prices_at_rebalance_end)
            self.portfolio_value = value_of_assets_at_step_end + cash_after_asset_alloc_and_costs

            step_portfolio_log_return = 0.0
            if portfolio_value_before_trade_and_cost > 1e-6:
                step_portfolio_log_return = np.log(self.portfolio_value / portfolio_value_before_trade_and_cost)

            if self.portfolio_value > 1e-6:
                self.current_weights[:-1] = (shares * prices_at_rebalance_end) / self.portfolio_value
                self.current_weights[-1] = cash_after_asset_alloc_and_costs / self.portfolio_value
                # Đảm bảo tổng tỷ trọng là 1 do lỗi làm tròn số
                self.current_weights /= np.sum(self.current_weights)
            else:
                self.current_weights[:-1] = 0; self.current_weights[-1] = 1.0

        # Tính toán phần thưởng cuối cùng
        self.episode_portfolio_log_returns.append(step_portfolio_log_return)
        
        if self.reward_use_log_return:
            reward = step_portfolio_log_return
        else: # Simple return
            reward = (self.portfolio_value / portfolio_value_before_trade_and_cost) - 1 if portfolio_value_before_trade_and_cost > 1e-6 else 0.0

        # Áp dụng hình phạt turnover nếu có
        if self.reward_turnover_penalty_factor > 0 and portfolio_value_before_trade_and_cost > 1e-6:
            # trades_value đã được tính ở trên cho trường hợp có giao dịch
            if np.isnan(prices_at_rebalance_start).any() or np.any(prices_at_rebalance_start <= 0): # Nếu không giao dịch
                turnover_ratio = 0.0
            else:
                turnover_ratio = trades_value / portfolio_value_before_trade_and_cost
            reward -= self.reward_turnover_penalty_factor * turnover_ratio
        
        self.episode_return_history.append(reward) # Lưu phần thưởng (có thể đã bao gồm penalty)

        self._current_prices_idx += self.rebalance_frequency_days
        self.current_step += 1

        terminated = self.portfolio_value < 0.01 * self.initial_capital
        if not terminated:
            terminated = self.current_step >= self.max_steps # >= vì max_steps là số bước tối đa (0-indexed)
        
        truncated = False
        if self._current_prices_idx + self.rebalance_frequency_days > len(self.prices_df) : # > thay vì >= để cho phép bước cuối cùng
             terminated = True

        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights,
            'transaction_costs': transaction_costs,
            'step_reward': reward,
            'raw_log_return': step_portfolio_log_return # Luôn lưu log return thô để phân tích
        }
        
        if terminated:
            info['cumulative_reward_shaped'] = np.sum(self.episode_return_history)
            info['cumulative_log_return'] = np.sum(self.episode_portfolio_log_returns)
            info['sharpe_ratio_from_log_returns'] = self.calculate_sharpe(self.episode_portfolio_log_returns)
            logger.debug(f"Episode terminated. Final Ptf Value: {self.portfolio_value:.2f}, Cum Rew (shaped): {info['cumulative_reward_shaped']:.4f}, Cum LogRet: {info['cumulative_log_return']:.4f}")


        logger.debug(
            f"Step: {self.current_step}/{self.max_steps}, PtfVal: {self.portfolio_value:.2f}, "
            f"ShapedRew: {reward:.5f}, LogRet: {step_portfolio_log_return:.5f}, TC: {transaction_costs:.2f}, "
            f"Wts: {[f'{w:.2f}' for w in self.current_weights]}"
        )
        return self._get_state(), reward, terminated, truncated, info
        
    def calculate_sharpe(self, log_returns_history, risk_free_rate_daily=0.0): # Nhận log returns
        if not log_returns_history or len(log_returns_history) < 2:
            return 0.0
        # Chuyển log return thành simple return để tính std chính xác hơn cho Sharpe truyền thống
        # Hoặc có thể tính trực tiếp từ log return (ít phổ biến hơn cho Sharpe nhưng vẫn có thể)
        # Đối với Sharpe, std của simple returns thường được sử dụng.
        # simple_returns = np.exp(np.array(log_returns_history)) - 1
        # mean_simple_return = np.mean(simple_returns)
        # std_simple_return = np.std(simple_returns)

        # Sử dụng trực tiếp log returns (xấp xỉ cho lợi nhuận nhỏ)
        log_returns_array = np.array(log_returns_history)
        mean_log_return = np.mean(log_returns_array)
        std_log_return = np.std(log_returns_array)

        if std_log_return < 1e-9: # Gần như không có biến động
            return 0.0 if abs(mean_log_return - risk_free_rate_daily) < 1e-9 else np.inf * np.sign(mean_log_return - risk_free_rate_daily)
        
        sharpe = (mean_log_return - risk_free_rate_daily) / std_log_return * np.sqrt(252) # Giả sử daily log returns
        return sharpe

    def render(self, mode='human'): # ... như cũ ...
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: {self.portfolio_value:.2f}")
            print(f"Weights: {self.current_weights}")
            print(f"Last shaped reward: {self.episode_return_history[-1] if self.episode_return_history else 'N/A'}")
            print(f"Last portfolio log return: {self.episode_portfolio_log_returns[-1] if self.episode_portfolio_log_returns else 'N/A'}")


    def close(self):
        pass