# marketml/portfolio_opt/backtesting.py
import pandas as pd
import numpy as np
import quantstats as qs
import logging # Import logging
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[2] # .ndmh/
sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

try:
    from marketml.configs import configs
except ImportError as e:
    print(f"CRITICAL ERROR in backtesting.py: Could not import configs. {e}")
    sys.exit(1)

# Lấy logger đã được thiết lập hoặc tạo logger riêng cho module này
logger = logging.getLogger(__name__)
if not logging.getLogger(__name__).hasHandlers(): # Kiểm tra logger cha
    # Fallback nếu logger "marketml_project" chưa được cấu hình bởi script chính
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PortfolioBacktester:
    def __init__(self, initial_capital, transaction_cost_bps=0):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps / 10000.0
        self.portfolio_history = []
        self.cash = initial_capital
        self.current_holdings = {}

    def _calculate_transaction_costs(self, trades_value):
        return trades_value * self.transaction_cost_bps

    def rebalance(self, date, new_weights: dict, current_prices: pd.Series):
        logger.debug(f"Attempting rebalance on {date} with target weights: {new_weights}")
        # 1. Tính giá trị danh mục hiện tại TRƯỚC khi tái cân bằng
        current_value_of_held_assets = 0
        for ticker_held, shares_held in self.current_holdings.items():
            if ticker_held in current_prices and pd.notna(current_prices[ticker_held]) and current_prices[ticker_held] > 0:
                current_value_of_held_assets += shares_held * current_prices[ticker_held]
            else:
                logger.warning(f"Price for currently held asset {ticker_held} not available or invalid on {date}. Assuming its value is 0 for this rebalance step.")
        
        portfolio_value_before_rebalance = self.cash + current_value_of_held_assets
        logger.debug(f"Portfolio value BEFORE rebalance on {date}: {portfolio_value_before_rebalance:.2f} (Cash: {self.cash:.2f}, Assets: {current_value_of_held_assets:.2f})")

        if portfolio_value_before_rebalance <= 0:
            if not self.portfolio_history: # Lần đầu tiên và giá trị âm/0 (rất hiếm)
                logger.warning(f"Portfolio value before rebalance is {portfolio_value_before_rebalance} on {date}. Resetting to initial capital for allocation.")
                portfolio_value_before_rebalance = self.initial_capital
            else: # Nếu không phải lần đầu mà giá trị âm/0
                logger.error(f"Critical: Portfolio value before rebalance is {portfolio_value_before_rebalance} on {date}. Cannot proceed with rebalance. Holding previous state.")
                # Ghi lại trạng thái không đổi
                if self.portfolio_history:
                    last_entry = self.portfolio_history[-1].copy()
                    last_entry['date'] = date
                    last_entry['returns'] = 0.0 # No change in value
                    self.portfolio_history.append(last_entry)
                return


        # 2. Lọc các ticker hợp lệ để giao dịch (có giá và có trong new_weights)
        valid_target_tickers = {
            t: w for t, w in new_weights.items()
            if t in current_prices.index and pd.notna(current_prices[t]) and current_prices[t] > 0
        }

        if not valid_target_tickers:
            logger.warning(f"No valid tickers with prices from new_weights for rebalancing on {date}. Holding cash/previous state.")
            # Ghi lại trạng thái không đổi (giữ nguyên cash và holdings hiện tại)
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value_before_rebalance, # Giá trị không đổi
                'weights': {t: (s*current_prices.get(t,0))/portfolio_value_before_rebalance if portfolio_value_before_rebalance > 0 else 0 for t,s in self.current_holdings.items()},
                'cash': self.cash,
                'returns': 0.0 # Không có thay đổi do không rebalance
            })
            return

        # 3. Tính toán số lượng cổ phiếu mục tiêu mới
        target_shares = {}
        total_allocated_to_assets = 0
        for ticker, target_weight in valid_target_tickers.items():
            target_value_for_ticker = portfolio_value_before_rebalance * target_weight
            target_shares[ticker] = target_value_for_ticker / current_prices[ticker]
            total_allocated_to_assets += target_value_for_ticker
        
        # 4. Tính toán giao dịch và chi phí
        trades_value_sell = 0
        trades_value_buy = 0

        # Những cổ phiếu cần bán hoặc giảm bớt
        for ticker_held, shares_currently_held in self.current_holdings.items():
            shares_to_hold_new = target_shares.get(ticker_held, 0) # Nếu ticker không có trong target_shares, bán hết
            if shares_currently_held > shares_to_hold_new:
                if ticker_held in current_prices and pd.notna(current_prices[ticker_held]) and current_prices[ticker_held] > 0:
                     trades_value_sell += (shares_currently_held - shares_to_hold_new) * current_prices[ticker_held]
        
        # Những cổ phiếu cần mua hoặc mua thêm
        for ticker_target, shares_to_buy_target in target_shares.items():
            shares_currently_held = self.current_holdings.get(ticker_target, 0)
            if shares_to_buy_target > shares_currently_held:
                 if ticker_target in current_prices and pd.notna(current_prices[ticker_target]) and current_prices[ticker_target] > 0:
                    trades_value_buy += (shares_to_buy_target - shares_currently_held) * current_prices[ticker_target]

        total_transaction_costs = self._calculate_transaction_costs(trades_value_sell + trades_value_buy)
        logger.debug(f"Transaction costs for rebalance on {date}: {total_transaction_costs:.2f}")

        # 5. Cập nhật tiền mặt và số lượng nắm giữ
        # Tiền mặt ban đầu cho tái cân bằng này là portfolio_value_before_rebalance
        # Sau khi phân bổ cho các tài sản mục tiêu (total_allocated_to_assets), phần còn lại là tiền mặt dự kiến
        # Sau đó trừ đi chi phí giao dịch
        self.cash = portfolio_value_before_rebalance - total_allocated_to_assets - total_transaction_costs
        self.current_holdings = target_shares.copy() # Cập nhật số lượng nắm giữ mới

        # 6. Ghi lại lịch sử sau tái cân bằng
        portfolio_value_after_rebalance = 0
        actual_weights_after_trade = {}
        
        value_of_assets_after_rebalance = 0
        for ticker, shares in self.current_holdings.items():
            if ticker in current_prices and pd.notna(current_prices[ticker]) and current_prices[ticker] > 0:
                value_of_assets_after_rebalance += shares * current_prices[ticker]
        
        portfolio_value_after_rebalance = self.cash + value_of_assets_after_rebalance
        logger.debug(f"Portfolio value AFTER rebalance on {date}: {portfolio_value_after_rebalance:.2f} (Cash: {self.cash:.2f}, Assets: {value_of_assets_after_rebalance:.2f})")

        if portfolio_value_after_rebalance > 0:
            for ticker, shares in self.current_holdings.items():
                if ticker in current_prices and pd.notna(current_prices[ticker]) and current_prices[ticker] > 0:
                    actual_weights_after_trade[ticker] = (shares * current_prices[ticker]) / portfolio_value_after_rebalance
        # Else: actual_weights_after_trade sẽ rỗng nếu portfolio_value là 0

        period_return = 0.0
        if len(self.portfolio_history) > 0:
            # Lấy giá trị của entry cuối cùng trong history, đó là giá trị tại điểm rebalance trước đó
            # hoặc giá trị tại ngày giao dịch cuối cùng trước ngày rebalance này.
            prev_value = self.portfolio_history[-1]['value']
            if prev_value > 0:
                period_return = (portfolio_value_after_rebalance - prev_value) / prev_value
        
        self.portfolio_history.append({
            'date': date,
            'value': portfolio_value_after_rebalance,
            'weights': actual_weights_after_trade,
            'cash': self.cash,
            'returns': period_return # Return này là tại điểm tái cân bằng, so với điểm trước đó
        })


    def get_portfolio_performance_df(self):
        if not self.portfolio_history:
            return pd.DataFrame()
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df

    def calculate_metrics(self, portfolio_returns_series, benchmark_returns=None):
        if portfolio_returns_series is None or portfolio_returns_series.empty or portfolio_returns_series.isnull().all():
            logger.error("Portfolio returns series is empty or all NaN. Cannot calculate metrics.")
            return None

        # Sử dụng quantstats
        qs.extend_pandas() # Mở rộng pandas với các hàm của quantstats
        
        # Tạo báo cáo HTML (hoặc chỉ lấy metrics)
        # output_path = configs.RESULTS_DIR / "portfolio_performance_report.html"
        # qs.reports.html(portfolio_returns_series, benchmark=benchmark_returns, output=str(output_path), title='Portfolio Performance')
        # print(f"QuantStats report saved to {output_path}")

        metrics_dict = {
            'Cumulative Return': qs.stats.comp(portfolio_returns_series) * 100,
            'Annualized Return': qs.stats.cagr(portfolio_returns_series, compounded=True) * 100, # compounded=True nếu returns là simple, False nếu đã là log returns
            'Annualized Volatility': qs.stats.volatility(portfolio_returns_series, annualize=True) * 100,
            'Sharpe Ratio': qs.stats.sharpe(portfolio_returns_series, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True), # compounded=True nếu returns là simple
            'Sortino Ratio': qs.stats.sortino(portfolio_returns_series, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True), # compounded=True nếu returns là simple
            'Max Drawdown': qs.stats.max_drawdown(portfolio_returns_series) * 100,
        }
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_returns = benchmark_returns.fillna(0.0).astype(float)
            try:
                common_index = portfolio_returns_series.index.intersection(benchmark_returns.index)
                if not common_index.empty:
                    # portfolio_aligned = portfolio_returns_series.loc[common_index] # Không cần thiết nếu dùng trực tiếp
                    benchmark_aligned = benchmark_returns.loc[common_index]

                    metrics_dict['Benchmark Cumulative Return'] = qs.stats.comp(benchmark_aligned) * 100
                    metrics_dict['Benchmark Annualized Return'] = qs.stats.cagr(benchmark_aligned, compounded=True) * 100
                    metrics_dict['Benchmark Annualized Volatility'] = qs.stats.volatility(benchmark_aligned, annualize=True, compounded=True) * 100
                else:
                    logger.warning("No common index between portfolio and benchmark returns for comparison.")
            except Exception as e_bench:
                logger.error(f"Error calculating benchmark metrics: {e_bench}")

        return pd.Series(metrics_dict)