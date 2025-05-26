# marketml/portfolio_opt/backtesting.py
import pandas as pd
import numpy as np
import quantstats as qs
import logging
# from pathlib import Path # Không cần nếu sys.path bị xóa
# import sys # XÓA DÒNG NÀY

try:
    from marketml.configs import configs
except ImportError:
    # Ghi log lỗi ban đầu ra stdout nếu logger chưa kịp thiết lập
    print("CRITICAL ERROR in backtesting.py: Could not import 'marketml.configs'. Ensure it's accessible.")
    raise # Re-raise để dừng hẳn

logger = logging.getLogger(__name__) # Lấy logger cục bộ cho module này

class PortfolioBacktester:
    """
    Handles portfolio rebalancing logic and performance calculation during a backtest.
    """
    def __init__(self, initial_capital: float, transaction_cost_bps: int = 0):
        """
        Initializes the PortfolioBacktester.

        Args:
            initial_capital (float): The starting capital for the portfolio.
            transaction_cost_bps (int): Transaction costs in basis points (e.g., 5 for 0.05%).
        """
        if initial_capital <= 0:
            logger.error(f"Initial capital must be positive. Received: {initial_capital}")
            raise ValueError("Initial capital must be positive.")

        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10000.0 # Convert bps to percentage
        self.portfolio_history = [] # Stores dicts of portfolio state at each rebalance/evaluation point
        self.cash = initial_capital
        self.current_holdings = {} # Stores {ticker: number_of_shares}
        logger.info(f"PortfolioBacktester initialized with Initial Capital: {self.initial_capital:.2f}, Tx Cost: {transaction_cost_bps} bps")

    def _calculate_transaction_costs(self, trades_value: float) -> float:
        """Calculates transaction costs based on the total value of trades."""
        return trades_value * self.transaction_cost_pct

    def rebalance(self, date: pd.Timestamp, new_weights: dict, current_prices: pd.Series):
        """
        Rebalances the portfolio to target new_weights at the given date and prices.

        Args:
            date (pd.Timestamp): The date of rebalancing.
            new_weights (dict): Target weights {ticker: weight}.
            current_prices (pd.Series): Prices of assets on the rebalance date, indexed by ticker.
        """
        logger.info(f"Attempting rebalance on {date.date()} with target weights sum: {sum(new_weights.values()):.4f}")
        logger.debug(f"Target weights for rebalance: {new_weights}")

        # 1. Calculate current portfolio value before rebalancing
        current_value_of_held_assets = 0
        for ticker_held, shares_held in self.current_holdings.items():
            asset_price = current_prices.get(ticker_held)
            if pd.notna(asset_price) and asset_price > 0:
                current_value_of_held_assets += shares_held * asset_price
            else:
                logger.warning(f"Price for currently held asset {ticker_held} is missing or invalid ({asset_price}) "
                               f"on {date.date()}. Its value is considered 0 for this rebalance.")
        
        portfolio_value_before_rebalance = self.cash + current_value_of_held_assets
        logger.debug(f"Portfolio value BEFORE rebalance on {date.date()}: {portfolio_value_before_rebalance:.2f} "
                     f"(Cash: {self.cash:.2f}, Assets: {current_value_of_held_assets:.2f})")

        if portfolio_value_before_rebalance <= 1e-6: # Effectively zero or negative
            if not self.portfolio_history: # First rebalance and broke
                logger.warning(f"Portfolio value before rebalance is {portfolio_value_before_rebalance:.2f} on {date.date()}. "
                               f"Resetting to initial capital for allocation attempt if it's the first operation.")
                portfolio_value_before_rebalance = self.initial_capital # Try to allocate from scratch
                self.cash = self.initial_capital
                self.current_holdings = {}
            else:
                logger.error(f"CRITICAL: Portfolio value ({portfolio_value_before_rebalance:.2f}) is zero or negative on {date.date()} "
                               f"and it's not the first rebalance. Holding previous state.")
                if self.portfolio_history:
                    last_entry = self.portfolio_history[-1].copy()
                    last_entry['date'] = date
                    last_entry['returns'] = 0.0 # No change if cannot rebalance
                    self.portfolio_history.append(last_entry)
                return

        # 2. Filter new_weights for valid tickers with positive prices
        valid_target_weights = {
            ticker: weight for ticker, weight in new_weights.items()
            if ticker in current_prices.index and pd.notna(current_prices[ticker]) and current_prices[ticker] > 0
        }

        if not valid_target_weights:
            logger.warning(f"No valid tickers with positive prices found in new_weights for rebalancing on {date.date()}. "
                           "Holding current cash/asset allocation.")
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value_before_rebalance,
                'weights': {t: (s * current_prices.get(t, 0)) / portfolio_value_before_rebalance
                            if portfolio_value_before_rebalance > 1e-6 else 0
                            for t, s in self.current_holdings.items()},
                'cash': self.cash,
                'returns': 0.0
            })
            return
        
        # Normalize valid_target_weights if their sum is not 1 (e.g. due to filtering)
        sum_valid_weights = sum(valid_target_weights.values())
        if abs(sum_valid_weights - 1.0) > 1e-6 and sum_valid_weights > 1e-6 : # If sum is not 1 and not zero
            logger.debug(f"Normalizing valid target weights from sum {sum_valid_weights:.4f} to 1.0")
            valid_target_weights = {t: w / sum_valid_weights for t, w in valid_target_weights.items()}


        # 3. Calculate target dollar values and shares for valid tickers
        target_shares = {}
        # Value available for assets is (1 - target cash weight) * portfolio_value_before_rebalance
        # Assuming new_weights dict includes cash implicitly or explicitly as (1 - sum of asset weights)
        # For PyPortfolioOpt, weights usually sum to 1 for assets.
        # If a cash weight is desired, it should be handled by the strategy providing new_weights.
        # Here, we assume new_weights are for assets and sum to 1 (or are normalized to sum to 1).
        
        # Iterate to calculate trades and costs
        current_asset_dollar_values_before_trade = {
            ticker: shares * current_prices.get(ticker, 0)
            for ticker, shares in self.current_holdings.items()
            if current_prices.get(ticker, 0) > 0
        }

        target_asset_dollar_values_ideal = {
            ticker: portfolio_value_before_rebalance * weight
            for ticker, weight in valid_target_weights.items()
        }
        
        trades_value_total = 0
        for ticker in set(current_asset_dollar_values_before_trade.keys()) | set(target_asset_dollar_values_ideal.keys()):
            current_val = current_asset_dollar_values_before_trade.get(ticker, 0)
            target_val = target_asset_dollar_values_ideal.get(ticker, 0)
            trades_value_total += abs(target_val - current_val)
            
        total_transaction_costs = self._calculate_transaction_costs(trades_value_total)
        logger.debug(f"Total value of trades: {trades_value_total:.2f}, Transaction costs for rebalance on {date.date()}: {total_transaction_costs:.2f}")

        # Portfolio value after transaction costs, before allocating to new weights
        portfolio_value_for_new_allocation = portfolio_value_before_rebalance - total_transaction_costs

        # 5. Update cash and holdings
        new_cash = portfolio_value_for_new_allocation # Start with all cash after costs
        new_holdings_shares = {}

        for ticker, target_weight in valid_target_weights.items():
            dollar_amount_for_ticker = portfolio_value_for_new_allocation * target_weight
            if current_prices[ticker] > 0:
                new_holdings_shares[ticker] = dollar_amount_for_ticker / current_prices[ticker]
                new_cash -= dollar_amount_for_ticker # Reduce cash by amount allocated to this asset
            else: # Should have been filtered by valid_target_weights, but as a safeguard
                new_holdings_shares[ticker] = 0
        
        self.cash = new_cash
        self.current_holdings = new_holdings_shares.copy()

        # 6. Record history after rebalance
        value_of_assets_after_rebalance = 0
        for ticker, shares in self.current_holdings.items():
            asset_price = current_prices.get(ticker) # Prices at rebalance point
            if pd.notna(asset_price) and asset_price > 0:
                value_of_assets_after_rebalance += shares * asset_price
        
        portfolio_value_after_rebalance_and_costs = self.cash + value_of_assets_after_rebalance
        logger.debug(f"Portfolio value AFTER rebalance & costs on {date.date()}: {portfolio_value_after_rebalance_and_costs:.2f} "
                      f"(Cash: {self.cash:.2f}, Assets: {value_of_assets_after_rebalance:.2f})")

        actual_weights_after_rebalance = {}
        if portfolio_value_after_rebalance_and_costs > 1e-6:
            for ticker, shares in self.current_holdings.items():
                asset_price = current_prices.get(ticker)
                if pd.notna(asset_price) and asset_price > 0:
                    actual_weights_after_rebalance[ticker] = (shares * asset_price) / portfolio_value_after_rebalance_and_costs
        
        period_return = 0.0
        if self.portfolio_history: # If not the very first operation
            # The 'return' is calculated from the value *before this rebalance* to the value *after this rebalance*
            # This captures the impact of this rebalance itself (costs) relative to the portfolio value just before it.
            # Note: update_portfolio_values_daily will calculate day-over-day returns.
            value_at_previous_point = self.portfolio_history[-1]['value']
            if value_at_previous_point > 1e-6:
                period_return = (portfolio_value_after_rebalance_and_costs - value_at_previous_point) / value_at_previous_point
        else: # First operation, return is vs initial capital if we consider this the first data point.
              # Or 0 if we only care about returns *between* rebalance points.
              # Let's make it return vs portfolio_value_before_rebalance to capture cost impact
            if portfolio_value_before_rebalance > 1e-6:
                period_return = (portfolio_value_after_rebalance_and_costs - portfolio_value_before_rebalance) / portfolio_value_before_rebalance


        self.portfolio_history.append({
            'date': date,
            'value': portfolio_value_after_rebalance_and_costs,
            'weights': actual_weights_after_rebalance,
            'cash': self.cash,
            'returns': period_return # This specific return reflects the rebalance event
        })

    def get_portfolio_performance_df(self) -> pd.DataFrame:
        """Returns the portfolio history as a DataFrame."""
        if not self.portfolio_history:
            logger.warning("Portfolio history is empty.")
            return pd.DataFrame()
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index() # Ensure sorted by date
        return df

    def calculate_metrics(self, portfolio_returns_series: pd.Series, benchmark_returns: pd.Series = None) -> pd.Series:
        """
        Calculates performance metrics using QuantStats.

        Args:
            portfolio_returns_series (pd.Series): Daily returns of the portfolio.
            benchmark_returns (pd.Series, optional): Daily returns of the benchmark.

        Returns:
            pd.Series: Series containing calculated performance metrics.
        """
        if portfolio_returns_series is None or portfolio_returns_series.empty or portfolio_returns_series.isnull().all():
            logger.error("Portfolio returns series is empty or all NaN. Cannot calculate metrics.")
            return pd.Series(dtype=float) # Return empty Series

        logger.info("Calculating performance metrics using QuantStats...")
        qs.extend_pandas()
        
        # Ensure returns are float
        portfolio_returns_series = portfolio_returns_series.astype(float).fillna(0.0)

        metrics_dict = {
            'Cumulative Return': qs.stats.comp(portfolio_returns_series) * 100,
            'Annualized Return': qs.stats.cagr(portfolio_returns_series, compounded=True) * 100,
            'Annualized Volatility': qs.stats.volatility(portfolio_returns_series, annualize=True, compounded=True) * 100, # compounded=True for daily simple returns
            'Sharpe Ratio': qs.stats.sharpe(portfolio_returns_series, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True, compounded=True),
            'Sortino Ratio': qs.stats.sortino(portfolio_returns_series, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True, compounded=True),
            'Max Drawdown': qs.stats.max_drawdown(portfolio_returns_series) * 100,
            'Skew': qs.stats.skew(portfolio_returns_series),
            'Kurtosis': qs.stats.kurtosis(portfolio_returns_series),
            'Calmar Ratio': qs.stats.calmar(portfolio_returns_series),
            'Value at Risk (VaR)': qs.stats.var(portfolio_returns_series) * 100, # Daily VaR
            'Conditional VaR (CVaR)': qs.stats.cvar(portfolio_returns_series) * 100, # Daily CVaR
        }

        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_returns = benchmark_returns.astype(float).fillna(0.0)
            logger.info("Calculating benchmark metrics...")
            try:
                common_index = portfolio_returns_series.index.intersection(benchmark_returns.index)
                if not common_index.empty:
                    portfolio_aligned = portfolio_returns_series.loc[common_index]
                    benchmark_aligned = benchmark_returns.loc[common_index]

                    metrics_dict['Benchmark Cumulative Return'] = qs.stats.comp(benchmark_aligned) * 100
                    metrics_dict['Benchmark Annualized Return'] = qs.stats.cagr(benchmark_aligned, compounded=True) * 100
                    metrics_dict['Benchmark Annualized Volatility'] = qs.stats.volatility(benchmark_aligned, annualize=True, compounded=True) * 100
                    metrics_dict['Beta'] = qs.stats.beta(portfolio_aligned, benchmark_aligned)
                    metrics_dict['Alpha (Annualized)'] = qs.stats.alpha(portfolio_aligned, benchmark_aligned, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True) * 100
                    metrics_dict['Information Ratio'] = qs.stats.information_ratio(portfolio_aligned, benchmark_aligned)
                else:
                    logger.warning("No common index found between portfolio and benchmark returns. Skipping benchmark-related metrics.")
            except Exception as e_bench:
                logger.error(f"Error calculating benchmark-related metrics: {e_bench}", exc_info=True)

        return pd.Series(metrics_dict)