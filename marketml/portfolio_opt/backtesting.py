# marketml/portfolio_opt/backtesting.py

import pandas as pd
import numpy as np
import quantstats as qs
import logging

try:
    from marketml.configs import configs
except ImportError:
    print("CRITICAL ERROR in backtesting.py: Could not import 'marketml.configs'. Ensure it's accessible.")
    raise

logger = logging.getLogger(__name__)

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
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.portfolio_history = []
        self.cash = initial_capital
        self.current_holdings = {}
        logger.info(f"PortfolioBacktester initialized with Initial Capital: {self.initial_capital:.2f}, Tx Cost: {transaction_cost_bps} bps")

    def _calculate_transaction_costs(self, trades_value: float) -> float:
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

        # Section: Calculate current portfolio value before rebalancing
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

        if portfolio_value_before_rebalance <= 1e-6:
            if not self.portfolio_history:
                logger.warning(f"Portfolio value before rebalance is {portfolio_value_before_rebalance:.2f} on {date.date()}. "
                               f"Resetting to initial capital for allocation attempt if it's the first operation.")
                portfolio_value_before_rebalance = self.initial_capital
                self.cash = self.initial_capital
                self.current_holdings = {}
            else:
                logger.error(f"CRITICAL: Portfolio value ({portfolio_value_before_rebalance:.2f}) is zero or negative on {date.date()} "
                               f"and it's not the first rebalance. Holding previous state.")
                if self.portfolio_history:
                    last_entry = self.portfolio_history[-1].copy()
                    last_entry['date'] = date
                    last_entry['returns'] = 0.0
                    self.portfolio_history.append(last_entry)
                return

        # Section: Filter new_weights for valid tickers with positive prices
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
        
        sum_valid_weights = sum(valid_target_weights.values())
        if abs(sum_valid_weights - 1.0) > 1e-6 and sum_valid_weights > 1e-6:
            logger.debug(f"Normalizing valid target weights from sum {sum_valid_weights:.4f} to 1.0")
            valid_target_weights = {t: w / sum_valid_weights for t, w in valid_target_weights.items()}

        # Section: Calculate target dollar values and shares for valid tickers
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

        portfolio_value_for_new_allocation = portfolio_value_before_rebalance - total_transaction_costs

        # Section: Update cash and holdings
        new_cash = portfolio_value_for_new_allocation
        new_holdings_shares = {}

        for ticker, target_weight in valid_target_weights.items():
            dollar_amount_for_ticker = portfolio_value_for_new_allocation * target_weight
            if current_prices[ticker] > 0:
                new_holdings_shares[ticker] = dollar_amount_for_ticker / current_prices[ticker]
                new_cash -= dollar_amount_for_ticker
            else:
                new_holdings_shares[ticker] = 0
        
        self.cash = new_cash
        self.current_holdings = new_holdings_shares.copy()

        # Section: Record history after rebalance
        value_of_assets_after_rebalance = 0
        for ticker, shares in self.current_holdings.items():
            asset_price = current_prices.get(ticker)
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
        if self.portfolio_history:
            value_at_previous_point = self.portfolio_history[-1]['value']
            if value_at_previous_point > 1e-6:
                period_return = (portfolio_value_after_rebalance_and_costs - value_at_previous_point) / value_at_previous_point
        else:
            if portfolio_value_before_rebalance > 1e-6:
                period_return = (portfolio_value_after_rebalance_and_costs - portfolio_value_before_rebalance) / portfolio_value_before_rebalance

        self.portfolio_history.append({
            'date': date,
            'value': portfolio_value_after_rebalance_and_costs,
            'weights': actual_weights_after_rebalance,
            'cash': self.cash,
            'returns': period_return
        })

    def get_portfolio_performance_df(self) -> pd.DataFrame:
        """Returns the portfolio history as a DataFrame."""
        if not self.portfolio_history:
            logger.warning("Portfolio history is empty.")
            return pd.DataFrame()
        df = pd.DataFrame(self.portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df

    def calculate_metrics(self, portfolio_returns_series: pd.Series, benchmark_returns: pd.Series = None, logger_instance: logging.Logger = None) -> pd.Series:
        """
        Calculates performance metrics using QuantStats with enhanced error handling and logging.

        Args:
            portfolio_returns_series (pd.Series): Daily returns of the portfolio.
            benchmark_returns (pd.Series, optional): Daily returns of the benchmark.
            logger_instance (logging.Logger, optional): Custom logger instance.

        Returns:
            pd.Series: Series containing calculated performance metrics.
        """
        current_logger = logger_instance if logger_instance else logger
        
        # Section: Input validation
        if portfolio_returns_series is None:
            current_logger.error("Portfolio returns series is None. Cannot calculate metrics.")
            return pd.Series(dtype=float)
        
        if not isinstance(portfolio_returns_series, pd.Series):
            current_logger.error(f"portfolio_returns_series is not a Pandas Series, but type {type(portfolio_returns_series)}. Cannot calculate metrics.")
            return pd.Series(dtype=float)
        
        if portfolio_returns_series.empty:
            current_logger.error("Portfolio returns series is empty. Cannot calculate metrics.")
            return pd.Series(dtype=float)
        
        # Section: Data preprocessing
        try:
            processed_returns = portfolio_returns_series.astype(float).fillna(0.0)
        except Exception as e:
            current_logger.error(f"Error converting returns to float: {e}", exc_info=True)
            return pd.Series(dtype=float)
        
        if processed_returns.isnull().any():
            current_logger.warning("Portfolio returns still contain NaN values after conversion, filling with zeros")
            processed_returns = processed_returns.fillna(0.0)
        
        current_logger.info("Calculating performance metrics using QuantStats...")
        qs.extend_pandas()
        
        # Section: Calculate base metrics
        try:
            metrics_dict = {
                'Cumulative Return': qs.stats.comp(processed_returns) * 100,
                'Annualized Return': qs.stats.cagr(processed_returns, compounded=True) * 100,
                'Annualized Volatility': qs.stats.volatility(processed_returns, annualize=True) * 100,
                'Sharpe Ratio': qs.stats.sharpe(processed_returns, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True),
                'Sortino Ratio': qs.stats.sortino(processed_returns, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True),
                'Max Drawdown': qs.stats.max_drawdown(processed_returns) * 100,
                'Skew': qs.stats.skew(processed_returns),
                'Kurtosis': qs.stats.kurtosis(processed_returns),
                'Calmar Ratio': qs.stats.calmar(processed_returns),
                'Value at Risk (VaR)': qs.stats.var(processed_returns) * 100,
                'Conditional VaR (CVaR)': qs.stats.cvar(processed_returns) * 100,
            }
        except Exception as e_metrics:
            current_logger.error(f"Error calculating base metrics: {e_metrics}", exc_info=True)
            return pd.Series(dtype=float)
        
        # Section: Calculate benchmark-related metrics if benchmark provided
        if benchmark_returns is not None:
            try:
                if not isinstance(benchmark_returns, pd.Series):
                    current_logger.warning("Benchmark returns is not a Pandas Series. Skipping benchmark metrics.")
                elif not benchmark_returns.empty:
                    benchmark_processed = benchmark_returns.astype(float).fillna(0.0)
                    
                    common_index = processed_returns.index.intersection(benchmark_processed.index)
                    if not common_index.empty:
                        portfolio_aligned = processed_returns.loc[common_index]
                        benchmark_aligned = benchmark_processed.loc[common_index]
                        
                        metrics_dict.update({
                            'Benchmark Cumulative Return': qs.stats.comp(benchmark_aligned) * 100,
                            'Benchmark Annualized Return': qs.stats.cagr(benchmark_aligned, compounded=True) * 100,
                            'Benchmark Annualized Volatility': qs.stats.volatility(benchmark_aligned, annualize=True, compounded=True) * 100,
                            'Beta': qs.stats.beta(portfolio_aligned, benchmark_aligned),
                            'Alpha (Annualized)': qs.stats.alpha(portfolio_aligned, benchmark_aligned, rf=configs.MARKOWITZ_RISK_FREE_RATE, annualize=True) * 100,
                            'Information Ratio': qs.stats.information_ratio(portfolio_aligned, benchmark_aligned),
                        })
                    else:
                        current_logger.warning("No common index between portfolio and benchmark. Skipping benchmark metrics.")
            except Exception as e_bench:
                current_logger.error(f"Error calculating benchmark metrics: {e_bench}", exc_info=True)
        
        return pd.Series(metrics_dict)
