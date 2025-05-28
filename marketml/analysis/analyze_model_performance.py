# /.ndmh/marketml/analysis/analyze_model_performance.py
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional, List

try:
    from marketml.configs import configs
    from marketml.utils import logger_setup
    from marketml.analysis.plot_utils import (
        plot_cumulative_portfolio_value,
        plot_drawdown_curves,
        plot_daily_returns_distribution,
        plot_key_metrics_comparison,
        plot_annualized_return_vs_volatility,
        plot_rolling_sharpe_ratio
    )
    from marketml.analysis import plot_utils as plotly_visualizer
except ImportError as e:
    print(f"CRITICAL ERROR in analyze_model_performance.py: {e}")
    
    # Fallback configuration
    class FallbackConfigs:
        RESULTS_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results_output"
        MODEL_PERF_SUMMARY_FILE = RESULTS_OUTPUT_DIR / "model_performance_summary.csv"
        MODEL_PERF_DETAILED_FILE = RESULTS_OUTPUT_DIR / "model_performance_detailed.csv"
        MARKOWITZ_RISK_FREE_RATE = 0.02
        RL_ALGORITHM = "PPO"
    
    configs = FallbackConfigs()
    
    # Fallback logger setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("analyzer_fallback")
    logger.warning("Using fallback configs and logger for analyze_model_performance.py")

    # DummyPlotter from new code
    class DummyPlotter:
        def __getattr__(self, name):
            def _missing_plot(*args, **kwargs):
                logger.error(f"Plotly plotting function '{name}' called, but plotly_visualizer could not be imported.")
            return _missing_plot
    plotly_visualizer = DummyPlotter()

# Initialize logger if not already set up
if 'logger' not in locals():
    logger = logger_setup.setup_basic_logging(log_file_name="analyze_model_performance.log") if 'logger_setup' in globals() else logging.getLogger(__name__)

# Path configuration
SUMMARY_FILE_PATH = configs.MODEL_PERF_SUMMARY_FILE
DETAILED_FILE_PATH = configs.MODEL_PERF_DETAILED_FILE
PLOTS_SAVE_DIR = configs.RESULTS_OUTPUT_DIR / "performance_plots"

# Section: Analysis constants
METRICS_SUFFIXES_TO_PLOT = getattr(configs, 'ANALYSIS_METRICS_SUFFIXES', [
    "_Accuracy", "_F1_Macro", "_F1_Weighted", "_Precision_Macro", "_Recall_Macro"
])
MODEL_NAMES_TO_ANALYZE = getattr(configs, 'ANALYSIS_MODEL_NAMES', [
    "ARIMA", "RandomForest", "XGBoost", "LSTM", "Transformer", "SVM"
])
PORTFOLIO_STRATEGIES = getattr(configs, 'PORTFOLIO_STRATEGIES', [
    "Markowitz", "BlackLitterman", "RL_Strategy"
])
PORTFOLIO_METRICS = getattr(configs, 'PORTFOLIO_KEY_METRICS', [
    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown'
])
ROLLING_WINDOWS = getattr(configs, 'ROLLING_SHARPE_WINDOWS', [30, 60, 90])

def load_performance_data(file_path: Path, index_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Loading performance data from: {file_path}")
        df = pd.read_csv(file_path, index_col=index_col, parse_dates=True if index_col == 'date' else False)
        if df.empty:
            logger.warning(f"Loaded empty DataFrame from {file_path}")
            return None
        return df
    except FileNotFoundError:
        logger.error(f"File not found at '{file_path}'")
    except Exception as e:
        logger.error(f"Error loading file '{file_path}': {e}", exc_info=True)
    return None

def analyze_forecasting_models() -> None:
    logger.info("--- Analyzing Forecasting Model Performance ---")
    
    df_summary = load_performance_data(SUMMARY_FILE_PATH, index_col=0)
    if df_summary is None:
        logger.warning("No summary data available for forecasting models.")
        return
    
    if 'mean' in df_summary.columns and 'std' in df_summary.columns:
        df_display = df_summary.dropna(subset=['mean']).copy()
        df_display['Mean ± Std'] = df_display.apply(
            lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1
        )
        logger.info(f"\nPerformance Summary:\n{df_display[['Mean ± Std']].to_string()}")
    
    try:
        if hasattr(plotly_visualizer, 'plot_forecasting_mean_performance'):
            plotly_visualizer.plot_forecasting_mean_performance(
                df_summary=df_summary,
                model_names=MODEL_NAMES_TO_ANALYZE,
                metric_suffixes=METRICS_SUFFIXES_TO_PLOT,
                save_dir=PLOTS_SAVE_DIR,
                save_filename="forecasting_mean_performance_plotly.png"
            )
        
        if DETAILED_FILE_PATH.exists():
            df_detailed = load_performance_data(DETAILED_FILE_PATH)
            if df_detailed is not None and len(df_detailed) > 1:
                if hasattr(plotly_visualizer, 'plot_forecasting_distribution'):
                    plotly_visualizer.plot_forecasting_distribution(
                        df_detailed=df_detailed,
                        model_names=MODEL_NAMES_TO_ANALYZE,
                        metric_suffixes=METRICS_SUFFIXES_TO_PLOT,
                        save_dir=PLOTS_SAVE_DIR,
                        save_filename="forecasting_distribution_plotly.png"
                    )
            elif df_detailed is not None:
                logger.info(f"Detailed forecasting results file has too few splits (<=1). Skipping distribution plot.")
        else:
            logger.info(f"Detailed forecasting results file not found at {DETAILED_FILE_PATH}, skipping distribution plot.")
    
    except Exception as e:
        logger.error(f"Error generating plotly forecasting plots: {e}")

def analyze_portfolio_strategies() -> None:
    """Section: Analyze and visualize portfolio strategy performance"""
    logger.info("--- Analyzing Portfolio Strategy Performance ---")
    
    strategy_dfs = {}
    portfolio_strategies_to_analyze = getattr(configs, 'PORTFOLIO_STRATEGIES_TO_ANALYZE', ["Markowitz", "BlackLitterman"])
    rl_algo_name = getattr(configs, 'RL_ALGORITHM', "PPO").upper()

    if getattr(configs, 'RL_STRATEGY_ENABLED', False) and rl_algo_name:
        rl_dict_key = f"RL ({rl_algo_name})"
        if rl_dict_key not in portfolio_strategies_to_analyze:
            portfolio_strategies_to_analyze.append(rl_dict_key)

    filename_map = {
        "Markowitz": getattr(configs, 'MARKOWITZ_PERF_DAILY_FILE_NAME', "markowitz_performance_daily.csv"),
        "BlackLitterman": getattr(configs, 'BLACKLITTERMAN_PERF_DAILY_FILE_NAME', "blacklitterman_performance_daily.csv"),
    }
    if getattr(configs, 'RL_STRATEGY_ENABLED', False) and rl_algo_name:
        rl_dict_key_for_map = f"RL ({rl_algo_name})"
        filename_map[rl_dict_key_for_map] = f"{configs.RL_ALGORITHM.lower()}_strategy_performance_daily.csv"

    for strategy_display_name in portfolio_strategies_to_analyze:
        actual_filename = filename_map.get(strategy_display_name)
        
        if not actual_filename:
            logger.warning(f"No filename mapping found for strategy: {strategy_display_name}. Skipping.")
            continue

        file_path = configs.RESULTS_OUTPUT_DIR / actual_filename
        df = load_performance_data(file_path, index_col='date')
        if df is not None:
            strategy_dfs[strategy_display_name] = df

    if not strategy_dfs:
        logger.warning("No portfolio strategy data available.")
        return

    summary_file = configs.RESULTS_OUTPUT_DIR / getattr(configs, 'PORTFOLIO_STRATEGIES_SUMMARY_FILE_NAME', "portfolio_strategies_summary.csv")
    df_summary = load_performance_data(summary_file, index_col=0)

    PLOTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Section: Portfolio plots
    plot_cumulative_portfolio_value(
        performance_dfs=strategy_dfs,
        save_dir=PLOTS_SAVE_DIR,
        title="Portfolio Value Over Time"
    )
    
    plot_drawdown_curves(
        performance_dfs=strategy_dfs,
        save_dir=PLOTS_SAVE_DIR,
        title="Portfolio Drawdown Analysis"
    )
    
    plot_daily_returns_distribution(
        performance_dfs=strategy_dfs,
        save_dir=PLOTS_SAVE_DIR,
        title="Daily Returns Distribution"
    )
    
    if df_summary is not None:
        plot_key_metrics_comparison(
            summary_metrics_df=df_summary,
            metrics_to_plot=PORTFOLIO_METRICS,
            save_dir=PLOTS_SAVE_DIR,
            title="Portfolio Performance Metrics Comparison"
        )
        
        plot_annualized_return_vs_volatility(
            summary_metrics_df=df_summary,
            save_dir=PLOTS_SAVE_DIR,
            title="Return vs Volatility Comparison"
        )
    
    risk_free_rate = getattr(configs, 'MARKOWITZ_RISK_FREE_RATE', 0.02)
    for window in ROLLING_WINDOWS:
        plot_rolling_sharpe_ratio(
            performance_dfs=strategy_dfs,
            rolling_window_days=window,
            risk_free_rate=risk_free_rate,
            save_dir=PLOTS_SAVE_DIR,
            title_suffix=f" ({window}-Day Window)"
        )

    try:
        if hasattr(plotly_visualizer, 'plot_cumulative_portfolio_value'):
            plotly_visualizer.plot_cumulative_portfolio_value(
                performance_dfs=strategy_dfs, 
                save_dir=PLOTS_SAVE_DIR
            )
        
        if hasattr(plotly_visualizer, 'plot_drawdown_curves'):
            plotly_visualizer.plot_drawdown_curves(
                performance_dfs=strategy_dfs, 
                save_dir=PLOTS_SAVE_DIR
            )
        
        if hasattr(plotly_visualizer, 'plot_daily_returns_distribution'):
            plotly_visualizer.plot_daily_returns_distribution(
                performance_dfs=strategy_dfs, 
                save_dir=PLOTS_SAVE_DIR
            )
        
        if df_summary is not None and not df_summary.empty:
            if hasattr(plotly_visualizer, 'plot_key_metrics_comparison'):
                plotly_visualizer.plot_key_metrics_comparison(
                    summary_metrics_df=df_summary,
                    metrics_to_plot=PORTFOLIO_METRICS,
                    save_dir=PLOTS_SAVE_DIR
                )
            
            if hasattr(plotly_visualizer, 'plot_annualized_return_vs_volatility'):
                plotly_visualizer.plot_annualized_return_vs_volatility(
                    summary_metrics_df=df_summary,
                    save_dir=PLOTS_SAVE_DIR
                )
        
        for window in ROLLING_WINDOWS:
            if hasattr(plotly_visualizer, 'plot_rolling_sharpe_ratio'):
                plotly_visualizer.plot_rolling_sharpe_ratio(
                    performance_dfs=strategy_dfs,
                    rolling_window_days=window,
                    risk_free_rate=risk_free_rate,
                    title_suffix=f" ({window}-Day)",
                    save_filename_base="portfolio_rolling_sharpe",
                    save_dir=PLOTS_SAVE_DIR
                )
    
    except Exception as e:
        logger.error(f"Error generating plotly portfolio plots: {e}")

def main() -> None:
    """Main analysis workflow"""
    logger.info("===== Starting Performance Analysis =====")
    analyze_forecasting_models()
    analyze_portfolio_strategies()
    
    logger.info("===== Analysis Completed =====")

if __name__ == "__main__":
    main()
