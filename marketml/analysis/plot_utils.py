# /.ndmh/marketml/analysis/plot_utils.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import logging
from typing import Dict, Optional, List

try:
    from marketml.configs import configs
    DEFAULT_PLOT_SAVE_DIR = getattr(configs, 'PLOTS_OUTPUT_DIR_PLOTLY', Path("plots_output_plotly_fallback"))
except ImportError:
    print("Warning: Could not import configs for plot_utils. Using fallback plot directory for Plotly.")
    DEFAULT_PLOT_SAVE_DIR = Path("plots_output_plotly_fallback")

logger = logging.getLogger(__name__)

def _ensure_save_dir(save_dir: Path) -> bool:
    """Ensure the save directory exists."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Could not create plot save directory {save_dir}: {e}")
        return False

def _save_plot(fig: go.Figure, save_filename: str, save_dir: Optional[Path], 
               width: int = 1200, height: int = 700) -> None:
    """Helper function to save plot to file."""
    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    if _ensure_save_dir(current_save_dir):
        try:
            filepath = current_save_dir / save_filename
            fig.write_image(filepath, width=width, height=height)
            logger.info(f"Plot saved as PNG to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Error saving plot as PNG: {e}", exc_info=True)

def _validate_performance_data(df: pd.DataFrame, required_col: str = 'value') -> bool:
    """Validate performance DataFrame structure."""
    return df is not None and not df.empty and required_col in df.columns and isinstance(df.index, pd.DatetimeIndex)

# ==============================================================================
# HÀM PLOT MỚI CHO FORECASTING MODEL PERFORMANCE (DÙNG PLOTLY)
# ==============================================================================

def plot_forecasting_mean_performance(
    df_summary: pd.DataFrame, # Index là ModelName_MetricName, cột 'mean', 'std'
    model_names: list,
    metric_suffixes: list, # Ví dụ: ["_Accuracy", "_F1_Macro"]
    title: str = "Mean Forecasting Model Performance",
    save_filename: str = "forecasting_mean_performance.png",
    save_dir: Path = None,
    width: int = 1200, height: int = 700
):
    """
    Vẽ biểu đồ cột so sánh hiệu suất trung bình của các model forecasting bằng Plotly.
    """
    if df_summary is None or df_summary.empty or 'mean' not in df_summary.columns:
        logger.warning("No valid summary data or 'mean' column for forecasting model performance plot.")
        return

    plot_data_list = []
    for model_prefix in model_names:
        for metric_suffix in metric_suffixes:
            full_metric_name = model_prefix + metric_suffix
            if full_metric_name in df_summary.index:
                plot_data_list.append({
                    'Model': model_prefix,
                    'Metric': metric_suffix.replace("_", " ").strip(), # Tên hiển thị đẹp hơn
                    'Mean Performance': df_summary.loc[full_metric_name, 'mean']
                })

    if not plot_data_list:
        logger.warning("No data to plot for forecasting mean performance after filtering.")
        return

    plot_df = pd.DataFrame(plot_data_list)
    # Sắp xếp để biểu đồ dễ nhìn hơn, ví dụ theo Metric rồi theo Model
    plot_df.sort_values(by=['Metric', 'Model'], ascending=[True, True], inplace=True)

    fig = px.bar(plot_df, x="Metric", y="Mean Performance", color="Model",
                 barmode="group", title=title,
                 labels={"Mean Performance": "Mean Score"},
                 color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Mean Score",
        legend_title="Forecasting Model",
        xaxis_tickangle=-45
    )

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    if _ensure_save_dir(current_save_dir):
        try:
            filepath = current_save_dir / save_filename
            fig.write_image(filepath, width=width, height=height)
            logger.info(f"Plotly forecasting mean performance plot saved to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Error saving Plotly forecasting mean performance plot: {e}", exc_info=True)
    # fig.show()
    # fig.close() # Plotly không có plt.close(), fig sẽ tự dọn dẹp


def plot_forecasting_distribution(
    df_detailed: pd.DataFrame, # DataFrame với mỗi cột là ModelName_MetricName
    model_names: list,
    metric_suffixes: list,
    title: str = "Forecasting Model Performance Distribution",
    save_filename: str = "forecasting_performance_distribution.png",
    save_dir: Path = None,
    width: int = 1600, height: int = 800
):
    """
    Vẽ box plot phân phối hiệu suất của các model forecasting qua các CV split bằng Plotly.
    """
    if df_detailed is None or df_detailed.empty or len(df_detailed) <= 1:
        logger.warning("Not enough data/splits in detailed forecasting results for distribution plot.")
        return

    metrics_to_plot_full = []
    for model_prefix in model_names:
        for metric_suffix in metric_suffixes:
            full_metric_name = model_prefix + metric_suffix
            if full_metric_name in df_detailed.columns:
                metrics_to_plot_full.append(full_metric_name)
    
    if not metrics_to_plot_full:
        logger.warning("No specified metrics found in the detailed forecasting data to plot distribution.")
        return

    # Melt DataFrame để phù hợp với Plotly Express
    plot_data_melted = pd.melt(df_detailed, value_vars=metrics_to_plot_full,
                               var_name='Metric_Model', value_name='Score')
    
    # Tách Model và Metric từ Metric_Model
    # Giả sử Metric_Model có dạng "ModelName_MetricSuffix"
    plot_data_melted['Model'] = plot_data_melted['Metric_Model'].apply(lambda x: x.split('_')[0])
    plot_data_melted['Metric'] = plot_data_melted['Metric_Model'].apply(lambda x: "_".join(x.split('_')[1:]).replace("_", " ").strip())

    # Lọc lại chỉ các model có trong model_names (đề phòng trường hợp Metric_Model có prefix lạ)
    plot_data_melted = plot_data_melted[plot_data_melted['Model'].isin(model_names)]
    if plot_data_melted.empty:
        logger.warning("No data left for Plotly forecasting distribution plot after filtering by model names.")
        return

    fig = px.box(plot_data_melted, x="Metric", y="Score", color="Model",
                 title=title,
                 labels={"Score": "Metric Score"},
                 color_discrete_sequence=px.colors.qualitative.Set2) # Hoặc một bảng màu khác

    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Score Distribution",
        legend_title="Forecasting Model",
        xaxis_tickangle=-45
    )

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    if _ensure_save_dir(current_save_dir):
        try:
            filepath = current_save_dir / save_filename
            fig.write_image(filepath, width=width, height=height)
            logger.info(f"Plotly forecasting performance distribution plot saved to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Error saving Plotly forecasting distribution plot: {e}", exc_info=True)
    # fig.show()

def plot_cumulative_portfolio_value(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Cumulative Portfolio Value Over Time",
    save_filename: str = "cumulative_portfolio_value.png",
    save_dir: Optional[Path] = None,
    width: int = 1200,
    height: int = 700
) -> None:
    """
    Plot cumulative portfolio value over time for multiple strategies.
    
    Args:
        performance_dfs: Dictionary of strategy names to DataFrames containing 'value' column
        title: Plot title
        save_filename: Output filename
        save_dir: Directory to save plot (uses default if None)
        width: Plot width in pixels
        height: Plot height in pixels
    """
    if not performance_dfs:
        logger.warning("No performance data provided for plotting cumulative value.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (strategy_name, df) in enumerate(performance_dfs.items()):
        if _validate_performance_data(df, 'value'):
            fig.add_trace(go.Scatter(
                x=df.index, y=df['value'], mode='lines', name=strategy_name,
                line=dict(color=colors[i % len(colors)])
            ))
        else:
            logger.warning(f"Skipping strategy '{strategy_name}' for cumulative value plot due to missing/invalid data.")

    if not fig.data:
        logger.warning("No valid data to plot for cumulative portfolio value.")
        return

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        legend_title="Strategy",
        hovermode="x unified"
    )
    _save_plot(fig, save_filename, save_dir, width, height)

def plot_drawdown_curves(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Portfolio Drawdown Curves",
    save_filename: str = "drawdown_curves.png",
    save_dir: Optional[Path] = None,
    width: int = 1200,
    height: int = 700
) -> None:
    """
    Plot portfolio drawdown curves for multiple strategies.
    
    Requires quantstats package and 'returns' column in DataFrames.
    """
    if not performance_dfs:
        logger.warning("No performance data provided for plotting drawdown curves.")
        return
    
    try:
        import quantstats as qs
        qs.extend_pandas()
    except ImportError:
        logger.error("QuantStats library not found. Cannot plot drawdown curves.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (strategy_name, df) in enumerate(performance_dfs.items()):
        if _validate_performance_data(df, 'returns'):
            returns_series = df['returns'].fillna(0.0).astype(float)
            if not returns_series.empty:
                drawdown_series = qs.stats.to_drawdown_series(returns_series)
                fig.add_trace(go.Scatter(
                    x=drawdown_series.index, y=drawdown_series * 100, 
                    mode='lines', name=strategy_name,
                    line=dict(color=colors[i % len(colors)])
                ))
            else:
                logger.warning(f"Return series for strategy '{strategy_name}' is empty. Skipping for drawdown plot.")
        else:
            logger.warning(f"Skipping strategy '{strategy_name}' for drawdown plot due to missing/invalid data or 'returns' column.")

    if not fig.data:
        logger.warning("No valid data to plot for drawdown curves.")
        return

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        legend_title="Strategy",
        hovermode="x unified"
    )
    _save_plot(fig, save_filename, save_dir, width, height)

def plot_daily_returns_distribution(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Distribution of Daily Returns",
    use_kde: bool = True,
    save_filename: str = "daily_returns_distribution.png",
    save_dir: Optional[Path] = None,
    width: int = 1200,
    height: int = 700
) -> None:
    """
    Plot distribution of daily returns, optionally using KDE.
    
    Args:
        performance_dfs: Dictionary of strategy names to DataFrames
        use_kde: Whether to show probability density (KDE) instead of frequency
    """
    if not performance_dfs:
        logger.warning("No performance data for plotting daily returns distribution.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (strategy_name, df) in enumerate(performance_dfs.items()):
        if df is not None and not df.empty and 'returns' in df.columns:
            returns_series = df['returns'].fillna(0.0).astype(float) * 100
            if not returns_series.empty:
                if use_kde:
                    fig.add_trace(go.Histogram(
                        x=returns_series, name=strategy_name, opacity=0.6,
                        marker_color=colors[i % len(colors)], histnorm='probability density'
                    ))
                else:
                    fig.add_trace(go.Histogram(
                        x=returns_series, name=strategy_name,
                        marker_color=colors[i % len(colors)]
                    ))
            else:
                logger.warning(f"Return series for strategy '{strategy_name}' is empty. Skipping for distribution plot.")
        else:
            logger.warning(f"Skipping strategy '{strategy_name}' for distribution plot due to missing/invalid data or 'returns' column.")
    
    if not fig.data:
        logger.warning("No valid data to plot for daily returns distribution.")
        return

    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Density" if use_kde else "Frequency",
        legend_title="Strategy",
        barmode='overlay'
    )
    if use_kde:
        fig.update_traces(opacity=0.70)

    _save_plot(fig, save_filename, save_dir, width, height)

def plot_key_metrics_comparison(
    summary_metrics_df: pd.DataFrame,
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Comparison of Key Performance Metrics",
    save_filename: str = "key_metrics_comparison.png",
    save_dir: Optional[Path] = None,
    width: int = 1200,
    height: int = 800
) -> None:
    """
    Plot comparison of key performance metrics across strategies.
    
    Args:
        summary_metrics_df: DataFrame with metrics as columns and strategies as index
        metrics_to_plot: List of metrics to include (default: Sharpe, Sortino, Calmar, Max Drawdown)
    """
    if summary_metrics_df.empty:
        logger.warning("Summary metrics DataFrame is empty. Cannot plot key metrics comparison.")
        return

    metrics_to_plot = metrics_to_plot or ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown']
    actual_metrics_to_plot = [m for m in metrics_to_plot if m in summary_metrics_df.columns]
    
    if not actual_metrics_to_plot:
        logger.warning(f"None of the specified metrics found in DataFrame columns: {summary_metrics_df.columns.tolist()}")
        return

    df_to_plot = summary_metrics_df[actual_metrics_to_plot].copy()
    
    # Convert Max Drawdown to absolute value if present
    if 'Max Drawdown' in df_to_plot.columns and 'Max Drawdown Abs' not in actual_metrics_to_plot:
        df_to_plot['Max Drawdown Abs'] = df_to_plot['Max Drawdown'].abs()
        actual_metrics_to_plot.append('Max Drawdown Abs')
        if 'Max Drawdown' in actual_metrics_to_plot:
            actual_metrics_to_plot.remove('Max Drawdown')

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    
    for i, metric_name in enumerate(actual_metrics_to_plot):
        fig.add_trace(go.Bar(
            x=df_to_plot.index,
            y=df_to_plot[metric_name],
            name=metric_name,
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Strategy",
        yaxis_title="Metric Value",
        barmode='group',
        legend_title="Metric"
    )
    _save_plot(fig, save_filename, save_dir, width, height)

def plot_annualized_return_vs_volatility(
    summary_metrics_df: pd.DataFrame,
    title: str = "Annualized Return vs. Annualized Volatility",
    save_filename: str = "return_vs_volatility.png",
    save_dir: Optional[Path] = None,
    width: int = 1000,
    height: int = 700
) -> None:
    """
    Plot comparison of annualized return vs volatility.
    
    Requires 'Annualized Return' and 'Annualized Volatility' columns in DataFrame.
    """
    required_cols = ['Annualized Return', 'Annualized Volatility']
    if summary_metrics_df.empty or not all(col in summary_metrics_df.columns for col in required_cols):
        logger.warning(f"Missing required columns {required_cols} in summary data. Cannot plot.")
        return

    df_to_plot = summary_metrics_df[required_cols].copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_to_plot.index,
        y=df_to_plot['Annualized Return'],
        name='Annualized Return (%)',
        marker_color='rgb(26, 118, 255)'
    ))
    fig.add_trace(go.Bar(
        x=df_to_plot.index,
        y=df_to_plot['Annualized Volatility'],
        name='Annualized Volatility (%)',
        marker_color='rgb(255, 127, 14)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Strategy",
        yaxis_title="Percentage (%)",
        yaxis_ticksuffix="%",
        barmode='group',
        legend_title="Metric"
    )
    _save_plot(fig, save_filename, save_dir, width, height)

def plot_rolling_sharpe_ratio(
    performance_dfs: Dict[str, pd.DataFrame],
    rolling_window_days: int = 60,
    risk_free_rate: float = 0.02,
    title_suffix: str = "",
    save_filename: str = "rolling_sharpe.png",
    save_dir: Optional[Path] = None,
    width: int = 1200,
    height: int = 700
) -> None:
    """
    Plot rolling Sharpe ratio for multiple strategies.
    
    Requires quantstats package and 'returns' column in DataFrames.
    
    Args:
        performance_dfs: Dictionary of strategy DataFrames
        rolling_window_days: Window size in days for rolling calculation
        risk_free_rate: Annualized risk-free rate
        title_suffix: Optional suffix for plot title
    """
    if not performance_dfs:
        logger.warning("No performance data for plotting rolling Sharpe.")
        return
        
    try:
        import quantstats as qs
        qs.extend_pandas()
    except ImportError:
        logger.error("QuantStats library not found. Cannot plot rolling Sharpe.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (strategy_name, df) in enumerate(performance_dfs.items()):
        if _validate_performance_data(df, 'returns'):
            returns_series = df['returns'].fillna(0.0).astype(float)
            if len(returns_series) >= rolling_window_days:
                rolling_sharpe = qs.stats.rolling_sharpe(
                    returns_series,
                    rf=risk_free_rate / 252.0,
                    rolling_period=rolling_window_days,
                    annualize=True,
                    periods_per_year=252,
                    prepare_returns=True
                )
                if rolling_sharpe is not None and not rolling_sharpe.empty:
                    fig.add_trace(go.Scatter(
                        x=rolling_sharpe.index, y=rolling_sharpe,
                        mode='lines', name=strategy_name,
                        line=dict(color=colors[i % len(colors)])
                    ))
                else:
                    logger.warning(f"Could not calculate rolling Sharpe for '{strategy_name}' (window {rolling_window_days}d).")
            else:
                logger.warning(f"Not enough data for strategy '{strategy_name}' (needs {rolling_window_days}d, has {len(returns_series)}).")
        else:
            logger.warning(f"Skipping strategy '{strategy_name}' - missing/invalid data or 'returns' column.")

    if not fig.data:
        logger.warning(f"No valid data to plot for rolling Sharpe ratio (window {rolling_window_days}d).")
        return

    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({rolling_window_days}-Day Window){title_suffix}",
        xaxis_title="Date",
        yaxis_title="Annualized Rolling Sharpe Ratio",
        legend_title="Strategy",
        hovermode="x unified"
    )
    _save_plot(fig, save_filename, save_dir, width, height)