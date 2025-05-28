# /.ndmh/marketml/analysis/plot_utils.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Kept for potential future use
import plotly.express as px
from pathlib import Path
import logging
from typing import Dict, Optional, List, Union
import numpy as np # Kept

# --- Configuration Handling ---
try:
    from marketml.configs import configs
    DEFAULT_PLOT_SAVE_DIR = getattr(configs, 'PLOTS_OUTPUT_DIR_PLOTLY', Path("plots_output_plotly_fallback"))
    FORECASTING_BASELINE_SCORES = getattr(configs, 'FORECASTING_BASELINE_SCORES', {})
    MARKET_EVENTS_FOR_PLOTS = getattr(configs, 'MARKET_EVENTS_FOR_PLOTS', [])
    DESIRED_MODEL_ORDER = getattr(configs, 'ANALYSIS_FORECASTING_MODEL_ORDER', [
        "ARIMA", "Prophet", "SVM", "RandomForest", "XGBoost", "LSTM", "Transformer"
    ])
    # Optional: Define a desired strategy order for portfolio plots if needed from configs
    DESIRED_STRATEGY_ORDER = getattr(configs, 'ANALYSIS_STRATEGY_ORDER', []) # Empty list if not defined
except ImportError:
    print("Warning: Could not import configs for plot_utils. Using fallback directories and default values.")
    DEFAULT_PLOT_SAVE_DIR = Path("plots_output_plotly_fallback")
    FORECASTING_BASELINE_SCORES = {}
    MARKET_EVENTS_FOR_PLOTS = []
    DESIRED_MODEL_ORDER = ["ARIMA", "Prophet", "SVM", "RandomForest", "XGBoost", "LSTM", "Transformer"]
    DESIRED_STRATEGY_ORDER = []

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _ensure_save_dir(save_dir: Path) -> bool:
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Could not create plot save directory {save_dir}: {e}")
        return False

def _save_plot(fig: go.Figure, save_filename: str, save_dir: Optional[Path],
               width: int = 1200, height: int = 700, scale: float = 1.8) -> None:
    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    if _ensure_save_dir(current_save_dir):
        try:
            filepath = current_save_dir / save_filename
            fig.write_image(filepath, width=width, height=height, scale=scale)
            logger.info(f"Plot saved as PNG to: {filepath.resolve()}")
        except Exception as e:
            logger.error(f"Error saving plot as PNG: {e}", exc_info=True)

def _validate_performance_data(df: pd.DataFrame, required_col: str = 'value') -> bool:
    return df is not None and not df.empty and required_col in df.columns and isinstance(df.index, pd.DatetimeIndex)

# --- Consistent Colors & Font Styling ---
FORECASTING_MODEL_COLORS = {
    "ARIMA": px.colors.qualitative.Plotly[0],
    "RandomForest": px.colors.qualitative.Plotly[1],
    "XGBoost": px.colors.qualitative.Plotly[2],
    "LSTM": px.colors.qualitative.Plotly[3],
    "Transformer": px.colors.qualitative.Plotly[4],
    "SVM": px.colors.qualitative.Plotly[5],
    "Prophet": px.colors.qualitative.Plotly[6],
}

PORTFOLIO_STRATEGY_COLORS = {
    "Markowitz": px.colors.qualitative.Set1[0],
    "BlackLitterman": px.colors.qualitative.Set1[1],
    "RL (PPO)": px.colors.qualitative.Set1[2],
    "Benchmark": px.colors.qualitative.Set1[3], # Example, adjust as needed
}

STRATEGY_NAME_MAPPING = {
    "RL_Strategy": "RL (PPO)",
    "RL_PPO": "RL (PPO)",
    # Add other explicit mappings if necessary
    # e.g., "MyCustomStrategy": "Custom Strategy Display Name"
}

def unify_strategy_name(name: str) -> str:
    return STRATEGY_NAME_MAPPING.get(name, name)

def get_color_map(
    names_in_data: List[str],
    base_color_dict: Dict[str, str],
    desired_order: Optional[List[str]] = None,
    default_palette=px.colors.qualitative.Plotly
) -> Dict[str, str]:
    color_map = {}
    available_palette_colors = list(default_palette)

    processing_order = []
    if desired_order:
        processing_order.extend([name for name in desired_order if name in names_in_data])
        processing_order.extend([name for name in names_in_data if name not in processing_order])
    else:
        processing_order = list(names_in_data)

    for name in processing_order:
        canonical_name = unify_strategy_name(name)
        assigned_color = None

        if canonical_name in base_color_dict:
            assigned_color = base_color_dict[canonical_name]
        elif name in base_color_dict:
            assigned_color = base_color_dict[name]

        if assigned_color:
            color_map[name] = assigned_color
            if assigned_color in available_palette_colors:
                try: available_palette_colors.remove(assigned_color)
                except ValueError: pass

    palette_idx_offset = 0
    for name in processing_order:
        if name not in color_map:
            if available_palette_colors:
                color_map[name] = available_palette_colors.pop(0)
            else:
                color_map[name] = default_palette[palette_idx_offset % len(default_palette)]
                palette_idx_offset += 1
    return color_map

COMMON_FONT_SIZE_TITLE = 18
COMMON_FONT_SIZE_AXIS_TITLE = 14
COMMON_FONT_SIZE_AXIS_TICK = 12
COMMON_FONT_SIZE_LEGEND = 12
COMMON_FONT_SIZE_ANNOTATION = 10
COMMON_FONT_SIZE_DATALABEL = 10

def apply_common_font_style(fig: go.Figure):
    fig.update_layout(
        title_font_size=COMMON_FONT_SIZE_TITLE,
        xaxis_title_font_size=COMMON_FONT_SIZE_AXIS_TITLE,
        yaxis_title_font_size=COMMON_FONT_SIZE_AXIS_TITLE,
        xaxis_tickfont_size=COMMON_FONT_SIZE_AXIS_TICK,
        yaxis_tickfont_size=COMMON_FONT_SIZE_AXIS_TICK,
        legend_font_size=COMMON_FONT_SIZE_LEGEND,
        legend_title_font_size=COMMON_FONT_SIZE_LEGEND + 1,
        font_family="Arial, sans-serif",
        title_x=0.5
    )
    if hasattr(fig.layout, 'annotations') and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if ann.font:
                ann.font.size = ann.font.size if ann.font.size is not None else COMMON_FONT_SIZE_ANNOTATION
                ann.font.family = ann.font.family or "Arial, sans-serif"
            else:
                ann.font = dict(size=COMMON_FONT_SIZE_ANNOTATION, family="Arial, sans-serif")
    
    def update_trace_textfont(trace):
        if hasattr(trace, 'textfont'):
            if trace.textfont is None:
                trace.textfont = dict(size=COMMON_FONT_SIZE_DATALABEL, family="Arial, sans-serif")
            else:
                current_size = getattr(trace.textfont, 'size', None)
                if current_size is None:
                    trace.textfont.size = COMMON_FONT_SIZE_DATALABEL
                trace.textfont.family = getattr(trace.textfont, 'family', None) or "Arial, sans-serif"
    fig.for_each_trace(update_trace_textfont)

# === FORECASTING PLOTS ===
def plot_forecasting_mean_performance(
    df_summary: pd.DataFrame, model_names: list, metric_suffixes: list,
    title: str = "Mean Forecasting Model Performance", save_filename: str = "forecasting_mean_performance.png",
    save_dir: Optional[Path] = None, width: int = 1200, height: int = 750
):
    if df_summary is None or df_summary.empty or 'mean' not in df_summary.columns:
        logger.warning("No valid summary data or 'mean' column for forecasting model performance plot."); return

    plot_data_list = []
    actual_models_to_plot_ordered = [m for m in DESIRED_MODEL_ORDER if m in model_names]
    actual_models_to_plot_ordered.extend([m for m in model_names if m not in actual_models_to_plot_ordered])
    metrics_display_ordered = []

    for model_prefix in actual_models_to_plot_ordered:
        for metric_suffix in metric_suffixes:
            # Assuming metric_suffix might start with '_' or not, consistent with plot_forecasting_distribution
            cleaned_metric_suffix = metric_suffix.lstrip('_')
            # Assuming df_summary index is ModelName + MetricSuffix (no underscore)
            full_metric_name = model_prefix + cleaned_metric_suffix 
            metric_display_name = cleaned_metric_suffix.replace("_", " ").strip()
            if full_metric_name in df_summary.index:
                plot_data_list.append({
                    'Model': model_prefix,
                    'Metric': metric_display_name,
                    'Mean Performance': df_summary.loc[full_metric_name, 'mean']
                })
                if metric_display_name not in metrics_display_ordered:
                    metrics_display_ordered.append(metric_display_name)
    
    if not plot_data_list: logger.warning("No data to plot for forecasting mean performance after filtering."); return
    plot_df = pd.DataFrame(plot_data_list)
    
    plot_df['Metric'] = pd.Categorical(plot_df['Metric'], categories=metrics_display_ordered, ordered=True)
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=actual_models_to_plot_ordered, ordered=True)
    plot_df.sort_values(by=['Metric', 'Model'], inplace=True)

    model_color_map = get_color_map(actual_models_to_plot_ordered, FORECASTING_MODEL_COLORS, desired_order=DESIRED_MODEL_ORDER)

    fig = px.bar(plot_df, x="Metric", y="Mean Performance", color="Model",
                 barmode="group", title=title, labels={"Mean Performance": "Mean Score"},
                 color_discrete_map=model_color_map,
                 text_auto='.3f',
                 category_orders={"Metric": metrics_display_ordered, "Model": actual_models_to_plot_ordered}
                )
    
    fig.update_traces(marker_opacity=0.8, textposition='outside')
    apply_common_font_style(fig)
    fig.update_layout(xaxis_tickangle=-30, height=height, hovermode="x unified")

    if len(metrics_display_ordered) > 1:
        for i in range(len(metrics_display_ordered) - 1):
            fig.add_vline(x=i + 0.5, line_width=1.5, line_dash="solid", line_color="darkgrey")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

# Updated plot_forecasting_distribution
def plot_forecasting_distribution(
    df_detailed: pd.DataFrame, model_names: list, metric_suffixes: list,
    title: str = "Forecasting Model Performance Distribution",
    save_filename: str = "forecasting_performance_distribution_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1600, height: int = 850
):
    if df_detailed is None or df_detailed.empty or len(df_detailed) <= 1:
        logger.warning("Not enough data/splits for distribution plot."); return
    
    actual_models_to_plot_ordered = [m for m in DESIRED_MODEL_ORDER if m in model_names]
    actual_models_to_plot_ordered.extend([m for m in model_names if m not in actual_models_to_plot_ordered])
    
    metrics_display_ordered = []
    metric_suffix_original_order = {suffix: i for i, suffix in enumerate(metric_suffixes)}
    temp_plot_data = []

    logger.debug(f"Detailed columns available: {df_detailed.columns.tolist()}")
    logger.debug(f"Models to iterate: {actual_models_to_plot_ordered}")
    logger.debug(f"Metric suffixes to iterate: {metric_suffixes}")

    for model_prefix_iter in actual_models_to_plot_ordered:
        for metric_suffix_iter in metric_suffixes:
            cleaned_metric_suffix = metric_suffix_iter.lstrip('_')
            # Assuming df_detailed columns are ModelName_MetricSuffix (e.g., ARIMA_RMSE)
            full_metric_key = f"{model_prefix_iter}_{cleaned_metric_suffix}"
            metric_display_name = cleaned_metric_suffix.replace("_", " ").strip()
            
            logger.debug(f"Checking for column: {full_metric_key}")
            if full_metric_key in df_detailed.columns:
                logger.debug(f"Found column: {full_metric_key}")
                if metric_display_name not in metrics_display_ordered:
                    metrics_display_ordered.append(metric_display_name)

                for score_val in df_detailed[full_metric_key]:
                    if pd.notna(score_val):
                         temp_plot_data.append({
                            'Model': model_prefix_iter,
                            'Metric_Display': metric_display_name,
                            'Score': score_val
                        })
            else:
                logger.debug(f"Column NOT FOUND: {full_metric_key}")

    if not temp_plot_data: 
        logger.warning("No data to plot for forecasting distribution after processing all models and metrics."); 
        return
    
    plot_df_long = pd.DataFrame(temp_plot_data)

    # Sort metrics_display_ordered based on the original order of metric_suffixes
    metrics_display_ordered.sort(key=lambda m_display: metric_suffix_original_order.get(
        m_display.replace(" ", "_"), 
        metric_suffix_original_order.get("_" + m_display.replace(" ", "_"), float('inf'))
    ))

    plot_df_long['Metric'] = pd.Categorical(plot_df_long['Metric_Display'], categories=metrics_display_ordered, ordered=True)
    plot_df_long['Model'] = pd.Categorical(plot_df_long['Model'], categories=actual_models_to_plot_ordered, ordered=True)
    plot_df_long.dropna(subset=['Metric', 'Model'], inplace=True)

    if plot_df_long.empty:
        logger.warning("DataFrame became empty after converting to categorical and dropping NaNs for forecasting distribution.")
        return

    model_color_map = get_color_map(actual_models_to_plot_ordered, FORECASTING_MODEL_COLORS, desired_order=DESIRED_MODEL_ORDER)

    fig = px.box(plot_df_long, x="Metric", y="Score", color="Model",
                 title=title, labels={"Score": "Metric Score", "Metric": "Metric"},
                 color_discrete_map=model_color_map,
                 points="all",
                 category_orders={"Metric": metrics_display_ordered, "Model": actual_models_to_plot_ordered}
                )

    apply_common_font_style(fig)
    fig.update_layout(xaxis_tickangle=-30, height=height, hovermode="closest")

    if len(metrics_display_ordered) > 1:
        for i in range(len(metrics_display_ordered) - 1):
            fig.add_vline(x=i + 0.5, line_width=1.5, line_dash="solid", line_color="darkgrey")

    has_baseline_plotted = False
    # Use the globally loaded FORECASTING_BASELINE_SCORES
    if FORECASTING_BASELINE_SCORES:
        for metric_name_iter in metrics_display_ordered:
            baseline_score = FORECASTING_BASELINE_SCORES.get(metric_name_iter) # Key is display name
            if baseline_score is not None:
                fig.add_hline(y=baseline_score, line_dash="dot", line_color="dimgray",
                              annotation_text=f"Baseline: {baseline_score:.3f}", annotation_position="bottom right",
                              annotation_font=dict(size=COMMON_FONT_SIZE_ANNOTATION))
                has_baseline_plotted = True
        if has_baseline_plotted and "Distribution" in title:
             fig.update_layout(title_text=title + " (with Baselines)")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

# === PORTFOLIO PLOTS ===
def plot_cumulative_portfolio_value(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Cumulative Portfolio Value",
    save_filename: str = "cumulative_portfolio_value_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1200, height: int = 750
):
    if not performance_dfs: logger.warning("No performance data for plotting cumulative value."); return
    
    all_perf_list = []
    min_date, max_date = pd.Timestamp.max, pd.Timestamp.min
    
    ordered_input_strategy_names_raw = list(performance_dfs.keys())
    unified_input_names = [unify_strategy_name(s) for s in ordered_input_strategy_names_raw]
    
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in unified_input_names])
        actual_strategies_to_plot_ordered.extend([s for s in unified_input_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = unified_input_names
    # Deduplicate while preserving order
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)

    for strategy_name_orig in ordered_input_strategy_names_raw:
        df_strat = performance_dfs.get(strategy_name_orig)
        if df_strat is None: continue
        strategy_display_name = unify_strategy_name(strategy_name_orig)

        if _validate_performance_data(df_strat, 'value'):
            df_temp = df_strat[['value']].copy()
            df_temp['Strategy'] = strategy_display_name
            df_temp.index.name = 'Date'
            all_perf_list.append(df_temp.reset_index())
            if not df_temp.empty:
                min_date = min(min_date, df_temp.index.min())
                max_date = max(max_date, df_temp.index.max())
        else:
            logger.warning(f"Skipping strategy '{strategy_name_orig}' for cumulative value: missing/invalid data.")
    
    if not all_perf_list: logger.warning("No valid data to plot for cumulative portfolio value."); return
    combined_df = pd.concat(all_perf_list)

    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)

    fig = px.line(combined_df, x="Date", y="value", color="Strategy", title=title,
                  labels={"value": "Portfolio Value", "Date": "Date"},
                  color_discrete_map=strategy_color_map,
                  category_orders={"Strategy": actual_strategies_to_plot_ordered}
                  )
    fig.update_traces(hovertemplate="<b>%{data.name}</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:,.2f}<extra></extra>",
                      line_width=2.5)

    apply_common_font_style(fig)
    fig.update_layout(height=height, hovermode="x unified")

    has_event_plotted = False
    if MARKET_EVENTS_FOR_PLOTS and min_date != pd.Timestamp.max:
        for event in MARKET_EVENTS_FOR_PLOTS:
            try:
                event_start = pd.to_datetime(event['start_date'])
                event_end = pd.to_datetime(event.get('end_date', event_start))
                if not (event_end < min_date or event_start > max_date):
                    ann_font = dict(size=COMMON_FONT_SIZE_ANNOTATION, color=event.get('font_color', 'black'))
                    if event_start == event_end:
                         fig.add_vline(x=event_start, line_dash=event.get('line_dash', "dashdot"), line_color=event.get('line_color', "darkgrey"),
                                       annotation_text=event['name'], annotation_position=event.get('annotation_position', "top left"), annotation_font=ann_font)
                    else:
                        fig.add_vrect(x0=event_start, x1=event_end, fillcolor=event.get('color', "rgba(128,128,128,0.1)"),
                                      layer="below", line_width=0, annotation_text=event['name'],
                                      annotation_position=event.get('annotation_position', "top left"), annotation_font=ann_font)
                    has_event_plotted = True
            except Exception as e: logger.warning(f"Could not plot market event '{event.get('name')}': {e}")
        if has_event_plotted and "Value" in title:
             fig.update_layout(title_text=title + " with Market Events")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

def plot_drawdown_curves(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Portfolio Drawdown Analysis",
    save_filename: str = "drawdown_curves_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1200, height: int = 750
):
    if not performance_dfs: logger.warning("No performance data for plotting drawdown curves."); return
    try: import quantstats as qs; qs.extend_pandas()
    except ImportError: logger.error("QuantStats library not found. Cannot plot drawdown curves."); return

    all_drawdowns_list = []
    strat_max_dd_info = {}
    
    ordered_input_strategy_names_raw = list(performance_dfs.keys())
    unified_input_names = [unify_strategy_name(s) for s in ordered_input_strategy_names_raw]

    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in unified_input_names])
        actual_strategies_to_plot_ordered.extend([s for s in unified_input_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = unified_input_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)

    for strategy_name_orig in ordered_input_strategy_names_raw:
        df = performance_dfs.get(strategy_name_orig)
        if df is None: continue
        strategy_display_name = unify_strategy_name(strategy_name_orig)

        if _validate_performance_data(df, 'returns'):
            returns_series = df['returns'].fillna(0.0).astype(float)
            if not returns_series.empty:
                drawdown_series = qs.stats.to_drawdown_series(returns_series) * 100
                if not drawdown_series.empty:
                    df_temp = drawdown_series.reset_index()
                    df_temp.columns = ['Date', 'Drawdown']
                    df_temp['Strategy'] = strategy_display_name
                    all_drawdowns_list.append(df_temp)
                    if pd.notna(drawdown_series.min()):
                        strat_max_dd_info[strategy_display_name] = {'date': drawdown_series.idxmin(), 'value': drawdown_series.min()}
            else: logger.warning(f"Return series for '{strategy_name_orig}' is empty. Skipping drawdown.")
        else: logger.warning(f"Skipping '{strategy_name_orig}' for drawdown: missing/invalid data.")

    if not all_drawdowns_list: logger.warning("No valid data to plot for drawdown curves."); return
    combined_df = pd.concat(all_drawdowns_list)

    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)

    fig = px.area(combined_df, x="Date", y="Drawdown", color="Strategy", title=title,
                  labels={"Drawdown": "Drawdown (%)"}, color_discrete_map=strategy_color_map,
                  line_shape='spline', category_orders={"Strategy": actual_strategies_to_plot_ordered})
    fig.update_traces(line_width=2)

    apply_common_font_style(fig)
    fig.update_layout(yaxis_ticksuffix="%", height=height, hovermode="x unified")

    has_marker_plotted = False
    for strategy_disp_name, dd_info in strat_max_dd_info.items():
        color_val = strategy_color_map.get(strategy_disp_name, px.colors.qualitative.Plotly[0])
        fig.add_scatter(
            x=[dd_info['date']], y=[dd_info['value']], mode="markers+text",
            marker=dict(color=color_val, size=11, symbol="diamond-open", line=dict(width=1.5, color='DarkSlateGrey')),
            text=[f"Max DD: {dd_info['value']:.1f}%"], textposition="bottom center",
            name=f"{strategy_disp_name} Max DD", legendgroup=strategy_disp_name, showlegend=False,
            textfont=dict(size=COMMON_FONT_SIZE_DATALABEL, family="Arial, sans-serif")
        )
        has_marker_plotted = True
    if has_marker_plotted and "Analysis" in title:
        fig.update_layout(title_text=title + " (with Max Drawdown Markers)")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

def plot_daily_returns_distribution(
    performance_dfs: Dict[str, pd.DataFrame],
    title: str = "Daily Returns Distribution",
    save_filename: str = "daily_returns_distribution_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1200, height: int = 750
):
    if not performance_dfs: logger.warning("No performance data for daily returns distribution."); return
    all_returns_list = []
    
    ordered_input_strategy_names_raw = list(performance_dfs.keys())
    unified_input_names = [unify_strategy_name(s) for s in ordered_input_strategy_names_raw]
    
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in unified_input_names])
        actual_strategies_to_plot_ordered.extend([s for s in unified_input_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = unified_input_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)
    
    for strategy_name_orig in ordered_input_strategy_names_raw:
        df_strat = performance_dfs.get(strategy_name_orig)
        if df_strat is None: continue
        
        strategy_display_name = unify_strategy_name(strategy_name_orig)
        if _validate_performance_data(df_strat, 'returns'):
            df_temp = pd.DataFrame()
            df_temp['Return'] = df_strat['returns'].fillna(0.0).astype(float) * 100
            df_temp['Strategy'] = strategy_display_name
            if not df_temp['Return'].empty:
                 all_returns_list.append(df_temp)
        else: logger.warning(f"Skipping strategy '{strategy_name_orig}' for distribution: missing 'returns' or invalid data.")
    
    if not all_returns_list: logger.warning("No valid data for daily returns distribution."); return
    combined_df = pd.concat(all_returns_list)

    final_ordered_strategies_for_plot = [s for s in actual_strategies_to_plot_ordered if s in combined_df['Strategy'].unique()]
    if not final_ordered_strategies_for_plot : logger.warning("No strategies left after data validation for daily returns distribution."); return

    combined_df['Strategy'] = pd.Categorical(combined_df['Strategy'], categories=final_ordered_strategies_for_plot, ordered=True)
    combined_df.sort_values(by="Strategy", inplace=True)

    strategy_color_map = get_color_map(final_ordered_strategies_for_plot, PORTFOLIO_STRATEGY_COLORS, desired_order=final_ordered_strategies_for_plot)

    fig = px.violin(combined_df, x="Strategy", y="Return", color="Strategy",
                    box=True, points=False, title=title,
                    labels={"Return": "Daily Return (%)", "Strategy": "Strategy"},
                    color_discrete_map=strategy_color_map,
                    category_orders={"Strategy": final_ordered_strategies_for_plot} 
                    )
    
    apply_common_font_style(fig)
    fig.update_layout(height=height, hovermode="closest")

    if len(final_ordered_strategies_for_plot) > 1:
        for i in range(len(final_ordered_strategies_for_plot) - 1):
            fig.add_vline(x=i + 0.5, line_width=1.5, line_dash="solid", line_color="darkgrey")

    has_stat_annotated = False
    for i, strategy_disp_name in enumerate(final_ordered_strategies_for_plot):
        strat_returns = combined_df[combined_df['Strategy'] == strategy_disp_name]['Return']
        if strat_returns.empty: continue
        mean_return = strat_returns.mean()
        median_return = strat_returns.median()
        color_val = strategy_color_map.get(strategy_disp_name, px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])
        
        common_ann_props = dict(showarrow=True, arrowhead=1, arrowwidth=1,
                                font=dict(size=COMMON_FONT_SIZE_ANNOTATION, color=color_val),
                                xref="x domain", # Use domain for consistent positioning across violins
                                bordercolor=color_val, borderwidth=1, bgcolor="rgba(255,255,255,0.85)")
        
        # Calculate x position based on index for domain referencing
        x_pos_domain = (i + 0.5) / len(final_ordered_strategies_for_plot)

        fig.add_annotation(x=x_pos_domain, y=mean_return, yshift=-30,
                           text=f"Mean: {mean_return:.2f}%", **common_ann_props, yref='y', xanchor="center")
        fig.add_annotation(x=x_pos_domain, y=median_return, yshift=30, 
                           text=f"Median: {median_return:.2f}%", **common_ann_props, yref='y', xanchor="center")
        has_stat_annotated = True
    if has_stat_annotated and "Distribution" in title:
        fig.update_layout(title_text=title + " with Mean/Median Stats")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

def plot_key_metrics_comparison(
    summary_metrics_df: pd.DataFrame, metrics_to_plot: Optional[List[str]] = None,
    title: str = "Key Performance Metrics Comparison",
    save_filename: str = "key_metrics_comparison_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1500, height: int = 1100
):
    if summary_metrics_df is None or summary_metrics_df.empty:
        logger.warning("Summary metrics DataFrame is empty for key metrics comparison."); return

    default_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Annualized Return', 'Annualized Volatility']
    input_metrics_to_plot_original = list(metrics_to_plot or default_metrics)
    
    df_plot_ready = summary_metrics_df.copy()
    df_plot_ready.index = df_plot_ready.index.map(unify_strategy_name)
    
    raw_index_names = list(df_plot_ready.index.unique())
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in raw_index_names])
        actual_strategies_to_plot_ordered.extend([s for s in raw_index_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = raw_index_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)

    processed_metrics_map = {} 
    final_metrics_for_df_cols = []

    for metric_orig in input_metrics_to_plot_original:
        if metric_orig == 'Max Drawdown' and 'Max Drawdown' in df_plot_ready.columns:
            display_name = 'Max Drawdown (% Abs)'
            df_plot_ready[display_name] = df_plot_ready['Max Drawdown'].abs() * 100
            processed_metrics_map[metric_orig] = display_name
            final_metrics_for_df_cols.append(display_name)
        elif metric_orig in ['Annualized Return', 'Annualized Volatility'] and metric_orig in df_plot_ready.columns:
            display_name = f'{metric_orig} (%)'
            # Check if already in percentage, heuristic: if values are small (e.g. < 1), multiply by 100
            if not (df_plot_ready[metric_orig].abs() > 5).any(): # If most values are small (likely decimals)
                 df_plot_ready[display_name] = df_plot_ready[metric_orig] * 100
            else: # Assume already in percentage or large enough not to scale
                 df_plot_ready[display_name] = df_plot_ready[metric_orig]
            processed_metrics_map[metric_orig] = display_name
            final_metrics_for_df_cols.append(display_name)
        elif metric_orig in df_plot_ready.columns:
            processed_metrics_map[metric_orig] = metric_orig
            final_metrics_for_df_cols.append(metric_orig)
            
    final_metrics_plot_order = [processed_metrics_map[m] for m in input_metrics_to_plot_original if m in processed_metrics_map]

    if not final_metrics_for_df_cols:
        logger.warning(f"None of the specified metrics for plotting resulted in valid columns."); return
    
    df_to_plot_final = df_plot_ready.reset_index().rename(columns={'index': 'Strategy'})
    plot_df_melted = pd.melt(df_to_plot_final, id_vars=['Strategy'], value_vars=final_metrics_for_df_cols,
                             var_name='Metric', value_name='Value')
    if plot_df_melted.empty: logger.warning("No data to plot after melting for key metrics comparison."); return
    
    plot_df_melted['Metric'] = pd.Categorical(plot_df_melted['Metric'], categories=final_metrics_plot_order, ordered=True)
    plot_df_melted['Strategy'] = pd.Categorical(plot_df_melted['Strategy'], categories=actual_strategies_to_plot_ordered, ordered=True)
    plot_df_melted.sort_values(by=['Metric', 'Strategy'], inplace=True)

    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)

    num_metrics = len(final_metrics_plot_order)
    facet_col_wrap = 3 if num_metrics > 2 else num_metrics if num_metrics > 0 else 1 # Adjust wrap
    if num_metrics > 6: facet_col_wrap = 3 # Max 3 columns for many metrics
    elif num_metrics > 4: facet_col_wrap = 2

    fig = px.bar(plot_df_melted, x="Strategy", y="Value", color="Strategy",
                 facet_col="Metric", facet_col_wrap=facet_col_wrap,
                 title=title, labels={"Value": "", "Strategy": ""},
                 color_discrete_map=strategy_color_map,
                 height=height, width=width, text_auto='.2f',
                 category_orders={
                     "Strategy": actual_strategies_to_plot_ordered,
                     "Metric": final_metrics_plot_order
                 })

    apply_common_font_style(fig)
    fig.update_traces(marker_opacity=0.8, textposition='outside')
    fig.update_layout(margin=dict(l=70, r=50, t=100, b=120),
                      legend_orientation="h", legend_yanchor="bottom", legend_y=1.02, legend_xanchor="right", legend_x=1)

    fig.update_yaxes(matches=None, title_text="Value", title_standoff=15) 
    fig.update_xaxes(showticklabels=True, title_text="Strategy", tickangle=-30, title_standoff=15)
    
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=COMMON_FONT_SIZE_AXIS_TITLE))

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

def plot_annualized_return_vs_volatility(
    summary_metrics_df: pd.DataFrame,
    title: str = "Risk-Return Profile of Strategies",
    save_filename: str = "return_vs_volatility_plotly.png",
    save_dir: Optional[Path] = None, width: int = 1000, height: int = 800
):
    required_cols = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']
    if summary_metrics_df is None or summary_metrics_df.empty or not all(col in summary_metrics_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in summary_metrics_df.columns and col in required_cols]
        logger.warning(f"Missing required columns {missing} for Risk-Return plot."); return
    
    df_plot = summary_metrics_df.copy()
    df_plot.index = df_plot.index.map(unify_strategy_name)
    df_plot['Strategy'] = df_plot.index
    
    raw_index_names = list(df_plot.index.unique())
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in raw_index_names])
        actual_strategies_to_plot_ordered.extend([s for s in raw_index_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = raw_index_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)

    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df_plot[col]):
            try: df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce').fillna(0)
            except ValueError: logger.error(f"Could not convert column '{col}' to numeric for Risk-Return plot."); return
    
    df_plot['Annualized Return Display'] = df_plot['Annualized Return'] * 100
    df_plot['Annualized Volatility Display'] = df_plot['Annualized Volatility'] * 100
        
    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)

    min_sharpe_abs = df_plot['Sharpe Ratio'].abs().min() if not df_plot['Sharpe Ratio'].empty else 0
    # Scale size by absolute Sharpe, ensure minimum size for visibility
    df_plot['Sharpe Ratio Size'] = (df_plot['Sharpe Ratio'].abs().fillna(0) - min_sharpe_abs + 0.5).clip(lower=0.5) * 15 + 7
    
    fig = px.scatter(
        data_frame=df_plot, x="Annualized Volatility Display", y="Annualized Return Display",
        color="Strategy", color_discrete_map=strategy_color_map,
        size="Sharpe Ratio Size", size_max=45,
        hover_name="Strategy",
        hover_data={
            'Strategy': False,
            'Annualized Return Display': ':.2f%', 
            'Annualized Volatility Display': ':.2f%', 
            'Sharpe Ratio': ':.2f',
            'Sharpe Ratio Size': False # Hide this auxiliary column from hover
        },
        text="Strategy", title=title,
        labels={"Annualized Volatility Display": "Annualized Volatility (%)", 
                "Annualized Return Display": "Annualized Return (%)"},
        category_orders={"Strategy": actual_strategies_to_plot_ordered}
    )
    
    apply_common_font_style(fig)
    fig.update_traces(textposition='top center', marker_opacity = 0.85, textfont=dict(size=COMMON_FONT_SIZE_DATALABEL -1, family="Arial, sans-serif"))
    fig.update_layout(height=height, hovermode="closest", legend_title_text="Portfolio Strategy")

    if not df_plot.empty:
        all_vals_vol = df_plot["Annualized Volatility Display"].dropna()
        all_vals_ret = df_plot["Annualized Return Display"].dropna()
        if not all_vals_vol.empty and not all_vals_ret.empty:
            # Determine range based on actual data points for the line
            min_coord = min(all_vals_vol.min(), all_vals_ret.min()) * 0.9
            max_coord = max(all_vals_vol.max(), all_vals_ret.max()) * 1.1
            min_coord = min(min_coord, 0) # Ensure 0 is included if values are positive

            fig.add_shape(type="line", x0=min_coord, y0=min_coord, x1=max_coord, y1=max_coord, 
                          line=dict(dash="dashdot", color="darkgrey"), name="Ref: Return = Volatility")

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

def plot_rolling_sharpe_ratio(
    performance_dfs: Dict[str, pd.DataFrame], rolling_window_days: int = 60,
    risk_free_rate: float = 0.02, 
    # title_suffix is not actively used in the current title logic below, but kept for signature compatibility
    title_suffix: str = "", 
    save_filename_base: str = "portfolio_rolling_sharpe",
    save_dir: Optional[Path] = None, width: int = 1200, height: int = 750
):
    if not performance_dfs: logger.warning("No performance data for rolling Sharpe."); return
    try: import quantstats as qs; qs.extend_pandas()
    except ImportError: logger.error("QuantStats library not found for rolling Sharpe."); return

    fig = go.Figure()
    
    ordered_input_strategy_names_raw = list(performance_dfs.keys())
    unified_input_names = [unify_strategy_name(s) for s in ordered_input_strategy_names_raw]
    
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in unified_input_names])
        actual_strategies_to_plot_ordered.extend([s for s in unified_input_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = unified_input_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)
    
    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)
    has_annotations = False

    for strategy_display_name in actual_strategies_to_plot_ordered:
        # Find original name to fetch df
        original_name_for_df = next((orig_name for orig_name in performance_dfs if unify_strategy_name(orig_name) == strategy_display_name), None)
        if not original_name_for_df: continue
        df = performance_dfs[original_name_for_df]

        if _validate_performance_data(df, 'returns'):
            returns_series = df['returns'].fillna(0.0).astype(float)
            if len(returns_series) >= rolling_window_days:
                daily_rf = risk_free_rate / 252.0 # Assuming 252 trading days
                rolling_sharpe = qs.stats.rolling_sharpe(returns_series, rf=daily_rf, rolling_period=rolling_window_days,
                                                       annualize=True, periods_per_year=252, prepare_returns=False) # prepare_returns=False as we pass returns
                if rolling_sharpe is not None and not rolling_sharpe.empty:
                    color_val = strategy_color_map.get(strategy_display_name, px.colors.qualitative.Plotly[0]) # Default color
                    fig.add_trace(go.Scattergl(x=rolling_sharpe.index, y=rolling_sharpe, mode='lines', name=strategy_display_name,
                                             line=dict(color=color_val, width=2)))
                    
                    peak_value = rolling_sharpe.max(); peak_date = rolling_sharpe.idxmax()
                    if pd.notna(peak_value) and pd.notna(peak_date):
                        yshift_val = 15 if peak_value > 0 else -25 
                        fig.add_annotation(x=peak_date, y=peak_value, text=f"Peak:{peak_value:.2f}", showarrow=True, arrowhead=2,
                                           font=dict(color=color_val, size=COMMON_FONT_SIZE_ANNOTATION -1),
                                           ax=0, ay=yshift_val * (-1), 
                                           align="center", borderpad=2, bgcolor="rgba(255,255,255,0.6)")
                        has_annotations = True
    
    if not fig.data: logger.warning(f"No valid data for rolling Sharpe (window {rolling_window_days}d)."); return

    final_title = f"Rolling Sharpe Ratio ({rolling_window_days}-Day Window)"
    if has_annotations:
        final_title += " with Peak Annotations"

    apply_common_font_style(fig)
    fig.update_layout(title_text=final_title, height=height, hovermode="x unified")

    save_filename = f"{save_filename_base}_{rolling_window_days}d.png"
    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)

# New function as per request
def plot_rolling_sharpe_ratio_faceted(
    performance_dfs: Dict[str, pd.DataFrame],
    rolling_windows: List[int],
    risk_free_rate: float = 0.02,
    title: str = "Rolling Sharpe Ratio Comparison (Faceted by Window)",
    save_filename: str = "rolling_sharpe_faceted.png",
    save_dir: Optional[Path] = None,
    width: int = 1600, height: int = 900 # Adjusted default size for faceted plot
):
    if not performance_dfs: logger.warning("No performance data for faceted rolling Sharpe."); return
    if not rolling_windows: logger.warning("No rolling windows specified for faceted rolling Sharpe."); return
    try: import quantstats as qs; qs.extend_pandas()
    except ImportError: logger.error("QuantStats library not found for faceted rolling Sharpe."); return

    all_rolling_sharpe_data = []
    daily_rf = risk_free_rate / 252.0 # Assuming 252 trading days per year
    
    # Determine overall strategy order for consistent coloring and legend
    ordered_input_strategy_names_raw = list(performance_dfs.keys())
    unified_input_names = [unify_strategy_name(s) for s in ordered_input_strategy_names_raw]
    
    actual_strategies_to_plot_ordered = []
    if DESIRED_STRATEGY_ORDER:
        desired_unified = [unify_strategy_name(s) for s in DESIRED_STRATEGY_ORDER]
        actual_strategies_to_plot_ordered.extend([s for s in desired_unified if s in unified_input_names])
        actual_strategies_to_plot_ordered.extend([s for s in unified_input_names if s not in actual_strategies_to_plot_ordered])
    else:
        actual_strategies_to_plot_ordered = unified_input_names
    actual_strategies_to_plot_ordered = sorted(list(set(actual_strategies_to_plot_ordered)), key=actual_strategies_to_plot_ordered.index)

    for strategy_name_orig in ordered_input_strategy_names_raw: # Iterate through original keys to get data
        df_strat = performance_dfs.get(strategy_name_orig)
        if df_strat is None: continue
        strategy_display_name = unify_strategy_name(strategy_name_orig)
        
        if not _validate_performance_data(df_strat, 'returns'):
            logger.warning(f"Skipping strategy '{strategy_name_orig}' for faceted rolling Sharpe - invalid 'returns' data.")
            continue
        
        returns_series = df_strat['returns'].fillna(0.0).astype(float)

        for window in rolling_windows:
            if len(returns_series) >= window:
                rolling_sharpe = qs.stats.rolling_sharpe(
                    returns_series, rf=daily_rf, rolling_period=window,
                    annualize=True, periods_per_year=252, prepare_returns=False # prepare_returns=False as we pass returns
                )
                if rolling_sharpe is not None and not rolling_sharpe.empty:
                    df_temp = rolling_sharpe.reset_index()
                    df_temp.columns = ['Date', 'Sharpe Ratio']
                    df_temp['Strategy'] = strategy_display_name
                    df_temp['Window'] = f"{window}-Day Window" # For facet title
                    all_rolling_sharpe_data.append(df_temp)
                else: 
                    logger.debug(f"Rolling Sharpe for '{strategy_display_name}' (window {window}d) was empty or None.")
            else: 
                logger.debug(f"Not enough data for '{strategy_display_name}' (window {window}d) for faceted plot. Has {len(returns_series)} points.")

    if not all_rolling_sharpe_data: 
        logger.warning("No data to plot for faceted rolling Sharpe after processing all strategies and windows."); 
        return
    
    combined_df = pd.concat(all_rolling_sharpe_data)
    
    # Ensure categorical order for windows and strategies
    ordered_windows_cat = [f"{w}-Day Window" for w in sorted(rolling_windows)]
    combined_df['Window'] = pd.Categorical(combined_df['Window'], categories=ordered_windows_cat, ordered=True)
    combined_df['Strategy'] = pd.Categorical(combined_df['Strategy'], categories=actual_strategies_to_plot_ordered, ordered=True)
    combined_df.sort_values(by=['Window', 'Strategy', 'Date'], inplace=True) # Sort for consistent line plotting
    
    strategy_color_map = get_color_map(actual_strategies_to_plot_ordered, PORTFOLIO_STRATEGY_COLORS, desired_order=actual_strategies_to_plot_ordered)

    # Determine facet_col_wrap dynamically, e.g., max 3 columns
    facet_col_wrap = min(len(ordered_windows_cat), 3)

    fig = px.line(
        combined_df, x="Date", y="Sharpe Ratio", color="Strategy",
        facet_col="Window",
        facet_col_wrap=facet_col_wrap,
        title=title, 
        labels={"Sharpe Ratio": "Annualized Sharpe Ratio", "Date": "Date"}, # Y-axis label for all facets
        color_discrete_map=strategy_color_map, 
        height=height, 
        width=width,
        category_orders={"Window": ordered_windows_cat, "Strategy": actual_strategies_to_plot_ordered}
    )
    fig.update_traces(line_width=1.5) # Slightly thinner lines for faceted plot

    apply_common_font_style(fig)
    fig.update_layout(legend_title_text="Strategy", hovermode="x unified")
    
    # Customize facet titles and axes
    fig.for_each_xaxis(lambda axis: axis.update(title_text="")) # Remove individual x-axis titles if Date is clear
    fig.for_each_yaxis(lambda axis: axis.update(title_text="Ann. Sharpe Ratio")) # Ensure y-axis title is consistent
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=COMMON_FONT_SIZE_AXIS_TITLE)) # Clean facet titles

    current_save_dir = save_dir if save_dir else DEFAULT_PLOT_SAVE_DIR
    _save_plot(fig, save_filename, current_save_dir, width, height)