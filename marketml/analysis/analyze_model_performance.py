# /.ndmh/marketml/analysis/analyze_model_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

try:
    from marketml.configs import configs
    from marketml.utils import logger_setup
except ImportError:
    print("CRITICAL ERROR in analyze_model_performance.py: Could not import 'marketml.configs' or 'marketml.utils.logger_setup'.")
    class FallbackConfigs:
        RESULTS_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results_output"
        MODEL_PERF_SUMMARY_FILE = RESULTS_OUTPUT_DIR / "model_performance_summary.csv"
        MODEL_PERF_DETAILED_FILE = RESULTS_OUTPUT_DIR / "model_performance_detailed.csv"
    configs = FallbackConfigs()
    # Fallback logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("analyzer_fallback")
    logger.warning("Using fallback configs and logger for analyze_model_performance.py.")


if 'logger' not in locals():
    logger = logger_setup.setup_basic_logging(log_file_name="analyze_performance.log")

# --- Get configuration from configs.py ---
SUMMARY_FILE_PATH = configs.MODEL_PERF_SUMMARY_FILE
DETAILED_FILE_PATH = configs.MODEL_PERF_DETAILED_FILE
PLOTS_SAVE_DIR = configs.RESULTS_OUTPUT_DIR # Save plots in the same directory as results CSV

# --- Constants that can be moved to configs.py if desired ---
# Example: configs.ANALYSIS_METRICS_TO_PLOT, configs.ANALYSIS_MODEL_NAMES
METRICS_SUFFIXES_TO_PLOT = getattr(configs, 'ANALYSIS_METRICS_SUFFIXES', [
    "_Accuracy", "_F1_Macro", "_F1_Weighted", "_Precision_Macro", "_Recall_Macro"
])
MODEL_NAMES_TO_ANALYZE = getattr(configs, 'ANALYSIS_MODEL_NAMES', ["ARIMA", "RandomForest", "XGBoost", "LSTM", "Transformer", "SVM"])


def load_summary_results(file_path: Path) -> pd.DataFrame | None:
    logger.info(f"Loading performance summary from: {file_path}")
    try:
        df_summary = pd.read_csv(file_path, index_col=0) # metric name is index
        logger.info("Summary loaded successfully.")
        return df_summary
    except FileNotFoundError:
        logger.error(f"Summary file not found at '{file_path}'. "
                     "Please run '02_train_forecasting_models.py' to generate results first.")
        return None
    except Exception as e:
        logger.error(f"Error loading summary file '{file_path}': {e}", exc_info=True)
        return None

def plot_mean_performance(df_summary: pd.DataFrame, metrics_suffixes: list, model_names: list, save_dir: Path):
    if df_summary is None or df_summary.empty or 'mean' not in df_summary.columns:
        logger.warning("No valid summary data or 'mean' column to plot for mean performance.")
        return

    plot_data_list = []
    for model_prefix in model_names:
        for metric_suffix in metrics_suffixes:
            full_metric_name = model_prefix + metric_suffix
            if full_metric_name in df_summary.index:
                 plot_data_list.append({
                     'Model': model_prefix,
                     'Metric': metric_suffix.replace("_", " "), # Clean name for plot
                     'Mean_Performance': df_summary.loc[full_metric_name, 'mean']
                 })

    if not plot_data_list:
        logger.warning("No data to plot for mean performance after filtering metrics and models.")
        return

    plot_df = pd.DataFrame(plot_data_list)
    plot_df.sort_values(by=['Metric', 'Mean_Performance'], ascending=[True, False], inplace=True)

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Metric', y='Mean_Performance', hue='Model', data=plot_df, palette='viridis')
    plt.title('Mean Model Performance Comparison Across CV Splits', fontsize=16)
    plt.ylabel('Mean Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='best') # 'center left' may be cut off
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=1.5) # Add padding

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_output_path = save_dir / "mean_performance_comparison.png"
        plt.savefig(plot_output_path)
        logger.info(f"Mean performance comparison plot saved to: {plot_output_path.resolve()}")
        plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error saving mean performance plot: {e}", exc_info=True)


def plot_performance_distribution(detailed_results_path: Path, metrics_suffixes: list, model_names: list, save_dir: Path):
    logger.info(f"Attempting to load detailed results from: {detailed_results_path}")
    try:
        df_detailed = pd.read_csv(detailed_results_path)
    except FileNotFoundError:
        logger.warning(f"Detailed results file not found at '{detailed_results_path}'. Skipping distribution plot.")
        return
    except Exception as e:
        logger.error(f"Error loading detailed results file '{detailed_results_path}': {e}", exc_info=True)
        return

    if df_detailed.empty or len(df_detailed) <= 1: # Need at least 2 splits for boxplot to be meaningful
        logger.warning("Not enough splits/data in detailed results to plot distribution (need at least 2 splits).")
        return

    # Filter columns to plot based on available model_metric combinations
    metrics_to_plot_full = []
    for model_prefix in model_names:
        for metric_suffix in metrics_suffixes:
            full_metric_name = model_prefix + metric_suffix
            if full_metric_name in df_detailed.columns:
                metrics_to_plot_full.append(full_metric_name)
    
    if not metrics_to_plot_full:
        logger.warning("No specified metrics found in the detailed data to plot distribution.")
        return

    plot_data_melted = pd.melt(df_detailed, value_vars=metrics_to_plot_full,
                               var_name='Metric_Model', value_name='Score')
    
    # Extract Model and Metric for hue and x-axis
    plot_data_melted['Model'] = plot_data_melted['Metric_Model'].apply(lambda x: x.split('_')[0])
    plot_data_melted['Metric'] = plot_data_melted['Metric_Model'].apply(lambda x: "_".join(x.split('_')[1:]).replace("_", " "))

    # Filter again for models we are interested in (in case Metric_Model had other prefixes)
    plot_data_melted = plot_data_melted[plot_data_melted['Model'].isin(model_names)]
    if plot_data_melted.empty:
        logger.warning("No data left to plot for distribution after filtering by model names.")
        return

    plt.figure(figsize=(18, 10))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=plot_data_melted, palette='Set2')
    plt.title('Model Performance Distribution Across CV Splits', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=1.5)

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_output_path = save_dir / "performance_distribution.png"
        plt.savefig(plot_output_path)
        logger.info(f"Performance distribution plot saved to: {plot_output_path.resolve()}")
        # plt.show()
        plt.close()
    except Exception as e:
        logger.error(f"Error saving performance distribution plot: {e}", exc_info=True)


def main():
    logger.info("--- Analyzing Model Performance ---")

    df_summary = load_summary_results(file_path=SUMMARY_FILE_PATH)

    if df_summary is not None and not df_summary.empty:
        logger.info("\n--- Performance Summary Table (Mean & Std Dev) ---")
        df_summary_display = df_summary.copy()
        if 'mean' in df_summary_display.columns and 'std' in df_summary_display.columns:
            # Filter to display only rows that have a valid 'mean'
            df_summary_display_valid = df_summary_display.dropna(subset=['mean'])
            if not df_summary_display_valid.empty:
                df_summary_display_valid['Mean +/- Std'] = df_summary_display_valid.apply(
                    lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1
                )
                logger.info(f"\n{df_summary_display_valid[['Mean +/- Std']].to_string()}")
            else:
                logger.info("No metrics with valid mean values found in the summary.")
        else:
            logger.info(f"\n{df_summary_display.to_string()}") # Print as is if no mean/std cols

        plot_mean_performance(df_summary,
                              metrics_suffixes=METRICS_SUFFIXES_TO_PLOT,
                              model_names=MODEL_NAMES_TO_ANALYZE,
                              save_dir=PLOTS_SAVE_DIR)

        if DETAILED_FILE_PATH.exists():
             try:
                df_detailed_check = pd.read_csv(DETAILED_FILE_PATH)
                if len(df_detailed_check) > 1 :
                    plot_performance_distribution(detailed_results_path=DETAILED_FILE_PATH,
                                                  metrics_suffixes=METRICS_SUFFIXES_TO_PLOT,
                                                  model_names=MODEL_NAMES_TO_ANALYZE,
                                                  save_dir=PLOTS_SAVE_DIR)
                else:
                    logger.info(f"Skipping distribution plot: Detailed results file '{DETAILED_FILE_PATH}' has too few splits (<=1).")
             except pd.errors.EmptyDataError:
                 logger.warning(f"Skipping distribution plot: Detailed results file '{DETAILED_FILE_PATH}' is empty.")
             except Exception as e_read_detail:
                 logger.error(f"Could not read detailed results file '{DETAILED_FILE_PATH}' for pre-check: {e_read_detail}")
        else:
            logger.info(f"Skipping distribution plot: Detailed results file '{DETAILED_FILE_PATH}' not found.")
    else:
        logger.warning("Summary data not loaded or empty. No analysis performed.")

    logger.info("--- Analysis Finished ---")

if __name__ == "__main__":
    main()