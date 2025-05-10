import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Cấu hình Đường dẫn ---
# Giả định script này nằm ở thư mục gốc dự án
project_root_from_cwd = Path.cwd()
RESULTS_DIR = project_root_from_cwd / "marketml" / "results" # Thư mục results nằm ở gốc
SUMMARY_FILE = RESULTS_DIR / "model_performance_summary.csv"
DETAILED_FILE = RESULTS_DIR / "model_performance_detailed.csv"

# --- Các metrics chính cần trực quan hóa ---
# Lấy từ tên cột trong file summary (ví dụ: ARIMA_Accuracy, RandomForest_F1_Macro)
# Bạn có thể tùy chỉnh danh sách này
METRICS_TO_PLOT = [
    "_Accuracy",
    "_F1_Macro",
    "_F1_Weighted",
    "_Precision_Macro",
    "_Recall_Macro"
]

MODEL_NAMES = ["ARIMA", "RandomForest", "XGBoost", "LSTM", "Transformer"] # Cập nhật nếu có thêm/bớt mô hình

def load_summary_results(file_path=SUMMARY_FILE):
    """Tải dữ liệu tóm tắt hiệu suất từ file CSV."""
    try:
        print(f"Loading performance summary from: {file_path}")
        # Index_col=0 để đọc cột đầu tiên (tên metric) làm index
        df_summary = pd.read_csv(file_path, index_col=0)
        print("Summary loaded successfully.")
        return df_summary
    except FileNotFoundError:
        print(f"Error: Summary file not found at '{file_path}'.")
        print("Please run 'run_experiment.py' to generate results first.")
        return None
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return None

def plot_mean_performance(df_summary, metrics_to_plot=None, model_names=None):
    """
    Vẽ biểu đồ cột so sánh hiệu suất trung bình của các mô hình cho các metrics được chọn.
    """
    if df_summary is None or 'mean' not in df_summary.columns:
        print("No valid summary data or 'mean' column to plot.")
        return

    if metrics_to_plot is None:
        # Tự động lấy tất cả các metrics có 'Accuracy' hoặc 'F1' nếu không được cung cấp
        metrics_to_plot = [idx for idx in df_summary.index if "Accuracy" in idx or "F1" in idx]
    else: # Lọc ra các metric thực sự tồn tại trong df_summary
        metrics_to_plot = [m for m in df_summary.index if any(suffix in m for suffix in metrics_to_plot)]


    if not metrics_to_plot:
        print("No specified metrics found in the summary data to plot.")
        return

    # Tạo DataFrame mới chỉ chứa giá trị mean cho các metric cần vẽ
    plot_data_list = []
    for model_prefix in model_names:
        for metric_suffix in METRICS_TO_PLOT: # Dùng METRICS_TO_PLOT gốc để tạo tên
            full_metric_name = model_prefix + metric_suffix
            if full_metric_name in df_summary.index:
                 plot_data_list.append({
                     'Model': model_prefix,
                     'Metric': metric_suffix.replace("_", " "), # Tên hiển thị đẹp hơn
                     'Mean_Performance': df_summary.loc[full_metric_name, 'mean']
                 })

    if not plot_data_list:
        print("No data to plot after filtering metrics and models.")
        return

    plot_df = pd.DataFrame(plot_data_list)

    # Sắp xếp để biểu đồ dễ nhìn hơn
    plot_df.sort_values(by=['Metric', 'Mean_Performance'], ascending=[True, False], inplace=True)

    plt.figure(figsize=(15, 8)) # Kích thước biểu đồ
    sns.barplot(x='Metric', y='Mean_Performance', hue='Model', data=plot_df, palette='viridis')
    plt.title('Mean Model Performance Comparison Across CV Splits', fontsize=16)
    plt.ylabel('Mean Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='center left', bbox_to_anchor=(-0.2, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Lưu biểu đồ
    plot_output_path = RESULTS_DIR / "mean_performance_comparison.png"
    plt.savefig(plot_output_path)
    print(f"Performance comparison plot saved to: {plot_output_path}")
    plt.show()


def plot_performance_distribution(detailed_results_path=DETAILED_FILE, metrics_to_plot=None, model_names=None):
    """
    Vẽ box plot để xem phân phối hiệu suất của các mô hình qua các split (nếu có nhiều split).
    """
    try:
        df_detailed = pd.read_csv(detailed_results_path)
        print(f"\nLoading detailed results from: {detailed_results_path}")
    except FileNotFoundError:
        print(f"Warning: Detailed results file not found at '{detailed_results_path}'. Skipping distribution plot.")
        return
    except Exception as e:
        print(f"Error loading detailed results file: {e}")
        return

    if df_detailed.empty or len(df_detailed) <= 1:
        print("Not enough splits in detailed results to plot distribution. Need at least 2 splits.")
        return

    if metrics_to_plot is None:
        metrics_to_plot = [col for col in df_detailed.columns if "Accuracy" in col or "F1_Macro" in col] # Chỉ lấy F1_Macro cho gọn
    else:
        metrics_to_plot_full = []
        for model_prefix in model_names:
            for metric_suffix in metrics_to_plot:
                if model_prefix + metric_suffix in df_detailed.columns:
                    metrics_to_plot_full.append(model_prefix + metric_suffix)
        metrics_to_plot = metrics_to_plot_full


    if not metrics_to_plot:
        print("No specified metrics found in the detailed data to plot distribution.")
        return

    # Chuẩn bị dữ liệu cho boxplot (melt dataframe)
    plot_data_melted = pd.melt(df_detailed, value_vars=metrics_to_plot, var_name='Metric_Model', value_name='Score')
    # Tách Model và Metric từ Metric_Model
    plot_data_melted['Model'] = plot_data_melted['Metric_Model'].apply(lambda x: x.split('_')[0])
    plot_data_melted['Metric'] = plot_data_melted['Metric_Model'].apply(lambda x: "_".join(x.split('_')[1:]).replace("_", " "))

    # Lọc lại chỉ các model có trong MODEL_NAMES
    plot_data_melted = plot_data_melted[plot_data_melted['Model'].isin(model_names)]


    plt.figure(figsize=(18, 10))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=plot_data_melted, palette='Set2')
    plt.title('Model Performance Distribution Across CV Splits', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_output_path = RESULTS_DIR / "performance_distribution.png"
    plt.savefig(plot_output_path)
    print(f"Performance distribution plot saved to: {plot_output_path}")
    plt.show()


def main():
    print("--- Analyzing Model Performance ---")

    # Tải dữ liệu tóm tắt
    df_summary = load_summary_results()

    if df_summary is not None:
        print("\n--- Performance Summary Table (Mean & Std Dev) ---")
        # Tạo lại cột mean_std_str để hiển thị nếu cần
        df_summary_display = df_summary.copy()
        if 'mean' in df_summary_display.columns and 'std' in df_summary_display.columns:
            df_summary_display['mean_std_str'] = df_summary_display.apply(
                lambda x: f"{x['mean']:.4f} +/- {x['std']:.4f}" if pd.notna(x['std']) else f"{x['mean']:.4f}", axis=1
            )
            print(df_summary_display[['mean_std_str']])
        else:
            print(df_summary_display)


        # Vẽ biểu đồ so sánh hiệu suất trung bình
        plot_mean_performance(df_summary, metrics_to_plot=METRICS_TO_PLOT, model_names=MODEL_NAMES)

        # Vẽ biểu đồ phân phối hiệu suất (nếu có nhiều split và file detailed tồn tại)
        if DETAILED_FILE.exists() and len(pd.read_csv(DETAILED_FILE)) > 1 :
            plot_performance_distribution(detailed_results_path=DETAILED_FILE, metrics_to_plot=METRICS_TO_PLOT, model_names=MODEL_NAMES)
        else:
            print(f"\nSkipping distribution plot: Detailed results file '{DETAILED_FILE}' not found or has too few splits.")

    print("\n--- Analysis Finished ---")

if __name__ == "__main__":
    main()