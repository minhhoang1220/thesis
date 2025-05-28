# marketml/configs/configs.py
from pathlib import Path
import pandas as pd # Giữ lại pandas nếu có các tham số dùng pd.Timedelta

# PROJECT_ROOT is the .ndmh/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Thư mục gốc của ứng dụng MarketML ---
MARKETML_APP_ROOT = PROJECT_ROOT / "marketml" # /.ndmh/marketml/

# --- Input Data Paths (Bên trong marketml/data_input) ---
INPUT_DATA_DIR = MARKETML_APP_ROOT / "data_input"
RAW_GLOBAL_PRICE_FILE = INPUT_DATA_DIR / "yahoo_price_data_fixed.csv"
RAW_GLOBAL_FINANCIAL_FILE = INPUT_DATA_DIR / "financial_data.csv"
FINANCIAL_DATA_FILE_PATH = INPUT_DATA_DIR / "financial_data.csv" # Thống nhất, đây là file financial đầu vào

# --- Processed Data Paths (Bên trong marketml/data_processed) ---
PROCESSED_DATA_DIR = MARKETML_APP_ROOT / "data_processed"
ENRICHED_DATA_FILE = PROCESSED_DATA_DIR / "price_data_enriched_v2.csv"
PRICE_DATA_FOR_PORTFOLIO_PATH = ENRICHED_DATA_FILE # Dùng cho portfolio
ENRICHED_DATA_FOR_FORECAST = ENRICHED_DATA_FILE # Dùng cho forecast_future

# --- Output Directories ---
RESULTS_OUTPUT_DIR = MARKETML_APP_ROOT / "results_output" # Đường dẫn chính cho output
FORECASTS_OUTPUT_DIR = MARKETML_APP_ROOT / "forecasts_output"
LOG_OUTPUT_DIR = PROJECT_ROOT / "logs" # Log của toàn bộ dự án (orchestrator, etc.)
PLOTS_OUTPUT_DIR = RESULTS_OUTPUT_DIR / "plots" # Sẽ là marketml/results_output/plots

# --- File Names for Outputs (Tất cả dựa trên RESULTS_OUTPUT_DIR hoặc FORECASTS_OUTPUT_DIR) ---
CLASSIFICATION_PROBS_FILE = RESULTS_OUTPUT_DIR / "classification_probabilities.csv"
CLASSIFICATION_PROBS_FILE_PATH = CLASSIFICATION_PROBS_FILE # Thống nhất biến (sử dụng một trong hai)

MODEL_PERF_SUMMARY_FILE = RESULTS_OUTPUT_DIR / "model_performance_summary.csv"
MODEL_PERF_DETAILED_FILE = RESULTS_OUTPUT_DIR / "model_performance_detailed.csv"

MARKOWITZ_PERF_DAILY_FILE_NAME = "markowitz_performance_daily.csv"
BLACKLITTERMAN_PERF_DAILY_FILE_NAME = "blacklitterman_performance_daily.csv"
PORTFOLIO_STRATEGIES_SUMMARY_FILE_NAME = "portfolio_strategies_summary.csv" # Dùng để lưu summary của các chiến lược

MARKOWITZ_PERF_DAILY_FILE = RESULTS_OUTPUT_DIR / MARKOWITZ_PERF_DAILY_FILE_NAME
BLACKLITTERMAN_PERF_DAILY_FILE = RESULTS_OUTPUT_DIR / BLACKLITTERMAN_PERF_DAILY_FILE_NAME
# Đường dẫn cho file PPO strategy performance sẽ được xây dựng động, ví dụ:
# RL_STRATEGY_PERF_DAILY_FILE_NAME = f"{'PPO'.lower()}_performance_daily.csv"
# RL_STRATEGY_PERF_DAILY_FILE = RESULTS_OUTPUT_DIR / RL_STRATEGY_PERF_DAILY_FILE_NAME

# --- Reinforcement Learning Model Paths ---
RL_MODEL_DIR = RESULTS_OUTPUT_DIR / "rl_models" # marketml/results_output/rl_models
RL_ALGORITHM = "PPO" # Thuật toán RL, ví dụ PPO, A2C, DDPG
RL_MODEL_NAME = f"{RL_ALGORITHM.lower()}_portfolio_agent.zip"
RL_MODEL_SAVE_PATH = RL_MODEL_DIR / RL_MODEL_NAME
RL_LOG_DIR_FOR_SB3 = RL_MODEL_DIR / "sb3_logs" # Logs riêng cho Stable Baselines3 Tensorboard
RL_TRAINING_LOG_DIR = RL_MODEL_DIR / "training_logs" # Logs của quá trình training RL (có thể khác LOG_OUTPUT_DIR)

# --- Model Performance Analysis ---
ANALYSIS_METRICS_SUFFIXES = [
    "_Accuracy", "_F1_Macro", "_F1_Weighted", "_Precision_Macro", "_Recall_Macro"
] # Suffixes for classification metrics
ANALYSIS_MODEL_NAMES = ["ARIMA", "RandomForest", "XGBoost", "LSTM", "Transformer", "SVM"] # Models for forecasting/classification analysis
# Thứ tự hiển thị mong muốn cho các mô hình forecasting trong biểu đồ
ANALYSIS_FORECASTING_MODEL_ORDER = ["ARIMA", "Prophet", "SVM", "RandomForest", "XGBoost", "LSTM", "Transformer"] # Giữ lại từ plot_utils hoặc định nghĩa ở đây
# Thứ tự hiển thị mong muốn cho chiến lược portfolio trong biểu đồ
ANALYSIS_STRATEGY_ORDER = ["Markowitz", "BlackLitterman", "RL (PPO)", "Benchmark"] # Thêm Benchmark nếu có

# --- Portfolio Performance Analysis ---
PORTFOLIO_KEY_METRICS_PLOT = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Annualized Return', 'Annualized Volatility']
PORTFOLIO_ROLLING_SHARPE_WINDOWS = [30, 60, 90]
# Tên file đã được dùng ở trên để tạo đường dẫn đầy đủ (e.g., PORTFOLIO_STRATEGIES_SUMMARY_FILE)

# --- DATA PARAMETERS ---
TIME_RANGE_START = "2020-01-01"
TIME_RANGE_END = "2024-12-31" # Đảm bảo dữ liệu bao phủ khoảng thời gian này

# --- FEATURE ENGINEERING PARAMETERS ---
GARCH_WINDOW = 252
GARCH_FORECAST_HORIZON = 1
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
SMA_WINDOW = 20
EMA_WINDOW = 20
ROLLING_STAT_WINDOWS = [5, 10]
PRICE_ZSCORE_WINDOW = 20
LAG_PERIODS = [1, 3, 5]
TREND_THRESHOLD = 0.002 # Ngưỡng % thay đổi để xác định trend tăng/giảm/giữ nguyên
INITIAL_TRAIN_YEARS = 1
TEST_YEARS = 1
STEP_YEARS = 1
USE_EXPANDING_WINDOW = True # True for expanding, False for rolling window in walk-forward
INITIAL_TRAIN_TIMEDELTA = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS)
TEST_TIMEDELTA = pd.Timedelta(days=365 * TEST_YEARS)
STEP_TIMEDELTA = pd.Timedelta(days=365 * STEP_YEARS)
TARGET_COL_PCT = 'target_pct_change' # Tên cột mục tiêu % thay đổi giá
TARGET_COL_TREND = 'target_trend' # Tên cột mục tiêu xu hướng giá (categorical)
N_ITER_SEARCH_SKLEARN = 30 # Số lần lặp cho RandomizedSearchCV
CV_FOLDS_TUNING_SKLEARN = 3 # Số fold cho cross-validation khi tuning
N_TIMESTEPS_SEQUENCE = 10 # Số bước thời gian cho mô hình sequence (LSTM, Transformer)
KERAS_EPOCHS = 50
KERAS_BATCH_SIZE = 64
KERAS_VALIDATION_SPLIT = 0.1
KERAS_EARLY_STOPPING_PATIENCE = 10
KERAS_REDUCE_LR_PATIENCE = 5
KERAS_REDUCE_LR_FACTOR = 0.2
KERAS_MIN_LR = 1e-6
LSTM_UNITS = 128
LSTM_DROPOUT_RATE = 0.25
LSTM_LEARNING_RATE = 5e-5
TRANSFORMER_NUM_BLOCKS = 2
TRANSFORMER_HEAD_SIZE = 128
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_DROPOUT_RATE = 0.35
TRANSFORMER_LEARNING_RATE = 5e-5
TRANSFORMER_WEIGHT_DECAY = 1e-4
BASE_FEATURE_COLS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 'EMA_20',
    'volume', 'garch_vol_forecast', 'OBV',
    'close_roll_mean_5', 'close_roll_std_5',
    'close_roll_mean_10', 'close_roll_std_10',
    'RSI_roll_mean_5', 'RSI_roll_std_5',
    'RSI_roll_mean_10', 'RSI_roll_std_10',
    'close_zscore_20'
] # Các cột feature cơ bản dùng cho model

# --- FORECAST_FUTURE.PY PARAMETERS ---
FORECAST_YEAR_TARGET = 2025 # Năm mục tiêu để dự báo, đảm bảo có đủ dữ liệu huấn luyện trước đó
FORECAST_TRAINING_YEARS = 3 # Số năm dữ liệu dùng để huấn luyện mô hình dự báo
FORECAST_TREND_THRESHOLD = TREND_THRESHOLD # Có thể dùng chung TREND_THRESHOLD
APPROX_TRADING_DAYS_PER_YEAR = 252
# ENRICHED_DATA_FOR_FORECAST đã được định nghĩa ở trên

# --- GENERAL PROJECT SETTINGS ---
RANDOM_SEED = 42

# --- PORTFOLIO OPTIMIZATION PARAMETERS ---
PORTFOLIO_ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'F', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'XOM', 'NFLX']
PORTFOLIO_START_DATE = "2023-01-01" # Ngày bắt đầu backtest portfolio
PORTFOLIO_END_DATE = "2023-12-31" # Ngày kết thúc backtest portfolio, đảm bảo dữ liệu trong PRICE_DATA_FOR_PORTFOLIO_PATH có đến ngày này
REBALANCE_FREQUENCY = 'BM' # Tần suất rebalance: 'BM' (cuối tháng kinh doanh), 'BQS' (đầu quý kinh doanh), etc.

# Các rolling window này có thể dùng cho tính toán covariance, returns trong portfolio
# Nếu trùng với feature engineering, cân nhắc mục đích. Nếu riêng cho portfolio thì giữ.
ROLLING_WINDOW_COVARIANCE_PORTFOLIO = 60 # Số ngày cho rolling covariance matrix
ROLLING_WINDOW_RETURNS_PORTFOLIO = 20 # Số ngày cho rolling returns (ví dụ, cho tính toán alpha/beta)

# Markowitz Parameters
MARKOWITZ_OBJECTIVE = 'max_sharpe' # Mục tiêu tối ưu: 'max_sharpe', 'min_volatility', 'efficient_return'
MARKOWITZ_RISK_FREE_RATE = 0.02
MARKOWITZ_WEIGHT_BOUNDS = (0.01, 0.3) # Giới hạn trọng số cho mỗi tài sản

# Black-Litterman Parameters
BL_TAU = 0.025 # Hệ số không chắc chắn của prior
BL_RISK_AVERSION = 2.5 # Mức độ ngại rủi ro của nhà đầu tư (delta)
BL_PROB_THRESHOLD_STRONG_VIEW = 0.65 # Ngưỡng xác suất từ model để coi là "strong view"
BL_VIEW_CONFIDENCE_STRONG = 0.8 # Độ tự tin cho "strong view" (dùng để tính Omega)
BL_EXPECTED_OUTPERFORMANCE_STRONG = 0.05 # Mức kỳ vọng outperform cho "strong view" (dùng cho vector Q)

# Backtesting Parameters
INITIAL_CAPITAL = 100000
TRANSACTION_COST_BPS = 5 # Phí giao dịch theo basis points (1 bps = 0.01%). 5 bps = 0.05%

# Soft Signal Generation (from classification model, used for BL and RL)
SOFT_SIGNAL_MODEL_NAME = 'XGBoost' # Model dùng để tạo tín hiệu mềm (xác suất)
SOFT_SIGNAL_TRAIN_END_DATE = "2022-12-31" # Ngày kết thúc huấn luyện cho model tạo soft signal (phải trước PORTFOLIO_START_DATE)
SOFT_SIGNAL_PREDICT_START_DATE = PORTFOLIO_START_DATE
SOFT_SIGNAL_PREDICT_END_DATE = PORTFOLIO_END_DATE

# --- Reinforcement Learning Portfolio Optimization ---
RL_STRATEGY_ENABLED = True # Đặt thành True/False để bật/tắt việc chạy và phân tích chiến lược RL
RL_TRAIN_DATA_START_DATE = "2020-01-01" # Ngày bắt đầu dữ liệu huấn luyện cho RL (nên trước SOFT_SIGNAL_TRAIN_END_DATE)
RL_TRAIN_DATA_END_DATE = SOFT_SIGNAL_TRAIN_END_DATE # Thống nhất với soft signal để RL có thể dùng output của nó

RL_LOOKBACK_WINDOW_SIZE = 30 # Số ngày lịch sử giá/features cho mỗi state của RL agent
RL_REBALANCE_FREQUENCY_DAYS = 5 # Số ngày giao dịch giữa các lần rebalance của RL agent
RL_TRANSACTION_COST_BPS = 10 # Phí giao dịch cho RL agent (0.1%)
RL_FINANCIAL_FEATURES = ['ROA', 'ROE', 'EPS', 'P/E Ratio', 'Debt/Equity', 'Dividend Yield'] # Các feature tài chính từ financial_data.csv
RL_PROB_FEATURES = [f'prob_increase_{SOFT_SIGNAL_MODEL_NAME}'] # Feature xác suất từ model classification

#  --- RL Reward Configuration ---
RL_REWARD_USE_LOG_RETURN = True # Sử dụng log return cho reward function
RL_REWARD_TURNOVER_PENALTY_FACTOR = 0.001 # Hệ số phạt cho portfolio turnover (khuyến khích ít giao dịch hơn)

# --- RL Algorithm and Training Configuration ---
# RL_ALGORITHM, RL_MODEL_NAME, RL_MODEL_SAVE_PATH, RL_LOG_DIR_FOR_SB3, RL_TRAINING_LOG_DIR đã định nghĩa ở trên
RL_TOTAL_TIMESTEPS = 50000 # Tổng số bước huấn luyện. Giảm (e.g., 10k-50k) để debug, tăng (e.g., 500k-2M) cho training thực tế.

# --- PPO Hyperparameters (can be adjusted if RL_ALGORITHM is "PPO") ---
RL_PPO_N_STEPS = 2048
RL_PPO_BATCH_SIZE = 64
RL_PPO_N_EPOCHS = 10
RL_PPO_GAMMA = 0.99
RL_PPO_GAE_LAMBDA = 0.95
RL_PPO_CLIP_RANGE = 0.2
RL_PPO_ENT_COEF = 0.0 # Hệ số entropy (khuyến khích exploration)
RL_PPO_VF_COEF = 0.5 # Hệ số cho value function loss
RL_PPO_MAX_GRAD_NORM = 0.5
RL_PPO_LEARNING_RATE = 3e-4 # Tốc độ học, 3e-4 là giá trị phổ biến cho PPO

# Network architecture for PPO. Ví dụ: 2 lớp ẩn, mỗi lớp 64 units.
# Cú pháp có thể thay đổi tùy phiên bản Stable Baselines3.
# Cho SB3 core:
RL_PPO_POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
# Hoặc có thể là: net_arch=[dict(pi=[64, 64], vf=[64, 64])] hoặc chỉ net_arch=[64,64]
# Cho SB3 Contrib (ví dụ RecurrentPPO với LSTM):
# RL_PPO_POLICY_KWARGS = dict(
#     net_arch=dict(pi=[64, 64], vf=[64, 64]),
#     features_extractor_kwargs=dict(features_dim=128), # Optional, if using custom feature extractor
#     # For RecurrentPPO / LSTMPolicy
#     # lstm_hidden_size=128,
#     # n_lstm_layers=1,
#     # shared_lstm=True, # Whether pi and vf share the LSTM
#     # enable_critic_lstm=True # Whether critic also uses LSTM
# )

# --- Plotting Specific Configurations (Loaded by plot_utils.py) ---
# Các config này có thể được plot_utils.py đọc trực tiếp
PLOTS_OUTPUT_DIR_PLOTLY = PLOTS_OUTPUT_DIR # Để plot_utils.py biết lưu vào đâu
FORECASTING_BASELINE_SCORES = { # Điểm baseline cho các metric dự báo, ví dụ:
    # "RMSE": 0.5,
    # "MAE": 0.3,
    # "R2 Score": 0.6
}
MARKET_EVENTS_FOR_PLOTS = [ # Các sự kiện thị trường để hiển thị trên biểu đồ
    # {"name": "COVID-19 Crash", "start_date": "2020-02-20", "end_date": "2020-03-23", "color": "rgba(255,0,0,0.1)", "line_color": "red"},
    # {"name": "Event X", "start_date": "2021-06-15", "line_dash": "dot", "annotation_position": "top right"}
]

# --- Fallback Configs (Cần đảm bảo các module khác có thể import an toàn) ---
# Ví dụ: trong một số file, nếu import configs thất bại, có thể dùng giá trị mặc định.
# Tuy nhiên, mục tiêu là tất cả các module đều import được file config này.
# Nếu bạn có các file sử dụng try-except để import configs, đảm bảo các biến
# fallback của chúng khớp với ý định của file config này.