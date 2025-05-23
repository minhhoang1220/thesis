# marketml/configs/configs.py
from pathlib import Path
import pandas as pd # Cần cho pd.Timedelta

# --- PATHS (Giữ nguyên đường dẫn tuyệt đối của bạn) ---
# Ví dụ: bạn có thể giữ nguyên cách bạn xác định base_path ở đây
# hoặc để chúng trong các module loader và các script chạy chính.
# Để đơn giản, tôi sẽ giả định các script chạy chính vẫn tự quản lý base_path.
# Tuy nhiên, các đường dẫn output có thể đặt ở đây.
PROJECT_ROOT = Path(__file__).resolve().parents[2] # .ndmh/
RESULTS_DIR = PROJECT_ROOT / "marketml" / "results" # Lưu kết quả vào .ndmh/marketml/results/
FORECASTS_DIR = PROJECT_ROOT / "marketml" / "forecasts" # Lưu dự báo vào .ndmh/marketml/forecasts/
ENRICHED_DATA_DIR = PROJECT_ROOT / "marketml" / "data" / "processed"
ENRICHED_DATA_FILE = ENRICHED_DATA_DIR / "price_data_enriched_v2.csv" # File enriched data chính

# --- DATA PARAMETERS ---
TIME_RANGE_START = "2020-01-01" # Có thể dùng để lọc dữ liệu nếu cần
TIME_RANGE_END = "2024-12-31"

# --- FEATURE ENGINEERING PARAMETERS ---
# (Các tham số cho add_technical_indicators, GARCH đã có trong create_enriched_data.py,
# nhưng có thể chuyển một phần ra đây nếu muốn linh hoạt hơn từ bên ngoài)
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
TREND_THRESHOLD = 0.002
INITIAL_TRAIN_YEARS = 1
TEST_YEARS = 1
STEP_YEARS = 1
USE_EXPANDING_WINDOW = True
INITIAL_TRAIN_TIMEDELTA = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS)
TEST_TIMEDELTA = pd.Timedelta(days=365 * TEST_YEARS)
STEP_TIMEDELTA = pd.Timedelta(days=365 * STEP_YEARS)
TARGET_COL_PCT = 'target_pct_change'
TARGET_COL_TREND = 'target_trend'
N_ITER_SEARCH_SKLEARN = 30
CV_FOLDS_TUNING_SKLEARN = 3
N_TIMESTEPS_SEQUENCE = 10
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
]

# --- FORECAST_FUTURE.PY PARAMETERS ---
FORECAST_YEAR_TARGET = 2025
FORECAST_TRAINING_YEARS = 3
FORECAST_TREND_THRESHOLD = 0.002
APPROX_TRADING_DAYS_PER_YEAR = 252
ENRICHED_DATA_FOR_FORECAST = ENRICHED_DATA_FILE # Dùng chung file enriched chính

# --- GENERAL PROJECT SETTINGS ---
RANDOM_SEED = 42

# --- PORTFOLIO OPTIMIZATION PARAMETERS ---
PORTFOLIO_ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'F', 'JNJ', 'JPM', 'V', 'PG', 'UNH', 'XOM', 'NFLX']
# PORTFOLIO_ASSETS = None # Để None nếu muốn dùng tất cả tickers có trong dữ liệu sau khi lọc theo ngày

PORTFOLIO_START_DATE = "2023-01-01"
PORTFOLIO_END_DATE = "2023-12-31" # Dùng dữ liệu 2023 để backtest
REBALANCE_FREQUENCY = 'BM' # 'BM': Business Month End, 'BQS': Business Quarter Start, 'W-FRI': Weekly Friday, or integer (days)

FINANCIAL_DATA_FILE_PATH = PROJECT_ROOT / "financial_data.csv" # Đảm bảo file này ở thư mục gốc .ndmh/
PRICE_DATA_FOR_PORTFOLIO_PATH = ENRICHED_DATA_FILE # Dùng file enriched cho dữ liệu giá
# File chứa xác suất sẽ được tạo ra
CLASSIFICATION_PROBS_FILE_PATH = RESULTS_DIR / "classification_probabilities.csv" # Sẽ lưu vào .ndmh/results/

ROLLING_WINDOW_COVARIANCE = 60 # ~3 tháng giao dịch
ROLLING_WINDOW_RETURNS = 20  # ~1 tháng giao dịch

# Markowitz Parameters
MARKOWITZ_OBJECTIVE = 'max_sharpe'
MARKOWITZ_RISK_FREE_RATE = 0.02 # Giả định lãi suất phi rủi ro 2%/năm
MARKOWITZ_WEIGHT_BOUNDS = (0.01, 0.3) # Min 1%, Max 30% cho mỗi tài sản

# Black-Litterman Parameters
BL_TAU = 0.025
BL_RISK_AVERSION = 2.5 # Delta (nếu tính market implied returns, hoặc dùng trong utility function của portfolio)
BL_PROB_THRESHOLD_STRONG_VIEW = 0.65
BL_VIEW_CONFIDENCE_STRONG = 0.8 # Độ tin cậy (1 - variance scale) cho view mạnh
BL_EXPECTED_OUTPERFORMANCE_STRONG = 0.05 # Kỳ vọng 5% outperform hàng năm cho view mạnh

# Backtesting Parameters
INITIAL_CAPITAL = 100000
TRANSACTION_COST_BPS = 5 # 5 bps = 0.05%
# BENCHMARK_TICKER = '^GSPC' # Cần lấy dữ liệu giá cho S&P 500 nếu muốn so sánh
# BENCHMARK_WEIGHTS = None # Hoặc danh mục 1/N

# Soft Signal Generation (từ classification model)
SOFT_SIGNAL_MODEL_NAME = 'XGBoost' # Model tốt nhất của bạn
SOFT_SIGNAL_TRAIN_END_DATE = "2022-12-31" # Huấn luyện model đến hết năm 2022
SOFT_SIGNAL_PREDICT_START_DATE = PORTFOLIO_START_DATE # Dự đoán cho năm 2023
SOFT_SIGNAL_PREDICT_END_DATE = PORTFOLIO_END_DATE

# --- Reinforcement Learning Portfolio Optimization ---
RL_STRATEGY_ENABLED = True # Đặt True để chạy chiến lược RL
RL_TRAIN_DATA_START_DATE = "2020-01-01" # Ví dụ: Sử dụng dữ liệu sớm hơn để huấn luyện
RL_TRAIN_DATA_END_DATE = "2022-12-31"   # Ví dụ: Huấn luyện đến trước giai đoạn backtest danh mục

RL_LOOKBACK_WINDOW_SIZE = 30 # Số ngày lịch sử giá cho trạng thái
RL_REBALANCE_FREQUENCY_DAYS = 5 # ví dụ: tái cân bằng hàng tuần (tác nhân RL thực hiện một bước mỗi 5 ngày giao dịch)
RL_TRANSACTION_COST_BPS = 10 # Điểm cơ bản cho môi trường RL
RL_FINANCIAL_FEATURES = ['ROA', 'ROE', 'EPS', 'P/E Ratio', 'Debt/Equity', 'Dividend Yield']
RL_PROB_FEATURES = [f'prob_increase_{SOFT_SIGNAL_MODEL_NAME}']

#  --- Cấu hình Phần thưởng RL ---
RL_REWARD_USE_LOG_RETURN = True # True để dùng log return, False để dùng simple return
RL_REWARD_TURNOVER_PENALTY_FACTOR = 0.001 # Hình phạt cho turnover. Đặt 0 để tắt.
                                        # Giá trị này cần tinh chỉnh, có thể là 0.0001, 0.0005, 0.001, 0.005

# --- Cấu hình Thuật toán và Huấn luyện RL ---
RL_ALGORITHM = "PPO"
RL_TOTAL_TIMESTEPS = 500000 # Tăng số bước huấn luyện
RL_MODEL_DIR = RESULTS_DIR / "rl_models"
RL_MODEL_NAME = f"{RL_ALGORITHM.lower()}_portfolio_agent.zip"
RL_MODEL_SAVE_PATH = RL_MODEL_DIR / RL_MODEL_NAME
RL_LOG_DIR = RL_MODEL_DIR / "logs"

# --- Siêu tham số PPO (có thể điều chỉnh) ---
RL_PPO_N_STEPS = 2048          # Mặc định của SB3 PPO. Số bước thu thập trước mỗi lần cập nhật.
                                # Nếu ep_len_mean nhỏ, có thể giảm (ví dụ 512, 1024)
RL_PPO_BATCH_SIZE = 64         # Mặc định 64. Thử 128, 256.
RL_PPO_N_EPOCHS = 10           # Mặc định 10.
RL_PPO_GAMMA = 0.99            # Hệ số chiết khấu. Thử 0.95, 0.98.
RL_PPO_GAE_LAMBDA = 0.95       # Factor for GAE. Thử 0.9, 0.98.
RL_PPO_CLIP_RANGE = 0.2        # Mặc định 0.2. Thử 0.1, 0.3.
RL_PPO_ENT_COEF = 0.0          # Hệ số Entropy. Mặc định 0.0. Thử 0.01, 0.005 để tăng khám phá.
RL_PPO_VF_COEF = 0.5           # Hệ số Value Function. Mặc định 0.5.
RL_PPO_MAX_GRAD_NORM = 0.5     # Mặc định 0.5.
RL_PPO_LEARNING_RATE = 0.0003  # Tốc độ học. Thử 1e-4, 5e-4.

# Kiến trúc mạng cho PPO (ví dụ: 2 lớp ẩn, mỗi lớp 64 units)
RL_PPO_POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
# Thử nghiệm: net_arch=dict(pi=[128, 128], vf=[128, 128])
# Hoặc: net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])