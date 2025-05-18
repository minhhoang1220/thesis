# marketml/configs/configs.py
from pathlib import Path
import pandas as pd # Cần cho pd.Timedelta

# --- PATHS (Giữ nguyên đường dẫn tuyệt đối của bạn) ---
# Ví dụ: bạn có thể giữ nguyên cách bạn xác định base_path ở đây
# hoặc để chúng trong các module loader và các script chạy chính.
# Để đơn giản, tôi sẽ giả định các script chạy chính vẫn tự quản lý base_path.
# Tuy nhiên, các đường dẫn output có thể đặt ở đây.
PROJECT_ROOT = Path(__file__).resolve().parents[2] # .ndmh/
RESULTS_DIR = PROJECT_ROOT / "results" # Lưu kết quả vào .ndmh/results/
FORECASTS_DIR = PROJECT_ROOT / "forecasts" # Lưu dự báo vào .ndmh/forecasts/
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

# --- CROSS-VALIDATION PARAMETERS for run_experiment.py ---
INITIAL_TRAIN_YEARS = 1
TEST_YEARS = 1
STEP_YEARS = 1
USE_EXPANDING_WINDOW = True
INITIAL_TRAIN_TIMEDELTA = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS)
TEST_TIMEDELTA = pd.Timedelta(days=365 * TEST_YEARS)
STEP_TIMEDELTA = pd.Timedelta(days=365 * STEP_YEARS)

# --- MODEL TRAINING & TUNING PARAMETERS for run_experiment.py ---
# General
TARGET_COL_PCT = 'target_pct_change'
TARGET_COL_TREND = 'target_trend'

# SKLearn-based models (RF, XGB, SVM)
N_ITER_SEARCH_SKLEARN = 30 # Số lần thử cho RandomizedSearchCV
CV_FOLDS_TUNING_SKLEARN = 3 # Số fold CV cho tuning nội bộ của RandomizedSearchCV

# Keras-based models (LSTM, Transformer)
N_TIMESTEPS_SEQUENCE = 10 # Số bước thời gian cho chuỗi
KERAS_EPOCHS = 50
KERAS_BATCH_SIZE = 64
KERAS_VALIDATION_SPLIT = 0.1 # Tỷ lệ dữ liệu dùng cho validation trong quá trình train Keras
KERAS_EARLY_STOPPING_PATIENCE = 10
KERAS_REDUCE_LR_PATIENCE = 5
KERAS_REDUCE_LR_FACTOR = 0.2
KERAS_MIN_LR = 1e-6

# LSTM Specific
LSTM_UNITS = 128
LSTM_DROPOUT_RATE = 0.25
LSTM_LEARNING_RATE = 5e-5

# Transformer Specific
TRANSFORMER_NUM_BLOCKS = 2
TRANSFORMER_HEAD_SIZE = 128
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_DROPOUT_RATE = 0.35 # KERAS_DROPOUT_RATE + 0.1
TRANSFORMER_LEARNING_RATE = 5e-5
TRANSFORMER_WEIGHT_DECAY = 1e-4

# --- FEATURE COLUMNS for run_experiment.py ---
# (Lưu ý: các cột lag sẽ được thêm tự động dựa trên LAG_PERIODS)
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
# (Lưu ý: file enriched cho forecast_future nên là file được tạo bởi create_enriched_data.py)
FORECAST_YEAR_TARGET = 2025
FORECAST_TRAINING_YEARS = 3 # Số năm dữ liệu lịch sử cuối cùng để huấn luyện ARIMA
FORECAST_TREND_THRESHOLD = 0.002 # Ngưỡng trend cho dự báo ARIMA
APPROX_TRADING_DAYS_PER_YEAR = 252
ENRICHED_DATA_FOR_FORECAST = PROJECT_ROOT / "marketml" / "data" / "processed" / "price_data_with_indicators.csv" # *Đảm bảo file này được tạo bởi create_enriched_data.py*