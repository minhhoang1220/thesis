import pandas as pd
from datetime import timedelta 
# Import các module cần thiết
from marketml.data.loader import price_loader
from marketml.data.loader import preprocess

# ==============================================================================
# HÀM TẠO CỬA SỔ ROLLING/EXPANDING DỰA TRÊN NGÀY THÁNG
# ==============================================================================
def create_time_series_cv_splits(
    df: pd.DataFrame,
    date_col: str,
    ticker_col: str,
    initial_train_period: pd.Timedelta,
    test_period: pd.Timedelta,
    step_period: pd.Timedelta,
    expanding: bool = False  # True cho expanding window, False cho rolling/sliding window
):
    """
    Tạo ra các cặp (train_df, test_df) cho cross-validation time series.
    Hoạt động trên từng ticker riêng lẻ và xử lý ngày tháng.
    """
    print(f"\nGenerating CV splits ({'Expanding' if expanding else 'Rolling'} Window):")
    print(f"  Initial Train: {initial_train_period}")
    print(f"  Test Period:   {test_period}")
    print(f"  Step Period:   {step_period}")

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    if min_date + initial_train_period + test_period > max_date:
        print("Warning: Not enough data for even one train-test split based on the provided periods.")
        return  # Không thể tạo split nào

    current_train_start_date = min_date
    current_train_end_date = min_date + initial_train_period  # Kết thúc train vào *cuối* ngày này

    split_index = 0
    while True:
        current_test_start_date = current_train_end_date  # Bắt đầu test *sau* ngày train cuối cùng
        current_test_end_date = current_test_start_date + test_period

        # Dừng nếu cửa sổ test vượt quá ngày cuối cùng có dữ liệu
        if current_test_end_date > max_date:
            print(f"Stopping: Test end date {current_test_end_date.date()} exceeds max data date {max_date.date()}.")
            break

        # Lọc dữ liệu cho train và test dựa trên khoảng thời gian hiện tại
        train_mask = (df[date_col] >= current_train_start_date) & (df[date_col] < current_train_end_date)
        test_mask = (df[date_col] >= current_test_start_date) & (df[date_col] < current_test_end_date)

        train_split_df = df.loc[train_mask].copy()
        test_split_df = df.loc[test_mask].copy()

        if train_split_df.empty or test_split_df.empty:
            print(f"Warning: Empty train or test split generated for period "
                  f"Train=[{current_train_start_date.date()}:{current_train_end_date.date()}), "
                  f"Test=[{current_test_start_date.date()}:{current_test_end_date.date()}). Skipping.")
        else:
            print(f"  Split {split_index}: "
                  f"Train=[{train_split_df[date_col].min().date()}:{train_split_df[date_col].max().date()}] ({len(train_split_df)} rows), "
                  f"Test=[{test_split_df[date_col].min().date()}:{test_split_df[date_col].max().date()}] ({len(test_split_df)} rows)")
            yield split_index, train_split_df, test_split_df
            split_index += 1

        if not expanding:  # Rolling window: cả start và end của train đều dịch chuyển
            current_train_start_date += step_period
        current_train_end_date += step_period

    print(f"Total splits generated: {split_index}")


# ==============================================================================
# SCRIPT CHÍNH (run_experiment.py)
# ==============================================================================

# --- Bước 1: Load dữ liệu thật ---
try:
    print("Loading real data...")
    raw_real_df = price_loader.load_price_data()
    print("Real data loaded successfully. Info:")
    raw_real_df.info()
    print("\nRaw data head:\n", raw_real_df.head())
    print("\nRaw data tail:\n", raw_real_df.tail())
except FileNotFoundError:
    print(f"Error: Data file not found. Check the path specified inside price_loader.py.")
    exit()
except Exception as e:
    print(f"Error loading data using price_loader: {e}")
    exit()

# --- Bước 2: Chuẩn hóa dữ liệu ---
print("\nStandardizing data...")
standardized_df = preprocess.standardize_data(raw_real_df)
print("Data standardized. Info:")
standardized_df.info()
print("\nStandardized data head:\n", standardized_df.head())

required_cols = ['date', 'ticker', 'close']
missing_cols = [col for col in required_cols if col not in standardized_df.columns]
if missing_cols:
    print(f"\nError: Missing required columns after standardization: {missing_cols}")
    exit()

if not pd.api.types.is_datetime64_any_dtype(standardized_df['date']):
    print("\nError: 'date' column is not datetime type after standardization.")
    exit()
if not pd.api.types.is_numeric_dtype(standardized_df['close']):
    print("\nError: 'close' column is not numeric after standardization.")
    exit()

# --- Bước 3: Thêm các chỉ báo kỹ thuật ---
print("\nAdding technical indicators (grouped by ticker)...")
try:
    all_indicators_df_list = []
    processed_tickers = 0
    for ticker, group_df in standardized_df.groupby('ticker'):
        group_df_sorted = group_df.sort_values('date').copy()
        group_with_indicators = preprocess.add_technical_indicators(group_df_sorted)
        all_indicators_df_list.append(group_with_indicators)
        processed_tickers += 1

    if not all_indicators_df_list:
        print("\nWarning: No ticker groups found or processed. Resulting DataFrame will be empty.")
        df_with_indicators = pd.DataFrame()
    else:
        df_with_indicators = pd.concat(all_indicators_df_list, ignore_index=True)
        print(f"\nTechnical indicators added successfully for {processed_tickers} tickers.")
except KeyError as e:
    print(f"\nError: Column not found during indicator calculation - {e}.")
    exit()
except Exception as e:
    print(f"\nError adding technical indicators: {e}")
    exit()

# --- Bước 4: Chia Train-Test với Rolling Window ---
if not df_with_indicators.empty:
    print("\nProceeding to Train-Test Split...")

    INITIAL_TRAIN_YEARS = 3
    TEST_YEARS = 1
    STEP_YEARS = 1
    USE_EXPANDING_WINDOW = False

    initial_train_td = pd.Timedelta(days=365 * INITIAL_TRAIN_YEARS)
    test_td = pd.Timedelta(days=365 * TEST_YEARS)
    step_td = pd.Timedelta(days=365 * STEP_YEARS)

    cv_splitter = create_time_series_cv_splits(
        df=df_with_indicators,
        date_col='date',
        ticker_col='ticker',
        initial_train_period=initial_train_td,
        test_period=test_td,
        step_period=step_td,
        expanding=USE_EXPANDING_WINDOW
    )

    for split_idx, train_df, test_df in cv_splitter:
        print(f"\n===== Processing CV Split {split_idx} =====")
        print(f"Train period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
        print(f"Test period:  {test_df['date'].min().date()} to {test_df['date'].max().date()}")
        print(f"===== Finished CV Split {split_idx} =====")
else:
    print("\nNo data with indicators available to perform train-test split.")

print("\nScript finished.")