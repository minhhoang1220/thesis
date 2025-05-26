import pandas as pd
import numpy as np
# Đảm bảo sys.path được cấu hình đúng nếu chạy trực tiếp file này
# Hoặc tốt nhất là chạy bằng `python -m tests.test_indicators` từ thư mục gốc
from marketml.data_handling.loader import preprocess

# --- SỬA Ở ĐÂY: Tăng dữ liệu mẫu ---
# Tạo dữ liệu mẫu lớn hơn để các chỉ báo có thể tính toán
# Tăng lên ví dụ 40 hoặc 50 dòng
N_SAMPLE_ROWS = 50 # <--- THAY ĐỔI Ở ĐÂY
np.random.seed(42)
dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=N_SAMPLE_ROWS, freq='D'))
raw_data = {
    "Date": dates,
    "Close": 100 + np.random.randn(N_SAMPLE_ROWS).cumsum(),
    "Volume": np.random.randint(500, 2000, size=N_SAMPLE_ROWS),
    "Ticker": ["AAPL"] * N_SAMPLE_ROWS,
    "Some Column%": np.random.rand(N_SAMPLE_ROWS) * 10,
    "Another-Col": ["value"] * N_SAMPLE_ROWS,
}
sample_df = pd.DataFrame(raw_data)

def test_standardize_data():
    # ... (Nội dung hàm này giữ nguyên) ...
    df_clean_remove = preprocess.standardize_data(sample_df.copy(), remove_columns=["Some Column%", "Another-Col"])
    df_clean_no_remove = preprocess.standardize_data(sample_df.copy())
    assert "date" in df_clean_remove.columns
    assert "close" in df_clean_remove.columns
    assert "volume" in df_clean_remove.columns
    assert "ticker" in df_clean_remove.columns
    assert "some_columnpercent" not in df_clean_remove.columns
    assert "another_col" not in df_clean_remove.columns
    assert pd.api.types.is_numeric_dtype(df_clean_remove["close"])
    assert df_clean_remove.shape[1] == 4
    assert "date" in df_clean_no_remove.columns
    assert "close" in df_clean_no_remove.columns
    assert "volume" in df_clean_no_remove.columns
    assert "ticker" in df_clean_no_remove.columns
    assert "some_columnpercent" in df_clean_no_remove.columns
    assert "another_col" in df_clean_no_remove.columns
    assert df_clean_no_remove.shape[1] == 6
    # pass # Không cần pass ở cuối hàm

def test_add_technical_indicators():
    df_clean = preprocess.standardize_data(sample_df.copy())
    # Gọi add_technical_indicators với các tham số mặc định của nó
    # (RSI=14, MACD=12,26,9, BB=20, SMA=20, EMA=20)
    df_with_indicators = preprocess.add_technical_indicators(df_clean)

    expected_indicator_columns = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Middle', 'BB_Upper', 'BB_Lower',
        'SMA_20', 'EMA_20',
    ]

    original_cols_expected = ['date', 'close', 'volume', 'ticker', 'some_columnpercent', 'another_col']
    for col in original_cols_expected:
         assert col in df_with_indicators.columns, f"Missing original column: {col}"

    for col in expected_indicator_columns:
        assert col in df_with_indicators.columns, f"Missing indicator column: {col}"
        assert pd.api.types.is_numeric_dtype(df_with_indicators[col]), f"Indicator column '{col}' is not numeric."
        # In ra số lượng NaN để debug nếu cần
        # num_nans = df_with_indicators[col].isna().sum()
        # print(f"Column {col}: {num_nans} NaNs out of {len(df_with_indicators)}")
        assert not df_with_indicators[col].isna().all(), f"Indicator column '{col}' is all NaN. Check window/min_periods vs data size ({N_SAMPLE_ROWS} rows)."

    # print("\n--- Sample DataFrame with Technical Indicators (Last 15 rows) ---")
    # print(df_with_indicators.tail(15).to_string()) # Có thể bỏ comment để xem output

if __name__ == "__main__":
    print("Running test_standardize_data...")
    test_standardize_data()
    print("✅ test_standardize_data passed")

    print("\nRunning test_add_technical_indicators...")
    test_add_technical_indicators()
    print("✅ test_add_technical_indicators passed")