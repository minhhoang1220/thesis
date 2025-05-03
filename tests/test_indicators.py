import pandas as pd
import numpy as np # Cần numpy để tạo dữ liệu mẫu lớn hơn
from marketml.data.loader import preprocess # Giữ nguyên import này, nhưng đảm bảo chạy test đúng cách

# --- SỬA Ở ĐÂY: Tăng dữ liệu mẫu ---
# Tạo dữ liệu mẫu lớn hơn (ví dụ 30 dòng) để các chỉ báo có thể tính toán
np.random.seed(42) # Để kết quả có thể tái lập
dates = pd.to_datetime(pd.date_range(start="2024-01-01", periods=30, freq='D'))
raw_data = {
    "Date": dates,
    "Close": 100 + np.random.randn(30).cumsum(), # Giá đóng cửa biến động ngẫu nhiên
    "Volume": np.random.randint(500, 2000, size=30),
    "Ticker": ["AAPL"] * 30,
    "Some Column%": np.random.rand(30) * 10,
    "Another-Col": ["value"] * 30, # Thêm cột để kiểm tra chuẩn hóa tên phức tạp hơn
}

sample_df = pd.DataFrame(raw_data)
# print("Sample DF Head:\n", sample_df.head()) # Debug xem dữ liệu mẫu

def test_standardize_data():
    # Test cả việc xóa cột và không xóa
    df_clean_remove = preprocess.standardize_data(sample_df.copy(), remove_columns=["Some Column%", "Another-Col"])
    df_clean_no_remove = preprocess.standardize_data(sample_df.copy())

    # Kiểm tra df_clean_remove
    assert "date" in df_clean_remove.columns
    assert "close" in df_clean_remove.columns
    assert "volume" in df_clean_remove.columns
    assert "ticker" in df_clean_remove.columns
    assert "some_columnpercent" not in df_clean_remove.columns # Đã bị xóa
    assert "another_col" not in df_clean_remove.columns # Đã bị xóa
    assert pd.api.types.is_numeric_dtype(df_clean_remove["close"])
    # --- SỬA Ở ĐÂY: Assertion shape ---
    # Dữ liệu gốc 6 cột, xóa 2 cột còn 4
    assert df_clean_remove.shape[1] == 4

    # Kiểm tra df_clean_no_remove
    assert "date" in df_clean_no_remove.columns
    assert "close" in df_clean_no_remove.columns
    assert "volume" in df_clean_no_remove.columns
    assert "ticker" in df_clean_no_remove.columns
    assert "some_columnpercent" in df_clean_no_remove.columns # Không bị xóa
    assert "another_col" in df_clean_no_remove.columns # Không bị xóa, đã chuẩn hóa tên
    assert df_clean_no_remove.shape[1] == 6 # Dữ liệu gốc 6 cột, không xóa cột nào
    pass

def test_add_technical_indicators():
    # Chuẩn hóa trước khi thêm chỉ báo
    # Không xóa cột nào để đảm bảo chúng vẫn còn sau khi thêm chỉ báo
    df_clean = preprocess.standardize_data(sample_df.copy())
    # print("\nClean DF before indicators:\n", df_clean.head()) # Debug

    # Thêm tất cả các chỉ báo mặc định
    df_with_indicators = preprocess.add_technical_indicators(df_clean) # Sử dụng mặc định
    # print("\nDF with indicators:\n", df_with_indicators.tail()) # Debug xem các giá trị cuối

    # --- SỬA Ở ĐÂY: Cập nhật tên cột mong đợi ---
    # Sử dụng tên chính xác từ ta_indicators.py
    expected_indicator_columns = [
        'RSI',          # Từ compute_rsi
        'MACD',         # Từ compute_macd
        'MACD_Signal',  # Từ compute_macd
        'MACD_Hist',    # Từ compute_macd
        'BB_Middle',    # Từ compute_bollinger_bands (window=20 mặc định trong preprocess)
        'BB_Upper',     # Từ compute_bollinger_bands
        'BB_Lower',     # Từ compute_bollinger_bands
        'SMA_20',       # Từ compute_sma (window=20 mặc định trong preprocess)
        'EMA_20',       # Từ compute_ema (window=20 mặc định trong preprocess)
    ]

    # Kiểm tra xem các cột gốc vẫn còn đó không
    original_cols_expected = ['date', 'close', 'volume', 'ticker', 'some_columnpercent', 'another_col']
    for col in original_cols_expected:
         assert col in df_with_indicators.columns, f"Missing original column after adding indicators: {col}"

    # Kiểm tra các cột chỉ báo mới
    for col in expected_indicator_columns:
        assert col in df_with_indicators.columns, f"Missing indicator column: {col}"
        # Kiểm tra xem cột có phải là số không (quan trọng)
        assert pd.api.types.is_numeric_dtype(df_with_indicators[col]), f"Indicator column '{col}' is not numeric."
        # Kiểm tra xem có ít nhất một giá trị không phải NaN không (do window)
        # Với 30 dòng dữ liệu, các chỉ báo window 20, 14, 26+9 sẽ có giá trị không NaN ở cuối
        assert not df_with_indicators[col].isna().all(), f"Indicator column '{col}' is all NaN. Check window size vs data size."
    
    print("\n--- Sample DataFrame with Technical Indicators (Last 15 rows) ---")
    print(df_with_indicators.tail(15).to_string())

if __name__ == "__main__":
    print("Running test_standardize_data...")
    test_standardize_data()
    print("✅ test_standardize_data passed")

    print("\nRunning test_add_technical_indicators...")
    test_add_technical_indicators() # Lệnh print sẽ được thực thi bên trong hàm này
    print("✅ test_add_technical_indicators passed")