import pandas as pd
from marketml.data.loader.preprocess import standardize_financial_data
from marketml.data.loader.finance_loader import load_financial_data

def test_standardize_financial_data():
    # Load dữ liệu thực
    df = load_financial_data()

    # Gọi chuẩn hóa
    df_standardized = standardize_financial_data(df.copy())

    # Kiểm tra tên cột đã chuẩn hóa
    for col in df_standardized.columns:
        assert ' ' not in col, f"Column {col} still contains spaces"
        assert col == col.lower(), f"Column {col} is not lowercase"

    # Kiểm tra kiểu dữ liệu có thể ép số
    for col in df_standardized.columns:
        if df_standardized[col].dtype == 'object':
            try:
                pd.to_numeric(df_standardized[col].dropna())
            except ValueError:
                assert False, f"Column {col} cannot be converted to numeric"

    # In kết quả ra bảng cho kiểm tra thủ công
    print("\n--- Financial Data (After Standardization) ---")
    print(df_standardized.head(10))
