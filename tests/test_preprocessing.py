import pandas as pd
from marketml.data.loader.preprocess import standardize_data
from marketml.data.loader.finance_loader import load_financial_data
from marketml.data.loader.price_loader import load_price_data

def test_standardize_financial_data():
    df = load_financial_data()
    df_standardized = standardize_data(df)

    expected_numeric_cols = [
        'roa', 'roe', 'eps', 'p_e_ratio', 'debt_equity', 'dividend_yield',
        'revenue', 'net_income', 'total_assets', 'total_stockholder_equity',
        'total_liabilities', 'cash_dividends_paid', 'year_end_price'
    ]

    for col in expected_numeric_cols:
        assert col in df_standardized.columns, f"Column {col} is missing"
        try:
            pd.to_numeric(df_standardized[col].dropna())
        except Exception:
            assert False, f"Column {col} cannot be converted to numeric"

    # Cột tickers tồn tại, là dạng object
    assert "ticker" in df_standardized.columns, "'ticker' column is missing"
    assert df_standardized["ticker"].dtype == object, "'ticker' column should be string"

    # share_outstanding bị loại bỏ
    assert "share_outstanding" not in df_standardized.columns, "'share_outstanding' should be removed"


    # In kết quả ra bảng cho kiểm tra thủ công
    print("\n--- Financial Data (After Standardization) ---")
    print(df_standardized.head(10))

    print("\n--- Data Types After Standardization ---")
    print(df.dtypes)
    print(df.describe(include='all'))
    

def test_standardize_price_data():
    df = load_price_data()
    df_standardized = standardize_data(df)

    expected_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

    for col in expected_cols:
        assert col in df_standardized.columns, f"Column {col} is missing"

    # Kiểm tra kiểu dữ liệu các cột số
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        try:
            pd.to_numeric(df_standardized[col].dropna())
        except Exception:
            assert False, f"Column {col} cannot be converted to numeric"

    # Kiểm tra kiểu ticker
    assert "ticker" in df_standardized.columns, "'ticker' column is missing"
    assert df_standardized["ticker"].dtype == object, "'ticker' column should be string"

    # In kết quả cho kiểm tra thủ công
    print("\n--- Price Data (After Standardization) ---")
    print(df_standardized.head(10))

    print("\n--- Data Types After Standardization ---")
    print(df.dtypes)
    print(df.describe(include='all'))
    

if __name__ == "__main__":
    test_standardize_financial_data()
    test_standardize_price_data()


