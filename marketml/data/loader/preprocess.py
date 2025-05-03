import pandas as pd
from marketml.indicators.ta_indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_sma,
    compute_ema,
)

def standardize_data(df: pd.DataFrame, remove_columns: list[str] = None) -> pd.DataFrame:
    df = df.copy()
    # Bước 1: Chuẩn hóa tên cột trong DataFrame (dùng .str.replace của Pandas)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_', regex=False) # Giữ nguyên regex=False ở đây
        .str.replace('(', '', regex=False)  # Giữ nguyên regex=False ở đây
        .str.replace(')', '', regex=False)  # Giữ nguyên regex=False ở đây
        .str.replace('%', 'percent', regex=False) # Giữ nguyên regex=False ở đây
        .str.replace('/', '_', regex=False)  # Giữ nguyên regex=False ở đây
    )
    standardized_df_columns = df.columns.tolist()

    # Bước 2: Xử lý việc xóa cột
    if remove_columns:
        # Chuẩn hóa các tên trong danh sách remove_columns (dùng .replace của string)
        cols_to_remove_standardized = [
            col.strip()
               .lower()
               .replace(' ', '_')
               .replace('-', '_') # <--- Xóa regex=False
               .replace('(', '')  # <--- Xóa regex=False
               .replace(')', '')  # <--- Xóa regex=False
               .replace('%', 'percent') # <--- Xóa regex=False
               .replace('/', '_')  # <--- Xóa regex=False
            for col in remove_columns
        ]

        # Xác định những cột cần xóa thực sự tồn tại trong DataFrame
        cols_to_drop = [col for col in cols_to_remove_standardized if col in standardized_df_columns]

        # Xóa các cột đã xác định
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

    # Bước 3: Chuyển đổi kiểu dữ liệu sang số
    exclude_cols = {"symbol", "year", "ticker", "date"}
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_technical_indicators(df: pd.DataFrame, indicators: list[str] = None) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame giá.

    Parameters:
        df (pd.DataFrame): Dữ liệu đã chuẩn hóa, yêu cầu có cột 'close'.
        indicators (list): Danh sách chỉ báo cần tính. Nếu None, mặc định tính tất cả.

    Returns:
        pd.DataFrame: Dữ liệu có thêm các cột chỉ báo.
    """
    if indicators is None:
        indicators = ['rsi', 'macd', 'bollinger', 'sma', 'ema']

    df = df.copy()

    if 'rsi' in indicators:
        df = compute_rsi(df)
    if 'macd' in indicators:
        df = compute_macd(df)
    if 'bollinger' in indicators:
        df = compute_bollinger_bands(df)
    if 'sma' in indicators:
        df = compute_sma(df)
    if 'ema' in indicators:
        df = compute_ema(df)

    return df
