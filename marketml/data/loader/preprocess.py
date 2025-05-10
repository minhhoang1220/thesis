import pandas as pd
import numpy as np # Cần numpy cho np.select

try:
    from marketml.indicators import ta_indicators
except ModuleNotFoundError:
    print("Error: Could not import 'marketml.indicators.ta_indicators' in preprocess.py.")
    print("Ensure the project structure allows this import.")
    # Nếu không import được, các hàm compute_... sẽ không tồn tại
    # Cần xử lý hoặc để script lỗi ở đây để người dùng biết
    ta_indicators = None # Hoặc raise lỗi


def standardize_data(df: pd.DataFrame, remove_columns: list[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_', regex=False)
                  .str.replace('-', '_', regex=False)
                  .str.replace('(', '', regex=False)
                  .str.replace(')', '', regex=False)
                  .str.replace('%', 'percent', regex=False)
                  .str.replace('/', '_', regex=False))
    standardized_df_columns = df.columns.tolist()
    if remove_columns:
        cols_to_remove_standardized = [
            col.strip().lower()
               .replace(' ', '_').replace('-', '_')
               .replace('(', '').replace(')', '')
               .replace('%', 'percent').replace('/', '_')
            for col in remove_columns
        ]
        cols_to_drop = [col for col in cols_to_remove_standardized if col in standardized_df_columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
    exclude_cols = {"symbol", "year", "ticker", "date"}
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# CẬP NHẬT HÀM NÀY
def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'close',
    indicators_to_add: list[str] = None,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_num_std: int = 2,
    sma_window: int = 20,
    ema_window: int = 20
) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật được chọn vào DataFrame.

    Parameters:
        df (pd.DataFrame): Dữ liệu đã chuẩn hóa, yêu cầu có cột 'price_col'.
        price_col (str): Tên cột giá dùng để tính chỉ báo (mặc định là 'close').
        indicators_to_add (list): Danh sách các chỉ báo cần tính (viết thường).
                                  Nếu None, mặc định tính tất cả: ['rsi', 'macd', 'bollinger', 'sma', 'ema'].
        rsi_window (int): Window cho RSI.
        macd_fast (int): Fast period cho MACD.
        macd_slow (int): Slow period cho MACD.
        macd_signal (int): Signal period cho MACD.
        bb_window (int): Window cho Bollinger Bands.
        bb_num_std (int): Số độ lệch chuẩn cho Bollinger Bands.
        sma_window (int): Window cho SMA.
        ema_window (int): Window cho EMA.

    Returns:
        pd.DataFrame: DataFrame có thêm các cột chỉ báo.
    """
    if ta_indicators is None:
        print("Error: ta_indicators module not loaded. Cannot add technical indicators.")
        return df # Trả về df gốc nếu không load được module chỉ báo

    df_out = df.copy()

    if price_col not in df_out.columns:
        raise ValueError(f"DataFrame must contain the price column '{price_col}' for technical indicators.")
    if not pd.api.types.is_numeric_dtype(df_out[price_col]):
         raise ValueError(f"Price column '{price_col}' must be numeric.")

    if indicators_to_add is None:
        indicators_to_add = ['rsi', 'macd', 'bollinger', 'sma', 'ema'] # Mặc định

    if 'rsi' in indicators_to_add:
        df_out = ta_indicators.compute_rsi(df_out, column=price_col, window=rsi_window)
    if 'macd' in indicators_to_add:
        df_out = ta_indicators.compute_macd(df_out, column=price_col, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if 'bollinger' in indicators_to_add:
        df_out = ta_indicators.compute_bollinger_bands(df_out, column=price_col, window=bb_window, num_std=bb_num_std)
    if 'sma' in indicators_to_add:
        # Sử dụng sma_window cho cả tên cột và tính toán
        # Lưu ý: Hàm compute_sma đã tự tạo tên cột f'SMA_{window}'
        df_out = ta_indicators.compute_sma(df_out, column=price_col, window=sma_window)
    if 'ema' in indicators_to_add:
        # Tương tự cho EMA
        df_out = ta_indicators.compute_ema(df_out, column=price_col, window=ema_window)

    return df_out