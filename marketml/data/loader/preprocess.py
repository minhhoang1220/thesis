#Xử lý missing data (NaN), tính các chỉ số,...
import pandas as pd

def standardize_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hóa tên cột: snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('(', '')
        .str.replace(')', '')
        .str.replace('%', 'percent')
        .str.replace('/', '_')
    )

    # Chuẩn hóa tất cả các cột dạng numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
