#Gọi yfinance.Ticker().financials để lấy báo cáo tài chính (Income Statement, Balance Sheet, Cash Flow Statement) của 30 công ty

import pandas as pd
from pathlib import Path

def load_financial_data(nrows=None) -> pd.DataFrame:
    base_path = Path("/mnt/shared-data/Khoa_Luan/")

    file_global = base_path / "Data_Global" / "financial" / "yahoo_financial_data.csv"
    file_vn = base_path / "Data_VN" / "financial" / "financial_data_vn.csv"

    df_global = pd.read_csv(file_global, nrows=nrows)
    df_vn = pd.read_csv(file_vn, nrows=nrows)

    df_global["Market"] = "Global"
    df_vn["Market"] = "Vietnam"

    df = pd.concat([df_global, df_vn], ignore_index=True)
    return df
