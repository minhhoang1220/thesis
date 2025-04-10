import pandas as pd
from pathlib import Path

def load_price_data(nrows=None) -> pd.DataFrame:
    base_path = Path("/mnt/shared-data/Khoa_Luan/")

    file_global = base_path / "Data_Global/price/yahoo_price_data_fixed.csv"
    file_vn = base_path / "Data_VN/price/yahoo_price_data_fixed.csv"

    df_global = pd.read_csv(file_global, parse_dates=["Date"], nrows=nrows)
    df_vn = pd.read_csv(file_vn, parse_dates=["Date"], nrows=nrows)

    df_global["Market"] = "Global"
    df_vn["Market"] = "Vietnam"

    df = pd.concat([df_global, df_vn], ignore_index=True)
    return df
