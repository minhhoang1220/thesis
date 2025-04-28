import pandas as pd
from pathlib import Path

def load_price_data(nrows=None) -> pd.DataFrame:
    base_path = Path("/media/sf_shared-data/Khoa_Luan/")

    file_global = base_path / "Data_Global/price/yahoo_price_data_fixed.csv"

    df_global = pd.read_csv(file_global, parse_dates=["Date"], nrows=nrows)

    df_global["Market"] = "Global"

    df = pd.concat([df_global], ignore_index=True)
    return df
