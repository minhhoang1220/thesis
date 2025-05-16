#.ndmh/marketml/data/loader/finance_loader.py
import pandas as pd
from pathlib import Path

def load_financial_data(nrows=None) -> pd.DataFrame:
    base_path = Path("/media/sf_shared-data/Khoa_Luan/")
    file_global = base_path / "Data_Global" / "financial" / "financial_data.csv"

    df_global = pd.read_csv(file_global, nrows=nrows)

    df = pd.concat([df_global], ignore_index=True)
    return df

