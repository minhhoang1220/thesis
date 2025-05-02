import pandas as pd

def standardize_data(df: pd.DataFrame, remove_columns: list[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('-', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', 'percent', regex=False)
        .str.replace('/', '_', regex=False)
    )

    if remove_columns:
        for col in remove_columns:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    exclude_cols = {"symbol", "year", "ticker","date"}
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df