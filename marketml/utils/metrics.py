# marketml/pipeline/build_features.py
import pandas as pd
from marketml.indicators.ta_indicators import add_all_indicators

def generate_features(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['date'])

    df = df.groupby('symbol').apply(lambda x: add_all_indicators(x.sort_values('date'))).reset_index(drop=True)

    df.dropna(inplace=True)  # Drop các dòng đầu bị NaN do rolling
    df.to_csv(output_path, index=False)

    print(f"✅ Saved to {output_path}")
