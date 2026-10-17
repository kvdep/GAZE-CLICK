import pandas as pd
df = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")
print("Columns:", list(df.columns))
print("First row:")
print(df.iloc[0])
