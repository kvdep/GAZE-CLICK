import pandas as pd, numpy as np

df = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")
row = df.iloc[0]
for k, v in row.items():
    print(f"{k}: {v}")
