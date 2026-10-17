import pandas as pd, numpy as np

df = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")

print("Checking RT-GENE head vectors vs angles:")
for i in [0, 100, 1000]:
    row = df.iloc[i]
    hx, hy, hz = row["head_vec_x"], row["head_vec_y"], row["head_vec_z"]
    hp_rad, hy_rad = row["head_pitch_rad"], row["head_yaw_rad"]
    hp_deg, hy_deg = row["head_pitch_deg"], row["head_yaw_deg"]
    print(f"\nSample {i}:")
    print(f"  head_pitch: {hp_deg:.2f} deg, head_yaw: {hy_deg:.2f} deg")
    print(f"  head_vec: [{hx:.4f}, {hy:.4f}, {hz:.4f}]")
    cx = -np.cos(hp_rad) * np.sin(hy_rad)
    cy = -np.sin(hp_rad)
    cz = -np.cos(hp_rad) * np.cos(hy_rad)
    print(f"  unif_vec: [{cx:.4f}, {cy:.4f}, {cz:.4f}]")
    print(f"  diff to head_vec: [{hx - cx:.4f}, {hy - cy:.4f}, {hz - cz:.4f}]")
