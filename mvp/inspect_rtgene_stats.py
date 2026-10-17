import pandas as pd
df = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")
for p in ["p002", "p007", "p008", "p009", "p005", "p012", "p014"]:
    sub = df[df["person"] == p]
    sess = sub["session"].unique().tolist()
    hp_m = sub["head_pitch_deg"].mean()
    hy_m = sub["head_yaw_deg"].mean()
    gp_m = sub["gaze_pitch_deg"].mean()
    gy_m = sub["gaze_yaw_deg"].mean()
    print(f"Subject {p}: count={len(sub)}, sessions={sess}")
    print(f"  Head Pitch mean={hp_m:.1f}, Head Yaw mean={hy_m:.1f}, Gaze Pitch mean={gp_m:.1f}, Gaze Yaw mean={gy_m:.1f}")
