import pandas as pd, numpy as np

df = pd.read_csv("mvp/benchmarks/benchmark_test_20261014_115554.csv")
print("=== ANALYSIS OF NEW BENCHMARK: benchmark_test_20261014_115554 ===")
for phase in df["phase"].unique():
    pdf = df[df["phase"] == phase]
    gt_x = pdf["target_gt_x"].mean()
    gt_y = pdf["target_gt_y"].mean()
    yaw = pdf["raw_yaw_deg"].mean()
    pitch = pdf["raw_pitch_deg"].mean()
    hyaw = pdf["head_yaw_deg"].mean()
    hpitch = pdf["head_pitch_deg"].mean()
    pred_x = pdf["pred_filt_x"].mean()
    pred_y = pdf["pred_filt_y"].mean()
    raw_x = pdf["pred_raw_x"].mean()
    raw_y = pdf["pred_raw_y"].mean()
    print(f"Phase {phase:15s} (N={len(pdf):2d}): GT=({gt_x:4.0f}, {gt_y:4.0f}) | RawGaze=({yaw:+6.2f}°, {pitch:+6.2f}°) | Head=({hyaw:+6.2f}°, {hpitch:+6.2f}°) | PredRaw=({raw_x:4.0f}, {raw_y:4.0f}) | PredFilt=({pred_x:4.0f}, {pred_y:4.0f})")

left_df = df[df["target_gt_x"] < 1000]
center_df = df[(df["target_gt_x"] >= 1000) & (df["target_gt_x"] <= 1800)]
right_df = df[df["target_gt_x"] > 1800]

print("\n--- HORIZONTAL RANGE BREAKDOWN ---")
print(f"LEFT   (GT_X={left_df['target_gt_x'].mean():.0f}): RawYaw={left_df['raw_yaw_deg'].mean():+6.2f}° | HeadYaw={left_df['head_yaw_deg'].mean():+6.2f}° | PredFiltX={left_df['pred_filt_x'].mean():.0f}")
print(f"CENTER (GT_X={center_df['target_gt_x'].mean():.0f}): RawYaw={center_df['raw_yaw_deg'].mean():+6.2f}° | HeadYaw={center_df['head_yaw_deg'].mean():+6.2f}° | PredFiltX={center_df['pred_filt_x'].mean():.0f}")
print(f"RIGHT  (GT_X={right_df['target_gt_x'].mean():.0f}): RawYaw={right_df['raw_yaw_deg'].mean():+6.2f}° | HeadYaw={right_df['head_yaw_deg'].mean():+6.2f}° | PredFiltX={right_df['pred_filt_x'].mean():.0f}")

top_df = df[df["target_gt_y"] < 700]
mid_y_df = df[(df["target_gt_y"] >= 700) & (df["target_gt_y"] <= 1100)]
bot_df = df[df["target_gt_y"] > 1100]

print("\n--- VERTICAL RANGE BREAKDOWN ---")
print(f"TOP    (GT_Y={top_df['target_gt_y'].mean():.0f}): RawPitch={top_df['raw_pitch_deg'].mean():+6.2f}° | HeadPitch={top_df['head_pitch_deg'].mean():+6.2f}° | PredFiltY={top_df['pred_filt_y'].mean():.0f}")
print(f"MIDDLE (GT_Y={mid_y_df['target_gt_y'].mean():.0f}): RawPitch={mid_y_df['raw_pitch_deg'].mean():+6.2f}° | HeadPitch={mid_y_df['head_pitch_deg'].mean():+6.2f}° | PredFiltY={mid_y_df['pred_filt_y'].mean():.0f}")
print(f"BOTTOM (GT_Y={bot_df['target_gt_y'].mean():.0f}): RawPitch={bot_df['raw_pitch_deg'].mean():+6.2f}° | HeadPitch={bot_df['head_pitch_deg'].mean():+6.2f}° | PredFiltY={bot_df['pred_filt_y'].mean():.0f}")
