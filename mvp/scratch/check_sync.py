import sys, os
sys.path.insert(0, os.path.abspath("mvp"))
import cv2, pandas as pd
from engine.face_pipeline import GazeTrackingPipeline

pipe = GazeTrackingPipeline()
cap = cv2.VideoCapture("mvp/benchmarks/benchmark_test_20261012_011259.mp4")
df = pd.read_parquet("mvp/benchmarks/benchmark_test_20261012_011259.parquet")

for i in range(25):
    ret, frame = cap.read()
    if not ret: break
    res = pipe.process_frame(frame)
    row = df.iloc[i]
    print(f"{i:02d}: Vid_yaw={res['raw_yaw_deg']:+.2f}, Parq_yaw={row['raw_yaw_deg']:+.2f}, Vid_pitch={res['raw_pitch_deg']:+.2f}, Parq_pitch={row['raw_pitch_deg']:+.2f}, Tgt=({row['target_gt_x']:.0f}, {row['target_gt_y']:.0f})")
