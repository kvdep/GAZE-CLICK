import os, sys, cv2, torch, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("mvp"))
from adapters.factory import create_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load user's video from 23:36:26
vid_path = "mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.mp4"
csv_path = "mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.csv"
df = pd.read_csv(csv_path)

yunet = cv2.FaceDetectorYN.create("mvp/models/weights/face_detection_yunet_2023mar.onnx", "", (1280, 720))
cap = cv2.VideoCapture(vid_path)

# Let's find sample frames for Far Left (pt 9), Far Right (pt 10), Far Top (pt 11), Far Bottom (pt 12)
# Since the video has 643 frames and CSV has 461 rows, let's find the frames corresponding to those points by reading CSV timestamps
# In CSV, point_idx 9 (Left), 10 (Right), 11 (Top), 12 (Bottom)
pt_indices = [9, 10, 11, 12]
pt_names = ["Far Left (X=173, Y=900)", "Far Right (X=2707, Y=900)", "Far Top (X=1440, Y=216)", "Far Bottom (X=1440, Y=1620)"]

# Let's inspect the CSV rows for these points
for p_idx, name in zip(pt_indices, pt_names):
    rows = df[df["point_idx"] == p_idx]
    if len(rows) > 0:
        mid_row = rows.iloc[len(rows)//2]
        print(f"CSV Point {p_idx} [{name}]: Target=({mid_row['target_gt_x']}, {mid_row['target_gt_y']})")

print("\n--- Testing GazeTR, PureGaze, YOLO26n, L2CS on sample frames ---")
# Let's check GazeTR
gazetr = create_adapter("gazetr", device)
puregaze = create_adapter("puregaze", device)
yolo = create_adapter("yolo26n", device)

# Let's grab 4 frames from the video where face is detected
frames = []
for i in range(min(500, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()

print(f"Read {len(frames)} frames from video.")
