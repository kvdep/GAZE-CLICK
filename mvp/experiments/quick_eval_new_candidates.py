import os, sys, time
import numpy as np
import pandas as pd
import torch
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from engine.calibration import GazeCalibration
from adapters.factory import create_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

sid = "20261013_182824"
vid_path = f"mvp/benchmarks/benchmark_test_{sid}.mp4"
csv_path = f"mvp/benchmarks/benchmark_test_{sid}.csv"
cache_path = f"mvp/experiments/_cache_face_{sid}.npz"

df = pd.read_csv(csv_path)
cache = np.load(cache_path)
bboxes = cache["bboxes"]
landmarks = cache["landmarks"]

cap = cv2.VideoCapture(vid_path)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames for session {sid}.")

candidates = [
    "puregaze",
    "ptgaze_eth_xgaze",
    "unigaze",
    "yakhyo_resnet50",
    "ptgaze_mpiifacegaze"
]

for model_name in candidates:
    print(f"\n==================== Testing '{model_name}' ====================")
    adapter = create_adapter(model_name, device)
    
    gaze_pitches = []
    gaze_yaws = []
    latencies = []
    
    t0 = time.time()
    for i, frame in enumerate(frames):
        bbox = tuple(bboxes[i])
        lms = landmarks[i]
        t_start = time.perf_counter()
        pred = adapter.predict_gaze(frame, bbox, lms)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
        gaze_pitches.append(pred["pitch_deg"])
        gaze_yaws.append(pred["yaw_deg"])
        
    df_eval = df.copy()
    df_eval["gaze_pitch_deg"] = gaze_pitches
    df_eval["gaze_yaw_deg"] = gaze_yaws
    
    is_dynamic = df_eval["phase"] == "CircleTrajectory"
    df_calib = df_eval[~is_dynamic]
    df_test = df_eval[is_dynamic]

    calib_samples = []
    for phase_name, grp in df_calib.groupby("phase"):
        tx = float(grp["target_gt_x"].iloc[0])
        ty = float(grp["target_gt_y"].iloc[0])
        y_med = float(grp["gaze_yaw_deg"].median())
        p_med = float(grp["gaze_pitch_deg"].median())
        calib_samples.append({
            "target_x": tx,
            "target_y": ty,
            "yaw_deg": y_med,
            "pitch_deg": p_med
        })

    calib = GazeCalibration(screen_w=2880, screen_h=1800, model_name=model_name)
    calib.fit(calib_samples, l2_reg=1.0)

    # Test phase evaluation
    pred_xs = []
    pred_ys = []
    for _, row in df_test.iterrows():
        px, py = calib.predict(row["gaze_pitch_deg"], row["gaze_yaw_deg"])
        pred_xs.append(px)
        pred_ys.append(py)

    gt_xs = df_test["target_gt_x"].values
    gt_ys = df_test["target_gt_y"].values
    pred_xs = np.array(pred_xs)
    pred_ys = np.array(pred_ys)

    px_per_mm = 2880.0 / 301.99
    err_px = np.sqrt((pred_xs - gt_xs)**2 + (pred_ys - gt_ys)**2)
    err_mm = err_px / px_per_mm
    ang_err_deg = np.degrees(np.arctan2(err_mm, 600.0))

    rx = np.corrcoef(gt_xs, pred_xs)[0, 1]
    ry = np.corrcoef(gt_ys, pred_ys)[0, 1]
    ss_tot = np.sum((gt_xs - np.mean(gt_xs))**2 + (gt_ys - np.mean(gt_ys))**2)
    ss_res = np.sum((gt_xs - pred_xs)**2 + (gt_ys - pred_ys)**2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    print(f"[{model_name:<18}] Mean AngErr={np.mean(ang_err_deg):.2f}° (Med={np.median(ang_err_deg):.2f}°), Rx={rx:+.3f}, Ry={ry:+.3f}, R2={r2:+.3f}, Latency={np.mean(latencies):.2f}ms in {time.time()-t0:.1f}s")
