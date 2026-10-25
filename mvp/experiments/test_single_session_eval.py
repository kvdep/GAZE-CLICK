import os, sys, time, math, glob
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath("mvp"))
import torch
from adapters.factory import create_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Evaluation configuration
SCREEN_W = 2880
SCREEN_H = 1800
PX_PER_MM = 9.536842105263158
DIST_MM = 600.0
CX = SCREEN_W / 2.0
CY = SCREEN_H / 2.0

def evaluate_session_for_model(session_id: str, adapter) -> dict:
    # Resolve file paths
    vid_path = None
    csv_path = None
    cache_path = None
    for pattern in [f"mvp/benchmarks/benchmark_test_*{session_id}*.mp4"]:
        matches = glob.glob(pattern)
        if matches:
            vid_path = matches[0]
            csv_path = vid_path.replace(".mp4", ".csv")
            break

    for pattern in [f"mvp/experiments/_cache_face_*{session_id}*.npz"]:
        matches = glob.glob(pattern)
        if matches:
            cache_path = matches[0]
            break

    if not vid_path or not os.path.exists(csv_path) or not cache_path:
        raise FileNotFoundError(f"Could not find files for session {session_id}: vid={vid_path}, csv={csv_path}, cache={cache_path}")


    df = pd.read_csv(csv_path)
    cache = np.load(cache_path)
    bboxes = cache["bboxes"]
    landmarks = cache["landmarks"]

    cap = cv2.VideoCapture(vid_path)
    total_frames = len(df)

    gaze_pitches = []
    gaze_yaws = []
    latencies = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bb = tuple(bboxes[i])
        lm = landmarks[i]

        t0 = time.perf_counter()
        res = adapter.predict_gaze(frame, bb, lm)
        t_ms = (time.perf_counter() - t0) * 1000.0

        p = res.get("pitch_deg", 0.0)
        y = res.get("yaw_deg", 0.0)
        if math.isnan(p): p = 0.0
        if math.isnan(y): y = 0.0

        gaze_pitches.append(p)
        gaze_yaws.append(y)
        latencies.append(t_ms)

    cap.release()

    df_eval = df.iloc[:len(gaze_pitches)].copy()
    df_eval["raw_pitch_deg"] = gaze_pitches
    df_eval["raw_yaw_deg"] = gaze_yaws
    df_eval["latency_ms"] = latencies

    # Split calibration points vs dynamic test points
    is_dynamic = df_eval["phase"] == "CircleTrajectory"
    df_calib = df_eval[~is_dynamic]
    df_test = df_eval[is_dynamic]

    if len(df_calib) < 10 or len(df_test) < 10:
        return {"error": "Insufficient frames"}

    # Fit calibration on static points
    calib_samples = []
    for phase_name, grp in df_calib.groupby("phase"):
        tx = float(grp["target_gt_x"].iloc[0])
        ty = float(grp["target_gt_y"].iloc[0])
        y_med = float(grp["raw_yaw_deg"].median())
        p_med = float(grp["raw_pitch_deg"].median())
        calib_samples.append((y_med, p_med, tx, ty))

    X_train = np.array([[s[0], s[1]] for s in calib_samples])
    Y_tx = np.array([s[2] for s in calib_samples])
    Y_ty = np.array([s[3] for s in calib_samples])

    reg_x = Ridge(alpha=1.0).fit(X_train, Y_tx)
    reg_y = Ridge(alpha=1.0).fit(X_train, Y_ty)

    # Predict test trajectory
    X_test = np.column_stack([df_test["raw_yaw_deg"].values, df_test["raw_pitch_deg"].values])
    pred_x = reg_x.predict(X_test)
    pred_y = reg_y.predict(X_test)

    gt_x = df_test["target_gt_x"].values
    gt_y = df_test["target_gt_y"].values

    # Pixel error
    err_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    err_mm = err_px / PX_PER_MM

    # Angular error
    v_gt_x = (gt_x - CX) / PX_PER_MM
    v_gt_y = (gt_y - CY) / PX_PER_MM
    v_gt_z = np.full_like(v_gt_x, -DIST_MM)
    norm_gt = np.sqrt(v_gt_x**2 + v_gt_y**2 + v_gt_z**2)
    ngt_x, ngt_y, ngt_z = v_gt_x / norm_gt, v_gt_y / norm_gt, v_gt_z / norm_gt

    v_pr_x = (pred_x - CX) / PX_PER_MM
    v_pr_y = (pred_y - CY) / PX_PER_MM
    v_pr_z = np.full_like(v_pr_x, -DIST_MM)
    norm_pr = np.sqrt(v_pr_x**2 + v_pr_y**2 + v_pr_z**2)
    npr_x, npr_y, npr_z = v_pr_x / norm_pr, v_pr_y / norm_pr, v_pr_z / norm_pr

    cos_sim = np.clip(ngt_x * npr_x + ngt_y * npr_y + ngt_z * npr_z, -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(cos_sim))

    # Pearson correlations
    rx = float(np.corrcoef(pred_x, gt_x)[0, 1]) if np.std(pred_x) > 1e-6 else 0.0
    ry = float(np.corrcoef(pred_y, gt_y)[0, 1]) if np.std(pred_y) > 1e-6 else 0.0

    # R2
    ss_res = np.sum((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    ss_tot = np.sum((gt_x - np.mean(gt_x))**2 + (gt_y - np.mean(gt_y))**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    return {
        "session_id": session_id,
        "n_test_frames": len(df_test),
        "mean_ang_err_deg": float(np.mean(ang_err_deg)),
        "median_ang_err_deg": float(np.median(ang_err_deg)),
        "std_ang_err_deg": float(np.std(ang_err_deg)),
        "mean_err_mm": float(np.mean(err_mm)),
        "median_err_mm": float(np.median(err_mm)),
        "mean_err_px": float(np.mean(err_px)),
        "median_err_px": float(np.median(err_px)),
        "r_x": rx,
        "r_y": ry,
        "r2": r2,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "slope_x_yaw": float(reg_x.coef_[0]),
        "slope_y_pitch": float(reg_y.coef_[1])
    }

models = ["yolo26n", "l2cs", "l2cs_diversity", "gazetr", "puregaze", "idiap_resnet18", "mediapipe"]
session = "165930"
print(f"--- Running single session {session} benchmark across {len(models)} models ---")

for m in models:
    try:
        t_start = time.time()
        adapter = create_adapter(m, device)
        res = evaluate_session_for_model(session, adapter)
        dur = time.time() - t_start
        print(f"[{m:<16}] AngErr: {res['mean_ang_err_deg']:.2f}° (Med: {res['median_ang_err_deg']:.2f}°) | Err: {res['mean_err_mm']:.1f} mm ({res['mean_err_px']:.1f} px) | Rx: {res['r_x']:+.3f}, Ry: {res['r_y']:+.3f}, R2: {res['r2']:+.3f} | Latency: {res['mean_latency_ms']:.2f} ms | Evaluated in {dur:.1f}s")
    except Exception as e:
        import traceback
        print(f"[{m:<16}] FAILED: {e}")
        traceback.print_exc()
