import os, sys, time, math, glob
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath("mvp"))
import torch
from adapters.factory import create_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Suite] Initializing 9-session benchmark suite on {device}...")

SCREEN_W = 2880
SCREEN_H = 1800
PX_PER_MM = 9.536842105263158
DIST_MM = 600.0
CX = SCREEN_W / 2.0
CY = SCREEN_H / 2.0

SESSIONS = [
    "20261013_182824",
    "20261014_115554",
    "20261017_140128",
    "20261017_143205",
    "20261017_143500",
    "20261017_143611",
    "20261017_152857",
    "20261017_165842",
    "20261017_165930"
]

MODELS = [
    "puregaze",
    "gazetr",
    "mediapipe",
    "yolo26n",
    "idiap_resnet18",
    "l2cs",
    "l2cs_diversity"
]

def eval_session(session_id: str, adapter, use_head_pose: bool = False) -> dict:
    vid_path = f"mvp/benchmarks/benchmark_test_{session_id}.mp4"
    csv_path = f"mvp/benchmarks/benchmark_test_{session_id}.csv"
    cache_path = f"mvp/experiments/_cache_face_{session_id}.npz"

    df = pd.read_csv(csv_path)
    cache = np.load(cache_path)
    bboxes = cache["bboxes"]
    landmarks = cache["landmarks"]

    cap = cv2.VideoCapture(vid_path)
    total_frames = len(df)

    gaze_pitches = []
    gaze_yaws = []
    head_pitches = []
    head_yaws = []
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
        hp = res.get("head_pitch_deg", 0.0)
        hy = res.get("head_yaw_deg", 0.0)

        if math.isnan(p): p = 0.0
        if math.isnan(y): y = 0.0
        if math.isnan(hp): hp = 0.0
        if math.isnan(hy): hy = 0.0

        gaze_pitches.append(p)
        gaze_yaws.append(y)
        head_pitches.append(hp)
        head_yaws.append(hy)
        latencies.append(t_ms)

    cap.release()

    df_eval = df.iloc[:len(gaze_pitches)].copy()
    df_eval["raw_pitch_deg"] = gaze_pitches
    df_eval["raw_yaw_deg"] = gaze_yaws
    df_eval["head_pitch_deg"] = head_pitches
    df_eval["head_yaw_deg"] = head_yaws
    df_eval["latency_ms"] = latencies

    is_dynamic = df_eval["phase"] == "CircleTrajectory"
    df_calib = df_eval[~is_dynamic]
    df_test = df_eval[is_dynamic]

    if len(df_calib) < 10 or len(df_test) < 10:
        return None

    # Calibration fit
    calib_samples = []
    for phase_name, grp in df_calib.groupby("phase"):
        tx = float(grp["target_gt_x"].iloc[0])
        ty = float(grp["target_gt_y"].iloc[0])
        y_med = float(grp["raw_yaw_deg"].median())
        p_med = float(grp["raw_pitch_deg"].median())
        hy_med = float(grp["head_yaw_deg"].median())
        hp_med = float(grp["head_pitch_deg"].median())
        calib_samples.append((y_med, p_med, hy_med, hp_med, tx, ty))

    if use_head_pose:
        X_train = np.array([[s[0], s[1], s[2], s[3]] for s in calib_samples])
        X_test = np.column_stack([
            df_test["raw_yaw_deg"].values,
            df_test["raw_pitch_deg"].values,
            df_test["head_yaw_deg"].values,
            df_test["head_pitch_deg"].values
        ])
    else:
        X_train = np.array([[s[0], s[1]] for s in calib_samples])
        X_test = np.column_stack([df_test["raw_yaw_deg"].values, df_test["raw_pitch_deg"].values])

    Y_tx = np.array([s[4] for s in calib_samples])
    Y_ty = np.array([s[5] for s in calib_samples])

    reg_x = Ridge(alpha=1.0).fit(X_train, Y_tx)
    reg_y = Ridge(alpha=1.0).fit(X_train, Y_ty)

    pred_x = reg_x.predict(X_test)
    pred_y = reg_y.predict(X_test)

    gt_x = df_test["target_gt_x"].values
    gt_y = df_test["target_gt_y"].values

    err_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    err_mm = err_px / PX_PER_MM

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

    rx = float(np.corrcoef(pred_x, gt_x)[0, 1]) if np.std(pred_x) > 1e-6 else 0.0
    ry = float(np.corrcoef(pred_y, gt_y)[0, 1]) if np.std(pred_y) > 1e-6 else 0.0

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
        "slope_x_yaw": float(reg_x.coef_[0]),
        "slope_y_pitch": float(reg_y.coef_[1])
    }

records = []
print(f"Starting 9-session benchmark across models ({len(MODELS)} models x {len(SESSIONS)} sessions = {len(MODELS)*len(SESSIONS)} evaluations).")

for model_name in MODELS:
    print(f"\n==================== Initializing model '{model_name}' ====================")
    try:
        adapter = create_adapter(model_name, device)
    except Exception as e:
        print(f"Failed to initialize {model_name}: {e}")
        continue

    for sid in SESSIONS:
        t0 = time.time()
        try:
            res = eval_session(sid, adapter, use_head_pose=False)
            if res:
                res["model_name"] = model_name
                records.append(res)
                print(f"[{model_name:<16}] {sid}: AngErr={res['mean_ang_err_deg']:.2f}° (Med={res['median_ang_err_deg']:.2f}°), Rx={res['r_x']:+.3f}, Ry={res['r_y']:+.3f}, R2={res['r2']:+.3f} in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[{model_name:<16}] {sid}: ERROR {e}")

    # For YOLO26n, also evaluate with 4D head pose calibration
    if model_name == "yolo26n":
        print(f"\n==================== Evaluating 'yolo26n (with 4D head compensation)' ====================")
        for sid in SESSIONS:
            t0 = time.time()
            try:
                res_hp = eval_session(sid, adapter, use_head_pose=True)
                if res_hp:
                    res_hp["model_name"] = "yolo26n_head_comp"
                    records.append(res_hp)
                    print(f"[{'yolo26n_head_comp':<16}] {sid}: AngErr={res_hp['mean_ang_err_deg']:.2f}° (Med={res_hp['median_ang_err_deg']:.2f}°), Rx={res_hp['r_x']:+.3f}, Ry={res_hp['r_y']:+.3f}, R2={res_hp['r2']:+.3f} in {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"[{'yolo26n_head_comp':<16}] {sid}: ERROR {e}")

df_results = pd.DataFrame(records)
out_csv = "mvp/experiments/multi_session_benchmark_results.csv"
df_results.to_csv(out_csv, index=False)
print(f"\nSaved raw results to {out_csv}")

# Aggregated summary
summary = []
for model_name, grp in df_results.groupby("model_name"):
    summary.append({
        "Model": model_name,
        "Sessions Count": len(grp),
        "Mean AngErr (deg)": f"{grp['mean_ang_err_deg'].mean():.2f} ± {grp['mean_ang_err_deg'].std():.2f}",
        "Median AngErr (deg)": f"{grp['median_ang_err_deg'].median():.2f}",
        "Min AngErr (deg)": f"{grp['mean_ang_err_deg'].min():.2f}",
        "Max AngErr (deg)": f"{grp['mean_ang_err_deg'].max():.2f}",
        "Mean Err (mm)": f"{grp['mean_err_mm'].mean():.1f} ± {grp['mean_err_mm'].std():.1f}",
        "Mean Err (px)": f"{grp['mean_err_px'].mean():.1f} ± {grp['mean_err_px'].std():.1f}",
        "Mean R_x": f"{grp['r_x'].mean():+.3f}",
        "Mean R_y": f"{grp['r_y'].mean():+.3f}",
        "Mean R2": f"{grp['r2'].mean():+.3f}",
        "Mean Latency (ms)": f"{grp['mean_latency_ms'].mean():.2f}"
    })

df_summary = pd.DataFrame(summary).sort_values("Mean AngErr (deg)")
print("\n" + "=" * 115)
print("AGGREGATED MULTI-SESSION BENCHMARK SUMMARY (9 User Sessions, 3344 Frames)")
print("=" * 115)
print(df_summary.to_string(index=False))
print("=" * 115)

out_summary_csv = "mvp/experiments/multi_session_summary.csv"
df_summary.to_csv(out_summary_csv, index=False)
