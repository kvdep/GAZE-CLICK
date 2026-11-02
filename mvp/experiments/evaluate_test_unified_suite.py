import os, sys
sys.stdout.reconfigure(line_buffering=True)
import os, sys, time, math
import cv2
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding='utf-8')

# Ensure no proxy blockage
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(k, None)

sys.path.insert(0, os.path.abspath("mvp"))
from adapters.factory import create_adapter
from engine.calibration import GazeCalibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Suite] Initializing Test Dataset Benchmark Suite on {device}...")

parquet_path = "mvp/experiments/test_unified.parquet"
cache_path = "mvp/experiments/_cache_face_test_unified.npz"

if not os.path.exists(cache_path):
    print(f"Error: Cache file {cache_path} does not exist.")
    sys.exit(1)

df = pd.read_parquet(parquet_path)
cache = np.load(cache_path)
bboxes = cache["bboxes"]
landmarks = cache["landmarks"]
detected_mask = cache["detected"]
total_samples = len(df)
print(f"Loaded {total_samples} samples from {parquet_path}. Detected faces: {np.sum(detected_mask)}/{total_samples}")

# Ground truth gaze vectors (unit vectors)
gt_vecs = np.zeros((total_samples, 3), dtype=np.float32)
for i in range(total_samples):
    row = df.iloc[i]
    gx, gy, gz = float(row["gaze_vec_x"]), float(row["gaze_vec_y"]), float(row["gaze_vec_z"])
    norm = math.sqrt(gx*gx + gy*gy + gz*gz)
    if norm > 1e-6:
        gt_vecs[i] = [gx / norm, gy / norm, gz / norm]
    else:
        p = float(row["gaze_pitch_rad"])
        y = float(row["gaze_yaw_rad"])
        vx = -math.cos(p) * math.sin(y)
        vy = -math.sin(p)
        vz = -math.cos(p) * math.cos(y)
        vnorm = math.sqrt(vx*vx + vy*vy + vz*vz)
        gt_vecs[i] = [vx / vnorm, vy / vnorm, vz / vnorm]

MODELS_TO_EVAL = [
    "puregaze",
    "unigaze",
    "ptgaze_eth_xgaze",
    "gazetr",
    "mediapipe",
    "yolo26n",
    "ptgaze_mpiifacegaze",
    "idiap_resnet18",
    "l2cs_diversity",
    "l2cs",
    "yakhyo_resnet50",
    "yakhyo_resnet18",
    "qualcomm_eyegaze"
]

results = []

# Pre-read image paths
image_paths = df["image_path"].tolist()
datasets = df["dataset_id"].tolist()
subjects = df["subject_id"].tolist()
has_screen_flags = df["has_screen_coords"].tolist()
screen_xs = df["screen_x_px"].to_numpy()
screen_ys = df["screen_y_px"].to_numpy()

for model_key in MODELS_TO_EVAL:
    print(f"\n=======================================================")
    print(f"Evaluating Model: {model_key} on {device}")
    print(f"=======================================================")
    
    try:
        adapter = create_adapter(model_key, device)
    except Exception as e:
        print(f"Failed to initialize adapter '{model_key}': {e}")
        continue

    pred_vecs = np.zeros((total_samples, 3), dtype=np.float32)
    pred_pitches = np.zeros(total_samples, dtype=np.float32)
    pred_yaws = np.zeros(total_samples, dtype=np.float32)
    pred_head_pitches = np.zeros(total_samples, dtype=np.float32)
    pred_head_yaws = np.zeros(total_samples, dtype=np.float32)
    latencies = []

    t_eval_start = time.time()
    for i in range(total_samples):
        img_p = image_paths[i]
        img = cv2.imread(img_p)
        if img is None:
            pred_vecs[i] = [0.0, 0.0, -1.0]
            continue
        
        bbox = tuple(bboxes[i])
        lms = landmarks[i]

        t0 = time.perf_counter()
        try:
            pred = adapter.predict_gaze(img, bbox, lms)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
        except Exception as e:
            pred = {"pitch_deg": 0.0, "yaw_deg": 0.0, "gaze_vector": np.array([0.0, 0.0, -1.0])}
            latencies.append(0.0)

        p_deg = pred.get("pitch_deg", 0.0)
        y_deg = pred.get("yaw_deg", 0.0)
        hp_deg = pred.get("head_pitch_deg", 0.0)
        hy_deg = pred.get("head_yaw_deg", 0.0)

        pred_pitches[i] = p_deg
        pred_yaws[i] = y_deg
        pred_head_pitches[i] = hp_deg
        pred_head_yaws[i] = hy_deg

        vec = pred.get("gaze_vector")
        if vec is not None and len(vec) == 3:
            vnorm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
            if vnorm > 1e-6:
                pred_vecs[i] = [vec[0] / vnorm, vec[1] / vnorm, vec[2] / vnorm]
            else:
                pred_vecs[i] = [0.0, 0.0, -1.0]
        else:
            p_rad = math.radians(p_deg)
            y_rad = math.radians(y_deg)
            vx = -math.cos(p_rad) * math.sin(y_rad)
            vy = -math.sin(p_rad)
            vz = -math.cos(p_rad) * math.cos(y_rad)
            pred_vecs[i] = [vx, vy, vz]

        if (i + 1) % 1000 == 0 or (i + 1) == total_samples:
            el = time.time() - t_eval_start
            print(f"[{model_key}] Inferred {i+1}/{total_samples} samples in {el:.1f}s ({(i+1)/el:.1f} FPS)")

    # 1. Compute Angular Errors across all 6041 samples
    dot_prods = np.sum(gt_vecs * pred_vecs, axis=1)
    cos_sims = np.clip(dot_prods, -1.0, 1.0)
    ang_errors_deg = np.degrees(np.arccos(cos_sims))
    
    mean_ang_err = float(np.mean(ang_errors_deg))
    median_ang_err = float(np.median(ang_errors_deg))
    std_ang_err = float(np.std(ang_errors_deg))
    mean_lat_ms = float(np.mean(latencies))

    # Helper function to compute calibrated screen error and R2 on MPIIFaceGaze
    def compute_screen_metrics(use_hp: bool = False):
        mpii_indices = [idx for idx, flg in enumerate(has_screen_flags) if flg]
        if not mpii_indices:
            return 0.0, 0.0
        
        # Group by subject (p00, p01)
        subj_preds_x = []
        subj_preds_y = []
        subj_gts_x = []
        subj_gts_y = []

        df_mpii = df.iloc[mpii_indices].copy()
        df_mpii["pred_yaw"] = pred_yaws[mpii_indices]
        df_mpii["pred_pitch"] = pred_pitches[mpii_indices]
        df_mpii["pred_head_yaw"] = pred_head_yaws[mpii_indices]
        df_mpii["pred_head_pitch"] = pred_head_pitches[mpii_indices]

        for subj_name, grp in df_mpii.groupby("subject_id"):
            n_grp = len(grp)
            if n_grp < 20:
                continue
            # Select 32 calibration anchor points evenly spaced across session
            calib_idx = np.linspace(0, n_grp - 1, min(32, n_grp // 2), dtype=int)
            test_mask = np.ones(n_grp, dtype=bool)
            test_mask[calib_idx] = False
            
            grp_calib = grp.iloc[calib_idx]
            grp_test = grp.iloc[test_mask]

            calib_samples = []
            for _, r in grp_calib.iterrows():
                entry = {
                    "target_x": float(r["screen_x_px"]),
                    "target_y": float(r["screen_y_px"]),
                    "yaw_deg": float(r["pred_yaw"]),
                    "pitch_deg": float(r["pred_pitch"])
                }
                if use_hp:
                    entry["head_yaw_deg"] = float(r["pred_head_yaw"])
                    entry["head_pitch_deg"] = float(r["pred_head_pitch"])
                calib_samples.append(entry)

            calib = GazeCalibration(screen_w=1440, screen_h=900, model_name=f"{adapter.name}_{subj_name}")
            calib.fit(calib_samples, l2_reg=1.0)

            for _, r in grp_test.iterrows():
                px, py = calib.predict(
                    pitch_deg=float(r["pred_pitch"]),
                    yaw_deg=float(r["pred_yaw"]),
                    head_pitch_deg=float(r["pred_head_pitch"]) if use_hp else 0.0,
                    head_yaw_deg=float(r["pred_head_yaw"]) if use_hp else 0.0
                )
                subj_preds_x.append(px)
                subj_preds_y.append(py)
                subj_gts_x.append(float(r["screen_x_px"]))
                subj_gts_y.append(float(r["screen_y_px"]))

        preds_x = np.array(subj_preds_x)
        preds_y = np.array(subj_preds_y)
        gts_x = np.array(subj_gts_x)
        gts_y = np.array(subj_gts_y)

        err_px = np.sqrt((preds_x - gts_x)**2 + (preds_y - gts_y)**2)
        mean_px = float(np.mean(err_px))

        # R2 calculation
        ss_res = np.sum((preds_x - gts_x)**2 + (preds_y - gts_y)**2)
        ss_tot = np.sum((gts_x - np.mean(gts_x))**2 + (gts_y - np.mean(gts_y))**2)
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))
        return mean_px, r2

    screen_px_err, r2_score = compute_screen_metrics(use_hp=False)

    print(f"[{model_key}] RESULT: MeanAng={mean_ang_err:.2f}°, MedAng={median_ang_err:.2f}°, ScreenPx={screen_px_err:.1f}px, R2={r2_score:.3f}, Latency={mean_lat_ms:.2f}ms")
    
    results.append({
        "Model": model_key,
        "Mean AngErr (deg)": f"{mean_ang_err:.2f}°",
        "Median AngErr (deg)": f"{median_ang_err:.2f}",
        "Mean Err (px)": f"{screen_px_err:.1f}",
        "Mean R2": f"{r2_score:.3f}",
        "Latency (ms)": f"{mean_lat_ms:.2f}"
    })

    # For YOLO26n, also compute with 4D head pose compensation
    if model_key == "yolo26n":
        screen_px_hp, r2_hp = compute_screen_metrics(use_hp=True)
        print(f"[yolo26n_head_comp] RESULT: MeanAng={mean_ang_err:.2f}°, MedAng={median_ang_err:.2f}°, ScreenPx={screen_px_hp:.1f}px, R2={r2_hp:.3f}")
        results.append({
            "Model": "yolo26n_head_comp",
            "Mean AngErr (deg)": f"{mean_ang_err:.2f}°",
            "Median AngErr (deg)": f"{median_ang_err:.2f}",
            "Mean Err (px)": f"{screen_px_hp:.1f}",
            "Mean R2": f"{r2_hp:.3f}",
            "Latency (ms)": f"{mean_lat_ms:.2f}"
        })

# Save results
out_csv = "mvp/experiments/test_dataset_summary.csv"
res_df = pd.DataFrame(results)
res_df.to_csv(out_csv, index=False)
print(f"\n=======================================================")
print(f"Evaluation complete! Saved empirical results to {out_csv}")
print(res_df)
print(f"=======================================================")
