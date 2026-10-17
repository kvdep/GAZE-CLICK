import os, sys, math, time, cv2, torch, numpy as np, pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.face_pipeline import GazeTrackingPipeline
from engine.calibration import GazeCalibration
from engine.filter import PointOneEuroFilter

def run_calibrated_benchmark(checkpoint_path: str, video_path: str, metadata_path: str) -> Dict[str, Any]:
    df = pd.read_parquet(metadata_path) if metadata_path.endswith(".parquet") else pd.read_csv(metadata_path)
    cap = cv2.VideoCapture(video_path)
    pipeline = GazeTrackingPipeline(checkpoint_path=checkpoint_path)
    calibration = GazeCalibration(screen_w=2880, screen_h=1800, profile_path="")
    filter_obj = PointOneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0, deadzone_radius_px=8.0)
    screen_w = 2880.0; screen_h = 1800.0
    phys_w_mm = 301.6; phys_h_mm = 188.5; dist_mm = 600.0
    px_per_mm = ((screen_w / phys_w_mm) + (screen_h / phys_h_mm)) / 2.0
    dist_px = dist_mm * px_per_mm
    frame_idx = 0
    records = []
    print(f"Extracting frame features using {os.path.basename(checkpoint_path)}...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame_idx >= len(df):
            break
        row = df.iloc[frame_idx]
        res = pipeline.process_frame(frame)
        p_deg = res.get("gaze_pitch_deg", 0.0)
        y_deg = res.get("gaze_yaw_deg", 0.0)
        hp_deg = res.get("head_pitch_deg", 0.0)
        hy_deg = res.get("head_yaw_deg", 0.0)
        records.append({
            "frame_idx": frame_idx,
            "target_gt_x": float(row["target_gt_x"]),
            "target_gt_y": float(row["target_gt_y"]),
            "timestamp_sec": float(row.get("timestamp_sec", frame_idx / 30.0)),
            "phase": str(row.get("phase", "")),
            "face_detected": res.get("face_detected", False),
            "pitch_deg": p_deg,
            "yaw_deg": y_deg,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "attention_variance": res.get("attention_variance", 0.020)
        })
        frame_idx += 1
    cap.release()
    eval_df = pd.DataFrame(records)
    valid_df = eval_df[eval_df["face_detected"]].copy()
    fix_phases = [p for p in valid_df["phase"].unique() if "StaticFixation" in p or "Grid Point" in p]
    calib_samples = []
    for fp in fix_phases:
        subset = valid_df[valid_df["phase"] == fp]
        if len(subset) >= 3:
            calib_samples.append({
                "target_x": float(subset["target_gt_x"].iloc[0]),
                "target_y": float(subset["target_gt_y"].iloc[0]),
                "pitch_deg": float(subset["pitch_deg"].median()),
                "yaw_deg": float(subset["yaw_deg"].median()),
                "head_pitch_deg": float(subset["head_pitch_deg"].median()),
                "head_yaw_deg": float(subset["head_yaw_deg"].median())
            })
    calib_res = calibration.fit(calib_samples, l2_reg=1.0)
    print(f"Calibration fit: R2={calib_res.get('r2_score', 0):.3f}, Residual={calib_res.get('mean_residual_px', 0):.1f} px across {len(calib_samples)} fixation points")
    test_subset = valid_df[valid_df["phase"] == "CircleTrajectory"].copy()
    if len(test_subset) == 0:
        test_subset = valid_df[~valid_df["phase"].isin(fix_phases)].copy()
    pred_x_list = []; pred_y_list = []
    for _, row in test_subset.iterrows():
        cx, cy = calibration.predict(
            row["pitch_deg"], row["yaw_deg"],
            head_pitch_deg=row["head_pitch_deg"],
            head_yaw_deg=row["head_yaw_deg"]
        )
        fx, fy = filter_obj.filter(cx, cy, row["timestamp_sec"])
        pred_x_list.append(fx); pred_y_list.append(fy)
    test_subset["pred_filt_x"] = pred_x_list
    test_subset["pred_filt_y"] = pred_y_list
    gt_pts = test_subset[["target_gt_x", "target_gt_y"]].to_numpy()
    pred_pts = test_subset[["pred_filt_x", "pred_filt_y"]].to_numpy()
    diffs = pred_pts - gt_pts
    err_px = np.linalg.norm(diffs, axis=-1)
    err_mm = err_px / px_per_mm
    cx_s = screen_w / 2.0; cy_s = screen_h / 2.0
    v_gt = np.stack([gt_pts[:, 0] - cx_s, gt_pts[:, 1] - cy_s, -np.full(len(gt_pts), dist_px)], axis=-1)
    v_pred = np.stack([pred_pts[:, 0] - cx_s, pred_pts[:, 1] - cy_s, -np.full(len(pred_pts), dist_px)], axis=-1)
    norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)
    norm_pred = v_pred / np.linalg.norm(v_pred, axis=-1, keepdims=True)
    dot = np.clip(np.sum(norm_gt * norm_pred, axis=-1), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))
    corr_x = float(np.corrcoef(pred_pts[:, 0], gt_pts[:, 0])[0, 1]) if len(pred_pts) > 2 else 0.0
    corr_y = float(np.corrcoef(pred_pts[:, 1], gt_pts[:, 1])[0, 1]) if len(pred_pts) > 2 else 0.0
    summary = {
        "checkpoint": os.path.basename(checkpoint_path),
        "test_frames": len(test_subset),
        "calib_r2": float(calib_res.get("r2_score", 0.0)),
        "angular_err_mean_deg": float(np.mean(ang_err_deg)),
        "angular_err_median_deg": float(np.median(ang_err_deg)),
        "screen_err_mean_mm": float(np.mean(err_mm)),
        "screen_err_median_mm": float(np.median(err_mm)),
        "screen_err_mean_px": float(np.mean(err_px)),
        "screen_err_median_px": float(np.median(err_px)),
        "mae_x_px": float(np.mean(np.abs(diffs[:, 0]))),
        "mae_y_px": float(np.mean(np.abs(diffs[:, 1]))),
        "bias_x_px": float(np.mean(diffs[:, 0])),
        "bias_y_px": float(np.mean(diffs[:, 1])),
        "corr_x": corr_x,
        "corr_y": corr_y
    }
    return summary

if __name__ == "__main__":
    v_path = "mvp/benchmarks/benchmark_test_20261014_115554.mp4"
    m_path = "mvp/benchmarks/benchmark_test_20261014_115554.parquet"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        cp = sys.argv[1]
        res = run_calibrated_benchmark(cp, v_path, m_path)
        print(f"--- EVAL: {cp} ---")
        for k, v in res.items():
            print(f"  {k}: {v}")
    else:
        checkpoints = [
            ("RUN 014 BEST (Epoch 6)", "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt"),
            ("RUN 014 LATEST", "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/latest_model.pt"),
            ("RUN 013 BEST (Epoch 1)", "mvp/experiments/yolo26n_gaze_net_yolo26n/run_013_20261016_182944/checkpoints/best_model.pt"),
            ("RUN 010 (Epoch 10)", "mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt")
        ]
        for tag, cp in checkpoints:
            if os.path.exists(cp):
                res = run_calibrated_benchmark(cp, v_path, m_path)
                print(f"--- {tag} ---")
                for k, v in res.items():
                    print(f"  {k}: {v}")
