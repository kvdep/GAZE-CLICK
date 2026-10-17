import os, sys, math, argparse, cv2, torch, numpy as np, pandas as pd
from typing import Dict, Any, Optional, Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.face_pipeline import GazeTrackingPipeline
from engine.calibration import GazeCalibration
from engine.filter import PointOneEuroFilter

def _compute_metrics(gt_pts: np.ndarray, pred_pts: np.ndarray, screen_w: float, screen_h: float, dist_px: float, px_per_mm: float) -> Dict[str, float]:
    diffs = pred_pts - gt_pts
    err_px = np.linalg.norm(diffs, axis=-1)
    err_mm = err_px / px_per_mm
    cx = screen_w / 2.0; cy = screen_h / 2.0
    v_gt = np.stack([gt_pts[:, 0] - cx, gt_pts[:, 1] - cy, -np.full(len(gt_pts), dist_px)], axis=-1)
    v_pred = np.stack([pred_pts[:, 0] - cx, pred_pts[:, 1] - cy, -np.full(len(pred_pts), dist_px)], axis=-1)
    norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)
    norm_pred = v_pred / np.linalg.norm(v_pred, axis=-1, keepdims=True)
    dot = np.clip(np.sum(norm_gt * norm_pred, axis=-1), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))
    same_x = (np.sign(pred_pts[:, 0] - cx) == np.sign(gt_pts[:, 0] - cx))
    same_y = (np.sign(pred_pts[:, 1] - cy) == np.sign(gt_pts[:, 1] - cy))
    corr_x = float(np.corrcoef(pred_pts[:, 0], gt_pts[:, 0])[0, 1]) if len(pred_pts) > 2 else 0.0
    corr_y = float(np.corrcoef(pred_pts[:, 1], gt_pts[:, 1])[0, 1]) if len(pred_pts) > 2 else 0.0
    return {
        "angular_err_mean_deg": float(np.mean(ang_err_deg)),
        "angular_err_median_deg": float(np.median(ang_err_deg)),
        "angular_err_std_deg": float(np.std(ang_err_deg)),
        "screen_err_mean_mm": float(np.mean(err_mm)),
        "screen_err_median_mm": float(np.median(err_mm)),
        "screen_err_mean_px": float(np.mean(err_px)),
        "mae_x_px": float(np.mean(np.abs(diffs[:, 0]))),
        "mae_y_px": float(np.mean(np.abs(diffs[:, 1]))),
        "bias_x_px": float(np.mean(diffs[:, 0])),
        "bias_y_px": float(np.mean(diffs[:, 1])),
        "corr_x": corr_x,
        "corr_y": corr_y,
        "dir_accuracy_x_pct": float(np.mean(same_x) * 100.0),
        "dir_accuracy_y_pct": float(np.mean(same_y) * 100.0),
        "dir_accuracy_both_pct": float(np.mean(same_x & same_y) * 100.0),
    }

def evaluate_video(video_path: str, table_path: str, checkpoint_path: str = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt", profile_path: str = "", auto_calibrate: bool = True) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Metadata table not found: {table_path}")
    df = pd.read_parquet(table_path) if table_path.endswith(".parquet") else pd.read_csv(table_path)
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    screen_w = 2880.0; screen_h = 1800.0
    phys_w_mm = 301.6; phys_h_mm = 188.5; dist_mm = 600.0
    px_per_mm = ((screen_w / phys_w_mm) + (screen_h / phys_h_mm)) / 2.0
    dist_px = dist_mm * px_per_mm
    pipeline = GazeTrackingPipeline(checkpoint_path=checkpoint_path)
    calibration = GazeCalibration(screen_w=int(screen_w), screen_h=int(screen_h), profile_path=profile_path)
    filter_obj = PointOneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0, deadzone_radius_px=8.0)
    frame_idx = 0
    records = []
    print(f"Evaluating benchmark video: {video_path} ({total_video_frames} frames) with checkpoint {os.path.basename(checkpoint_path)}...")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame_idx >= len(df):
            break
        row = df.iloc[frame_idx]
        gt_x = float(row["target_gt_x"]); gt_y = float(row["target_gt_y"])
        t_sec = float(row.get("timestamp_sec", frame_idx / fps))
        res = pipeline.process_frame(frame)
        p_deg = res.get("gaze_pitch_deg", 0.0); y_deg = res.get("gaze_yaw_deg", 0.0)
        hp_deg = res.get("head_pitch_deg", 0.0); hy_deg = res.get("head_yaw_deg", 0.0)
        raw_x, raw_y = calibration.predict_uncalibrated(p_deg, y_deg)
        records.append({
            "frame_idx": frame_idx,
            "target_gt_x": gt_x,
            "target_gt_y": gt_y,
            "timestamp_sec": t_sec,
            "pred_raw_x": raw_x,
            "pred_raw_y": raw_y,
            "face_detected": res.get("face_detected", False),
            "gaze_pitch_deg": p_deg,
            "gaze_yaw_deg": y_deg,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "phase": str(row.get("phase", ""))
        })
        frame_idx += 1
    cap.release()
    eval_df = pd.DataFrame(records)
    valid_df = eval_df[eval_df["face_detected"]].copy()
    if len(valid_df) == 0:
        return {"error": "No faces detected in video"}
    calib_performed = False
    calib_r2 = 0.0
    calib_phases = [p for p in valid_df["phase"].unique() if "StaticFixation" in p or "Grid Point" in p]
    if auto_calibrate and len(calib_phases) >= 5:
        calib_samples = []
        for cp_phase in calib_phases:
            subset = valid_df[valid_df["phase"] == cp_phase]
            if len(subset) >= 3:
                calib_samples.append({
                    "target_x": float(subset["target_gt_x"].iloc[0]),
                    "target_y": float(subset["target_gt_y"].iloc[0]),
                    "pitch_deg": float(subset["gaze_pitch_deg"].median()),
                    "yaw_deg": float(subset["gaze_yaw_deg"].median()),
                    "head_pitch_deg": float(subset["head_pitch_deg"].median()),
                    "head_yaw_deg": float(subset["head_yaw_deg"].median())
                })
        fit_res = calibration.fit(calib_samples, l2_reg=1.0)
        calib_performed = bool(fit_res.get("status", 0) == 1)
        calib_r2 = float(fit_res.get("r2_score", 0.0))
        print(f"Auto-calibration fitted: R2={calib_r2:.3f}, points={len(calib_samples)}")
    pred_calib_x = []; pred_calib_y = []
    for _, r in valid_df.iterrows():
        if calibration.is_calibrated:
            cx, cy = calibration.predict(r["gaze_pitch_deg"], r["gaze_yaw_deg"], head_pitch_deg=r["head_pitch_deg"], head_yaw_deg=r["head_yaw_deg"])
        else:
            cx, cy = r["pred_raw_x"], r["pred_raw_y"]
        fx, fy = filter_obj.filter(cx, cy, r["timestamp_sec"])
        pred_calib_x.append(fx); pred_calib_y.append(fy)
    valid_df["pred_calib_x"] = pred_calib_x
    valid_df["pred_calib_y"] = pred_calib_y
    all_gt = valid_df[["target_gt_x", "target_gt_y"]].to_numpy()
    all_raw = valid_df[["pred_raw_x", "pred_raw_y"]].to_numpy()
    all_calib = valid_df[["pred_calib_x", "pred_calib_y"]].to_numpy()
    raw_metrics = _compute_metrics(all_gt, all_raw, screen_w, screen_h, dist_px, px_per_mm)
    calib_metrics = _compute_metrics(all_gt, all_calib, screen_w, screen_h, dist_px, px_per_mm)
    test_phase_df = valid_df[valid_df["phase"] == "CircleTrajectory"].copy()
    if len(test_phase_df) == 0:
        test_phase_df = valid_df[~valid_df["phase"].isin(grid_phases)].copy()
    if len(test_phase_df) > 0:
        test_gt = test_phase_df[["target_gt_x", "target_gt_y"]].to_numpy()
        test_pred = test_phase_df[["pred_calib_x", "pred_calib_y"]].to_numpy()
        circle_metrics = _compute_metrics(test_gt, test_pred, screen_w, screen_h, dist_px, px_per_mm)
    else:
        circle_metrics = calib_metrics
    results = {
        "video_path": video_path,
        "checkpoint": os.path.basename(checkpoint_path),
        "total_evaluated_frames": len(eval_df),
        "valid_frames": len(valid_df),
        "test_trajectory_frames": len(test_phase_df),
        "auto_calibrate": calib_performed,
        "calib_r2": calib_r2,
        "raw_metrics": raw_metrics,
        "calib_metrics": calib_metrics,
        "circle_metrics": circle_metrics,
        "angular_err_mean_deg": circle_metrics["angular_err_mean_deg"] if calib_performed else raw_metrics["angular_err_mean_deg"],
        "angular_err_median_deg": circle_metrics["angular_err_median_deg"] if calib_performed else raw_metrics["angular_err_median_deg"],
        "screen_err_mean_mm": circle_metrics["screen_err_mean_mm"] if calib_performed else raw_metrics["screen_err_mean_mm"],
        "screen_err_mean_px": circle_metrics["screen_err_mean_px"] if calib_performed else raw_metrics["screen_err_mean_px"],
        "corr_x": circle_metrics["corr_x"] if calib_performed else raw_metrics["corr_x"],
        "corr_y": circle_metrics["corr_y"] if calib_performed else raw_metrics["corr_y"],
        "mae_x_px": circle_metrics["mae_x_px"] if calib_performed else raw_metrics["mae_x_px"],
        "mae_y_px": circle_metrics["mae_y_px"] if calib_performed else raw_metrics["mae_y_px"],
        "bias_x_px": circle_metrics["bias_x_px"] if calib_performed else raw_metrics["bias_x_px"],
        "bias_y_px": circle_metrics["bias_y_px"] if calib_performed else raw_metrics["bias_y_px"],
        "dir_accuracy_x_pct": circle_metrics["dir_accuracy_x_pct"] if calib_performed else raw_metrics["dir_accuracy_x_pct"],
        "dir_accuracy_y_pct": circle_metrics["dir_accuracy_y_pct"] if calib_performed else raw_metrics["dir_accuracy_y_pct"],
        "dir_accuracy_both_pct": circle_metrics["dir_accuracy_both_pct"] if calib_performed else raw_metrics["dir_accuracy_both_pct"],
    }
    plot_path = video_path.replace(".mp4", "_evaluated_plot.png")
    _save_evaluation_plot(plot_path, int(screen_w), int(screen_h), test_gt if len(test_phase_df) > 0 else all_gt, test_pred if len(test_phase_df) > 0 else all_calib, results)
    results["plot_path"] = plot_path
    report_path = video_path.replace(".mp4", "_evaluated_report.md")
    _save_evaluation_report(report_path, results)
    results["report_path"] = report_path
    print(f"Evaluation Complete: Test Trajectory Angular Error = {circle_metrics['angular_err_mean_deg']:.2f} deg, Screen = {circle_metrics['screen_err_mean_mm']:.1f} mm ({circle_metrics['screen_err_mean_px']:.1f} px)")
    return results

def _save_evaluation_plot(plot_path: str, screen_w: int, screen_h: int, gt_pts: np.ndarray, pred_pts: np.ndarray, res: Dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
        ax.set_xlim(0, screen_w); ax.set_ylim(screen_h, 0)
        m = res.get("circle_metrics", res)
        ax.set_title(f"Video Ground Truth Benchmark (CircleTrajectory)\nAngular Error: {m['angular_err_mean_deg']:.2f} deg | Screen Error: {m['screen_err_mean_mm']:.1f} mm ({m['screen_err_mean_px']:.1f} px) | R_y: {m['corr_y']:.3f}", fontsize=11)
        ax.set_xlabel("Screen X (px)"); ax.set_ylabel("Screen Y (px)")
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], color="green", linewidth=2.5, label="Ground Truth Targets", alpha=0.85)
        ax.scatter(gt_pts[::10, 0], gt_pts[::10, 1], color="darkgreen", s=35, zorder=4)
        ax.plot(pred_pts[:, 0], pred_pts[:, 1], color="blue", linewidth=1.2, label="Calibrated Trajectory", alpha=0.7)
        ax.scatter(pred_pts[::10, 0], pred_pts[::10, 1], color="darkblue", s=15, alpha=0.5, zorder=3)
        ax.axvline(screen_w / 2.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(screen_h / 2.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
    except Exception:
        pass

def _save_evaluation_report(report_path: str, res: Dict[str, Any]) -> None:
    circ = res.get("circle_metrics", res)
    raw = res.get("raw_metrics", res)
    content = f"""=== ОТЧЕТ ЭКСПЕРТИЗЫ ФИНАЛЬНОГО ТЕСТОВОГО ВИДЕО (VIDEO BENCHMARK AUDIT) ===

> **СТАТУС:** Официальная верификация на видеозаписи лица пользователя.
> **ЗАПРЕТ:** Обучение на этом видео или дообучение гиперпараметров строго запрещено.
> **ЧЕКПОИНТ:** `{res.get('checkpoint', '')}`
> **АВТОКАЛИБРОВКА (5-параметрическая Ridge):** {'ВКЛЮЧЕНА (R2=' + str(round(res.get('calib_r2', 0), 3)) + ')' if res.get('auto_calibrate') else 'ОТКЛЮЧЕНА'}

---

== РАЗДЕЛ 1. МЕТРИКИ ТОЧНОСТИ НА НЕЗАВИСИМОЙ ТРАЕКТОРИИ (CircleTrajectory, {res.get('test_trajectory_frames', 0)} кадров)

| Метрика | Калиброванное значение | Некалиброванное значение | Норматив ТЗ | Вердикт (Калибр.) |
|---|---|---|---|---|
| **Угловая погрешность (Mean)** | **{circ['angular_err_mean_deg']:.2f} deg** | {raw['angular_err_mean_deg']:.2f} deg | $\\le 5.20^\\circ$ | {'PASS' if circ['angular_err_mean_deg'] <= 5.20 else 'FAIL'} |
| **Угловая погрешность (Median)** | **{circ['angular_err_median_deg']:.2f} deg** | {raw['angular_err_median_deg']:.2f} deg | - | - |
| **Экранная ошибка (Mean, mm)** | **{circ['screen_err_mean_mm']:.1f} mm** | {raw['screen_err_mean_mm']:.1f} mm | $\\le 50.0\\text{{ мм}}$ | {'PASS' if circ['screen_err_mean_mm'] <= 50.0 else 'FAIL'} |
| **Экранная ошибка (Mean, px)** | **{circ['screen_err_mean_px']:.1f} px** | {raw['screen_err_mean_px']:.1f} px | - | - |
| **Корреляция по X (R_x)** | **{circ['corr_x']:.3f}** | {raw['corr_x']:.3f} | $\\ge 0.70$ | {'PASS' if circ['corr_x'] >= 0.70 else 'FAIL'} |
| **Корреляция по Y (R_y)** | **{circ['corr_y']:.3f}** | {raw['corr_y']:.3f} | $\\ge 0.70$ | {'PASS' if circ['corr_y'] >= 0.70 else 'FAIL'} |
| **Точность знака по X (Лево/Право)** | **{circ['dir_accuracy_x_pct']:.1f}%** | {raw['dir_accuracy_x_pct']:.1f}% | $\\ge 80.0\\%$ | {'PASS' if circ['dir_accuracy_x_pct'] >= 80.0 else 'FAIL'} |
| **Точность знака по Y (Верх/Низ)** | **{circ['dir_accuracy_y_pct']:.1f}%** | {raw['dir_accuracy_y_pct']:.1f}% | $\\ge 80.0\\%$ | {'PASS' if circ['dir_accuracy_y_pct'] >= 80.0 else 'FAIL'} |
| **Точность направления (Оба квадранта)** | **{circ['dir_accuracy_both_pct']:.1f}%** | {raw['dir_accuracy_both_pct']:.1f}% | $\\ge 70.0\\%$ | {'PASS' if circ['dir_accuracy_both_pct'] >= 70.0 else 'FAIL'} |

---

== РАЗДЕЛ 2. ПОКОМПОНЕНТНЫЕ СМЕЩЕНИЯ НА ТЕСТОВОЙ ТРАЕКТОРИИ
- **MAE по X:** `{circ['mae_x_px']:.1f}` px
- **MAE по Y:** `{circ['mae_y_px']:.1f}` px
- **Систематический сдвиг по X:** `{circ['bias_x_px']:+.1f}` px
- **Систематический сдвиг по Y:** `{circ['bias_y_px']:+.1f}` px
- **Файл графика:** `{res.get('plot_path', '')}`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaze Model on Ground Truth Benchmark Video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--no-auto-calibrate", dest="auto_calibrate", action="store_false")
    args = parser.parse_args()
    evaluate_video(args.video, args.metadata, args.checkpoint, args.profile, auto_calibrate=args.auto_calibrate)

if __name__ == "__main__":
    main()
