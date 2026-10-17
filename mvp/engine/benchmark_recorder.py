import os, sys, math, time, cv2, numpy as np, pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from engine.calibration import GazeCalibration

class BenchmarkRecorder:
    def __init__(self, output_dir: str = "mvp/benchmarks", screen_w: int = 2880, screen_h: int = 1800, cam_w: int = 1280, cam_h: int = 720, fps: float = 30.0, user_distance_mm: float = 600.0, physical_screen_w_mm: float = 300.0, physical_screen_h_mm: float = 190.0) -> None:
        self.output_dir = output_dir
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.target_fps = fps
        self.user_distance_mm = user_distance_mm
        self.physical_screen_w_mm = physical_screen_w_mm
        self.physical_screen_h_mm = physical_screen_h_mm
        self.px_per_mm_x = self.screen_w / max(self.physical_screen_w_mm, 1.0)
        self.px_per_mm_y = self.screen_h / max(self.physical_screen_h_mm, 1.0)
        self.dist_px = self.user_distance_mm * ((self.px_per_mm_x + self.px_per_mm_y) / 2.0)
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_recording = False
        self.video_writer = None
        self.session_id = ""
        self.records: List[Dict[str, Any]] = []
        self.start_time = 0.0
        self.static_points = self._build_static_points()
        self.static_idx = 0
        self.frames_per_static_point = 15
        self.static_recording_counter = 0
        self.is_capturing_static = False
        self.trajectory_active = False
        self.trajectory_angle = 0.0
        self.trajectory_speed = (2.0 * math.pi) / 10.0
        self.trajectory_radius = float(min(self.screen_w, self.screen_h) * 0.32)
        self.trajectory_center = (float(self.screen_w / 2.0), float(self.screen_h / 2.0))
        self.trajectory_progress = 0.0
        self.current_target_pt = (self.screen_w / 2.0, self.screen_h / 2.0)
        self.current_phase_name = ""
        self.session_calibration: Optional[GazeCalibration] = None
        self.on_calibration_updated: Optional[Callable[[GazeCalibration], None]] = None

    def _build_static_points(self) -> List[Dict[str, Any]]:
        pts = []
        nx, ny = 3, 3
        margin_x = 0.12
        xs = np.linspace(margin_x * self.screen_w, (1.0 - margin_x) * self.screen_w, nx)
        ys = np.linspace(0.18 * self.screen_h, 0.86 * self.screen_h, ny)
        for r_idx, y in enumerate(ys):
            for c_idx, x in enumerate(xs):
                pts.append({
                    "target_x": float(x),
                    "target_y": float(y),
                    "desc": f"Grid Point ({c_idx+1},{r_idx+1})"
                })
        extreme_points = [
            (float(0.06 * self.screen_w), float(0.50 * self.screen_h), "Far Left Edge"),
            (float(0.94 * self.screen_w), float(0.50 * self.screen_h), "Far Right Edge"),
            (float(0.50 * self.screen_w), float(0.14 * self.screen_h), "Far Top Edge"),
            (float(0.50 * self.screen_w), float(0.89 * self.screen_h), "Far Bottom Edge"),
        ]
        for ex_x, ex_y, name in extreme_points:
            pts.append({
                "target_x": ex_x,
                "target_y": ex_y,
                "desc": name
            })
        return pts

    def start_recording(self) -> str:
        self.session_id = f"benchmark_test_{time.strftime('%Y%m%d_%H%M%S')}"
        video_path = os.path.join(self.output_dir, f"{self.session_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.target_fps, (self.cam_w, self.cam_h))
        self.records = []
        self.static_idx = 0
        self.static_recording_counter = 0
        self.is_capturing_static = False
        self.trajectory_active = False
        self.trajectory_angle = 0.0
        self.trajectory_progress = 0.0
        self.start_time = time.perf_counter()
        self.is_recording = True
        first_pt = self.static_points[0]
        self.current_target_pt = (first_pt["target_x"], first_pt["target_y"])
        self.current_phase_name = f"Static: {first_pt['desc']}"
        return self.session_id

    def trigger_space_press(self) -> None:
        if not self.is_recording:
            return
        if self.static_idx < len(self.static_points) and not self.is_capturing_static:
            self.is_capturing_static = True
            self.static_recording_counter = 0

    def update_state(self, is_holding_trajectory_key: bool, dt: float) -> Tuple[float, float, str, float, bool]:
        if not self.is_recording:
            return (self.screen_w / 2.0, self.screen_h / 2.0, "Idle", 0.0, False)
        total_steps = len(self.static_points) + 1
        if self.static_idx < len(self.static_points):
            curr_pt = self.static_points[self.static_idx]
            tx = curr_pt["target_x"]; ty = curr_pt["target_y"]
            self.current_target_pt = (tx, ty)
            overall_progress = float(self.static_idx) / float(total_steps)
            if self.is_capturing_static:
                status = f"RECORDING [{self.static_recording_counter}/{self.frames_per_static_point}]: {curr_pt['desc']}"
            else:
                status = f"Point {self.static_idx + 1} of {len(self.static_points)}: Look at target, PRESS SPACE to record"
            return (tx, ty, status, overall_progress, False)
        else:
            self.current_phase_name = "Continuous Trajectory"
            cx, cy = self.trajectory_center
            if is_holding_trajectory_key:
                self.trajectory_angle += self.trajectory_speed * dt
                if self.trajectory_angle >= 2.0 * math.pi:
                    self.trajectory_angle = 2.0 * math.pi
                    self.trajectory_progress = 1.0
                    return (cx, cy, "Benchmark Finished! Release key to save.", 1.0, True)
            self.trajectory_progress = min(1.0, self.trajectory_angle / (2.0 * math.pi))
            tx = cx + self.trajectory_radius * math.cos(self.trajectory_angle - math.pi / 2.0)
            ty = cy + self.trajectory_radius * math.sin(self.trajectory_angle - math.pi / 2.0)
            self.current_target_pt = (float(tx), float(ty))
            overall_progress = (len(self.static_points) + self.trajectory_progress) / float(total_steps)
            if is_holding_trajectory_key:
                status = f"TRAJECTORY ACTIVE: Circle Tracking ({int(self.trajectory_progress * 100)}%)"
            else:
                status = "HOLD [SHIFT] or [G] to move target and track with eyes. Release to pause."
            return (tx, ty, status, overall_progress, False)

    def record_frame(self, frame_bgr: np.ndarray, pred_raw: Tuple[float, float], pred_calib: Tuple[float, float], pred_filtered: Tuple[float, float], model_res: Dict[str, Any], is_holding_trajectory_key: bool) -> bool:
        if not self.is_recording:
            return False
        should_record = False
        phase_label = ""
        if self.static_idx < len(self.static_points):
            if self.is_capturing_static:
                should_record = True
                curr_desc = self.static_points[self.static_idx]["desc"]
                phase_label = f"StaticFixation: {curr_desc}"
    def _fit_session_calibration(self) -> None:
        static_recs = [r for r in self.records if r.get("phase", "").startswith("StaticFixation") and r.get("face_detected", False)]
        if len(static_recs) < 15:
            return
        df = pd.DataFrame(static_recs)
        samples = []
        for p, group in df.groupby("phase"):
            samples.append({
                "target_x": float(group["target_gt_x"].iloc[0]),
                "target_y": float(group["target_gt_y"].iloc[0]),
                "yaw_deg": float(group["gaze_yaw_deg"].median()),
                "pitch_deg": float(group["gaze_pitch_deg"].median()),
                "head_yaw_deg": float(group["head_yaw_deg"].median()),
                "head_pitch_deg": float(group["head_pitch_deg"].median())
            })
        if len(samples) >= 5:
            calib = GazeCalibration(screen_w=self.screen_w, screen_h=self.screen_h, profile_path="mvp/calibration_profile.json")
            calib.fit(samples, l2_reg=1.0)
            self.session_calibration = calib
            if self.on_calibration_updated is not None:
                try:
                    self.on_calibration_updated(calib)
                except Exception:
                    pass

    def record_frame(self, frame_bgr: np.ndarray, pred_raw: Tuple[float, float], pred_calib: Tuple[float, float], pred_filtered: Tuple[float, float], model_res: Dict[str, Any], is_holding_trajectory_key: bool) -> bool:
        if not self.is_recording:
            return False
        should_record = False
        phase_label = ""
        if self.static_idx < len(self.static_points):
            if self.is_capturing_static:
                should_record = True
                curr_desc = self.static_points[self.static_idx]["desc"]
                phase_label = f"StaticFixation: {curr_desc}"
                self.static_recording_counter += 1
                if self.static_recording_counter >= self.frames_per_static_point:
                    self.is_capturing_static = False
                    self.static_recording_counter = 0
                    self.static_idx += 1
                    if self.static_idx >= len(self.static_points):
                        self._fit_session_calibration()
        else:
            if is_holding_trajectory_key:
                should_record = True
                phase_label = "CircleTrajectory"
        if not should_record:
            return False
        t_now = time.perf_counter()
        elapsed_sec = t_now - self.start_time
        if self.video_writer is not None:
            if frame_bgr.shape[1] != self.cam_w or frame_bgr.shape[0] != self.cam_h:
                write_frame = cv2.resize(frame_bgr, (self.cam_w, self.cam_h))
            else:
                write_frame = frame_bgr
            self.video_writer.write(write_frame)
        gt_x, gt_y = self.current_target_pt
        if self.session_calibration is not None and self.session_calibration.is_calibrated:
            p_calib_x, p_calib_y = self.session_calibration.predict(
                float(model_res.get("gaze_pitch_deg", 0.0)),
                float(model_res.get("gaze_yaw_deg", 0.0)),
                head_pitch_deg=float(model_res.get("head_pitch_deg", 0.0)),
                head_yaw_deg=float(model_res.get("head_yaw_deg", 0.0))
            )
            p_filt_x, p_filt_y = p_calib_x, p_calib_y
        else:
            p_calib_x, p_calib_y = pred_calib
            p_filt_x, p_filt_y = pred_filtered
        rec = {
            "video_frame_idx": len(self.records),
            "timestamp_sec": elapsed_sec,
            "phase": phase_label,
            "target_gt_x": gt_x,
            "target_gt_y": gt_y,
            "pred_raw_x": pred_raw[0],
            "pred_raw_y": pred_raw[1],
            "pred_calib_x": p_calib_x,
            "pred_calib_y": p_calib_y,
            "pred_filt_x": p_filt_x,
            "pred_filt_y": p_filt_y,
            "gaze_pitch_deg": model_res.get("gaze_pitch_deg", 0.0),
            "gaze_yaw_deg": model_res.get("gaze_yaw_deg", 0.0),
            "raw_pitch_deg": model_res.get("raw_pitch_deg", 0.0),
            "raw_yaw_deg": model_res.get("raw_yaw_deg", 0.0),
            "head_pitch_deg": model_res.get("head_pitch_deg", 0.0),
            "head_yaw_deg": model_res.get("head_yaw_deg", 0.0),
            "face_detected": model_res.get("face_detected", False)
        }
        self.records.append(rec)
        return True

    def stop_recording_and_evaluate(self) -> Dict[str, Any]:
        if not self.is_recording:
            return {"error": "Not recording"}
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if len(self.records) == 0:
            return {"error": "No frames recorded during benchmark"}
        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.output_dir, f"{self.session_id}.csv")
        parquet_path = os.path.join(self.output_dir, f"{self.session_id}.parquet")
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            pass
        valid_df = df[df["face_detected"]].copy()
        if len(valid_df) == 0:
            return {"error": "Face was never detected in benchmark"}
        eval_subset = valid_df[valid_df["phase"] == "CircleTrajectory"].copy()
        if len(eval_subset) < 10:
            eval_subset = valid_df.copy()
        if self.session_calibration is not None and self.session_calibration.is_calibrated:
            eval_calib_list = []
            for _, r in eval_subset.iterrows():
                px, py = self.session_calibration.predict(
                    float(r["gaze_pitch_deg"]), float(r["gaze_yaw_deg"]),
                    head_pitch_deg=float(r["head_pitch_deg"]),
                    head_yaw_deg=float(r["head_yaw_deg"])
                )
                eval_calib_list.append([px, py])
            eval_calib_arr = np.array(eval_calib_list, dtype=np.float64)
            calib_pts = eval_calib_arr
            filt_pts = eval_calib_arr
        else:
            calib_pts = eval_subset[["pred_calib_x", "pred_calib_y"]].to_numpy()
            filt_pts = eval_subset[["pred_filt_x", "pred_filt_y"]].to_numpy()
        gt_pts = eval_subset[["target_gt_x", "target_gt_y"]].to_numpy()
        raw_pts = eval_subset[["pred_raw_x", "pred_raw_y"]].to_numpy()
        diff_filt = filt_pts - gt_pts
        diff_calib = calib_pts - gt_pts
        diff_raw = raw_pts - gt_pts
        err_filt_px = np.linalg.norm(diff_filt, axis=-1)
        err_calib_px = np.linalg.norm(diff_calib, axis=-1)
        err_raw_px = np.linalg.norm(diff_raw, axis=-1)
        err_filt_mm = err_filt_px / ((self.px_per_mm_x + self.px_per_mm_y) / 2.0)
        cx = self.screen_w / 2.0; cy = self.screen_h / 2.0
        v_gt = np.stack([gt_pts[:, 0] - cx, gt_pts[:, 1] - cy, -np.full(len(gt_pts), self.dist_px)], axis=-1)
        v_pred = np.stack([filt_pts[:, 0] - cx, filt_pts[:, 1] - cy, -np.full(len(filt_pts), self.dist_px)], axis=-1)
        norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)
        norm_pred = v_pred / np.linalg.norm(v_pred, axis=-1, keepdims=True)
        dot_prod = np.clip(np.sum(norm_gt * norm_pred, axis=-1), -1.0, 1.0)
        ang_errors_deg = np.degrees(np.arccos(dot_prod))
        same_dir_x = (np.sign(filt_pts[:, 0] - cx) == np.sign(gt_pts[:, 0] - cx))
        same_dir_y = (np.sign(filt_pts[:, 1] - cy) == np.sign(gt_pts[:, 1] - cy))
        acc_dir_x = float(np.mean(same_dir_x) * 100.0)
        acc_dir_y = float(np.mean(same_dir_y) * 100.0)
        acc_dir_both = float(np.mean(same_dir_x & same_dir_y) * 100.0)
        corr_x = float(np.corrcoef(filt_pts[:, 0], gt_pts[:, 0])[0, 1]) if len(filt_pts) > 2 else 0.0
        corr_y = float(np.corrcoef(filt_pts[:, 1], gt_pts[:, 1])[0, 1]) if len(filt_pts) > 2 else 0.0
        results = {
            "session_id": self.session_id,
            "total_frames": len(df),
            "valid_frames": len(valid_df),
            "duration_sec": float(df["timestamp_sec"].iloc[-1]),
            "screen_err_mean_px": float(np.mean(err_filt_px)),
            "screen_err_median_px": float(np.median(err_filt_px)),
            "screen_err_std_px": float(np.std(err_filt_px)),
            "screen_err_mean_mm": float(np.mean(err_filt_mm)),
            "screen_err_median_mm": float(np.median(err_filt_mm)),
            "angular_err_mean_deg": float(np.mean(ang_errors_deg)),
            "angular_err_median_deg": float(np.median(ang_errors_deg)),
            "angular_err_std_deg": float(np.std(ang_errors_deg)),
            "mae_x_px": float(np.mean(np.abs(diff_filt[:, 0]))),
            "mae_y_px": float(np.mean(np.abs(diff_filt[:, 1]))),
            "bias_x_px": float(np.mean(diff_filt[:, 0])),
            "bias_y_px": float(np.mean(diff_filt[:, 1])),
            "corr_x": corr_x,
            "corr_y": corr_y,
            "dir_accuracy_x_pct": acc_dir_x,
            "dir_accuracy_y_pct": acc_dir_y,
            "dir_accuracy_both_pct": acc_dir_both,
            "unfiltered_err_mean_px": float(np.mean(err_calib_px)),
            "uncalibrated_err_mean_px": float(np.mean(err_raw_px)),
            "video_path": os.path.join(self.output_dir, f"{self.session_id}.mp4"),
            "csv_path": csv_path,
            "parquet_path": parquet_path,
            "report_path": os.path.join(self.output_dir, f"{self.session_id}_report.md"),
            "plot_path": os.path.join(self.output_dir, f"{self.session_id}_trajectory.png")
        }
        self._generate_report(results, df, valid_df)
        self._generate_plot(results, valid_df)
        return results

    def _generate_plot(self, results: Dict[str, Any], valid_df: pd.DataFrame) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
            ax.set_xlim(0, self.screen_w)
            ax.set_ylim(self.screen_h, 0)
            ax.set_title(f"Benchmark Gaze Tracking: GT vs Predicted Trajectory\nAngular Error: {results['angular_err_mean_deg']:.2f} deg | Screen: {results['screen_err_mean_mm']:.1f} mm ({results['screen_err_mean_px']:.1f} px)", fontsize=11)
            ax.set_xlabel("Screen X (px)")
            ax.set_ylabel("Screen Y (px)")
            gt_x = valid_df["target_gt_x"].to_numpy()
            gt_y = valid_df["target_gt_y"].to_numpy()
            pred_x = valid_df["pred_filt_x"].to_numpy()
            pred_y = valid_df["pred_filt_y"].to_numpy()
            ax.plot(gt_x, gt_y, color="green", linewidth=2.5, label="Ground Truth Path", alpha=0.85)
            ax.scatter(gt_x[::10], gt_y[::10], color="darkgreen", s=35, zorder=4)
            ax.plot(pred_x, pred_y, color="blue", linewidth=1.2, label="Predicted (Filtered)", alpha=0.7)
            ax.scatter(pred_x[::10], pred_y[::10], color="darkblue", s=15, alpha=0.5, zorder=3)
            ax.axvline(self.screen_w / 2.0, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(self.screen_h / 2.0, color="gray", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(results["plot_path"])
            plt.close(fig)
        except Exception:
            pass

    def _generate_report(self, res: Dict[str, Any], all_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        rep_content = f"""=== ПАСПОРТ ФИНАЛЬНОГО ТЕСТОВОГО БЕНЧМАРКА (TEST DATASET BENCHMARK) ===

> **СТРОГИЙ СТАТУС:** Финальный тестовый набор данных пользователя.
> **ЗАПРЕТ:** Обучение моделей, подгонка гиперпараметров или дообучение на данных сессии `{res['session_id']}` строго запрещены.

---

== РАЗДЕЛ 1. СВОДНЫЕ ХАРАКТЕРИСТИКИ СЕССИИ
- **Идентификатор сессии:** `{res['session_id']}`
- **Длительность теста:** `{res['duration_sec']:.2f}` с
- **Всего кадров в датасете:** `{res['total_frames']}`
- **Кадров с обнаруженным лицом:** `{res['valid_frames']}` ({res['valid_frames']/max(1,res['total_frames'])*100.0:.1f}%)
- **Разрешение дисплея:** `{self.screen_w} x {self.screen_h}` px (`{self.physical_screen_w_mm:.0f} x {self.physical_screen_h_mm:.0f}` мм)
- **Файл видеозаписи лица (Ground Truth Source):** `{res['video_path']}`
- **Файл таблицы разметки:** `{res['csv_path']}`
- **Файл графика траектории:** `{res['plot_path']}`

---

== РАЗДЕЛ 2. ЦЕЛЕВЫЕ МЕТРИКИ ТОЧНОСТИ

| Метрика | Значение | Норматив ТЗ / Стандарт | Вердикт |
|---|---|---|---|
| **Угловая ошибка (Mean)** | **{res['angular_err_mean_deg']:.2f} deg** | $\\le 5.20^\\circ$ | {'PASS' if res['angular_err_mean_deg'] <= 5.20 else 'FAIL'} |
| **Угловая ошибка (Median)** | **{res['angular_err_median_deg']:.2f} deg** | - | - |
| **Экранная ошибка (Mean, px)** | **{res['screen_err_mean_px']:.1f} px** | - | - |
| **Экранная ошибка (Mean, mm)** | **{res['screen_err_mean_mm']:.1f} mm** | $\\le 50.0\\text{{ мм}}$ | {'PASS' if res['screen_err_mean_mm'] <= 50.0 else 'FAIL'} |
| **Экранная ошибка (Median, mm)** | **{res['screen_err_median_mm']:.1f} mm** | - | - |
| **Корреляция по оси X (R_x)** | **{res['corr_x']:.3f}** | $\\ge 0.70$ | {'PASS' if res['corr_x'] >= 0.70 else 'FAIL'} |
| **Корреляция по оси Y (R_y)** | **{res['corr_y']:.3f}** | $\\ge 0.70$ | {'PASS' if res['corr_y'] >= 0.70 else 'FAIL'} |
| **Точность знака по X (Лево/Право)** | **{res['dir_accuracy_x_pct']:.1f}%** | $\\ge 80.0\\%$ | {'PASS' if res['dir_accuracy_x_pct'] >= 80.0 else 'FAIL'} |
| **Точность знака по Y (Верх/Низ)** | **{res['dir_accuracy_y_pct']:.1f}%** | $\\ge 80.0\\%$ | {'PASS' if res['dir_accuracy_y_pct'] >= 80.0 else 'FAIL'} |
| **Точность направления (Оба квадранта)** | **{res['dir_accuracy_both_pct']:.1f}%** | $\\ge 70.0\\%$ | {'PASS' if res['dir_accuracy_both_pct'] >= 70.0 else 'FAIL'} |

---

== РАЗДЕЛ 3. ПОКОМПОНЕНТНЫЕ НЕВЯЗКИ
- **MAE по оси X:** `{res['mae_x_px']:.1f}` px (`{res['mae_x_px'] / self.px_per_mm_x:.1f}` мм)
- **MAE по оси Y:** `{res['mae_y_px']:.1f}` px (`{res['mae_y_px'] / self.px_per_mm_y:.1f}` мм)
- **Систематический сдвиг (Bias X):** `{res['bias_x_px']:+.1f}` px
- **Систематический сдвиг (Bias Y):** `{res['bias_y_px']:+.1f}` px
- **Ошибка без темпорального фильтра:** `{res['unfiltered_err_mean_px']:.1f}` px
- **Ошибка до калибровки:** `{res['uncalibrated_err_mean_px']:.1f}` px

---
"""
        with open(res["report_path"], "w", encoding="utf-8") as f:
            f.write(rep_content)
