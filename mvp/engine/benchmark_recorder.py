import os, sys, math, time, cv2, re, psutil, torch, numpy as np, pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
try:
    from engine.calibration import GazeCalibration
except ImportError:
    from mvp.engine.calibration import GazeCalibration

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-6 or sy < 1e-6:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return 0.0
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    if ss_tot < 1e-6:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - (ss_res / ss_tot))

class BenchmarkRecorder:
    def __init__(
        self,
        output_dir: str = "mvp/benchmarks",
        screen_w: int = 2880,
        screen_h: int = 1800,
        cam_w: int = 1280,
        cam_h: int = 720,
        fps: float = 30.0,
        user_distance_mm: float = 600.0,
        physical_screen_w_mm: float = 300.0,
        physical_screen_h_mm: float = 190.0,
        model_name: str = "l2cs"
    ) -> None:
        self.output_dir = output_dir
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.target_fps = fps
        self.user_distance_mm = user_distance_mm
        self.physical_screen_w_mm = physical_screen_w_mm
        self.physical_screen_h_mm = physical_screen_h_mm
        self.model_name = model_name
        self.process = psutil.Process()

        self.px_per_mm_x = self.screen_w / max(self.physical_screen_w_mm, 1.0)
        self.px_per_mm_y = self.screen_h / max(self.physical_screen_h_mm, 1.0)
        self.dist_px = self.user_distance_mm * ((self.px_per_mm_x + self.px_per_mm_y) / 2.0)
        os.makedirs(self.output_dir, exist_ok=True)

        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.session_id = ""
        self.session_timestamp = ""
        self.temp_video_path = ""
        self.records: List[Dict[str, Any]] = []
        self.start_time = 0.0

        # Phase 1: Static Calibration Points
        self.static_points = self._build_static_points()
        self.static_idx = 0
        self.frames_per_static_point = 20
        self.static_recording_counter = 0
        self.is_capturing_static = False
        self.waiting_for_key_release = False

        # Phase 2: Dynamic Moving Trajectory Test
        self.stage = "calibration"  # "calibration" -> "dynamic_test" -> "finished"
        self.trajectory_angle = 0.0
        self.trajectory_speed = (2.0 * math.pi) / 12.0  # complete circle in 12s of active hold
        self.trajectory_radius = float(min(self.screen_w, self.screen_h) * 0.35)
        self.trajectory_center = (float(self.screen_w / 2.0), float(self.screen_h / 2.0))
        self.trajectory_progress = 0.0
        self.is_holding_trajectory_key = False

        self.is_capturing_active = False
        self.current_target_pt = (self.screen_w / 2.0, self.screen_h / 2.0)
        self.session_calibration: Optional[GazeCalibration] = None
        self.on_calibration_updated: Optional[Callable[[GazeCalibration], None]] = None

    def _build_static_points(self) -> List[Dict[str, Any]]:
        pts = []
        nx, ny = 3, 3
        margin_x = 0.15
        xs = np.linspace(margin_x * self.screen_w, (1.0 - margin_x) * self.screen_w, nx)
        ys = np.linspace(0.18 * self.screen_h, 0.86 * self.screen_h, ny)
        for r_idx, y in enumerate(ys):
            for c_idx, x in enumerate(xs):
                pts.append({
                    "target_x": float(x),
                    "target_y": float(y),
                    "desc": f"Сетка ({c_idx+1},{r_idx+1})"
                })
        extreme_points = [
            (float(0.06 * self.screen_w), float(0.50 * self.screen_h), "Край слева"),
            (float(0.94 * self.screen_w), float(0.50 * self.screen_h), "Край справа"),
            (float(0.50 * self.screen_w), float(0.12 * self.screen_h), "Край сверху"),
            (float(0.50 * self.screen_w), float(0.90 * self.screen_h), "Край снизу"),
        ]
        for ex_x, ex_y, name in extreme_points:
            pts.append({
                "target_x": ex_x,
                "target_y": ex_y,
                "desc": name
            })
        return pts

    def start_recording(self, model_name: Optional[str] = None) -> str:
        if model_name:
            self.model_name = model_name
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"test_{self.session_timestamp}"
        self.temp_video_path = os.path.join(self.output_dir, f"_temp_{self.session_timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, self.target_fps, (self.cam_w, self.cam_h))
        self.records = []
        self.static_idx = 0
        self.static_recording_counter = 0
        self.is_capturing_static = False
        self.waiting_for_key_release = False
        self.stage = "calibration"
        self.trajectory_angle = 0.0
        self.trajectory_progress = 0.0
        self.is_holding_trajectory_key = False
        self.is_capturing_active = False

        self.start_time = time.perf_counter()
        self.is_recording = True

        first_pt = self.static_points[0]
        self.current_target_pt = (first_pt["target_x"], first_pt["target_y"])
        return self.session_id

    def trigger_space_press(self) -> None:
        if not self.is_recording:
            return
        if self.stage == "calibration":
            if self.waiting_for_key_release:
                # User has not released space yet from previous point
                return
            if self.static_idx < len(self.static_points) and not self.is_capturing_static:
                self.is_capturing_static = True
                self.static_recording_counter = 0
        elif self.stage == "dynamic_test":
            self.is_holding_trajectory_key = True

    def trigger_space_release(self) -> None:
        if not self.is_recording:
            return
        if self.stage == "calibration":
            self.waiting_for_key_release = False
        elif self.stage == "dynamic_test":
            self.is_holding_trajectory_key = False

    def update_state(self, is_holding_trajectory_key: bool = False, dt: float = 0.0) -> Tuple[float, float, str, float, bool]:
        if not self.is_recording:
            return (self.screen_w / 2.0, self.screen_h / 2.0, "Ожидание запуска", 0.0, False)

        total_static_pts = len(self.static_points)
        total_steps = total_static_pts + 1  # static points + dynamic pursuit trajectory

        if self.stage == "calibration":
            if self.static_idx < total_static_pts:
                curr_pt = self.static_points[self.static_idx]
                tx = curr_pt["target_x"]
                ty = curr_pt["target_y"]
                self.current_target_pt = (tx, ty)

                pt_fraction = float(self.static_recording_counter) / float(self.frames_per_static_point)
                overall_progress = (float(self.static_idx) + pt_fraction) / float(total_steps)

                if self.is_capturing_static:
                    self.is_capturing_active = True
                    status = (f"[КАЛИБРОВКА] Точка {self.static_idx + 1}/{total_static_pts} "
                              f"[{curr_pt['desc']}]: ЗАПИСЬ [{self.static_recording_counter}/{self.frames_per_static_point}] — НЕ СМЕЩАЙТЕ ВЗГЛЯД")
                elif self.waiting_for_key_release:
                    self.is_capturing_active = False
                    status = (f"[КАЛИБРОВКА] Точка {self.static_idx} записана! "
                              f"ОТПУСТИТЕ [ПРОБЕЛ] и переведите взгляд на точку {self.static_idx + 1}")
                else:
                    self.is_capturing_active = False
                    status = (f"[КАЛИБРОВКА] Точка {self.static_idx + 1}/{total_static_pts} "
                              f"[{curr_pt['desc']}]: Зафиксируйте взгляд на цели и НАЖМИТЕ [ПРОБЕЛ]")

                return (tx, ty, status, overall_progress, False)
            else:
                # Finished static calibration points!
                self._fit_session_calibration()
                self.stage = "dynamic_test"
                self.waiting_for_key_release = False

        if self.stage == "dynamic_test":
            is_active = is_holding_trajectory_key or self.is_holding_trajectory_key
            cx, cy = self.trajectory_center
            if is_active:
                self.trajectory_angle += self.trajectory_speed * dt
                if self.trajectory_angle >= 2.0 * math.pi:
                    self.trajectory_angle = 2.0 * math.pi
                    self.trajectory_progress = 1.0
                    self.stage = "finished"
                    return (cx, cy, "Тест завершен! Формирование отчета...", 1.0, True)

            self.trajectory_progress = min(1.0, self.trajectory_angle / (2.0 * math.pi))
            tx = cx + self.trajectory_radius * math.cos(self.trajectory_angle - math.pi / 2.0)
            ty = cy + self.trajectory_radius * math.sin(self.trajectory_angle - math.pi / 2.0)
            self.current_target_pt = (float(tx), float(ty))
            self.is_capturing_active = is_active

            overall_progress = (float(total_static_pts) + self.trajectory_progress) / float(total_steps)
            pct = int(self.trajectory_progress * 100)

            if is_active:
                status = f"[ЭТАП 2: ДИНАМИЧЕСКИЙ ТЕСТ] ДВИЖЕНИЕ ТОЧКИ ({pct}%): Следите за точкой глазами"
            else:
                status = f"[ПАУЗА ({pct}%)] УДЕРЖИВАЙТЕ [SHIFT], [G] или [ПРОБЕЛ] для движения точки"

            return (tx, ty, status, overall_progress, False)

        # Finished
        return (self.screen_w / 2.0, self.screen_h / 2.0, "Тест завершен! Сохранение результатов...", 1.0, True)

    def _fit_session_calibration(self) -> None:
        calib_recs = [r for r in self.records if r.get("stage") == "calibration" and r.get("face_detected", 0) == 1]
        if len(calib_recs) < 10:
            return

        df_calib = pd.DataFrame(calib_recs)
        samples = []
        for p_idx, grp in df_calib.groupby("point_idx"):
            entry = {
                "target_x": float(grp["target_gt_x"].iloc[0]),
                "target_y": float(grp["target_gt_y"].iloc[0]),
                "yaw_deg": float(grp["gaze_yaw_deg"].median()),
                "pitch_deg": float(grp["gaze_pitch_deg"].median()),
                "head_yaw_deg": float(grp["head_yaw_deg"].median()),
                "head_pitch_deg": float(grp["head_pitch_deg"].median())
            }
            if "ix" in grp and grp["ix"].notna().all() and grp["ix"].iloc[0] is not None:
                entry["ix"] = float(grp["ix"].median())
                entry["iy"] = float(grp["iy"].median())
                entry["eye_diff_x"] = float(grp["eye_diff_x"].median()) if "eye_diff_x" in grp else 0.0
                entry["eye_diff_y"] = float(grp["eye_diff_y"].median()) if "eye_diff_y" in grp else 0.0
            samples.append(entry)

        if len(samples) >= 5:
            calib = GazeCalibration(
                screen_w=self.screen_w,
                screen_h=self.screen_h,
                model_name=self.model_name
            )
            calib.fit(samples, l2_reg=1.0)
            self.session_calibration = calib
            if self.on_calibration_updated is not None:
                try:
                    self.on_calibration_updated(calib)
                except Exception:
                    pass

    def record_frame(
        self,
        frame_bgr: np.ndarray,
        pred_raw: Tuple[float, float],
        pred_calib: Tuple[float, float],
        pred_filtered: Tuple[float, float],
        model_res: Dict[str, Any],
        fps: float = 30.0,
        mirror_camera: bool = True,
        is_holding_trajectory_key: bool = False
    ) -> bool:
        if not self.is_recording:
            return False

        t_now = time.perf_counter()
        elapsed_sec = t_now - self.start_time

        # Target GT
        gt_x, gt_y = self.current_target_pt

        # Determine if this frame should be captured into dataset
        should_record = False
        current_stage = self.stage
        point_idx = -1
        point_desc = ""

        if self.stage == "calibration":
            if self.static_idx < len(self.static_points):
                curr_desc = self.static_points[self.static_idx]["desc"]
                point_desc = curr_desc
                point_idx = self.static_idx
                if self.is_capturing_static:
                    should_record = True
                    self.static_recording_counter += 1
                    if self.static_recording_counter >= self.frames_per_static_point:
                        self.static_recording_counter = 0
                        self.is_capturing_static = False
                        self.static_idx += 1
                        self.waiting_for_key_release = True
                        if self.static_idx >= len(self.static_points):
                            self._fit_session_calibration()
                            self.stage = "dynamic_test"
                            self.waiting_for_key_release = False
            else:
                self._fit_session_calibration()
                self.stage = "dynamic_test"
                self.waiting_for_key_release = False

        elif self.stage == "dynamic_test":
            point_desc = f"Траектория ({int(self.trajectory_progress * 100)}%)"
            point_idx = 100
            if is_holding_trajectory_key or self.is_holding_trajectory_key:
                should_record = True

        # Telemetry info
        ram_gb = float(psutil.virtual_memory().used / (1024 ** 3))
        proc_mb = float(self.process.memory_info().rss / (1024 ** 2))
        vram_mb = 0.0
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram_mb = float((total_b - free_b) / (1024 ** 2))

        # Instant error metrics (Calibrated)
        curr_pred_calib = pred_calib if self.session_calibration is None else self.session_calibration.predict(
            model_res.get("gaze_pitch_deg", 0.0),
            model_res.get("gaze_yaw_deg", 0.0),
            head_pitch_deg=model_res.get("head_pitch_deg", 0.0),
            head_yaw_deg=model_res.get("head_yaw_deg", 0.0),
            ix=model_res.get("ix"),
            iy=model_res.get("iy"),
            eye_diff_x=model_res.get("eye_diff_x"),
            eye_diff_y=model_res.get("eye_diff_y")
        )
        curr_err_px = float(math.sqrt((curr_pred_calib[0] - gt_x)**2 + (curr_pred_calib[1] - gt_y)**2))
        curr_err_mm = float(curr_err_px / ((self.px_per_mm_x + self.px_per_mm_y) / 2.0))

        # 3D Angular error for calibrated prediction
        cx = self.screen_w / 2.0
        cy = self.screen_h / 2.0
        tgt_x_mm = (gt_x - cx) / self.px_per_mm_x
        tgt_y_mm = (gt_y - cy) / self.px_per_mm_y
        v_gt = np.array([tgt_x_mm, tgt_y_mm, -self.user_distance_mm], dtype=np.float64)
        n_gt = v_gt / np.linalg.norm(v_gt)

        pred_x_mm = (curr_pred_calib[0] - cx) / self.px_per_mm_x
        pred_y_mm = (curr_pred_calib[1] - cy) / self.px_per_mm_y
        v_pred = np.array([pred_x_mm, pred_y_mm, -self.user_distance_mm], dtype=np.float64)
        n_pred = v_pred / np.linalg.norm(v_pred)
        calib_ang_err = float(np.degrees(np.arccos(np.clip(np.dot(n_gt, n_pred), -1.0, 1.0))))

        # Build Annotated Telemetry Video Frame
        vis = frame_bgr.copy()
        if mirror_camera:
            vis = cv2.flip(vis, 1)

        h_cam, w_cam = vis.shape[:2]
        face_det = model_res.get("face_detected", False)
        if face_det and "face_bbox" in model_res and model_res["face_bbox"]:
            fx, fy, bw, bh = model_res["face_bbox"]
            draw_fx = w_cam - 1.0 - (fx + bw) if mirror_camera else fx
            cv2.rectangle(vis, (int(draw_fx), int(fy)), (int(draw_fx + bw), int(fy + bh)), (0, 255, 0), 2)

            lm = model_res.get("landmarks") or {}
            re_pt = lm.get("right_eye")
            le_pt = lm.get("left_eye")
            p_deg = model_res.get("gaze_pitch_deg", 0.0)
            y_deg = model_res.get("gaze_yaw_deg", 0.0)
            p_rad = math.radians(p_deg)
            y_rad = math.radians(y_deg)

            arrow_len = 65.0
            dx = arrow_len * math.sin(y_rad) if mirror_camera else -arrow_len * math.sin(y_rad)
            dy = arrow_len * math.sin(p_rad)

            if re_pt and le_pt:
                if mirror_camera:
                    draw_rex, draw_rey = int(w_cam - 1.0 - re_pt[0]), int(re_pt[1])
                    draw_lex, draw_ley = int(w_cam - 1.0 - le_pt[0]), int(le_pt[1])
                else:
                    draw_rex, draw_rey = int(re_pt[0]), int(re_pt[1])
                    draw_lex, draw_ley = int(le_pt[0]), int(le_pt[1])

                cv2.circle(vis, (draw_rex, draw_rey), 4, (255, 128, 0), -1)
                cv2.circle(vis, (draw_lex, draw_ley), 4, (255, 128, 0), -1)
                cv2.arrowedLine(vis, (draw_rex, draw_rey), (int(draw_rex + dx), int(draw_rey + dy)), (0, 0, 255), 2, tipLength=0.25)
                cv2.arrowedLine(vis, (draw_lex, draw_ley), (int(draw_lex + dx), int(draw_ley + dy)), (0, 0, 255), 2, tipLength=0.25)

        # Draw HUD Telemetry Box
        overlay = vis.copy()
        box_w, box_h = 440, 215
        cv2.rectangle(overlay, (12, 12), (12 + box_w, 12 + box_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.82, vis, 0.18, 0, vis)
        cv2.rectangle(vis, (12, 12), (12 + box_w, 12 + box_h), (50, 60, 80), 1)

        stage_name = "КАЛИБРОВКА" if current_stage == "calibration" else "ДИНАМИЧЕСКИЙ ТЕСТ"
        rec_status = "ЗАПИСЬ КАДРА" if should_record else "ОЖИДАНИЕ КЛАВИШИ"
        rec_color = (0, 255, 0) if should_record else (0, 165, 255)

        hud_lines = [
            (f"МОДЕЛЬ: {self.model_name}", (255, 215, 0)),
            (f"ЭТАП: {stage_name} | {rec_status}", rec_color),
            (f"ЦЕЛЬ: {point_desc} (X={int(gt_x)}, Y={int(gt_y)})", (255, 255, 255)),
            (f"FPS: {fps:.1f} | Задержка инференса: {model_res.get('latency_model_ms', 0.0):.2f} мс", (255, 255, 255)),
            (f"RAM: {ram_gb:.2f} ГБ (Процесс: {proc_mb:.0f} МБ) | VRAM: {vram_mb:.0f} МБ", (0, 255, 255)),
            (f"ВЗГЛЯД: Pitch={model_res.get('gaze_pitch_deg', 0.0):+.1f}° | Yaw={model_res.get('gaze_yaw_deg', 0.0):+.1f}°", (200, 200, 200)),
            (f"ОШИБКА: {calib_ang_err:.2f}° | {curr_err_mm:.1f} мм ({curr_err_px:.1f} px)", (0, 255, 128) if curr_err_mm < 50 else (0, 128, 255)),
            (f"ПРЕДСКАЗАНИЕ: X={int(curr_pred_calib[0])} Y={int(curr_pred_calib[1])}", (200, 200, 200))
        ]

        y_offset = 34
        for text, col in hud_lines:
            cv2.putText(vis, text, (22, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
            y_offset += 23

        # Write to video
        if self.video_writer is not None:
            if vis.shape[1] != self.cam_w or vis.shape[0] != self.cam_h:
                write_frame = cv2.resize(vis, (self.cam_w, self.cam_h))
            else:
                write_frame = vis
            self.video_writer.write(write_frame)

        # Store record if active
        if should_record:
            rec = {
                "frame_idx": len(self.records),
                "timestamp_sec": elapsed_sec,
                "stage": current_stage,
                "point_idx": point_idx,
                "point_desc": point_desc,
                "target_gt_x": gt_x,
                "target_gt_y": gt_y,
                "pred_raw_x": pred_raw[0],
                "pred_raw_y": pred_raw[1],
                "pred_calib_x": curr_pred_calib[0],
                "pred_calib_y": curr_pred_calib[1],
                "pred_filt_x": pred_filtered[0],
                "pred_filt_y": pred_filtered[1],
                "calib_screen_err_px": curr_err_px,
                "calib_screen_err_mm": curr_err_mm,
                "calib_ang_err_deg": calib_ang_err,
                "gaze_pitch_deg": model_res.get("gaze_pitch_deg", 0.0),
                "gaze_yaw_deg": model_res.get("gaze_yaw_deg", 0.0),
                "head_pitch_deg": model_res.get("head_pitch_deg", 0.0),
                "head_yaw_deg": model_res.get("head_yaw_deg", 0.0),
                "ix": model_res.get("ix"),
                "iy": model_res.get("iy"),
                "eye_diff_x": model_res.get("eye_diff_x"),
                "eye_diff_y": model_res.get("eye_diff_y"),
                "face_detected": int(bool(face_det)),
                "fps": fps,
                "ram_gb": ram_gb,
                "proc_ram_mb": proc_mb,
                "vram_mb": vram_mb,
                "latency_model_ms": model_res.get("latency_model_ms", 0.0)
            }
            self.records.append(rec)
            return True

        return False

    def stop_recording_and_evaluate(self) -> Dict[str, Any]:
        if not self.is_recording:
            return {"error": "Not recording"}
        self.is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if len(self.records) == 0:
            if os.path.exists(self.temp_video_path):
                try:
                    os.remove(self.temp_video_path)
                except Exception:
                    pass
            return {"error": "No frames recorded during accuracy test"}

        df = pd.DataFrame(self.records)
        valid_df = df[df["face_detected"] == 1].copy()
        if len(valid_df) == 0:
            return {"error": "Face was never detected during test"}

        # 1. Compute Raw 3D Angular Errors across all valid frames
        cx = self.screen_w / 2.0
        cy = self.screen_h / 2.0
        gt_x = valid_df["target_gt_x"].to_numpy()
        gt_y = valid_df["target_gt_y"].to_numpy()

        tgt_x_mm = (gt_x - cx) / self.px_per_mm_x
        tgt_y_mm = (gt_y - cy) / self.px_per_mm_y
        v_gt = np.stack([tgt_x_mm, tgt_y_mm, -np.full(len(gt_x), self.user_distance_mm)], axis=-1)
        norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)

        # 2. Fit / verify Calibration Model
        if self.session_calibration is None:
            self._fit_session_calibration()

        calib_model = self.session_calibration or GazeCalibration(
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            model_name=self.model_name
        )

        p_deg = valid_df["gaze_pitch_deg"].to_numpy()
        y_deg = valid_df["gaze_yaw_deg"].to_numpy()
        rp = float(calib_model.resting_pitch_deg if calib_model.resting_pitch_deg is not None else 0.0)
        ry = float(calib_model.resting_yaw_deg if calib_model.resting_yaw_deg is not None else 0.0)
        y_rel = np.radians(y_deg - ry)
        p_rel = np.radians(p_deg - rp)

        # In camera/screen frame (+X right, +Y down, -Z looking at screen):
        gx = np.sin(y_rel) * np.cos(p_rel)
        gy = np.sin(p_rel)
        gz = -np.cos(y_rel) * np.cos(p_rel)
        v_pred_raw = np.stack([gx, gy, gz], axis=-1)
        norm_pred_raw = v_pred_raw / np.linalg.norm(v_pred_raw, axis=-1, keepdims=True)

        raw_dot = np.clip(np.sum(norm_gt * norm_pred_raw, axis=-1), -1.0, 1.0)
        valid_df["raw_ang_err_deg"] = np.degrees(np.arccos(raw_dot))

        # Compute Raw Screen Errors using consistent uncalibrated geometric projection
        raw_eval_x = []
        raw_eval_y = []
        for _, row in valid_df.iterrows():
            rx_val, ry_val = calib_model.predict_uncalibrated(row["gaze_pitch_deg"], row["gaze_yaw_deg"])
            raw_eval_x.append(rx_val)
            raw_eval_y.append(ry_val)
        valid_df["pred_raw_x"] = np.array(raw_eval_x)
        valid_df["pred_raw_y"] = np.array(raw_eval_y)

        diff_raw_x = valid_df["pred_raw_x"].to_numpy() - gt_x
        diff_raw_y = valid_df["pred_raw_y"].to_numpy() - gt_y
        raw_scr_px = np.sqrt(diff_raw_x**2 + diff_raw_y**2)
        valid_df["raw_scr_err_px"] = raw_scr_px
        valid_df["raw_scr_err_mm"] = raw_scr_px * (self.physical_screen_w_mm / self.screen_w)

        calib_preds = []
        for _, row in valid_df.iterrows():
            px_val, py_val = calib_model.predict(
                row["gaze_pitch_deg"],
                row["gaze_yaw_deg"],
                head_pitch_deg=row["head_pitch_deg"],
                head_yaw_deg=row["head_yaw_deg"],
                ix=row.get("ix"),
                iy=row.get("iy"),
                eye_diff_x=row.get("eye_diff_x"),
                eye_diff_y=row.get("eye_diff_y")
            )
            calib_preds.append((px_val, py_val))
        calib_preds = np.array(calib_preds)
        valid_df["pred_calib_eval_x"] = calib_preds[:, 0]
        valid_df["pred_calib_eval_y"] = calib_preds[:, 1]

        diff_calib = calib_preds - np.column_stack([gt_x, gt_y])
        calib_scr_px = np.linalg.norm(diff_calib, axis=-1)
        calib_scr_mm = calib_scr_px * (self.physical_screen_w_mm / self.screen_w)
        valid_df["calib_scr_err_px"] = calib_scr_px
        valid_df["calib_scr_err_mm"] = calib_scr_mm

        v_pred_calib = np.stack([(calib_preds[:, 0] - cx) / self.px_per_mm_x,
                                (calib_preds[:, 1] - cy) / self.px_per_mm_y,
                                -np.full(len(calib_preds), self.user_distance_mm)], axis=-1)
        norm_pred_calib = v_pred_calib / np.linalg.norm(v_pred_calib, axis=-1, keepdims=True)
        calib_dot = np.clip(np.sum(norm_gt * norm_pred_calib, axis=-1), -1.0, 1.0)
        valid_df["calib_ang_err_deg"] = np.degrees(np.arccos(calib_dot))

        # 3. Partition into Stage 1 (Calibration Points) and Stage 2 (Dynamic Test)
        df_calib_stage = valid_df[valid_df["stage"] == "calibration"].copy()
        df_test_stage = valid_df[valid_df["stage"] == "dynamic_test"].copy()

        # Primary evaluation dataset:
        # If dynamic test was performed, metrics are evaluated on dynamic test!
        # If user stopped before dynamic test, fallback to calibration stage.
        eval_df = df_test_stage if len(df_test_stage) >= 15 else valid_df
        eval_source_name = "ДИНАМИЧЕСКИЙ ТЕСТ ТРАЕКТОРИИ" if len(df_test_stage) >= 15 else "СТАТИЧЕСКИЕ ТОЧКИ КАЛИБРОВКИ"

        # Global Evaluation Metrics
        mean_calib_ang = float(eval_df["calib_ang_err_deg"].mean())
        median_calib_ang = float(eval_df["calib_ang_err_deg"].median())
        mean_raw_ang = float(eval_df["raw_ang_err_deg"].mean())
        median_raw_ang = float(eval_df["raw_ang_err_deg"].median())

        mean_calib_mm = float(eval_df["calib_scr_err_mm"].mean())
        median_calib_mm = float(eval_df["calib_scr_err_mm"].median())
        mean_calib_px = float(eval_df["calib_scr_err_px"].mean())

        mean_raw_mm = float(eval_df["raw_scr_err_mm"].mean())
        mean_raw_px = float(eval_df["raw_scr_err_px"].mean())

        corr_x_calib = safe_corr(eval_df["pred_calib_eval_x"].to_numpy(), eval_df["target_gt_x"].to_numpy())
        corr_y_calib = safe_corr(eval_df["pred_calib_eval_y"].to_numpy(), eval_df["target_gt_y"].to_numpy())
        corr_x_raw = safe_corr(eval_df["pred_raw_x"].to_numpy(), eval_df["target_gt_x"].to_numpy())
        corr_y_raw = safe_corr(eval_df["pred_raw_y"].to_numpy(), eval_df["target_gt_y"].to_numpy())

        r2_fit = float(calib_model.r2_score)

        # Directional Sign Accuracy
        gt_dx = eval_df["target_gt_x"].to_numpy() - cx
        gt_dy = eval_df["target_gt_y"].to_numpy() - cy
        pred_dx = eval_df["pred_calib_eval_x"].to_numpy() - cx
        pred_dy = eval_df["pred_calib_eval_y"].to_numpy() - cy

        dir_acc_x = float(np.mean(np.sign(gt_dx) == np.sign(pred_dx)) * 100.0) if len(gt_dx) > 0 else 0.0
        dir_acc_y = float(np.mean(np.sign(gt_dy) == np.sign(pred_dy)) * 100.0) if len(gt_dy) > 0 else 0.0
        dir_acc_both = float(np.mean((np.sign(gt_dx) == np.sign(pred_dx)) & (np.sign(gt_dy) == np.sign(pred_dy))) * 100.0) if len(gt_dx) > 0 else 0.0

        # Calibration Stability / Drift Analysis
        # Compare error during Stage 1 vs Stage 2
        calib_stage_mean_ang = float(df_calib_stage["calib_ang_err_deg"].mean()) if len(df_calib_stage) > 0 else 0.0
        test_stage_mean_ang = float(df_test_stage["calib_ang_err_deg"].mean()) if len(df_test_stage) > 0 else 0.0
        drift_ang_deg = test_stage_mean_ang - calib_stage_mean_ang if len(df_test_stage) > 0 and len(df_calib_stage) > 0 else 0.0
        drift_ratio = (test_stage_mean_ang / max(1e-3, calib_stage_mean_ang)) if calib_stage_mean_ang > 1e-3 and len(df_test_stage) > 0 else 1.0

        mean_fps = float(df["fps"].mean())
        peak_ram_gb = float(df["ram_gb"].max())
        peak_vram_mb = float(df["vram_mb"].max())

        # Exact naming convention: {model_name}_{recording_time}_{metric_name}_{metric_value}
        clean_model = re.sub(r"[^a-zA-Z0-9_\-]", "_", self.model_name).strip("_").lower()
        metric_name = "ang_err_deg"
        final_prefix = f"{clean_model}_{self.session_timestamp}_{metric_name}_{mean_calib_ang:.2f}"

        final_video_path = os.path.join(self.output_dir, f"{final_prefix}.mp4")
        final_csv_path = os.path.join(self.output_dir, f"{final_prefix}.csv")
        final_report_path = os.path.join(self.output_dir, f"{final_prefix}_report.md")
        final_plot_path = os.path.join(self.output_dir, f"{final_prefix}_plot.png")

        # Rename temp video
        if os.path.exists(self.temp_video_path):
            if os.path.exists(final_video_path):
                try:
                    os.remove(final_video_path)
                except Exception:
                    pass
            try:
                os.rename(self.temp_video_path, final_video_path)
            except Exception:
                pass

        # Save CSV
        df.to_csv(final_csv_path, index=False)

        # Per-point breakdown for calibration stage
        point_stats = []
        for p_idx, grp in df_calib_stage.groupby("point_idx"):
            p_desc = grp["point_desc"].iloc[0]
            point_stats.append({
                "point_idx": int(p_idx),
                "point_desc": p_desc,
                "target_x": float(grp["target_gt_x"].iloc[0]),
                "target_y": float(grp["target_gt_y"].iloc[0]),
                "samples": len(grp),
                "calib_ang_err_mean_deg": float(grp["calib_ang_err_deg"].mean()),
                "raw_ang_err_mean_deg": float(grp["raw_ang_err_deg"].mean()),
                "screen_err_mean_mm": float(grp["calib_scr_err_mm"].mean()),
                "screen_err_mean_px": float(grp["calib_scr_err_px"].mean())
            })

        results = {
            "session_id": final_prefix,
            "model_name": self.model_name,
            "timestamp": self.session_timestamp,
            "metric_name": metric_name,
            "metric_value": mean_calib_ang,
            "eval_source": eval_source_name,
            "total_frames": len(df),
            "valid_frames": len(valid_df),
            "calib_frames": len(df_calib_stage),
            "test_frames": len(df_test_stage),
            "duration_sec": float(df["timestamp_sec"].iloc[-1]),
            "angular_err_mean_deg": mean_calib_ang,
            "angular_err_median_deg": median_calib_ang,
            "raw_angular_err_mean_deg": mean_raw_ang,
            "raw_angular_err_median_deg": median_raw_ang,
            "screen_err_mean_mm": mean_calib_mm,
            "screen_err_median_mm": median_calib_mm,
            "screen_err_mean_px": mean_calib_px,
            "raw_screen_err_mean_mm": mean_raw_mm,
            "raw_screen_err_mean_px": mean_raw_px,
            "r2_score": r2_fit,
            "corr_x": corr_x_calib,
            "corr_y": corr_y_calib,
            "raw_corr_x": corr_x_raw,
            "raw_corr_y": corr_y_raw,
            "dir_acc_x": dir_acc_x,
            "dir_acc_y": dir_acc_y,
            "dir_acc_both": dir_acc_both,
            "calib_stage_mean_ang": calib_stage_mean_ang,
            "test_stage_mean_ang": test_stage_mean_ang,
            "drift_ang_deg": drift_ang_deg,
            "drift_ratio": drift_ratio,
            "mean_fps": mean_fps,
            "peak_ram_gb": peak_ram_gb,
            "peak_vram_mb": peak_vram_mb,
            "point_stats": point_stats,
            "video_path": final_video_path,
            "csv_path": final_csv_path,
            "report_path": final_report_path,
            "plot_path": final_plot_path
        }

        self._generate_report(results, point_stats)
        self._generate_plot(results, valid_df, df_calib_stage, df_test_stage)
        return results

    def _generate_plot(self, results: Dict[str, Any], valid_df: pd.DataFrame, df_calib: pd.DataFrame, df_test: pd.DataFrame) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=140)

            # Left Plot: Screen Space Trajectory & Calibration Points
            ax1.set_xlim(0, self.screen_w)
            ax1.set_ylim(self.screen_h, 0)
            ax1.set_title(
                f"Траектория взгляда: {results['model_name']}\n"
                f"Ошибка: {results['angular_err_mean_deg']:.2f}° ({results['screen_err_mean_mm']:.1f} мм) | "
                f"Дрейф: {results['drift_ang_deg']:+.2f}°",
                fontsize=11, fontweight="bold"
            )
            ax1.set_xlabel("Дисплей X (px)")
            ax1.set_ylabel("Дисплей Y (px)")

            # Draw static calibration points
            if len(df_calib) > 0:
                ax1.scatter(df_calib["target_gt_x"], df_calib["target_gt_y"], color="darkred", s=80, marker="s", label="Точки калибровки GT", zorder=5)
                ax1.scatter(df_calib["pred_calib_eval_x"], df_calib["pred_calib_eval_y"], color="orange", s=15, alpha=0.45, label="Калибровочные фиксации", zorder=4)

            # Draw dynamic moving test trajectory
            if len(df_test) > 0:
                gt_tx = df_test["target_gt_x"].to_numpy()
                gt_ty = df_test["target_gt_y"].to_numpy()
                pred_tx = df_test["pred_calib_eval_x"].to_numpy()
                pred_ty = df_test["pred_calib_eval_y"].to_numpy()

                ax1.plot(gt_tx, gt_ty, color="green", linewidth=2.8, label="Движущаяся цель GT", alpha=0.9, zorder=6)
                ax1.plot(pred_tx, pred_ty, color="#2563eb", linewidth=1.5, label="Калиброванный трек", alpha=0.8, zorder=7)
                ax1.scatter(pred_tx[::5], pred_ty[::5], color="#1d4ed8", s=20, alpha=0.6, zorder=8)

            ax1.axvline(self.screen_w / 2.0, color="gray", linestyle="--", alpha=0.5)
            ax1.axhline(self.screen_h / 2.0, color="gray", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, linestyle=":", alpha=0.6)

            # Right Plot: Comparison of Raw Model vs Calibrated Model Error
            eval_data = df_test if len(df_test) >= 15 else valid_df
            time_axis = eval_data["timestamp_sec"].to_numpy()
            raw_err = eval_data["raw_ang_err_deg"].to_numpy()
            calib_err = eval_data["calib_ang_err_deg"].to_numpy()

            ax2.plot(time_axis, raw_err, color="#dc2626", linewidth=1.4, label=f"Сырая ошибка модели (ср. {results['raw_angular_err_mean_deg']:.2f}°)", alpha=0.75)
            ax2.plot(time_axis, calib_err, color="#16a34a", linewidth=1.8, label=f"Калиброванная ошибка (ср. {results['angular_err_mean_deg']:.2f}°)", alpha=0.9)
            ax2.axhline(5.20, color="#9333ea", linestyle="--", label="Порог ТЗ (5.20°)")

            ax2.set_title(f"Сравнение ошибки во времени (Сырая модель vs Калибровка)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Время сессии (с)")
            ax2.set_ylabel("Угловая ошибка (°)")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.savefig(results["plot_path"])
            plt.close(fig)
        except Exception as e:
            print(f"[Plot Warning] Could not generate plot: {e}")

    def _generate_report(self, res: Dict[str, Any], point_stats: List[Dict[str, Any]]) -> None:
        rows_points = []
        for p in point_stats:
            rows_points.append(
                f"| **{p['point_desc']}** | ({int(p['target_x'])}, {int(p['target_y'])}) | "
                f"**{p['calib_ang_err_mean_deg']:.2f}°** | {p['raw_ang_err_mean_deg']:.2f}° | "
                f"{p['screen_err_mean_mm']:.1f} мм | {p['screen_err_mean_px']:.1f} px | {p['samples']} |"
            )
        table_points = "\n".join(rows_points) if rows_points else "| Нет данных | - | - | - | - | - | - |"

        rep_content = f"""# ПАСПОРТ ТЕСТА ТОЧНОСТИ МОДЕЛИ: {res['model_name']}

* **Идентификатор сессии:** `{res['session_id']}`
* **Исследуемая модель:** `{res['model_name']}`
* **Дата и время тестирования:** `{res['timestamp']}`
* **Основная метрика оценки (`{res['metric_name']}`):** **{res['metric_value']:.2f}°**
* **База для основной метрики:** `{res['eval_source']}`
* **Длительность теста:** `{res['duration_sec']:.2f}` с
* **Всего кадров:** `{res['total_frames']}` (Калибровка: `{res['calib_frames']}`, Дин. тест: `{res['test_frames']}`)
* **Файл видеозаписи с телеметрией:** `{res['video_path']}`
* **Файл датасета:** `{res['csv_path']}`
* **Файл графика:** `{res['plot_path']}`

---

## 1. Сравнительные метрики точности (Сырая модель vs Калиброванная)

| Показатель | Сырая модель (Raw) | Калиброванная модель | Норматив ТЗ | Вердикт |
|---|---|---|---|---|
| **Угловая ошибка (Mean)** | **{res['raw_angular_err_mean_deg']:.2f}°** | **{res['angular_err_mean_deg']:.2f}°** | $\\le 5.20^\\circ$ | {'PASS' if res['angular_err_mean_deg'] <= 5.20 else 'FAIL'} |
| **Угловая ошибка (Median)** | **{res['raw_angular_err_median_deg']:.2f}°** | **{res['angular_err_median_deg']:.2f}°** | - | - |
| **Экранная ошибка (Mean, мм)** | **{res['raw_screen_err_mean_mm']:.1f} мм** | **{res['screen_err_mean_mm']:.1f} мм** | $\\le 50.0\\text{{ мм}}$ | {'PASS' if res['screen_err_mean_mm'] <= 50.0 else 'FAIL'} |
| **Экранная ошибка (Median, мм)** | - | **{res['screen_err_median_mm']:.1f} мм** | - | - |
| **Экранная ошибка (Mean, px)** | **{res['raw_screen_err_mean_px']:.1f} px** | **{res['screen_err_mean_px']:.1f} px** | - | - |
| **Корреляция по оси X (R_x)** | **{res['raw_corr_x']:.3f}** | **{res['corr_x']:.3f}** | $\\ge 0.70$ | {'PASS' if res['corr_x'] >= 0.70 else 'FAIL'} |
| **Корреляция по оси Y (R_y)** | **{res['raw_corr_y']:.3f}** | **{res['corr_y']:.3f}** | $\\ge 0.70$ | {'PASS' if res['corr_y'] >= 0.70 else 'FAIL'} |
| **Точность квадранта (Оба направления)** | - | **{res['dir_acc_both']:.1f}%** | $\\ge 70.0\\%$ | {'PASS' if res['dir_acc_both'] >= 70.0 else 'FAIL'} |
| **Качество регрессии (R² Score)** | - | **{res['r2_score']:.3f}** | $\\ge 0.70$ | {'PASS' if res['r2_score'] >= 0.70 else 'FAIL'} |

---

## 2. Анализ стабильности и устойчивости к смещениям (Calibration Drift)

| Параметр стабильности | Значение | Описание |
|---|---|---|
| **Ошибка этапа 1 (Калибровка)** | **{res['calib_stage_mean_ang']:.2f}°** | Ошибка на статических контрольных точках калибровки |
| **Ошибка этапа 2 (Дин. тест)** | **{res['test_stage_mean_ang']:.2f}°** | Ошибка при динамическом слежении за движущейся точкой |
| **Величина дрейфа (Δ Drift)** | **{res['drift_ang_deg']:+.2f}°** | Разница между динамическим тестом и калибровкой |
| **Коэффициент деградации** | **{res['drift_ratio']:.2f}x** | Отношение ошибки теста к ошибке калибровки |

---

## 3. Системные ресурсы
| Параметр | Значение |
|---|---|
| **Средний FPS** | **{res['mean_fps']:.1f} кадров/с** |
| **Пиковое потребление RAM** | **{res['peak_ram_gb']:.2f} ГБ** |
| **Пиковое потребление VRAM (GPU)** | **{res['peak_vram_mb']:.0f} МБ** |

---

## 4. Результаты калибровки по статическим контрольным точкам

| Контрольная точка | Координаты GT (px) | Калиброванная ошибка | Сырая ошибка | Экранная ошибка (мм) | Экранная ошибка (px) | Кадров |
|---|---|---|---|---|---|---|
{table_points}
"""
        with open(res["report_path"], "w", encoding="utf-8") as f:
            f.write(rep_content)
