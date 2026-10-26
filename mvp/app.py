import os, sys, math, time, argparse, ctypes, cv2, torch, numpy as np
from typing import Optional, Dict, Any, Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.face_pipeline import create_gaze_pipeline, GazeTrackingPipeline, MediaPipeIrisPipeline
from engine.calibration import GazeCalibration
from engine.filter import PointOneEuroFilter
from engine.cursor_controller import CursorController
from engine.telemetry_logger import TelemetryLogger
from engine.benchmark_recorder import BenchmarkRecorder

class GazeApplication:
    def __init__(
        self,
        camera_id: int = 0,
        cam_w: int = 1280,
        cam_h: int = 720,
        enable_mouse: bool = False,
        model_name: str = "l2cs",
        checkpoint_path: Optional[str] = None,
        yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx",
        task_path: str = "mvp/models/weights/face_landmarker.task",
        log_dir: str = "mvp/logs",
        headless: bool = False,
        test_mode: bool = False
    ) -> None:
        self.camera_id = camera_id; self.req_cam_w = cam_w; self.req_cam_h = cam_h
        self.headless = headless
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.test_mode = test_mode
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDPIAware()
        except Exception:
            pass
        self.screen_w = int(self.user32.GetSystemMetrics(0))
        self.screen_h = int(self.user32.GetSystemMetrics(1))
        if self.screen_w <= 0 or self.screen_h <= 0:
            self.screen_w = 2880; self.screen_h = 1800
        self.pipeline = create_gaze_pipeline(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            yunet_path=yunet_path,
            task_path=task_path
        )
        self.calibration = GazeCalibration(
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            profile_path="mvp/calibration_profile.json",
            model_name=self.model_name
        )
        self.filter = PointOneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0, deadzone_radius_px=25.0)
        self.cursor_ctrl = CursorController(screen_w=self.screen_w, screen_h=self.screen_h, enable_control=enable_mouse)
        self.logger = TelemetryLogger(log_dir=log_dir, session_prefix="session")
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            alt_id = 1 if self.camera_id == 0 else 0
            self.cap = cv2.VideoCapture(alt_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(alt_id)
            if self.cap.isOpened():
                self.camera_id = alt_id
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_cam_h)
            self.actual_cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.actual_cam_w = self.req_cam_w; self.actual_cam_h = self.req_cam_h
        self.is_running = False
        self.in_calibration_mode = False
        self.calib_points = self._generate_calibration_grid()
        self.calib_idx = 0
        self.calib_point_start_time = 0.0
        self.calib_duration_per_point = 2.2
        self.calib_settle_sec = 0.7
        self.calib_point_samples = []
        self.calib_final_dataset = []
        self.prev_space_down = False
        self.window_hud_name = "GAZE-CLICK MVP [Tracker & Telemetry]"
        self.window_calib_name = "GAZE-CLICK Calibration"
        self.window_test_name = "GAZE-CLICK Accuracy Test"
        self.benchmark = BenchmarkRecorder(
            output_dir="mvp/benchmarks",
            screen_w=self.screen_w,
            screen_h=self.screen_h,
            cam_w=self.actual_cam_w,
            cam_h=self.actual_cam_h,
            model_name=getattr(self.pipeline, "model_name", self.model_name)
        )
        if self.test_mode:
            self.benchmark.start_recording(model_name=getattr(self.pipeline, "model_name", self.model_name))

    def _generate_calibration_grid(self) -> list:
        pts = []
        nx, ny = 3, 3
        margin_x = 0.15; margin_y = 0.18
        xs = np.linspace(margin_x * self.screen_w, (1.0 - margin_x) * self.screen_w, nx)
        ys = np.linspace(margin_y * self.screen_h, (1.0 - margin_y) * self.screen_h, ny)
        for y in ys:
            for x in xs:
                pts.append((float(x), float(y)))
        return pts

    def _draw_hud(self, frame: np.ndarray, result: dict, screen_pt: tuple, fps: float, lat_total: float) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()
        if result.get("face_detected", False):
            fx, fy, fw, fh = result["face_bbox"]
            cv2.rectangle(vis, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 255, 0), 2)
            score = result.get("face_score", 0.0)
            cv2.putText(vis, f"Face: {score:.2f}", (int(fx), max(18, int(fy) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            lm = result.get("landmarks", {})
            r_eye = lm.get("right_eye")
            l_eye = lm.get("left_eye")
            if r_eye and l_eye:
                rex, rey = int(r_eye[0]), int(r_eye[1])
                lex, ley = int(l_eye[0]), int(l_eye[1])
                cv2.circle(vis, (rex, rey), 4, (255, 128, 0), -1)
                cv2.circle(vis, (lex, ley), 4, (255, 128, 0), -1)
                p_rad = math.radians(result.get("gaze_pitch_deg", 0.0))
                y_rad = math.radians(result.get("gaze_yaw_deg", 0.0))
                arrow_len = 65.0
                dx = -arrow_len * math.sin(y_rad)
                dy = arrow_len * math.sin(p_rad)
                cv2.arrowedLine(vis, (rex, rey), (int(rex + dx), int(rey + dy)), (0, 0, 255), 2, tipLength=0.25)
                cv2.arrowedLine(vis, (lex, ley), (int(lex + dx), int(ley + dy)), (0, 0, 255), 2, tipLength=0.25)
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (420, 246), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        vram_mb = 0.0
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram_mb = float((total_b - free_b) / (1024 ** 2))
        ctrl_status = "ENABLED [ON]" if self.cursor_ctrl.enable_control else "PREVIEW [OFF]"
        ctrl_color = (0, 255, 0) if self.cursor_ctrl.enable_control else (0, 165, 255)
        calib_status = f"CALIBRATED (R2={self.calibration.r2_score:.2f})" if self.calibration.is_calibrated else "UNCALIBRATED (Press C)"
        calib_color = (0, 255, 0) if self.calibration.is_calibrated else (0, 0, 255)
        model_display = getattr(self.pipeline, "model_name", self.model_name)
        lines = [
            (f"Model: {model_display}", (255, 215, 0)),
            (f"FPS: {fps:.1f} | Pipeline: {lat_total:.1f} ms", (255, 255, 255)),
            (f"Model Forward: {result.get('latency_model_ms', 0.0):.2f} ms", (255, 255, 255)),
            (f"VRAM: {vram_mb:.1f} MB | Cam: {w}x{h}", (255, 255, 255)),
            (f"Gaze Pitch: {result.get('gaze_pitch_deg', 0.0):+.1f} deg | Yaw: {result.get('gaze_yaw_deg', 0.0):+.1f} deg", (0, 255, 255)),
            (f"Head Pitch: {result.get('head_pitch_deg', 0.0):+.1f} deg | Yaw: {result.get('head_yaw_deg', 0.0):+.1f} deg", (200, 200, 200)),
            (f"Screen: X={int(screen_pt[0])} Y={int(screen_pt[1])}", (0, 255, 255)),
            (f"Mouse Control: {ctrl_status}", ctrl_color),
            (f"Calibration: {calib_status}", calib_color),
        ]
        y_pos = 32
        for text, color in lines:
            cv2.putText(vis, text, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y_pos += 22
        cv2.putText(vis, "[M]: Toggle Mouse | [C]: Calibrate | [R]: Reset Calib | [Q/ESC]: Quit", (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _render_calibration_frame(self, target_pt: tuple, progress: float) -> np.ndarray:
        calib_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        tx, ty = int(target_pt[0]), int(target_pt[1])
        radius_outer = int(55.0 * (1.0 - 0.7 * progress))
        cv2.circle(calib_img, (tx, ty), radius_outer, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(calib_img, (tx, ty), 7, (0, 0, 255), -1, cv2.LINE_AA)
        title = f"Gaze Calibration: Point {self.calib_idx + 1} of {len(self.calib_points)}"
        cv2.putText(calib_img, title, (int(self.screen_w / 2 - 260), 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        hint = "Look steadily at the red center dot. Keep your head natural."
        cv2.putText(calib_img, hint, (int(self.screen_w / 2 - 340), 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        prog_bar_w = 400; prog_bar_h = 10
        prog_bar_x = int(self.screen_w / 2 - prog_bar_w / 2)
        prog_bar_y = self.screen_h - 60
        cv2.rectangle(calib_img, (prog_bar_x, prog_bar_y), (prog_bar_x + prog_bar_w, prog_bar_y + prog_bar_h), (60, 60, 60), -1)
        fill_w = int(prog_bar_w * ((self.calib_idx + progress) / len(self.calib_points)))
        cv2.rectangle(calib_img, (prog_bar_x, prog_bar_y), (prog_bar_x + fill_w, prog_bar_y + prog_bar_h), (0, 255, 0), -1)
        return calib_img

    def start_calibration(self) -> None:
        self.in_calibration_mode = True
        self.calib_idx = 0
        self.calib_point_samples = []
        self.calib_final_dataset = []
        self.calib_point_start_time = time.perf_counter()
        if not self.headless:
            cv2.namedWindow(self.window_calib_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_calib_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def finish_calibration(self) -> None:
        self.in_calibration_mode = False
        if not self.headless:
            try:
                cv2.destroyWindow(self.window_calib_name)
            except Exception:
                pass
        fit_res = self.calibration.fit(self.calib_final_dataset)
        print(f"Calibration completed: {fit_res}")

    def run(self, max_frames: Optional[int] = None) -> None:
        self.is_running = True
        fps_hist = []
        t_prev = time.perf_counter()
        frame_counter = 0
        if not self.headless:
            cv2.namedWindow(self.window_hud_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_hud_name, 960, 540)
        while self.is_running:
            t_frame_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                frame = np.zeros((self.actual_cam_h, self.actual_cam_w, 3), dtype=np.uint8)
            frame_counter += 1
            result = self.pipeline.process_frame(frame)
            now_mono = time.perf_counter()
            dt = max(now_mono - t_prev, 1e-6)
            t_prev = now_mono
            fps_inst = 1.0 / dt
            fps_hist.append(fps_inst)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps_avg = float(np.mean(fps_hist))
            p_deg = result.get("gaze_pitch_deg", 0.0)
            y_deg = result.get("gaze_yaw_deg", 0.0)
            hp_deg = result.get("head_pitch_deg", 0.0)
            hy_deg = result.get("head_yaw_deg", 0.0)
            raw_x, raw_y = self.calibration.predict_uncalibrated(p_deg, y_deg)
            calib_x, calib_y = self.calibration.predict(
                p_deg, y_deg, hp_deg, hy_deg,
                ix=result.get("ix"),
                iy=result.get("iy"),
                eye_diff_x=result.get("eye_diff_x"),
                eye_diff_y=result.get("eye_diff_y")
            )
            filt_x, filt_y = self.filter.filter(calib_x, calib_y, now_mono)
            click_event = False
            if result.get("face_detected", False):
                self.cursor_ctrl.move_cursor(filt_x, filt_y)
                click_event = self.cursor_ctrl.check_dwell_click(filt_x, filt_y)
            t_frame_end = time.perf_counter()
            lat_total = (t_frame_end - t_frame_start) * 1000.0
            lm = result.get("landmarks") or {}
            re = lm.get("right_eye") or (0.0, 0.0)
            le = lm.get("left_eye") or (0.0, 0.0)
            fb = result.get("face_bbox") or (0.0, 0.0, 0.0, 0.0)
            log_data = {
                "fps": fps_avg,
                "latency_model_ms": result.get("latency_model_ms", 0.0),
                "latency_total_ms": lat_total,
                "cam_w": self.actual_cam_w,
                "cam_h": self.actual_cam_h,
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "face_detected": result.get("face_detected", False),
                "face_x": fb[0], "face_y": fb[1], "face_w": fb[2], "face_h": fb[3],
                "face_score": result.get("face_score", 0.0),
                "right_eye_x": re[0], "right_eye_y": re[1],
                "left_eye_x": le[0], "left_eye_y": le[1],
                "gaze_pitch_deg": p_deg, "gaze_yaw_deg": y_deg,
                "head_pitch_deg": hp_deg, "head_yaw_deg": hy_deg,
                "raw_screen_x": raw_x, "raw_screen_y": raw_y,
                "calibrated_screen_x": calib_x, "calibrated_screen_y": calib_y,
                "filtered_screen_x": filt_x, "filtered_screen_y": filt_y,
                "mouse_active": self.cursor_ctrl.enable_control,
                "click_event": click_event
            }
            self.logger.log_frame(log_data)
            if self.test_mode:
                if self.headless:
                    if self.benchmark.stage == "calibration":
                        if self.benchmark.waiting_for_key_release:
                            self.benchmark.trigger_space_release()
                        else:
                            self.benchmark.trigger_space_press()
                    elif self.benchmark.stage == "dynamic_test":
                        self.benchmark.is_holding_trajectory_key = True

                tx, ty, status_str, prog, is_finished = self.benchmark.update_state(
                    is_holding_trajectory_key=self.benchmark.is_holding_trajectory_key,
                    dt=dt
                )
                self.benchmark.record_frame(
                    frame_bgr=frame,
                    pred_raw=(raw_x, raw_y),
                    pred_calib=(calib_x, calib_y),
                    pred_filtered=(filt_x, filt_y),
                    model_res=result,
                    fps=fps_avg,
                    mirror_camera=False,
                    is_holding_trajectory_key=self.benchmark.is_holding_trajectory_key
                )
                if not self.headless:
                    test_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    itx, ity = int(tx), int(ty)
                    col = (0, 255, 0) if self.benchmark.is_capturing_active else (0, 165, 255)
                    cv2.circle(test_img, (itx, ity), 42, col, 2, cv2.LINE_AA)
                    cv2.circle(test_img, (itx, ity), 8, col, -1, cv2.LINE_AA)
                    cv2.putText(test_img, status_str, (int(self.screen_w / 2 - 380), 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(self.window_test_name, test_img)
                if is_finished:
                    res = self.benchmark.stop_recording_and_evaluate()
                    if self.benchmark.session_calibration is not None:
                        self.calibration = self.benchmark.session_calibration
                    print("\n" + "="*75)
                    print(f" ПАСПОРТ ТЕСТА ТОЧНОСТИ: {res.get('model_name')}")
                    print(f" Источник оценки:          {res.get('eval_source')}")
                    print(f" Калиброванная угловая:    {res.get('angular_err_mean_deg', 0.0):.2f}°")
                    print(f" Сырая угловая ошибка:     {res.get('raw_angular_err_mean_deg', 0.0):.2f}°")
                    print(f" Экранная ошибка (мм):     {res.get('screen_err_mean_mm', 0.0):.1f} мм ({res.get('screen_err_mean_px', 0.0):.1f} px)")
                    print(f" Дрейф калибровки (Δ):     {res.get('drift_ang_deg', 0.0):+.2f}° (коэф. {res.get('drift_ratio', 1.0):.2f}x)")
                    print(f" Качество регрессии R²:    {res.get('r2_score', 0.0):.3f}")
                    print(f" Корреляция X / Y:         {res.get('corr_x', 0.0):.3f} / {res.get('corr_y', 0.0):.3f}")
                    print(f" Видеозапись с телеметрией:{res.get('video_path')}")
                    print(f" Полный Markdown отчет:    {res.get('report_path')}")
                    print("="*75 + "\n")
                    self.is_running = False

            if self.in_calibration_mode:
                elapsed_pt = now_mono - self.calib_point_start_time
                target = self.calib_points[self.calib_idx]
                if elapsed_pt >= self.calib_settle_sec and result.get("face_detected", False):
                    self.calib_point_samples.append((p_deg, y_deg))
                if elapsed_pt >= self.calib_duration_per_point:
                    if self.calib_point_samples:
                        med_p = float(np.median([s[0] for s in self.calib_point_samples]))
                        med_y = float(np.median([s[1] for s in self.calib_point_samples]))
                        self.calib_final_dataset.append({
                            "pitch_deg": med_p,
                            "yaw_deg": med_y,
                            "target_x": target[0],
                            "target_y": target[1]
                        })
                    self.calib_point_samples = []
                    self.calib_idx += 1
                    if self.calib_idx >= len(self.calib_points):
                        self.finish_calibration()
                    else:
                        self.calib_point_start_time = time.perf_counter()
                elif not self.headless:
                    progress = min(1.0, elapsed_pt / self.calib_duration_per_point)
                    calib_view = self._render_calibration_frame(target, progress)
                    cv2.imshow(self.window_calib_name, calib_view)
            if not self.headless:
                hud_frame = self._draw_hud(frame, result, (filt_x, filt_y), fps_avg, lat_total)
                cv2.imshow(self.window_hud_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    self.is_running = False

                if self.test_mode:
                    # Physical hardware keyboard state via Windows API to prevent auto-repeat bugs
                    space_down = bool(self.user32.GetAsyncKeyState(0x20) & 0x8000)
                    shift_down = bool(self.user32.GetAsyncKeyState(0x10) & 0x8000)
                    g_down = bool(self.user32.GetAsyncKeyState(0x47) & 0x8000)

                    if self.benchmark.stage == "calibration":
                        # Transition from UP to DOWN: start capture
                        if space_down and not self.prev_space_down:
                            self.benchmark.trigger_space_press()
                        # Transition from DOWN to UP: advance to next point
                        elif not space_down and self.prev_space_down:
                            self.benchmark.trigger_space_release()
                    elif self.benchmark.stage == "dynamic_test":
                        self.benchmark.is_holding_trajectory_key = (space_down or shift_down or g_down)

                    self.prev_space_down = space_down
                else:
                    if key == 32:  # Space
                        self.cursor_ctrl.toggle_control()
                    elif key in (ord('m'), ord('M')):
                        self.cursor_ctrl.toggle_control()
                    elif key in (ord('c'), ord('C')):
                        if not self.in_calibration_mode:
                            self.start_calibration()
                    elif key in (ord('r'), ord('R')):
                        self.calibration.reset()
            if max_frames is not None and frame_counter >= max_frames:
                self.is_running = False
        self.cleanup()

    def cleanup(self) -> None:
        if hasattr(self, "benchmark") and self.benchmark and getattr(self.benchmark, "is_recording", False):
            try:
                res = self.benchmark.stop_recording_and_evaluate()
                if res and "error" not in res:
                    print(f"\n[Benchmark] Accuracy Test finalized: Mean Angular Error = {res.get('angular_err_mean_deg', 0.0):.2f} deg")
            except Exception as e:
                print(f"[Benchmark] Cleanup warning: {e}")
        if self.cap.isOpened():
            self.cap.release()
        if not self.headless:
            cv2.destroyAllWindows()
        summary_path = self.logger.close()
        print(f"Session closed. Summary report saved to: {summary_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="GAZE-CLICK Real-Time System")
    parser.add_argument("--model_name", "--model-name", "--model", dest="model_name", type=str, default="puregaze",
                        help="Gaze model: 'puregaze', 'unigaze', 'gazetr', 'ptgaze_eth_xgaze', 'mediapipe', 'yolo26n', 'qualcomm_eyegaze', 'yakhyo_resnet50', 'ptgaze_mpiifacegaze', 'idiap_resnet18', 'l2cs', or path to checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional override path to custom checkpoint (.pt/.pth/.safetensors/.pkl)")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--enable-mouse", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--test", "--benchmark", action="store_true",
                        help="Run automated accuracy test, recording visualized video, computing per-point metrics, and outputting named report/video files")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    # If --checkpoint was given and --model-name was default, and file is YOLO/ResNet, adjust appropriately
    if args.checkpoint and args.model_name == "l2cs":
        if "yolo" in args.checkpoint.lower():
            args.model_name = "yolo26n"

    if not args.cv and not args.headless:
        import app_qt
        qt_app = app_qt.QApplication(sys.argv)
        qt_app.setApplicationName("GAZE-CLICK")
        win = app_qt.GazeMainWindow(
            camera_id=args.camera_id,
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            enable_mouse=args.enable_mouse
        )
        if args.test:
            win.showFullScreen()
            win.start_benchmark()
        elif args.fullscreen or args.calibrate:
            win.showFullScreen()
            if args.calibrate:
                win.start_calibration()
        else:
            win.show()
        sys.exit(qt_app.exec())

    app = GazeApplication(
        camera_id=args.camera_id,
        cam_w=args.width,
        cam_h=args.height,
        enable_mouse=args.enable_mouse,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        headless=args.headless,
        test_mode=args.test
    )
    if args.calibrate:
        app.start_calibration()
    app.run(max_frames=args.max_frames)

if __name__ == "__main__":
    main()
