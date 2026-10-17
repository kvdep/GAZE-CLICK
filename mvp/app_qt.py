import os, sys, math, time, ctypes, argparse, cv2, torch, numpy as np
from typing import Optional, Dict, Any, Tuple, List

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QStackedWidget,
    QProgressBar, QMessageBox, QDialog, QTextEdit
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.face_pipeline import GazeTrackingPipeline
from engine.calibration import GazeCalibration
from engine.filter import PointOneEuroFilter
from engine.cursor_controller import CursorController
from engine.telemetry_logger import TelemetryLogger
from engine.benchmark_recorder import BenchmarkRecorder

class CalibrationCanvas(QWidget):
    def __init__(self, parent=None, ref_screen_w: float = 2880.0, ref_screen_h: float = 1800.0):
        super().__init__(parent)
        self.ref_screen_w = float(ref_screen_w)
        self.ref_screen_h = float(ref_screen_h)
        self.points: List[Tuple[float, float]] = []
        self.current_idx: int = 0
        self.progress: float = 0.0
        self.status_text: str = ""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_state(self, points: List[Tuple[float, float]], idx: int, progress: float, status_text: str):
        self.points = points
        self.current_idx = idx
        self.progress = progress
        self.status_text = status_text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(14, 16, 20))
        w = self.width()
        h = self.height()
        if not self.points or self.current_idx >= len(self.points):
            return
        pt = self.points[self.current_idx]
        tx, ty = pt[0], pt[1]
        draw_x = (tx / max(self.ref_screen_w, 1.0)) * w
        draw_y = (ty / max(self.ref_screen_h, 1.0)) * h
        radius_outer = 60.0 * (1.0 - 0.65 * self.progress)
        pen_outer = QPen(QColor(245, 158, 11), 3)
        painter.setPen(pen_outer)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(draw_x, draw_y), radius_outer, radius_outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(239, 68, 68)))
        painter.drawEllipse(QPointF(draw_x, draw_y), 9.0, 9.0)
        painter.setPen(QColor(255, 255, 255))
        font_title = QFont("Segoe UI", 20, QFont.Weight.Bold)
        painter.setFont(font_title)
        title_text = f"Calibration Point {self.current_idx + 1} of {len(self.points)}"
        painter.drawText(QRectF(0, 30, float(w), 36), Qt.AlignmentFlag.AlignCenter, title_text)
        font_sub = QFont("Segoe UI", 13)
        painter.setFont(font_sub)
        painter.setPen(QColor(209, 213, 219))
        sub_text = "Look at the red dot and PRESS SPACE [Пробел] to record (or wait for auto-advance)"
        painter.drawText(QRectF(0, 70, float(w), 30), Qt.AlignmentFlag.AlignCenter, sub_text)
        bar_w = 460.0; bar_h = 10.0
        bar_x = (w - bar_w) / 2.0; bar_y = h - 60.0
        painter.setBrush(QBrush(QColor(31, 41, 55)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 5.0, 5.0)
        completed_ratio = (self.current_idx + self.progress) / len(self.points)
        fill_w = bar_w * min(1.0, max(0.0, completed_ratio))
        painter.setBrush(QBrush(QColor(16, 185, 129)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 5.0, 5.0)

class BenchmarkCanvas(QWidget):
    def __init__(self, parent=None, ref_screen_w: float = 2880.0, ref_screen_h: float = 1800.0):
        super().__init__(parent)
        self.ref_screen_w = float(ref_screen_w)
        self.ref_screen_h = float(ref_screen_h)
        self.target_pt: Tuple[float, float] = (0.0, 0.0)
        self.progress: float = 0.0
        self.status_text: str = ""
        self.is_holding_key: bool = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_state(self, target_pt: Tuple[float, float], progress: float, status_text: str, is_holding_key: bool = False):
        self.target_pt = target_pt
        self.progress = progress
        self.status_text = status_text
        self.is_holding_key = is_holding_key
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(10, 12, 16))
        w = self.width(); h = self.height()
        draw_x = (self.target_pt[0] / max(self.ref_screen_w, 1.0)) * w
        draw_y = (self.target_pt[1] / max(self.ref_screen_h, 1.0)) * h
        pen_cross = QPen(QColor(59, 130, 246, 160), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen_cross)
        painter.drawLine(QPointF(0, draw_y), QPointF(w, draw_y))
        painter.drawLine(QPointF(draw_x, 0), QPointF(draw_x, h))
        target_color = QColor(16, 185, 129) if "RECORDING" in self.status_text or self.is_holding_key else QColor(239, 68, 68)
        pen_outer = QPen(target_color, 3)
        painter.setPen(pen_outer)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(draw_x, draw_y), 36.0, 36.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(target_color))
        painter.drawEllipse(QPointF(draw_x, draw_y), 10.0, 10.0)
        font_warn = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font_warn)
        painter.setPen(QColor(239, 68, 68))
        warn_text = "[GROUND TRUTH BENCHMARK VIDEO RECORDING - NO TRAINING ALLOWED]"
        painter.drawText(QRectF(0, 15, float(w), 24), Qt.AlignmentFlag.AlignCenter, warn_text)
        font_info = QFont("Segoe UI", 20, QFont.Weight.Bold)
        painter.setFont(font_info)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(0, 42, float(w), 36), Qt.AlignmentFlag.AlignCenter, self.status_text)
        font_hint = QFont("Segoe UI", 13)
        painter.setFont(font_hint)
        painter.setPen(QColor(156, 163, 175))
        if "TRAJECTORY" in self.status_text or "SHIFT" in self.status_text:
            hint = "HOLD [SHIFT] or [G] to move circle marker and follow with eyes. Release to pause. [ESC] to exit."
        else:
            hint = "PRESS SPACE [Пробел] when looking directly at target to record fixation. [ESC] to exit."
        painter.drawText(QRectF(0, 80, float(w), 26), Qt.AlignmentFlag.AlignCenter, hint)
        bar_w = 600.0; bar_h = 12.0
        bar_x = (w - bar_w) / 2.0; bar_y = h - 60.0
        painter.setBrush(QBrush(QColor(31, 41, 55)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 6.0, 6.0)
        fill_w = bar_w * min(1.0, max(0.0, self.progress))
        painter.setBrush(QBrush(QColor(59, 130, 246)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 6.0, 6.0)

class GazeMainWindow(QMainWindow):
    def __init__(self, camera_id: int = 0, checkpoint_path: str = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt", yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx", log_dir: str = "mvp/logs") -> None:
        super().__init__()
        self.setWindowTitle("GAZE-CLICK MVP [Real-Time Gaze Tracking System]")
        self.camera_id = camera_id
        self.checkpoint_path = checkpoint_path
        self.yunet_path = yunet_path
        self.log_dir = log_dir
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDPIAware()
        except Exception:
            pass
        self.screen_w = int(self.user32.GetSystemMetrics(0))
        self.screen_h = int(self.user32.GetSystemMetrics(1))
        if self.screen_w <= 0 or self.screen_h <= 0:
            self.screen_w = 2880; self.screen_h = 1800
        self.pipeline = GazeTrackingPipeline(checkpoint_path=self.checkpoint_path, yunet_path=self.yunet_path)
        self.calibration = GazeCalibration(screen_w=self.screen_w, screen_h=self.screen_h, profile_path="mvp/calibration_profile.json")
        self.filter = PointOneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0, deadzone_radius_px=8.0)
        self.cursor_ctrl = CursorController(screen_w=self.screen_w, screen_h=self.screen_h, enable_control=False)
        self.logger = TelemetryLogger(log_dir=self.log_dir, session_prefix="session")
        self.benchmark = BenchmarkRecorder(output_dir="mvp/benchmarks", screen_w=self.screen_w, screen_h=self.screen_h)
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.actual_cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mirror_camera: bool = True
        self.invert_gaze_x: bool = False
        self.is_holding_trajectory_key: bool = False
        self.latest_gaze_angles: Tuple[float, float] = (-8.99, 0.0)
        self.mode: str = "tracking"
        self.calib_points = self._generate_calibration_grid()
        self.calib_idx = 0
        self.calib_start_t = 0.0
        self.calib_point_samples = []
        self.calib_final_dataset = []
        self.prev_frame_t = time.perf_counter()
        self.fps_val = 0.0
        self.frame_count = 0
        self.cached_mem_str = "RAM: 0.0 GB | VRAM: 0 MB"
        self.last_benchmark_results = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame_tick)
        self.timer.start(10)

    def _generate_calibration_grid(self) -> List[Tuple[float, float]]:
        pts = []
        nx, ny = 3, 3
        margin_x = 0.12
        xs = np.linspace(margin_x * self.screen_w, (1.0 - margin_x) * self.screen_w, nx)
        ys = np.linspace(0.18 * self.screen_h, 0.86 * self.screen_h, ny)
        for y in ys:
            for x in xs:
                pts.append((float(x), float(y)))
        return pts

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f1115; }
            QWidget { color: #f3f4f6; font-family: 'Segoe UI', Arial, sans-serif; }
            QFrame.card { background-color: #171a21; border: 1px solid #262c36; border-radius: 8px; }
            QLabel.title { font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 4px; }
            QLabel.metric_label { font-size: 12px; color: #9ca3af; }
            QLabel.metric_val { font-size: 13px; font-weight: bold; color: #60a5fa; }
            QPushButton {
                background-color: #1f2430; border: 1px solid #374151; border-radius: 6px;
                padding: 10px 16px; font-size: 13px; font-weight: 600; color: #ffffff;
            }
            QPushButton:hover { background-color: #2b3242; border-color: #4b5563; }
            QPushButton:pressed { background-color: #151922; }
            QPushButton.primary { background-color: #2563eb; border-color: #1d4ed8; }
            QPushButton.primary:hover { background-color: #3b82f6; }
            QPushButton.danger { background-color: #dc2626; border-color: #b91c1c; }
            QPushButton.danger:hover { background-color: #ef4444; }
            QPushButton.success { background-color: #059669; border-color: #047857; }
            QPushButton.success:hover { background-color: #10b981; }
        """)
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)
        self.view_dashboard = QWidget(self)
        dash_layout = QHBoxLayout(self.view_dashboard)
        dash_layout.setContentsMargins(18, 18, 18, 18)
        dash_layout.setSpacing(18)
        left_box = QVBoxLayout()
        left_box.setSpacing(12)
        video_card = QFrame()
        video_card.setProperty("class", "card")
        video_card_layout = QVBoxLayout(video_card)
        video_card_layout.setContentsMargins(10, 10, 10, 10)
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(720, 480)
        self.lbl_video.setStyleSheet("background-color: #090a0d; border-radius: 6px;")
        video_card_layout.addWidget(self.lbl_video)
        left_box.addWidget(video_card, stretch=1)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)
        telemetry_card = QFrame()
        telemetry_card.setProperty("class", "card")
        telem_layout = QVBoxLayout(telemetry_card)
        telem_layout.setContentsMargins(16, 16, 16, 16)
        telem_layout.setSpacing(10)
        lbl_telem_head = QLabel("SYSTEM TELEMETRY & METRICS")
        lbl_telem_head.setProperty("class", "title")
        telem_layout.addWidget(lbl_telem_head)
        grid_metrics = QGridLayout()
        grid_metrics.setHorizontalSpacing(14)
        grid_metrics.setVerticalSpacing(8)
        self.val_fps = QLabel("0.0 FPS")
        self.val_fps.setProperty("class", "metric_val")
        self.val_latency = QLabel("0.0 ms (Model: 0.0 ms)")
        self.val_latency.setProperty("class", "metric_val")
        self.val_gaze = QLabel("Pitch: +0.0 deg | Yaw: +0.0 deg")
        self.val_gaze.setProperty("class", "metric_val")
        self.val_head = QLabel("Pitch: +0.0 deg | Yaw: +0.0 deg")
        self.val_head.setProperty("class", "metric_val")
        self.val_screen = QLabel("X: 0 px | Y: 0 px")
        self.val_screen.setProperty("class", "metric_val")
        self.val_gain = QLabel(f"Gain: {self.calibration.gain_x:.2f}x (Press +/-)")
        self.val_gain.setProperty("class", "metric_val")
        self.val_calib_status = QLabel("Uncalibrated")
        self.val_calib_status.setProperty("class", "metric_val")
        self.val_mouse_status = QLabel("Disabled [Preview]")
        self.val_mouse_status.setProperty("class", "metric_val")
        self.val_memory = QLabel("RAM: 0.0 GB | VRAM: 0 MB")
        self.val_memory.setProperty("class", "metric_val")
        metrics_rows = [
            ("Throughput:", self.val_fps),
            ("Pipeline Latency:", self.val_latency),
            ("Gaze Angles:", self.val_gaze),
            ("Head Pose:", self.val_head),
            ("Screen Point:", self.val_screen),
            ("Gaze Gain (Sensitivity):", self.val_gain),
            ("Calibration Profile:", self.val_calib_status),
            ("Mouse Control:", self.val_mouse_status),
            ("Memory Consumption:", self.val_memory),
        ]
        for row_i, (lbl_name, val_widget) in enumerate(metrics_rows):
            lbl_k = QLabel(lbl_name)
            lbl_k.setProperty("class", "metric_label")
            grid_metrics.addWidget(lbl_k, row_i, 0)
            grid_metrics.addWidget(val_widget, row_i, 1)
        telem_layout.addLayout(grid_metrics)
        right_panel.addWidget(telemetry_card)
        controls_card = QFrame()
        controls_card.setProperty("class", "card")
        ctrl_layout = QVBoxLayout(controls_card)
        ctrl_layout.setContentsMargins(16, 16, 16, 16)
        ctrl_layout.setSpacing(10)
        lbl_ctrl_head = QLabel("CONTROL ACTIONS")
        lbl_ctrl_head.setProperty("class", "title")
        ctrl_layout.addWidget(lbl_ctrl_head)
        self.btn_calibrate = QPushButton("Start 9-Point Calibration [C]")
        self.btn_calibrate.setProperty("class", "primary")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        ctrl_layout.addWidget(self.btn_calibrate)
        self.btn_mouse = QPushButton("Enable Mouse Control [M]")
        self.btn_mouse.clicked.connect(self.toggle_mouse)
        ctrl_layout.addWidget(self.btn_mouse)
        self.btn_benchmark = QPushButton("Record Final Test Dataset [B]")
        self.btn_benchmark.setProperty("class", "danger")
        self.btn_benchmark.clicked.connect(self.start_benchmark)
        ctrl_layout.addWidget(self.btn_benchmark)
        self.btn_report = QPushButton("View Benchmark Report [V]")
        self.btn_report.clicked.connect(self.show_benchmark_report)
        ctrl_layout.addWidget(self.btn_report)
        gain_layout = QHBoxLayout()
        btn_gain_down = QPushButton("Sensitivity Gain - [-]")
        btn_gain_down.clicked.connect(lambda: self.adjust_gain(-0.15))
        btn_gain_up = QPushButton("Sensitivity Gain + [+/=]")
        btn_gain_up.clicked.connect(lambda: self.adjust_gain(+0.15))
        gain_layout.addWidget(btn_gain_down)
        gain_layout.addWidget(btn_gain_up)
        ctrl_layout.addLayout(gain_layout)
        toggles_layout = QHBoxLayout()
        self.btn_mirror = QPushButton("Mirror Feed: ON")
        self.btn_mirror.clicked.connect(self.toggle_mirror)
        toggles_layout.addWidget(self.btn_mirror)
        self.btn_invert_x = QPushButton("Invert X: OFF")
        self.btn_invert_x.clicked.connect(self.toggle_invert_x)
        toggles_layout.addWidget(self.btn_invert_x)
        ctrl_layout.addLayout(toggles_layout)
        self.btn_recenter = QPushButton("Zero Recenter [Z]")
        self.btn_recenter.clicked.connect(self.recenter_zero)
        ctrl_layout.addWidget(self.btn_recenter)
        self.btn_reset_calib = QPushButton("Reset Calibration [R]")
        self.btn_reset_calib.clicked.connect(self.reset_calibration)
        ctrl_layout.addWidget(self.btn_reset_calib)
        self.btn_fullscreen = QPushButton("Toggle Fullscreen [F11]")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        ctrl_layout.addWidget(self.btn_fullscreen)
        right_panel.addWidget(controls_card)
        right_panel.addStretch(1)
        dash_layout.addLayout(left_box, stretch=3)
        dash_layout.addLayout(right_panel, stretch=2)
        self.view_calibration = CalibrationCanvas(self, ref_screen_w=self.screen_w, ref_screen_h=self.screen_h)
        self.view_benchmark = BenchmarkCanvas(self, ref_screen_w=self.screen_w, ref_screen_h=self.screen_h)
        self.stacked_widget.addWidget(self.view_dashboard)
        self.stacked_widget.addWidget(self.view_calibration)
        self.stacked_widget.addWidget(self.view_benchmark)
        self.stacked_widget.setCurrentIndex(0)
        self.resize(1440, 900)

    def adjust_gain(self, delta: float):
        new_gx = self.calibration.gain_x + delta
        new_gy = self.calibration.gain_y + delta * 0.85
        self.calibration.set_gain(new_gx, new_gy)
        self.val_gain.setText(f"Gain: {self.calibration.gain_x:.2f}x (Press +/-)")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif key == Qt.Key.Key_Space:
            if self.mode == "benchmark":
                self.benchmark.trigger_space_press()
            elif self.mode == "calibration":
                self._confirm_calib_point()
            elif self.mode == "tracking":
                self.toggle_mouse()
        elif key in (Qt.Key.Key_Shift, Qt.Key.Key_G, Qt.Key.Key_H):
            self.is_holding_trajectory_key = True
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.adjust_gain(0.15)
        elif key == Qt.Key.Key_Minus:
            self.adjust_gain(-0.15)
        elif key == Qt.Key.Key_M:
            if self.mode == "tracking":
                self.toggle_mouse()
        elif key == Qt.Key.Key_C:
            if self.mode == "tracking":
                self.start_calibration()
        elif key == Qt.Key.Key_B:
            if self.mode == "tracking":
                self.start_benchmark()
        elif key == Qt.Key.Key_V:
            self.show_benchmark_report()
        elif key == Qt.Key.Key_R:
            if self.mode == "tracking":
                self.reset_calibration()
        elif key == Qt.Key.Key_Z:
            if self.mode == "tracking":
                self.recenter_zero()
        elif key == Qt.Key.Key_I:
            self.toggle_invert_x()
        elif key == Qt.Key.Key_F:
            self.toggle_fullscreen()
        elif key == Qt.Key.Key_Escape:
            if self.mode == "calibration":
                self.cancel_calibration()
            elif self.mode == "benchmark":
                self.finish_benchmark()
            elif self.isFullScreen():
                self.showNormal()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key.Key_Shift, Qt.Key.Key_G, Qt.Key.Key_H):
            self.is_holding_trajectory_key = False
        super().keyReleaseEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_mirror(self):
        self.mirror_camera = not self.mirror_camera
        state = "ON" if self.mirror_camera else "OFF"
        self.btn_mirror.setText(f"Mirror Feed: {state}")

    def toggle_invert_x(self):
        self.invert_gaze_x = not self.invert_gaze_x
        state = "ON" if self.invert_gaze_x else "OFF"
        self.btn_invert_x.setText(f"Invert X: {state}")

    def toggle_mouse(self):
        new_state = not self.cursor_ctrl.enable_control
        self.cursor_ctrl.set_enabled(new_state)
        if new_state:
            self.btn_mouse.setText("Disable Mouse Control [M]")
            self.btn_mouse.setProperty("class", "success")
            self.val_mouse_status.setText("ENABLED [Active Control]")
            self.val_mouse_status.setStyleSheet("color: #10b981;")
        else:
            self.btn_mouse.setText("Enable Mouse Control [M]")
            self.btn_mouse.setProperty("class", "")
            self.val_mouse_status.setText("Disabled [Preview]")
            self.val_mouse_status.setStyleSheet("color: #60a5fa;")
        self.btn_mouse.style().unpolish(self.btn_mouse)
        self.btn_mouse.style().polish(self.btn_mouse)

    def start_calibration(self):
        self.mode = "calibration"
        self.calib_idx = 0
        self.calib_point_samples = []
        self.calib_final_dataset = []
        self.calib_start_t = time.perf_counter()
        self.stacked_widget.setCurrentIndex(1)

    def cancel_calibration(self):
        self.mode = "tracking"
        self.stacked_widget.setCurrentIndex(0)

    def _confirm_calib_point(self):
        if self.mode != "calibration" or self.calib_idx >= len(self.calib_points):
            return
        curr_pt = self.calib_points[self.calib_idx]
        if len(self.calib_point_samples) >= 3:
            s_arr = np.array(self.calib_point_samples)
            med_p = float(np.median(s_arr[:, 0]))
            med_y = float(np.median(s_arr[:, 1]))
            med_hp = float(np.median(s_arr[:, 2])) if s_arr.shape[1] >= 4 else 0.0
            med_hy = float(np.median(s_arr[:, 3])) if s_arr.shape[1] >= 4 else 0.0
            self.calib_final_dataset.append({
                "target_x": curr_pt[0], "target_y": curr_pt[1],
                "pitch_deg": med_p, "yaw_deg": med_y,
                "head_pitch_deg": med_hp, "head_yaw_deg": med_hy
            })
        self.calib_point_samples = []
        self.calib_idx += 1
        self.calib_start_t = time.perf_counter()
        if self.calib_idx >= len(self.calib_points):
            self.finish_calibration()

    def finish_calibration(self):
        res = self.calibration.fit(self.calib_final_dataset, l2_reg=0.5)
        self.mode = "tracking"
        self.stacked_widget.setCurrentIndex(0)
        if res.get("status", 0) == 1:
            QMessageBox.information(
                self, "Calibration Complete",
                f"Ridge Regression calibrated successfully:\n"
                f"- R2 Score: {res['r2_score']:.3f}\n"
                f"- Mean Residual: {res['mean_residual_px']:.1f} px\n"
                f"- Calibration points: {res['samples']}"
            )
        else:
            QMessageBox.warning(self, "Calibration Failed", f"Reason: {res.get('error', 'Unknown')}")

    def reset_calibration(self):
        self.calibration.reset()
        QMessageBox.information(self, "Calibration Reset", "Calibration profile cleared. System restored to geometric projection.")

    def recenter_zero(self):
        p, y = self.latest_gaze_angles
        self.calibration.center_at(p, y)
        QMessageBox.information(self, "Zero Centered", f"Zero centered at Pitch={p:+.2f} deg, Yaw={y:+.2f} deg")

    def start_benchmark(self):
        self.mode = "benchmark"
        session_id = self.benchmark.start_recording()
        self.stacked_widget.setCurrentIndex(2)

    def finish_benchmark(self):
        results = self.benchmark.stop_recording_and_evaluate()
        self.last_benchmark_results = results
        self.mode = "tracking"
        self.stacked_widget.setCurrentIndex(0)
        if "error" in results:
            QMessageBox.warning(self, "Benchmark Incomplete", f"Benchmark ended with notice: {results['error']}")
        else:
            self.show_benchmark_report()

    def show_benchmark_report(self):
        rep_text = ""
        plot_path = ""
        if self.last_benchmark_results and "report_path" in self.last_benchmark_results:
            p = self.last_benchmark_results["report_path"]
            plot_path = self.last_benchmark_results.get("plot_path", "")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    rep_text = f.read()
        else:
            rep_files = sorted([os.path.join("mvp/benchmarks", f) for f in os.listdir("mvp/benchmarks") if f.endswith("_report.md")])
            if rep_files:
                with open(rep_files[-1], "r", encoding="utf-8") as f:
                    rep_text = f.read()
                plot_candidate = rep_files[-1].replace("_report.md", "_trajectory.png")
                if os.path.exists(plot_candidate):
                    plot_path = plot_candidate
        if not rep_text:
            QMessageBox.information(self, "No Report Found", "No benchmark tests recorded yet. Click 'Record Final Test Dataset' first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Ground Truth Benchmark Report")
        dlg.resize(920, 720)
        dlg_layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(rep_text)
        txt.setStyleSheet("background-color: #111827; color: #f9fafb; font-family: 'Consolas', monospace; font-size: 13px;")
        dlg_layout.addWidget(txt)
        if plot_path and os.path.exists(plot_path):
            lbl_img = QLabel()
            pix = QPixmap(plot_path)
            lbl_img.setPixmap(pix.scaled(880, 420, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dlg_layout.addWidget(lbl_img)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dlg_layout.addWidget(btn_close)
        dlg.exec()

    def _process_frame_tick(self):
        t_now = time.perf_counter()
        dt = t_now - self.prev_frame_t
        self.prev_frame_t = t_now
        inst_fps = 1.0 / max(dt, 0.001)
        self.fps_val = 0.9 * self.fps_val + 0.1 * inst_fps if self.fps_val > 0.0 else inst_fps
        self.frame_count += 1
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return
        t_pipeline_start = time.perf_counter()
        res = self.pipeline.process_frame(frame_bgr)
        lat_model = res.get("latency_model_ms", 0.0)
        p_deg = res.get("gaze_pitch_deg", 0.0)
        y_deg = res.get("gaze_yaw_deg", 0.0)
        hp_deg = res.get("head_pitch_deg", 0.0)
        hy_deg = res.get("head_yaw_deg", 0.0)
        if self.invert_gaze_x:
            y_deg = -y_deg
        self.latest_gaze_angles = (p_deg, y_deg)
        raw_x, raw_y = self.calibration.predict_uncalibrated(p_deg, y_deg)
        if self.calibration.is_calibrated:
            calib_x, calib_y = self.calibration.predict(p_deg, y_deg, head_pitch_deg=hp_deg, head_yaw_deg=hy_deg)
        else:
            calib_x, calib_y = raw_x, raw_y
        filt_x, filt_y = self.filter.filter(calib_x, calib_y, t_now)
        lat_pipeline = (time.perf_counter() - t_pipeline_start) * 1000.0
        if self.mode == "tracking":
            click_detected = self.cursor_ctrl.update_position(filt_x, filt_y, variance=res.get("attention_variance"))
            self._update_tracking_ui(frame_bgr, res, (raw_x, raw_y), (calib_x, calib_y), (filt_x, filt_y), lat_model, lat_pipeline)
            lm = res.get("landmarks") or {}
            re = lm.get("right_eye") or (0.0, 0.0)
            le = lm.get("left_eye") or (0.0, 0.0)
            fb = res.get("face_bbox") or (0.0, 0.0, 0.0, 0.0)
            self.logger.log_frame({
                "fps": self.fps_val,
                "latency_model_ms": lat_model,
                "latency_total_ms": lat_pipeline,
                "cam_w": self.actual_cam_w,
                "cam_h": self.actual_cam_h,
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "face_detected": res.get("face_detected", False),
                "face_x": fb[0], "face_y": fb[1], "face_w": fb[2], "face_h": fb[3],
                "face_score": res.get("face_score", 0.0),
                "right_eye_x": re[0], "right_eye_y": re[1],
                "left_eye_x": le[0], "left_eye_y": le[1],
                "gaze_pitch_deg": p_deg,
                "gaze_yaw_deg": y_deg,
                "head_pitch_deg": res.get("head_pitch_deg", 0.0),
                "head_yaw_deg": res.get("head_yaw_deg", 0.0),
                "raw_screen_x": raw_x, "raw_screen_y": raw_y,
                "calibrated_screen_x": calib_x, "calibrated_screen_y": calib_y,
                "filtered_screen_x": filt_x, "filtered_screen_y": filt_y,
                "mouse_active": self.cursor_ctrl.enable_control,
                "click_event": click_detected
            })
        elif self.mode == "calibration":
            self._handle_calibration_step(t_now, p_deg, y_deg, hp_deg, hy_deg)
        elif self.mode == "benchmark":
            tx, ty, status_str, progress_val, is_finished = self.benchmark.update_state(self.is_holding_trajectory_key, dt)
            self.benchmark.record_frame(
                frame_bgr, pred_raw=(raw_x, raw_y), pred_calib=(calib_x, calib_y),
                pred_filtered=(filt_x, filt_y), model_res=res,
                is_holding_trajectory_key=self.is_holding_trajectory_key
            )
            self.view_benchmark.set_state((tx, ty), progress_val, status_str, self.is_holding_trajectory_key)
            if is_finished:
                self.finish_benchmark()

    def _handle_calibration_step(self, now: float, pitch_deg: float, yaw_deg: float, head_pitch_deg: float = 0.0, head_yaw_deg: float = 0.0):
        elapsed = now - self.calib_start_t
        duration_per_point = 2.4
        settle_sec = 0.6
        progress = min(1.0, elapsed / duration_per_point)
        if elapsed >= settle_sec:
            self.calib_point_samples.append((pitch_deg, yaw_deg, head_pitch_deg, head_yaw_deg))
        curr_pt = self.calib_points[self.calib_idx]
        self.view_calibration.set_state(self.calib_points, self.calib_idx, progress, f"Point {self.calib_idx + 1}")
        if elapsed >= duration_per_point:
            self._confirm_calib_point()

    def _update_tracking_ui(self, frame_bgr: np.ndarray, res: Dict[str, Any], raw_pt: Tuple[float, float], calib_pt: Tuple[float, float], filt_pt: Tuple[float, float], lat_model: float, lat_pipe: float):
        self.val_fps.setText(f"{self.fps_val:.1f} FPS")
        self.val_latency.setText(f"{lat_pipe:.1f} ms (Model: {lat_model:.2f} ms)")
        p = res.get("gaze_pitch_deg", 0.0); y = res.get("gaze_yaw_deg", 0.0)
        self.val_gaze.setText(f"Pitch: {p:+.1f} deg | Yaw: {y:+.1f} deg")
        hp = res.get("head_pitch_deg", 0.0); hy = res.get("head_yaw_deg", 0.0)
        self.val_head.setText(f"Pitch: {hp:+.1f} deg | Yaw: {hy:+.1f} deg")
        self.val_screen.setText(f"X: {int(filt_pt[0])} px | Y: {int(filt_pt[1])} px")
        if self.calibration.is_calibrated:
            self.val_calib_status.setText(f"Calibrated (R2={self.calibration.r2_score:.2f})")
            self.val_calib_status.setStyleSheet("color: #10b981;")
        else:
            self.val_calib_status.setText("Geometric Projection [Uncalibrated]")
            self.val_calib_status.setStyleSheet("color: #f59e0b;")
        if self.frame_count % 30 == 0:
            vram_mb = 0.0
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                vram_mb = (total_b - free_b) / (1024 ** 2)
            import psutil
            ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            self.cached_mem_str = f"RAM: {ram_gb:.1f} GB | VRAM: {vram_mb:.0f} MB"
            self.val_memory.setText(self.cached_mem_str)
        if self.mirror_camera:
            vis = cv2.flip(frame_bgr, 1)
            frame_w = vis.shape[1]
            if res.get("face_detected", False):
                fx, fy, fw, fh = res["face_bbox"]
                m_fx = frame_w - 1.0 - (fx + fw)
                cv2.rectangle(vis, (int(m_fx), int(fy)), (int(m_fx + fw), int(fy + fh)), (16, 185, 129), 2)
                lm = res.get("landmarks", {})
                r_eye = lm.get("right_eye"); l_eye = lm.get("left_eye")
                if r_eye and l_eye:
                    rex, rey = int(frame_w - 1.0 - r_eye[0]), int(r_eye[1])
                    lex, ley = int(frame_w - 1.0 - l_eye[0]), int(l_eye[1])
                    cv2.circle(vis, (rex, rey), 4, (59, 130, 246), -1)
                    cv2.circle(vis, (lex, ley), 4, (59, 130, 246), -1)
                    p_rad = math.radians(p - self.calibration.resting_pitch_deg); y_rad = math.radians(y)
                    arrow_len = 55.0
                    dx = arrow_len * math.sin(y_rad)
                    dy = -arrow_len * math.sin(p_rad)
                    cv2.arrowedLine(vis, (rex, rey), (int(rex + dx), int(rey + dy)), (239, 68, 68), 2, tipLength=0.25)
                    cv2.arrowedLine(vis, (lex, ley), (int(lex + dx), int(ley + dy)), (239, 68, 68), 2, tipLength=0.25)
        else:
            vis = frame_bgr.copy()
            if res.get("face_detected", False):
                fx, fy, fw, fh = res["face_bbox"]
                cv2.rectangle(vis, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (16, 185, 129), 2)
                lm = res.get("landmarks", {})
                r_eye = lm.get("right_eye"); l_eye = lm.get("left_eye")
                if r_eye and l_eye:
                    rex, rey = int(r_eye[0]), int(r_eye[1])
                    lex, ley = int(l_eye[0]), int(l_eye[1])
                    cv2.circle(vis, (rex, rey), 4, (59, 130, 246), -1)
                    cv2.circle(vis, (lex, ley), 4, (59, 130, 246), -1)
                    p_rad = math.radians(p - self.calibration.resting_pitch_deg); y_rad = math.radians(y)
                    arrow_len = 55.0
                    dx = arrow_len * math.sin(y_rad)
                    dy = -arrow_len * math.sin(p_rad)
                    cv2.arrowedLine(vis, (rex, rey), (int(rex + dx), int(rey + dy)), (239, 68, 68), 2, tipLength=0.25)
                    cv2.arrowedLine(vis, (lex, ley), (int(lex + dx), int(ley + dy)), (239, 68, 68), 2, tipLength=0.25)
        rgb_frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.lbl_video.setPixmap(pix.scaled(
            self.lbl_video.width(), self.lbl_video.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        ))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.logger.close()
        super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser(description="GAZE-CLICK PyQt6 Real-Time System")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="mvp/experiments/yolo26n_gaze_net_yolo26n/run_008_20261014_024920/checkpoints/best_model.pt")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("GAZE-CLICK")
    win = GazeMainWindow(camera_id=args.camera_id, checkpoint_path=args.checkpoint)
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
