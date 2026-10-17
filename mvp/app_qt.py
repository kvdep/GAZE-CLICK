import os, sys, math, time, ctypes, argparse, cv2, torch, numpy as np
from typing import Optional, Dict, Any, Tuple, List

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QStackedWidget,
    QProgressBar, QMessageBox, QDialog, QTextEdit, QScrollArea
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.face_pipeline import GazeTrackingPipeline
from engine.calibration import GazeCalibration
from engine.filter import PointOneEuroFilter
from engine.cursor_controller import CursorController
from engine.telemetry_logger import TelemetryLogger
from engine.benchmark_recorder import BenchmarkRecorder

GLOBAL_QSS = """
QMainWindow {
    background-color: #0c0e12;
}
QWidget {
    color: #f1f5f9;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame.card {
    background-color: #131720;
    border: 1px solid #232a38;
    border-radius: 8px;
}
QFrame.subcard {
    background-color: #0d1017;
    border: 1px solid #1e2430;
    border-radius: 6px;
}
QLabel.title {
    font-size: 15px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.5px;
}
QLabel.section_tag {
    font-size: 11px;
    font-weight: 700;
    color: #38bdf8;
    background-color: #082f49;
    border: 1px solid #0369a1;
    border-radius: 4px;
    padding: 2px 6px;
}
QLabel.metric_label {
    font-size: 12px;
    color: #94a3b8;
}
QLabel.metric_val {
    font-size: 13px;
    font-weight: 600;
    color: #38bdf8;
}
QPushButton {
    background-color: #1a202c;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 9px 14px;
    font-size: 13px;
    font-weight: 600;
    color: #f1f5f9;
}
QPushButton:hover {
    background-color: #242c3d;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #131822;
}
QPushButton.primary {
    background-color: #2563eb;
    border: 1px solid #3b82f6;
    color: #ffffff;
}
QPushButton.primary:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}
QPushButton.success {
    background-color: #059669;
    border: 1px solid #10b981;
    color: #ffffff;
}
QPushButton.success:hover {
    background-color: #047857;
}
QPushButton.danger {
    background-color: #dc2626;
    border: 1px solid #ef4444;
    color: #ffffff;
}
QPushButton.danger:hover {
    background-color: #b91c1c;
}
QPushButton.accent {
    background-color: #0891b2;
    border: 1px solid #06b6d4;
    color: #ffffff;
}
QPushButton.accent:hover {
    background-color: #0e7490;
}
QDialog, QMessageBox {
    background-color: #13161c;
    color: #ffffff;
    border: 1px solid #2d3748;
    border-radius: 8px;
}
QDialog QWidget, QMessageBox QWidget {
    background-color: #13161c;
    color: #ffffff;
}
QDialog QLabel, QMessageBox QLabel {
    color: #ffffff;
    background-color: transparent;
    font-size: 13px;
}
QDialog QPushButton, QMessageBox QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    min-width: 80px;
}
QDialog QPushButton:hover, QMessageBox QPushButton:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}
QDialog QPushButton:pressed, QMessageBox QPushButton:pressed {
    background-color: #1e40af;
}
"""

def show_alert(parent: Optional[QWidget], title: str, message: str, icon: QMessageBox.Icon = QMessageBox.Icon.Information) -> int:
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setIcon(icon)
    msg.setStyleSheet("""
        QMessageBox {
            background-color: #13161c;
            border: 1px solid #2d3748;
            border-radius: 8px;
        }
        QLabel {
            color: #ffffff;
            font-size: 13px;
            background-color: transparent;
        }
        QPushButton {
            background-color: #2563eb;
            color: #ffffff;
            border: 1px solid #3b82f6;
            border-radius: 6px;
            padding: 8px 22px;
            font-size: 13px;
            font-weight: 600;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #1d4ed8;
            border-color: #60a5fa;
        }
        QPushButton:pressed {
            background-color: #1e40af;
        }
    """)
    return msg.exec()

class BenchmarkReportDialog(QDialog):
    def __init__(self, parent=None, report_text: str = "", plot_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Ground Truth Benchmark Report")
        self.resize(1000, 800)
        self.setStyleSheet("""
            QDialog {
                background-color: #13161c;
                border: 1px solid #2d3748;
                border-radius: 10px;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 9px 24px;
                font-size: 13px;
                font-weight: 600;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header_lbl = QLabel("BENCHMARK PERFORMANCE EVALUATION")
        header_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(header_lbl)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet("""
            QTextEdit {
                background-color: #0b0d13;
                color: #e2e8f0;
                border: 1px solid #232936;
                border-radius: 8px;
                font-family: 'Consolas', 'Cascadia Code', monospace;
                font-size: 13px;
                padding: 12px;
            }
        """)
        html_content = self._format_report_to_html(report_text)
        txt.setHtml(html_content)
        layout.addWidget(txt, stretch=3)

        if plot_path and os.path.exists(plot_path):
            img_container = QFrame()
            img_container.setStyleSheet("background-color: #0b0d13; border: 1px solid #232936; border-radius: 8px;")
            img_layout = QVBoxLayout(img_container)
            img_layout.setContentsMargins(8, 8, 8, 8)
            lbl_title = QLabel("Target vs Prediction Trajectory Plot:")
            lbl_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #94a3b8; margin-bottom: 4px;")
            img_layout.addWidget(lbl_title)
            lbl_img = QLabel()
            pix = QPixmap(plot_path)
            lbl_img.setPixmap(pix.scaled(940, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_layout.addWidget(lbl_img)
            layout.addWidget(img_container, stretch=2)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("Close [ESC]")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _format_report_to_html(self, rep_text: str) -> str:
        lines = rep_text.splitlines()
        html_lines = [
            "<html><body style='background-color: #0b0d13; color: #e2e8f0; font-family: Consolas, monospace;'>"
        ]
        in_table = False
        table_html = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("|"):
                if not in_table:
                    in_table = True
                    table_html = ["<table style='width: 100%; border-collapse: collapse; margin: 12px 0; background-color: #11141c; border: 1px solid #2d3748;'>"]
                cells = [c.strip() for c in line_str.split("|")[1:-1]]
                if all(set(c).issubset({"-", " ", ":"}) for c in cells):
                    continue
                row_cells = []
                for cell in cells:
                    cell_clean = cell.replace("**", "").replace("$", "").replace("\\le", "<=").replace("\\ge", ">=").replace("\\circ", "°").replace("\\text{ мм}", " mm")
                    if "PASS" in cell_clean:
                        cell_fmt = "<span style='background-color: #064e3b; color: #34d399; font-weight: bold; padding: 2px 8px; border-radius: 4px;'>PASS</span>"
                    elif "FAIL" in cell_clean:
                        cell_fmt = "<span style='background-color: #7f1d1d; color: #f87171; font-weight: bold; padding: 2px 8px; border-radius: 4px;'>FAIL</span>"
                    else:
                        cell_fmt = cell_clean
                    row_cells.append(f"<td style='padding: 8px 12px; border: 1px solid #232936; color: #cbd5e1;'>{cell_fmt}</td>")
                table_html.append(f"<tr>{''.join(row_cells)}</tr>")
                continue
            else:
                if in_table:
                    table_html.append("</table>")
                    html_lines.append("".join(table_html))
                    in_table = False
                    table_html = []
            if line_str.startswith("==="):
                title = line_str.replace("===", "").strip()
                html_lines.append(f"<div style='font-size: 16px; font-weight: bold; color: #38bdf8; margin: 10px 0; border-bottom: 2px solid #0284c7; padding-bottom: 4px;'>{title}</div>")
            elif line_str.startswith("=="):
                title = line_str.replace("==", "").strip()
                html_lines.append(f"<div style='font-size: 14px; font-weight: bold; color: #818cf8; margin: 14px 0 6px 0;'>{title}</div>")
            elif line_str.startswith(">"):
                quote_text = line_str.replace(">", "").strip().replace("**", "")
                html_lines.append(f"<div style='background-color: #171d2b; border-left: 4px solid #3b82f6; padding: 6px 12px; margin: 6px 0; color: #93c5fd; font-size: 12px;'>{quote_text}</div>")
            elif line_str.startswith("- **"):
                parts = line_str[2:].split(":**", 1)
                if len(parts) == 2:
                    k = parts[0].replace("**", "").strip()
                    v = parts[1].strip().replace("`", "")
                    html_lines.append(f"<div style='margin: 4px 0;'><span style='color: #94a3b8; font-weight: bold;'>{k}:</span> <span style='color: #38bdf8;'>{v}</span></div>")
                else:
                    html_lines.append(f"<div style='margin: 4px 0; color: #cbd5e1;'>{line_str}</div>")
            elif line_str == "---":
                html_lines.append("<hr style='border: none; border-top: 1px solid #1e293b; margin: 10px 0;'/>")
            elif line_str:
                html_lines.append(f"<div style='margin: 3px 0; color: #cbd5e1;'>{line_str}</div>")
        if in_table:
            table_html.append("</table>")
            html_lines.append("".join(table_html))
        html_lines.append("</body></html>")
        return "".join(html_lines)

class PipelineWorker(QThread):
    frame_processed = pyqtSignal(QImage, dict)
    worker_error = pyqtSignal(str)

    def __init__(self, camera_id: int = 0, checkpoint_path: str = "", yunet_path: str = "", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.checkpoint_path = checkpoint_path
        self.yunet_path = yunet_path
        self.running = False
        self.mirror_camera = True
        self.invert_gaze_x = False
        self.resting_pitch_deg = -8.99
        self.actual_cam_w = 1280
        self.actual_cam_h = 720

    def run(self):
        try:
            pipeline = GazeTrackingPipeline(checkpoint_path=self.checkpoint_path, yunet_path=self.yunet_path)
        except Exception as e:
            self.worker_error.emit(f"Failed to initialize AI Pipeline: {str(e)}")
            return
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_cam_w = w if w > 0 else 1280
        self.actual_cam_h = h if h > 0 else 720
        self.running = True
        prev_t = time.perf_counter()
        fps_ema = 0.0
        while self.running:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            dt = max(1e-4, now - prev_t)
            prev_t = now
            inst_fps = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps if fps_ema > 0.0 else inst_fps
            t_pipe_start = time.perf_counter()
            res = pipeline.process_frame(frame_bgr)
            lat_pipe = (time.perf_counter() - t_pipe_start) * 1000.0
            p = res.get("gaze_pitch_deg", 0.0)
            y = res.get("gaze_yaw_deg", 0.0)
            if self.invert_gaze_x:
                y = -y
            if self.mirror_camera:
                vis = cv2.flip(frame_bgr, 1)
                fw = vis.shape[1]
                if res.get("face_detected", False):
                    fx, fy, bw, bh = res["face_bbox"]
                    m_fx = fw - 1.0 - (fx + bw)
                    cv2.rectangle(vis, (int(m_fx), int(fy)), (int(m_fx + bw), int(fy + bh)), (16, 185, 129), 2)
                    lm = res.get("landmarks", {})
                    re = lm.get("right_eye"); le = lm.get("left_eye")
                    if re and le:
                        rex, rey = int(fw - 1.0 - re[0]), int(re[1])
                        lex, ley = int(fw - 1.0 - le[0]), int(le[1])
                        cv2.circle(vis, (rex, rey), 4, (59, 130, 246), -1)
                        cv2.circle(vis, (lex, ley), 4, (59, 130, 246), -1)
                        p_rad = math.radians(p - self.resting_pitch_deg); y_rad = math.radians(y)
                        arrow_len = 55.0
                        dx = arrow_len * math.sin(y_rad)
                        dy = -arrow_len * math.sin(p_rad)
                        cv2.arrowedLine(vis, (rex, rey), (int(rex + dx), int(rey + dy)), (239, 68, 68), 2, tipLength=0.25)
                        cv2.arrowedLine(vis, (lex, ley), (int(lex + dx), int(ley + dy)), (239, 68, 68), 2, tipLength=0.25)
            else:
                vis = frame_bgr.copy()
                if res.get("face_detected", False):
                    fx, fy, bw, bh = res["face_bbox"]
                    cv2.rectangle(vis, (int(fx), int(fy)), (int(fx + bw), int(fy + bh)), (16, 185, 129), 2)
                    lm = res.get("landmarks", {})
                    re = lm.get("right_eye"); le = lm.get("left_eye")
                    if re and le:
                        rex, rey = int(re[0]), int(re[1])
                        lex, ley = int(le[0]), int(le[1])
                        cv2.circle(vis, (rex, rey), 4, (59, 130, 246), -1)
                        cv2.circle(vis, (lex, ley), 4, (59, 130, 246), -1)
                        p_rad = math.radians(p - self.resting_pitch_deg); y_rad = math.radians(y)
                        arrow_len = 55.0
                        dx = -arrow_len * math.sin(y_rad)
                        dy = -arrow_len * math.sin(p_rad)
                        cv2.arrowedLine(vis, (rex, rey), (int(rex + dx), int(rey + dy)), (239, 68, 68), 2, tipLength=0.25)
                        cv2.arrowedLine(vis, (lex, ley), (int(lex + dx), int(ley + dy)), (239, 68, 68), 2, tipLength=0.25)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            vh, vw, vc = rgb.shape
            qimg = QImage(rgb.data, vw, vh, vc * vw, QImage.Format.Format_RGB888).copy()
            telemetry = {
                "fps": fps_ema,
                "latency_pipeline_ms": lat_pipe,
                "latency_model_ms": res.get("latency_model_ms", 0.0),
                "gaze_pitch_deg": p,
                "gaze_yaw_deg": y,
                "head_pitch_deg": res.get("head_pitch_deg", 0.0),
                "head_yaw_deg": res.get("head_yaw_deg", 0.0),
                "face_detected": res.get("face_detected", False),
                "face_bbox": res.get("face_bbox", (0.0, 0.0, 0.0, 0.0)),
                "face_score": res.get("face_score", 0.0),
                "landmarks": res.get("landmarks", {}),
                "attention_variance": res.get("attention_variance", None),
                "raw_frame": frame_bgr,
                "model_res": res,
                "cam_w": self.actual_cam_w,
                "cam_h": self.actual_cam_h
            }
            self.frame_processed.emit(qimg, telemetry)
        cap.release()

    def stop(self):
        self.running = False

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
        painter.fillRect(self.rect(), QColor(11, 13, 18))
        w = self.width()
        h = self.height()
        if not self.points or self.current_idx >= len(self.points):
            return
        pt = self.points[self.current_idx]
        tx, ty = pt[0], pt[1]
        draw_x = (tx / max(self.ref_screen_w, 1.0)) * w
        draw_y = (ty / max(self.ref_screen_h, 1.0)) * h
        radius_outer = 64.0 * (1.0 - 0.65 * self.progress)
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
        title_text = f"Calibration Target Point {self.current_idx + 1} of {len(self.points)}"
        painter.drawText(QRectF(0, 30, float(w), 36), Qt.AlignmentFlag.AlignCenter, title_text)
        font_sub = QFont("Segoe UI", 13)
        painter.setFont(font_sub)
        painter.setPen(QColor(209, 213, 219))
        sub_text = "Fixate on the red marker. Press [SPACE] to record or wait for countdown"
        painter.drawText(QRectF(0, 70, float(w), 30), Qt.AlignmentFlag.AlignCenter, sub_text)
        bar_w = 480.0; bar_h = 10.0
        bar_x = (w - bar_w) / 2.0; bar_y = h - 60.0
        painter.setBrush(QBrush(QColor(30, 41, 59)))
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
        pen_cross = QPen(QColor(59, 130, 246, 140), 1, Qt.PenStyle.DashLine)
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
        warn_text = "[GROUND TRUTH BENCHMARK - STRICT EVALUATION - NO TRAINING DATA]"
        painter.drawText(QRectF(0, 15, float(w), 24), Qt.AlignmentFlag.AlignCenter, warn_text)
        font_info = QFont("Segoe UI", 20, QFont.Weight.Bold)
        painter.setFont(font_info)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(0, 42, float(w), 36), Qt.AlignmentFlag.AlignCenter, self.status_text)
        font_hint = QFont("Segoe UI", 13)
        painter.setFont(font_hint)
        painter.setPen(QColor(156, 163, 175))
        if "TRAJECTORY" in self.status_text or "SHIFT" in self.status_text or "Circle" in self.status_text:
            hint = "HOLD [SHIFT] or [G] to move target continuously. Follow with eyes. Release to pause. [ESC] to exit."
        else:
            hint = "Fixate on the red target, then PRESS [SPACE] to capture ground truth. [ESC] to exit."
        painter.drawText(QRectF(0, 80, float(w), 26), Qt.AlignmentFlag.AlignCenter, hint)
        bar_w = 620.0; bar_h = 12.0
        bar_x = (w - bar_w) / 2.0; bar_y = h - 60.0
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 6.0, 6.0)
        fill_w = bar_w * min(1.0, max(0.0, self.progress))
        painter.setBrush(QBrush(QColor(59, 130, 246)))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 6.0, 6.0)

class GazeMainWindow(QMainWindow):
    def __init__(self, camera_id: int = 0, checkpoint_path: str = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt", yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx", log_dir: str = "mvp/logs") -> None:
        super().__init__()
        self.setWindowTitle("GAZE-CLICK // NEURAL REAL-TIME GAZE CONTROL")
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

        self.calibration = GazeCalibration(screen_w=self.screen_w, screen_h=self.screen_h, profile_path="mvp/calibration_profile.json")
        self.filter = PointOneEuroFilter(min_cutoff=0.5, beta=0.005, d_cutoff=1.0, deadzone_radius_px=8.0)
        self.cursor_ctrl = CursorController(screen_w=self.screen_w, screen_h=self.screen_h, enable_control=False)
        self.logger = TelemetryLogger(log_dir=self.log_dir, session_prefix="session")
        self.benchmark = BenchmarkRecorder(output_dir="mvp/benchmarks", screen_w=self.screen_w, screen_h=self.screen_h)

        self.mirror_camera: bool = True
        self.invert_gaze_x: bool = False
        self.is_holding_trajectory_key: bool = False
        self.latest_gaze_angles: Tuple[float, float] = (-8.99, 0.0)
        self.latest_head_angles: Tuple[float, float] = (0.0, 0.0)
        self.mode: str = "tracking"
        self.calib_points = self._generate_calibration_grid()
        self.calib_idx = 0
        self.calib_start_t = 0.0
        self.calib_point_samples = []
        self.calib_final_dataset = []
        self.fps_val = 0.0
        self.frame_count = 0
        self.cached_mem_str = "RAM: 0.0 GB | VRAM: 0 MB"
        self.last_benchmark_results = None

        self._build_ui()

        self.worker = PipelineWorker(
            camera_id=self.camera_id,
            checkpoint_path=self.checkpoint_path,
            yunet_path=self.yunet_path,
            parent=self
        )
        self.worker.mirror_camera = self.mirror_camera
        self.worker.invert_gaze_x = self.invert_gaze_x
        self.worker.resting_pitch_deg = self.calibration.resting_pitch_deg
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.worker_error.connect(self._on_worker_error)
        self.worker.start()

        self.prev_anim_t = time.perf_counter()
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start(16)

    def _generate_calibration_grid(self) -> List[Tuple[float, float]]:
        pts = []
        nx, ny = 3, 3
        margin_x = 0.12
        xs = np.linspace(margin_x * self.screen_w, (1.0 - margin_x) * self.screen_w, nx)
        ys = np.linspace(0.18 * self.screen_h, 0.86 * self.screen_h, ny)
        for y in ys:
            for x in xs:
                pts.append((float(x), float(y)))
        extreme_points = [
            (float(0.06 * self.screen_w), float(0.50 * self.screen_h)),
            (float(0.94 * self.screen_w), float(0.50 * self.screen_h)),
            (float(0.50 * self.screen_w), float(0.14 * self.screen_h)),
            (float(0.50 * self.screen_w), float(0.89 * self.screen_h)),
        ]
        for ex_x, ex_y in extreme_points:
            pts.append((ex_x, ex_y))
        return pts

    def _build_ui(self):
        self.setStyleSheet(GLOBAL_QSS)
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.view_dashboard = QWidget(self)
        dash_layout = QHBoxLayout(self.view_dashboard)
        dash_layout.setContentsMargins(20, 20, 20, 20)
        dash_layout.setSpacing(18)

        left_box = QVBoxLayout()
        left_box.setSpacing(14)

        video_card = QFrame()
        video_card.setProperty("class", "card")
        video_card_layout = QVBoxLayout(video_card)
        video_card_layout.setContentsMargins(14, 14, 14, 14)
        video_card_layout.setSpacing(10)

        video_head_row = QHBoxLayout()
        lbl_video_title = QLabel("OPTICAL FEED // LANDMARKS & GAZE VECTOR")
        lbl_video_title.setProperty("class", "title")
        video_head_row.addWidget(lbl_video_title)
        video_head_row.addStretch(1)
        self.lbl_pipeline_tag = QLabel("WORKER: ACTIVE (60 FPS GUI)")
        self.lbl_pipeline_tag.setProperty("class", "section_tag")
        video_head_row.addWidget(self.lbl_pipeline_tag)
        video_card_layout.addLayout(video_head_row)

        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setMinimumSize(720, 480)
        self.lbl_video.setStyleSheet("background-color: #08090d; border: 1px solid #1c222e; border-radius: 6px;")
        video_card_layout.addWidget(self.lbl_video, stretch=1)

        vid_btn_row = QHBoxLayout()
        vid_btn_row.setSpacing(10)
        self.btn_mirror = QPushButton("Mirror Feed: ON")
        self.btn_mirror.clicked.connect(self.toggle_mirror)
        vid_btn_row.addWidget(self.btn_mirror)

        self.btn_invert_x = QPushButton("Invert X: OFF")
        self.btn_invert_x.clicked.connect(self.toggle_invert_x)
        vid_btn_row.addWidget(self.btn_invert_x)

        btn_gain_down = QPushButton("Gain - [-]")
        btn_gain_down.clicked.connect(lambda: self.adjust_gain(-0.15))
        vid_btn_row.addWidget(btn_gain_down)

        btn_gain_up = QPushButton("Gain + [+/=]")
        btn_gain_up.clicked.connect(lambda: self.adjust_gain(+0.15))
        vid_btn_row.addWidget(btn_gain_up)

        video_card_layout.addLayout(vid_btn_row)
        left_box.addWidget(video_card, stretch=1)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(14)

        telem_card = QFrame()
        telem_card.setProperty("class", "card")
        telem_layout = QVBoxLayout(telem_card)
        telem_layout.setContentsMargins(16, 16, 16, 16)
        telem_layout.setSpacing(12)

        lbl_telem_head = QLabel("SYSTEM TELEMETRY & INFERENCE")
        lbl_telem_head.setProperty("class", "title")
        telem_layout.addWidget(lbl_telem_head)

        grid_telem = QGridLayout()
        grid_telem.setHorizontalSpacing(14)
        grid_telem.setVerticalSpacing(8)

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
        self.val_gain = QLabel(f"Gain: {self.calibration.gain_x:.2f}x")
        self.val_gain.setProperty("class", "metric_val")
        self.val_calib_status = QLabel("Uncalibrated")
        self.val_calib_status.setProperty("class", "metric_val")
        self.val_mouse_status = QLabel("Disabled [Preview Mode]")
        self.val_mouse_status.setProperty("class", "metric_val")
        self.val_memory = QLabel("RAM: 0.0 GB | VRAM: 0 MB")
        self.val_memory.setProperty("class", "metric_val")

        metrics = [
            ("Pipeline Throughput:", self.val_fps),
            ("Processing Latency:", self.val_latency),
            ("Gaze Angle Vector:", self.val_gaze),
            ("Head Pose Orientation:", self.val_head),
            ("Target Screen Point:", self.val_screen),
            ("Sensitivity Multiplier:", self.val_gain),
            ("Calibration Profile:", self.val_calib_status),
            ("Mouse Control Mode:", self.val_mouse_status),
            ("Memory Footprint:", self.val_memory),
        ]
        for row_i, (k_name, v_wid) in enumerate(metrics):
            lbl_k = QLabel(k_name)
            lbl_k.setProperty("class", "metric_label")
            grid_telem.addWidget(lbl_k, row_i, 0)
            grid_telem.addWidget(v_wid, row_i, 1)

        telem_layout.addLayout(grid_telem)
        right_panel.addWidget(telem_card)

        ctrl_card = QFrame()
        ctrl_card.setProperty("class", "card")
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(16, 16, 16, 16)
        ctrl_layout.setSpacing(10)

        lbl_ctrl_head = QLabel("CONTROL ACTIONS")
        lbl_ctrl_head.setProperty("class", "title")
        ctrl_layout.addWidget(lbl_ctrl_head)

        self.btn_recenter = QPushButton("Recenter View [R]")
        self.btn_recenter.setProperty("class", "accent")
        self.btn_recenter.clicked.connect(self.recenter_view)
        ctrl_layout.addWidget(self.btn_recenter)

        self.btn_mouse = QPushButton("Enable Mouse Control [M]")
        self.btn_mouse.clicked.connect(self.toggle_mouse)
        ctrl_layout.addWidget(self.btn_mouse)

        self.btn_calibrate = QPushButton("Start 13-Point Calibration [C]")
        self.btn_calibrate.setProperty("class", "primary")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        ctrl_layout.addWidget(self.btn_calibrate)

        self.btn_benchmark = QPushButton("Record Final Test Dataset [B]")
        self.btn_benchmark.setProperty("class", "danger")
        self.btn_benchmark.clicked.connect(self.start_benchmark)
        ctrl_layout.addWidget(self.btn_benchmark)

        self.btn_report = QPushButton("View Benchmark Report [V]")
        self.btn_report.clicked.connect(self.show_benchmark_report)
        ctrl_layout.addWidget(self.btn_report)

        self.btn_reset_calib = QPushButton("Reset Calibration Profile")
        self.btn_reset_calib.clicked.connect(self.reset_calibration)
        ctrl_layout.addWidget(self.btn_reset_calib)

        self.btn_fullscreen = QPushButton("Toggle Fullscreen [F11]")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        ctrl_layout.addWidget(self.btn_fullscreen)

        right_panel.addWidget(ctrl_card)

        helper_card = QFrame()
        helper_card.setProperty("class", "subcard")
        help_layout = QVBoxLayout(helper_card)
        help_layout.setContentsMargins(12, 10, 12, 10)
        lbl_help = QLabel("HOTKEYS: [R] Recenter | [M] Mouse | [C] Calibrate | [B] Benchmark | [V] Report | [Shift/G] Track")
        lbl_help.setStyleSheet("font-size: 11px; color: #64748b; font-weight: bold;")
        lbl_help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_layout.addWidget(lbl_help)
        right_panel.addWidget(helper_card)

        right_panel.addStretch(1)

        dash_layout.addLayout(left_box, stretch=3)
        dash_layout.addLayout(right_panel, stretch=2)

        self.view_calibration = CalibrationCanvas(self, ref_screen_w=self.screen_w, ref_screen_h=self.screen_h)
        self.view_benchmark = BenchmarkCanvas(self, ref_screen_w=self.screen_w, ref_screen_h=self.screen_h)

        self.stacked_widget.addWidget(self.view_dashboard)
        self.stacked_widget.addWidget(self.view_calibration)
        self.stacked_widget.addWidget(self.view_benchmark)
        self.stacked_widget.setCurrentIndex(0)
        self.resize(1440, 920)

    def _on_worker_error(self, err_msg: str):
        show_alert(self, "Pipeline Error", err_msg, QMessageBox.Icon.Critical)

    def _on_anim_tick(self):
        t_now = time.perf_counter()
        dt = max(1e-4, t_now - self.prev_anim_t)
        self.prev_anim_t = t_now
        if self.mode == "benchmark":
            tx, ty, status_str, progress_val, is_finished = self.benchmark.update_state(self.is_holding_trajectory_key, dt)
            self.view_benchmark.set_state((tx, ty), progress_val, status_str, self.is_holding_trajectory_key)
            if is_finished:
                self.finish_benchmark()
        elif self.mode == "calibration":
            elapsed = t_now - self.calib_start_t
            duration_per_point = 2.4
            progress = min(1.0, elapsed / duration_per_point)
            if self.calib_idx < len(self.calib_points):
                self.view_calibration.set_state(self.calib_points, self.calib_idx, progress, f"Point {self.calib_idx + 1}")
            if elapsed >= duration_per_point:
                self._confirm_calib_point()

    def _on_frame_processed(self, qimg: QImage, telem: dict):
        p = telem["gaze_pitch_deg"]
        y = telem["gaze_yaw_deg"]
        hp = telem["head_pitch_deg"]
        hy = telem["head_yaw_deg"]
        self.latest_gaze_angles = (p, y)
        self.latest_head_angles = (hp, hy)
        raw_x, raw_y = self.calibration.predict_uncalibrated(p, y)
        if self.calibration.is_calibrated:
            calib_x, calib_y = self.calibration.predict(p, y, head_pitch_deg=hp, head_yaw_deg=hy)
        else:
            calib_x, calib_y = raw_x, raw_y
        filt_x, filt_y = self.filter.filter(calib_x, calib_y, time.perf_counter())
        lat_model = telem.get("latency_model_ms", 0.0)
        lat_pipe = telem.get("latency_pipeline_ms", 0.0)
        worker_fps = telem.get("fps", 0.0)
        self.fps_val = worker_fps
        self.frame_count += 1

        if self.mode == "tracking":
            click_detected = self.cursor_ctrl.update_position(filt_x, filt_y, variance=telem.get("attention_variance"))
            self._update_tracking_ui(qimg, telem, (raw_x, raw_y), (calib_x, calib_y), (filt_x, filt_y), lat_model, lat_pipe)
            lm = telem.get("landmarks") or {}
            re = lm.get("right_eye") or (0.0, 0.0)
            le = lm.get("left_eye") or (0.0, 0.0)
            fb = telem.get("face_bbox") or (0.0, 0.0, 0.0, 0.0)
            self.logger.log_frame({
                "fps": self.fps_val,
                "latency_model_ms": lat_model,
                "latency_total_ms": lat_pipe,
                "cam_w": telem.get("cam_w", 1280),
                "cam_h": telem.get("cam_h", 720),
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "face_detected": telem.get("face_detected", False),
                "face_x": fb[0], "face_y": fb[1], "face_w": fb[2], "face_h": fb[3],
                "face_score": telem.get("face_score", 0.0),
                "right_eye_x": re[0], "right_eye_y": re[1],
                "left_eye_x": le[0], "left_eye_y": le[1],
                "gaze_pitch_deg": p,
                "gaze_yaw_deg": y,
                "head_pitch_deg": hp,
                "head_yaw_deg": hy,
                "raw_screen_x": raw_x, "raw_screen_y": raw_y,
                "calibrated_screen_x": calib_x, "calibrated_screen_y": calib_y,
                "filtered_screen_x": filt_x, "filtered_screen_y": filt_y,
                "mouse_active": self.cursor_ctrl.enable_control,
                "click_event": click_detected
            })
        elif self.mode == "calibration":
            now = time.perf_counter()
            elapsed = now - self.calib_start_t
            if elapsed >= 0.6:
                self.calib_point_samples.append((p, y, hp, hy))
        elif self.mode == "benchmark":
            self.benchmark.record_frame(
                telem["raw_frame"],
                pred_raw=(raw_x, raw_y),
                pred_calib=(calib_x, calib_y),
                pred_filtered=(filt_x, filt_y),
                model_res=telem["model_res"],
                is_holding_trajectory_key=self.is_holding_trajectory_key
            )

    def _update_tracking_ui(self, qimg: QImage, telem: dict, raw_pt: Tuple[float, float], calib_pt: Tuple[float, float], filt_pt: Tuple[float, float], lat_model: float, lat_pipe: float):
        self.val_fps.setText(f"{self.fps_val:.1f} FPS")
        self.val_latency.setText(f"{lat_pipe:.1f} ms (Model: {lat_model:.2f} ms)")
        p = telem.get("gaze_pitch_deg", 0.0); y = telem.get("gaze_yaw_deg", 0.0)
        self.val_gaze.setText(f"Pitch: {p:+.1f} deg | Yaw: {y:+.1f} deg")
        hp = telem.get("head_pitch_deg", 0.0); hy = telem.get("head_yaw_deg", 0.0)
        self.val_head.setText(f"Pitch: {hp:+.1f} deg | Yaw: {hy:+.1f} deg")
        self.val_screen.setText(f"X: {int(filt_pt[0])} px | Y: {int(filt_pt[1])} px")
        if self.calibration.is_calibrated:
            self.val_calib_status.setText(f"Calibrated (R2={self.calibration.r2_score:.2f}, Offset: {self.calibration.offset_x:+.0f},{self.calibration.offset_y:+.0f})")
            self.val_calib_status.setStyleSheet("color: #10b981;")
        else:
            self.val_calib_status.setText(f"Geometric Projection [Offset: {self.calibration.offset_x:+.0f},{self.calibration.offset_y:+.0f}]")
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
        pix = QPixmap.fromImage(qimg)
        self.lbl_video.setPixmap(pix.scaled(
            self.lbl_video.width(), self.lbl_video.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        ))

    def adjust_gain(self, delta: float):
        new_gx = self.calibration.gain_x + delta
        new_gy = self.calibration.gain_y + delta * 0.85
        self.calibration.set_gain(new_gx, new_gy)
        self.val_gain.setText(f"Gain: {self.calibration.gain_x:.2f}x")

    def recenter_view(self):
        p, y = self.latest_gaze_angles
        hp, hy = self.latest_head_angles
        ox, oy = self.calibration.recenter(
            p, y,
            target_x=self.screen_w / 2.0,
            target_y=self.screen_h / 2.0,
            head_pitch_deg=hp,
            head_yaw_deg=hy
        )
        self.filter.reset()
        if hasattr(self, "worker"):
            self.worker.resting_pitch_deg = self.calibration.resting_pitch_deg
        status_msg = f"Recentered [Offset X: {ox:+.0f}px, Y: {oy:+.0f}px]"
        self.val_calib_status.setText(status_msg)
        self.val_calib_status.setStyleSheet("color: #38bdf8; font-weight: bold;")

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
                self.recenter_view()
        elif key == Qt.Key.Key_Z:
            if self.mode == "tracking":
                self.recenter_view()
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
        if hasattr(self, "worker"):
            self.worker.mirror_camera = self.mirror_camera
        state = "ON" if self.mirror_camera else "OFF"
        self.btn_mirror.setText(f"Mirror Feed: {state}")

    def toggle_invert_x(self):
        self.invert_gaze_x = not self.invert_gaze_x
        if hasattr(self, "worker"):
            self.worker.invert_gaze_x = self.invert_gaze_x
        state = "ON" if self.invert_gaze_x else "OFF"
        self.btn_invert_x.setText(f"Invert X: {state}")

    def toggle_mouse(self):
        new_state = not self.cursor_ctrl.enable_control
        self.cursor_ctrl.set_enabled(new_state)
        if new_state:
            self.btn_mouse.setText("Disable Mouse Control [M]")
            self.btn_mouse.setProperty("class", "success")
            self.val_mouse_status.setText("ENABLED [Active Control]")
            self.val_mouse_status.setStyleSheet("color: #10b981; font-weight: bold;")
        else:
            self.btn_mouse.setText("Enable Mouse Control [M]")
            self.btn_mouse.setProperty("class", "")
            self.val_mouse_status.setText("Disabled [Preview Mode]")
            self.val_mouse_status.setStyleSheet("color: #60a5fa; font-weight: bold;")
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
        res = self.calibration.fit(self.calib_final_dataset, l2_reg=1.0)
        self.mode = "tracking"
        self.stacked_widget.setCurrentIndex(0)
        if res.get("status", 0) == 1:
            show_alert(
                self, "Calibration Complete",
                f"Ridge Regression calibrated successfully:\n"
                f"- R2 Score: {res['r2_score']:.3f}\n"
                f"- Mean Residual: {res['mean_residual_px']:.1f} px\n"
                f"- Calibration points: {res['samples']}"
            )
        else:
            show_alert(self, "Calibration Failed", f"Reason: {res.get('error', 'Unknown')}", QMessageBox.Icon.Warning)

    def reset_calibration(self):
        self.calibration.reset()
        self.filter.reset()
        show_alert(self, "Calibration Reset", "Calibration profile cleared. System restored to geometric projection.")
        self.val_calib_status.setText("Geometric Projection [Uncalibrated]")
        self.val_calib_status.setStyleSheet("color: #f59e0b;")

    def start_benchmark(self):
        self.mode = "benchmark"
        self.benchmark.on_calibration_updated = self._on_benchmark_calibration_updated
        session_id = self.benchmark.start_recording()
        self.stacked_widget.setCurrentIndex(2)

    def _on_benchmark_calibration_updated(self, calib):
        self.calibration = calib
        self.val_calib_status.setText(f"Calibrated ({calib.points_recorded} pts, R2={calib.r2_score:.2f})")
        self.val_calib_status.setStyleSheet("color: #10b981; font-weight: bold;")

    def finish_benchmark(self):
        results = self.benchmark.stop_recording_and_evaluate()
        self.last_benchmark_results = results
        self.mode = "tracking"
        self.stacked_widget.setCurrentIndex(0)
        if self.benchmark.session_calibration is not None:
            self.calibration = self.benchmark.session_calibration
            self.val_calib_status.setText(f"Calibrated ({self.calibration.points_recorded} pts, R2={self.calibration.r2_score:.2f})")
            self.val_calib_status.setStyleSheet("color: #10b981; font-weight: bold;")
        if "error" in results:
            show_alert(self, "Benchmark Incomplete", f"Benchmark ended with notice: {results['error']}", QMessageBox.Icon.Warning)
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
            show_alert(self, "No Report Found", "No benchmark tests recorded yet. Click 'Record Final Test Dataset' first.")
            return
        dlg = BenchmarkReportDialog(self, report_text=rep_text, plot_path=plot_path)
        dlg.exec()

    def closeEvent(self, event):
        self.anim_timer.stop()
        if hasattr(self, "worker"):
            try:
                self.worker.frame_processed.disconnect()
            except Exception:
                pass
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
        self.logger.close()
        super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser(description="GAZE-CLICK PyQt6 Real-Time System")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("GAZE-CLICK")
    app.setStyleSheet(GLOBAL_QSS)
    win = GazeMainWindow(camera_id=args.camera_id, checkpoint_path=args.checkpoint)
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
