import os, sys, time, csv, psutil, torch, numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

class TelemetryLogger:
    def __init__(self, log_dir: str = "mvp/logs", session_prefix: str = "session") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        msk_tz = timezone(timedelta(hours=3))
        self.start_dt = datetime.now(msk_tz)
        ts_str = self.start_dt.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{session_prefix}_{ts_str}"
        self.csv_path = os.path.join(self.log_dir, f"{self.session_id}.csv")
        self.summary_md_path = os.path.join(self.log_dir, f"{self.session_id}_summary.md")
        self.process = psutil.Process()
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.fieldnames = [
            "timestamp_msk", "elapsed_sec", "frame_idx", "fps",
            "latency_model_ms", "latency_total_ms",
            "ram_used_gb", "ram_free_gb", "vram_used_mb", "vram_free_mb",
            "cam_w", "cam_h", "screen_w", "screen_h",
            "face_detected", "face_x", "face_y", "face_w", "face_h", "face_score",
            "right_eye_x", "right_eye_y", "left_eye_x", "left_eye_y",
            "gaze_pitch_deg", "gaze_yaw_deg",
            "head_pitch_deg", "head_yaw_deg",
            "raw_screen_x", "raw_screen_y",
            "calibrated_screen_x", "calibrated_screen_y",
            "filtered_screen_x", "filtered_screen_y",
            "mouse_active", "click_event"
        ]
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()
        self.start_mono = time.perf_counter()
        self.frames_logged = 0
        self.fps_records = []
        self.model_latencies = []
        self.total_latencies = []
        self.ram_records = []
        self.vram_records = []
        self.pitch_records = []
        self.yaw_records = []
        self.clicks_count = 0

    def log_frame(self, data: Dict[str, Any]) -> None:
        now_mono = time.perf_counter()
        elapsed = now_mono - self.start_mono
        self.frames_logged += 1
        fps = float(data.get("fps", 0.0))
        lat_model = float(data.get("latency_model_ms", 0.0))
        lat_total = float(data.get("latency_total_ms", 0.0))
        ram_gb = float(self.process.memory_info().rss / (1024 ** 3))
        ram_free_gb = float(psutil.virtual_memory().available / (1024 ** 3))
        vram_mb = 0.0; vram_free_mb = 0.0
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            vram_mb = float((total_b - free_b) / (1024 ** 2))
            vram_free_mb = float(free_b / (1024 ** 2))
        self.fps_records.append(fps)
        self.model_latencies.append(lat_model)
        self.total_latencies.append(lat_total)
        self.ram_records.append(ram_gb)
        self.vram_records.append(vram_mb)
        if data.get("face_detected", False):
            self.pitch_records.append(float(data.get("gaze_pitch_deg", 0.0)))
            self.yaw_records.append(float(data.get("gaze_yaw_deg", 0.0)))
        if data.get("click_event", False):
            self.clicks_count += 1
        msk_tz = timezone(timedelta(hours=3))
        row = {
            "timestamp_msk": datetime.now(msk_tz).strftime("%H:%M:%S.%f")[:-3],
            "elapsed_sec": round(elapsed, 3),
            "frame_idx": self.frames_logged,
            "fps": round(fps, 1),
            "latency_model_ms": round(lat_model, 2),
            "latency_total_ms": round(lat_total, 2),
            "ram_used_gb": round(ram_gb, 2),
            "ram_free_gb": round(ram_free_gb, 2),
            "vram_used_mb": round(vram_mb, 1),
            "vram_free_mb": round(vram_free_mb, 1),
            "cam_w": data.get("cam_w", 0),
            "cam_h": data.get("cam_h", 0),
            "screen_w": data.get("screen_w", 0),
            "screen_h": data.get("screen_h", 0),
            "face_detected": int(bool(data.get("face_detected", False))),
            "face_x": round(float(data.get("face_x", 0.0)), 1),
            "face_y": round(float(data.get("face_y", 0.0)), 1),
            "face_w": round(float(data.get("face_w", 0.0)), 1),
            "face_h": round(float(data.get("face_h", 0.0)), 1),
            "face_score": round(float(data.get("face_score", 0.0)), 3),
            "right_eye_x": round(float(data.get("right_eye_x", 0.0)), 1),
            "right_eye_y": round(float(data.get("right_eye_y", 0.0)), 1),
            "left_eye_x": round(float(data.get("left_eye_x", 0.0)), 1),
            "left_eye_y": round(float(data.get("left_eye_y", 0.0)), 1),
            "gaze_pitch_deg": round(float(data.get("gaze_pitch_deg", 0.0)), 2),
            "gaze_yaw_deg": round(float(data.get("gaze_yaw_deg", 0.0)), 2),
            "head_pitch_deg": round(float(data.get("head_pitch_deg", 0.0)), 2),
            "head_yaw_deg": round(float(data.get("head_yaw_deg", 0.0)), 2),
            "raw_screen_x": round(float(data.get("raw_screen_x", 0.0)), 1),
            "raw_screen_y": round(float(data.get("raw_screen_y", 0.0)), 1),
            "calibrated_screen_x": round(float(data.get("calibrated_screen_x", 0.0)), 1),
            "calibrated_screen_y": round(float(data.get("calibrated_screen_y", 0.0)), 1),
            "filtered_screen_x": round(float(data.get("filtered_screen_x", 0.0)), 1),
            "filtered_screen_y": round(float(data.get("filtered_screen_y", 0.0)), 1),
            "mouse_active": int(bool(data.get("mouse_active", False))),
            "click_event": int(bool(data.get("click_event", False)))
        }
        if self.csv_file is None or self.csv_file.closed:
            return
        self.writer.writerow(row)
        if self.frames_logged % 30 == 0:
            self.csv_file.flush()

    def close(self) -> str:
        if self.csv_file is None or self.csv_file.closed:
            return ""
        self.csv_file.flush(); self.csv_file.close()
        total_time = time.perf_counter() - self.start_mono
        mean_fps = float(np.mean(self.fps_records)) if self.fps_records else 0.0
        min_fps = float(np.min(self.fps_records)) if self.fps_records else 0.0
        max_fps = float(np.max(self.fps_records)) if self.fps_records else 0.0
        mean_lat_model = float(np.mean(self.model_latencies)) if self.model_latencies else 0.0
        mean_lat_total = float(np.mean(self.total_latencies)) if self.total_latencies else 0.0
        p95_lat_total = float(np.percentile(self.total_latencies, 95)) if self.total_latencies else 0.0
        mean_ram = float(np.mean(self.ram_records)) if self.ram_records else 0.0
        peak_ram = float(np.max(self.ram_records)) if self.ram_records else 0.0
        mean_vram = float(np.mean(self.vram_records)) if self.vram_records else 0.0
        peak_vram = float(np.max(self.vram_records)) if self.vram_records else 0.0
        mean_pitch = float(np.mean(self.pitch_records)) if self.pitch_records else 0.0
        std_pitch = float(np.std(self.pitch_records)) if self.pitch_records else 0.0
        mean_yaw = float(np.mean(self.yaw_records)) if self.yaw_records else 0.0
        std_yaw = float(np.std(self.yaw_records)) if self.yaw_records else 0.0
        summary_md = f"""# Отчет тестовой сессии MVP: {self.session_id}

* **Дата и время запуска (MSK):** {self.start_dt.strftime("%Y-%m-%d %H:%M:%S")}
* **Длительность сессии:** {total_time:.2f} с
* **Всего обработано кадров:** {self.frames_logged}
* **Путь к полному CSV-логу:** `{self.csv_path}`

## 1. Производительность и вычислительные задержки
| Показатель | Среднее значение | Минимум | Максимум | 95-й перцентиль |
| :--- | :---: | :---: | :---: | :---: |
| **FPS (кадров/сек)** | **{mean_fps:.1f}** | {min_fps:.1f} | {max_fps:.1f} | — |
| **Задержка модели (мс)** | **{mean_lat_model:.2f}** | {min(self.model_latencies) if self.model_latencies else 0:.2f} | {max(self.model_latencies) if self.model_latencies else 0:.2f} | — |
| **Полная задержка пайплайна (мс)** | **{mean_lat_total:.2f}** | {min(self.total_latencies) if self.total_latencies else 0:.2f} | {max(self.total_latencies) if self.total_latencies else 0:.2f} | **{p95_lat_total:.2f}** |

## 2. Потребление системных ресурсов
| Ресурс | Среднее значение | Пиковое значение |
| :--- | :---: | :---: |
| **RAM (ОЗУ процесса)** | {mean_ram:.2f} ГБ | {peak_ram:.2f} ГБ |
| **VRAM (Видеопамять GPU)** | {mean_vram:.1f} МБ | {peak_vram:.1f} МБ |

## 3. Статистика отслеживания взгляда
* **Тангаж взгляда (Pitch):** среднее {mean_pitch:+.2f}°, std {std_pitch:.2f}°
* **Рыскание взгляда (Yaw):** среднее {mean_yaw:+.2f}°, std {std_yaw:.2f}°
* **Зарегистрировано кликов:** {self.clicks_count}
"""
        with open(self.summary_md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        return self.summary_md_path
