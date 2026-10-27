#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uvc_stream.py
Потокобезопасный захват видеопотока с UVC Веб-камеры (включая режим Веб-камеры Samsung One UI)
через бэкенд OpenCV Media Foundation (CAP_MSMF).

Особенности:
- Полное отсутствие зависаний, характерных для CAP_DSHOW при наличии виртуальных устройств (Проблема 49).
- Фоновый тред с непрерывным cap.grab() для исключения накопления задержки в системном FIFO-буфере Windows.
"""

import sys
import threading
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

# Реконфигурация UTF-8 потоков
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class UVCCameraStream:
    """
    Класс асинхронного захвата кадров с UVC-камеры с защитой от накопления буфера задержки.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._fps_measured: float = 0.0
        self._last_fps_calc_time: float = 0.0
        self._last_fps_frame_count: int = 0

    def start(self, timeout: float = 8.0) -> bool:
        """Инициализация камеры и запуск фонового треда захвата."""
        print(f"[*] Открытие UVC камеры индекс {self.camera_index} через CAP_MSMF...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть UVC камеру с индексом {self.camera_index}.")

        # Установка свойств
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._capture_loop,
            name=f"UVC-CaptureThread-{self.camera_index}",
            daemon=True
        )
        self.worker_thread.start()

        # Ожидание первого кадра
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            with self._lock:
                if self._latest_frame is not None:
                    print(f"[+] UVC камера {self.camera_index} инициализирована ({actual_w}x{actual_h} @ {actual_fps} FPS).")
                    return True
            time.sleep(0.05)

        self.stop()
        raise TimeoutError(f"Истекло время ожидания первого кадра от UVC камеры {self.camera_index}.")

    def _capture_loop(self):
        """Непрерывный захват кадров в цикле для удержания буфера в актуальном состоянии."""
        self._last_fps_calc_time = time.perf_counter()
        while self.running and self.cap and self.cap.isOpened():
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.005)
                continue

            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                t_recv = time.perf_counter()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = t_recv
                    self._frame_count += 1

                    dt = t_recv - self._last_fps_calc_time
                    if dt >= 1.0:
                        df = self._frame_count - self._last_fps_frame_count
                        self._fps_measured = df / dt
                        self._last_fps_calc_time = t_recv
                        self._last_fps_frame_count = self._frame_count

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Возвращает актуальный кадр и timestamp."""
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame, self._latest_timestamp

    def get_telemetry(self) -> Dict[str, Any]:
        """Метрики производительности."""
        with self._lock:
            return {
                "camera_index": self.camera_index,
                "frame_count": self._frame_count,
                "fps_measured": round(self._fps_measured, 2),
                "resolution": (self._latest_frame.shape[1], self._latest_frame.shape[0]) if self._latest_frame is not None else None,
                "latest_timestamp": self._latest_timestamp,
                "is_alive": self.running and (self.cap.isOpened() if self.cap else False)
            }

    def stop(self):
        """Остановка треда и освобождение интерфейса камеры."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print(f"[*] UVC камера {self.camera_index} освобождена.")


if __name__ == "__main__":
    print("uvc_stream.py - модуль захвата с UVC камер через OpenCV MSMF.")
