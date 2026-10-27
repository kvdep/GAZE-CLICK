#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrcpy_stream.py
Низколатентный захват аппаратного видеопотока с камер Android (Samsung Galaxy S25 Ultra)
через ADB, scrcpy и Windows Named Pipe напрямую в память (NumPy BGR24) через PyAV.

Оптимизации нулевой задержки (Zero Buffer Bloat):
1. Размер Named Pipe уменьшен до 64 КБ (вместо 2 МБ) для исключения очередей в ядре ОС.
2. Флаги libavformat: fflags=nobuffer, flags=low_delay, probesize=32, analyzeduration=0.
3. Атомарное обновление _latest_frame: потребитель всегда получает строго последний физический кадр.
4. Поддержка нативных высоких разрешений: 1080p, 2K (2560x1440), 4K UHD (3840x2160).
"""

import sys
import os
import io
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import av
import win32pipe
import win32file

# Реконфигурация UTF-8 потоков
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
SCRCPY_BIN = TOOLS_DIR / "scrcpy.exe"
ADB_BIN = TOOLS_DIR / "adb.exe"


class _PipeStream(io.RawIOBase):
    """Потоковый адаптер Win32 Named Pipe для PyAV контейнера."""

    def __init__(self, handle):
        self.handle = handle

    def readable(self):
        return True

    def readinto(self, b):
        try:
            hr, data = win32file.ReadFile(self.handle, len(b))
            n = len(data)
            b[:n] = data
            return n
        except Exception:
            return 0


class ScrcpyCameraStream:
    """
    Класс низколатентного захвата видеопотока с камеры Android устройства через scrcpy.
    """

    def __init__(
        self,
        camera_id: str = "0",
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        bit_rate: str = "20M",
        port: int = 27183,
        camera_facing: Optional[str] = None
    ):
        self.camera_id = str(camera_id)
        self.width = width
        self.height = height
        self.fps = fps
        self.bit_rate = bit_rate
        self.port = port
        self.camera_facing = camera_facing

        self.pipe_name = rf"\\.\pipe\scrcpy_cam_{self.camera_id}_{uuid.uuid4().hex[:8]}"
        self.pipe_handle = None

        self.proc: Optional[subprocess.Popen] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Буфер кадра
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._latest_pts: int = 0
        self._frame_count: int = 0
        self._fps_measured: float = 0.0
        self._last_fps_calc_time: float = 0.0
        self._last_fps_frame_count: int = 0

    def start(self, timeout: float = 12.0) -> bool:
        """Создает именованный канал малого размера, запускает scrcpy и начинает декодирование."""
        if not SCRCPY_BIN.exists():
            raise FileNotFoundError(f"Исполняемый файл scrcpy не найден по пути: {SCRCPY_BIN}")

        # Буфер pipe уменьшен до 64 КБ (исключение задержки буферизации в ядре)
        self.pipe_handle = win32pipe.CreateNamedPipe(
            self.pipe_name,
            win32pipe.PIPE_ACCESS_INBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
            1,
            65536,
            65536,
            0,
            None
        )

        cmd = [
            str(SCRCPY_BIN),
            "--video-source=camera",
            f"--camera-size={self.width}x{self.height}",
            f"--camera-fps={self.fps}",
            f"--video-bit-rate={self.bit_rate}",
            "--no-playback",
            "--no-audio",
            "--verbosity=error",
            f"--record={self.pipe_name}",
            "--record-format=mkv",
            f"--port={self.port}:{self.port+10}",
            "--tunnel-host=127.0.0.1"
        ]

        if self.camera_facing:
            cmd.append(f"--camera-facing={self.camera_facing}")
        else:
            cmd.append(f"--camera-id={self.camera_id}")

        print(f"[*] Запуск scrcpy для камеры ID={self.camera_id} ({self.width}x{self.height} @ {self.fps} FPS, порт {self.port})...")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._decoder_loop,
            name=f"ScrcpyDecoder-Cam{self.camera_id}",
            daemon=True
        )
        self.worker_thread.start()

        # Ожидание первого кадра
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if not self.running or self.proc.poll() is not None:
                stderr_out = self.proc.stderr.read().decode('utf-8', errors='replace') if self.proc.stderr else ""
                self.stop()
                raise RuntimeError(f"Сбой процесса scrcpy (ID={self.camera_id}). Код: {self.proc.poll()}. Ошибка:\n{stderr_out}")
            with self._lock:
                if self._latest_frame is not None:
                    h, w = self._latest_frame.shape[:2]
                    print(f"[+] Поток камеры ID={self.camera_id} активен ({w}x{h} @ {self.fps} FPS).")
                    return True
            time.sleep(0.02)

        self.stop()
        raise TimeoutError(f"Истекло время ожидания первого кадра от камеры ID={self.camera_id} ({timeout} с).")

    def _decoder_loop(self):
        """Низколатентный цикл разбора и декодирования пакетов."""
        try:
            win32pipe.ConnectNamedPipe(self.pipe_handle, None)
            raw_stream = _PipeStream(self.pipe_handle)
            buffered_stream = io.BufferedReader(raw_stream, buffer_size=16384)

            # Опции libavformat для исключения буферизации
            container = av.open(
                buffered_stream,
                format='matroska',
                options={
                    'probesize': '32',
                    'analyzeduration': '0',
                    'fflags': 'nobuffer',
                    'flags': 'low_delay'
                }
            )
            self._last_fps_calc_time = time.perf_counter()

            for frame in container.decode(video=0):
                if not self.running:
                    break

                t_recv = time.perf_counter()
                bgr_array = frame.to_ndarray(format='bgr24')

                with self._lock:
                    self._latest_frame = bgr_array
                    self._latest_timestamp = t_recv
                    self._latest_pts = frame.pts if frame.pts is not None else 0
                    self._frame_count += 1

                    dt = t_recv - self._last_fps_calc_time
                    if dt >= 1.0:
                        df = self._frame_count - self._last_fps_frame_count
                        self._fps_measured = df / dt
                        self._last_fps_calc_time = t_recv
                        self._last_fps_frame_count = self._frame_count

        except Exception as e:
            if self.running:
                print(f"[!] Исключение в потоке декодирования камеры ID={self.camera_id}: {e}", file=sys.stderr)
        finally:
            self.running = False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Получение самого свежего кадра.
        :return: (успех, массив BGR ndarray, системная метка времени time.perf_counter()).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame, self._latest_timestamp

    def get_telemetry(self) -> Dict[str, Any]:
        """Метрики потока."""
        with self._lock:
            return {
                "camera_id": self.camera_id,
                "frame_count": self._frame_count,
                "fps_measured": round(self._fps_measured, 2),
                "resolution": (self._latest_frame.shape[1], self._latest_frame.shape[0]) if self._latest_frame is not None else None,
                "latest_timestamp": self._latest_timestamp,
                "is_alive": self.running and (self.proc.poll() is None if self.proc else False)
            }

    def stop(self):
        """Штатная остановка процесса и освобождение ресурсов."""
        self.running = False

        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        if self.pipe_handle:
            try:
                win32file.CloseHandle(self.pipe_handle)
            except Exception:
                pass
            self.pipe_handle = None

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        print(f"[*] Поток камеры ID={self.camera_id} остановлен.")
