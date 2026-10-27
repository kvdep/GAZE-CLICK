#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_camera_manager.py
Менеджер одновременного захвата и синхронизации нескольких видеопотоков
для подачи на вход моделей компьютерного зрения (CV).

Поддерживает:
1. Одновременный захват двух сенсоров (например, Front Camera ID 1 + Rear Camera ID 0).
2. Контроль рассинхронизации меток времени: Delta_t = |t1 - t2|.
3. Единый интерфейс доступа для последующего инференса CV (YuNet, L2CS, YOLO).
"""

import sys
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Реконфигурация UTF-8 потоков
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from scrcpy_stream import ScrcpyCameraStream
from uvc_stream import UVCCameraStream


class MultiCameraManager:
    """
    Координатор синхронного захвата нескольких видеопотоков с телефона и локальных камер.
    """

    def __init__(self):
        self.streams: Dict[str, Any] = {}
        self.is_running = False

    def add_scrcpy_stream(
        self,
        name: str,
        camera_id: str = "1",
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        port: int = 27183
    ):
        """
        Добавление потока камеры телефона через ADB scrcpy.
        :param name: Логическое имя потока (например, 'front_cam', 'rear_cam').
        :param camera_id: Идентификатор сенсора в Camera2 API ('1' - фронтальная, '0' - основная).
        :param port: Уникальный локальный TCP порт туннеля (каждый поток требует свой порт).
        """
        if name in self.streams:
            raise ValueError(f"Поток с именем '{name}' уже зарегистрирован.")

        stream = ScrcpyCameraStream(
            camera_id=camera_id,
            width=width,
            height=height,
            fps=fps,
            port=port
        )
        self.streams[name] = stream

    def add_uvc_stream(
        self,
        name: str,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30
    ):
        """Добавление потока стандартной UVC веб-камеры."""
        if name in self.streams:
            raise ValueError(f"Поток с именем '{name}' уже зарегистрирован.")

        stream = UVCCameraStream(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps
        )
        self.streams[name] = stream

    def start_all(self, timeout: float = 12.0) -> bool:
        """Запуск всех сконфигурированных видеопотоков."""
        if not self.streams:
            raise RuntimeError("Список потоков пуст. Добавьте хотя бы один поток перед запуском.")

        print(f"[*] Запуск {len(self.streams)} видеопотоков...")
        for name, stream in self.streams.items():
            print(f"  -> Инициализация потока '{name}'...")
            stream.start(timeout=timeout)

        self.is_running = True
        print("[+] Все видеопотоки успешно запущены.")
        return True

    def get_synchronized_frames(self, max_allowed_desync_ms: float = 33.3) -> Tuple[bool, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Захват текущей группы кадров со всех потоков с расчетом межпотоковой рассинхронизации.

        :param max_allowed_desync_ms: Максимально допустимая разница времени между кадрами (мс).
        :return: (успех, словарь кадров {имя: ndarray}, словарь телеметрии синхронизации).
        """
        frames: Dict[str, np.ndarray] = {}
        timestamps: Dict[str, float] = {}

        for name, stream in self.streams.items():
            ok, frame, t_stamp = stream.get_frame()
            if not ok or frame is None:
                return False, {}, {"error": f"Кадр недоступен в потоке '{name}'"}
            frames[name] = frame
            timestamps[name] = t_stamp

        # Расчет разброса меток времени (Delta t)
        t_values = list(timestamps.values())
        delta_t_ms = (max(t_values) - min(t_values)) * 1000.0 if len(t_values) > 1 else 0.0
        is_synced = delta_t_ms <= max_allowed_desync_ms

        telemetry = {
            "timestamps": timestamps,
            "desync_ms": round(delta_t_ms, 2),
            "is_synchronized": is_synced,
            "stream_telemetry": {k: v.get_telemetry() for k, v in self.streams.items()}
        }

        return True, frames, telemetry

    def stop_all(self):
        """Штатная остановка всех видеопотоков."""
        self.is_running = False
        print("[*] Остановка всех видеопотоков...")
        for name, stream in self.streams.items():
            try:
                stream.stop()
            except Exception as e:
                print(f"[!] Ошибка при остановке потока '{name}': {e}", file=sys.stderr)
        print("[+] Все потоки остановлены.")


if __name__ == "__main__":
    print("multi_camera_manager.py - модуль оркестрации синхронных видеопотоков.")
