#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dual_stream.py
Проверка одновременного параллельного захвата двух сенсоров (Front Camera 1 + Rear Camera 0)
на Samsung Galaxy S25 Ultra через Camera2 API и MultiCameraManager.
"""

import sys
import time
from pathlib import Path

# Реконфигурация UTF-8 потоков
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from multi_camera_manager import MultiCameraManager


def test_concurrent_dual_camera():
    print("=" * 80)
    print(" ТЕСТ ОДНОВРЕМЕННОГО ЗАХВАТА ДВУХ КАМЕР (CONCURRENT DUAL CAMERA STREAM)")
    print("=" * 80)

    manager = MultiCameraManager()
    # Поток 1: Фронтальная камера (Camera ID 1, порт 27183)
    manager.add_scrcpy_stream(
        name="front_cam",
        camera_id="1",
        width=1280,
        height=720,
        fps=30,
        port=27183
    )

    # Поток 2: Тыловая камера (Camera ID 0, порт 27185)
    manager.add_scrcpy_stream(
        name="rear_cam",
        camera_id="0",
        width=1280,
        height=720,
        fps=30,
        port=27185
    )

    try:
        manager.start_all(timeout=12.0)
    except Exception as e:
        print(f"[!] Сбой одновременного запуска двух камер: {e}")
        return False

    print("\n[+] Оба потока активны. Сбор 10 синхронных пар кадров...")
    samples = 0
    desync_list = []

    t_start = time.perf_counter()
    while samples < 10:
        ok, frames, telemetry = manager.get_synchronized_frames(max_allowed_desync_ms=50.0)
        if not ok:
            time.sleep(0.01)
            continue

        f_front = frames.get("front_cam")
        f_rear = frames.get("rear_cam")

        if f_front is not None and f_rear is not None:
            samples += 1
            desync_ms = telemetry["desync_ms"]
            desync_list.append(desync_ms)
            print(f"  [Пара {samples:02d}] Front shape: {f_front.shape} | Rear shape: {f_rear.shape} | Рассинхронизация: {desync_ms:.2f} мс")

        time.sleep(0.03)

    total_time = time.perf_counter() - t_start
    manager.stop_all()

    avg_desync = sum(desync_list) / len(desync_list) if desync_list else 0.0
    print("\n" + "=" * 80)
    print(f" ИТОГИ ТЕСТА DUAL CAMERA:")
    print(f"  • Успешно захвачено пар кадров: {samples}/10")
    print(f"  • Средняя рассинхронизация Delta_t: {avg_desync:.2f} мс")
    print(f"  • Время захвата 10 пар: {total_time:.2f} с")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_concurrent_dual_camera()
    sys.exit(0 if success else 1)
