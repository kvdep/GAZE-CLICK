#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_rear_cameras.py
Экспериментальная проверка одновременного запуска двух тыловых камер (Camera 0 + Camera 2)
на Samsung Galaxy S25 Ultra.
"""

import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrcpy_stream import ScrcpyCameraStream


def run_test():
    print("=" * 80)
    print(" ТЕСТ ОДНОВРЕМЕННОГО ЗАХВАТА ДВУХ ТЫЛОВЫХ КАМЕР (REAR 0 + REAR 2)")
    print("=" * 80)

    print("[*] Шаг 1: Запуск тыловой камеры ID=0 (Основная широкоугольная)...")
    s0 = ScrcpyCameraStream(camera_id='0', width=640, height=480, fps=30, port=27183)
    s0.start(timeout=8.0)
    print("[+] Камера ID=0 успешно запущена.")

    print("\n[*] Шаг 2: Попытка запуска тыловой камеры ID=2 (Телефото)...")
    s2 = ScrcpyCameraStream(camera_id='2', width=640, height=480, fps=30, port=27185)
    s2_started = False
    try:
        s2.start(timeout=8.0)
        s2_started = True
        print("[+] Камера ID=2 успешно запущена.")
    except Exception as e:
        print(f"[!] Ошибка запуска камеры ID=2: {e}")

    print("\n[*] Шаг 3: Мониторинг параллельного поступления кадров (10 итераций по 150 мс)...")
    c0_frames = []
    c2_frames = []

    for i in range(10):
        time.sleep(0.15)
        telem0 = s0.get_telemetry()
        telem2 = s2.get_telemetry()
        cnt0 = telem0["frame_count"]
        cnt2 = telem2["frame_count"]
        c0_frames.append(cnt0)
        c2_frames.append(cnt2)
        alive0 = telem0["is_alive"]
        alive2 = telem2["is_alive"]
        print(f"  Итерация {i+1:02d}: Cam 0 кадров={cnt0} (жива: {alive0}) | Cam 2 кадров={cnt2} (жива: {alive2})")

    s0.stop()
    s2.stop()

    print("\n" + "=" * 80)
    print(" АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТА:")
    delta_c0 = c0_frames[-1] - c0_frames[0]
    delta_c2 = c2_frames[-1] - c2_frames[0]
    print(f"  • Новых кадров с Камеры 0 за время теста: {delta_c0}")
    print(f"  • Новых кадров с Камеры 2 за время теста: {delta_c2}")

    if delta_c0 > 0 and delta_c2 > 0:
        print("  [+] ВЫВОД: Обе тыловые камеры могут захватываться одновременно!")
    elif delta_c0 == 0 and delta_c2 > 0:
        print("  [!] ВЫВОД: Камера 0 была заблокирована/вытеснена при старте Камеры 2.")
    elif delta_c2 == 0 and delta_c0 > 0:
        print("  [!] ВЫВОД: Камера 2 не смогла генерировать кадры при активной Камере 0.")
    else:
        print("  [!] ВЫВОД: Обе камеры остановились.")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
