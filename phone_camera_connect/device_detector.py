#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
device_detector.py
Диагностика подключения Samsung Galaxy S25 Ultra, статуса USB-интерфейсов,
ADB-авторизации, UVC-вебкамер и аппаратных сенсоров камеры.
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

# Реконфигурация UTF-8 потоков согласно Проблеме 10/20 в SYSTEM_AND_ENVIRONMENT_GUIDE.md
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Пути к встроенным утилитам
SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR / "tools"
ADB_BIN = TOOLS_DIR / "adb.exe"
SCRCPY_BIN = TOOLS_DIR / "scrcpy.exe"

# Аппаратные характеристики Samsung Galaxy S25 Ultra (SM-S938x)
S25_ULTRA_CAMERA_SPECS = {
    "front": [
        {
            "sensor": "ISOCELL S5K3LU / Dual Pixel",
            "resolution": "12 MP (4000x3000)",
            "focal_length_equiv": "26mm",
            "aperture": "f/2.2",
            "pixel_size_um": 1.12,
            "sensor_size": "1/3.2 inch",
            "autofocus": "Dual Pixel PDAF",
            "notes": "Единственный физический фронтальный оптический модуль на S25 Ultra."
        }
    ],
    "rear": [
        {
            "id": 0,
            "name": "Основная широкоугольная (Main Wide)",
            "sensor": "ISOCELL HP2",
            "resolution": "200 MP (16320x12240) -> 12.5 MP binning",
            "focal_length_equiv": "24mm",
            "aperture": "f/1.7",
            "pixel_size_um": 0.6,
            "sensor_size": "1/1.3 inch",
            "ois": True
        },
        {
            "id": 1,
            "name": "Телефото 5x перископ (Periscope Telephoto)",
            "sensor": "Sony IMX854 / ISOCELL JN3",
            "resolution": "50 MP (8192x6144) -> 12.5 MP binning",
            "focal_length_equiv": "115mm",
            "aperture": "f/3.4",
            "pixel_size_um": 0.7,
            "sensor_size": "1/2.52 inch",
            "ois": True
        },
        {
            "id": 2,
            "name": "Телефото 3x (Telephoto)",
            "sensor": "Sony IMX754",
            "resolution": "10 MP (3648x2736)",
            "focal_length_equiv": "67mm",
            "aperture": "f/2.4",
            "pixel_size_um": 1.12,
            "sensor_size": "1/3.52 inch",
            "ois": True
        },
        {
            "id": 3,
            "name": "Сверхширокоугольная (Ultra-Wide)",
            "sensor": "ISOCELL JN3",
            "resolution": "50 MP (8192x6144) -> 12.5 MP binning",
            "focal_length_equiv": "13mm (120 deg)",
            "aperture": "f/1.9",
            "pixel_size_um": 0.7,
            "sensor_size": "1/2.5 inch",
            "autofocus": "Dual Pixel PDAF"
        }
    ]
}


def query_pnp_devices():
    """Запрос Win32 PnP-устройств, связанных с Samsung (VID 04E8)."""
    ps_cmd = (
        'Get-PnpDevice | '
        'Where-Object { $_.InstanceId -like "*04E8*" } | '
        'Select-Object Status, Class, FriendlyName, InstanceId | '
        'ConvertTo-Json'
    )
    try:
        proc = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        data = json.loads(proc.stdout)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[!] Ошибка вызова PowerShell PnP: {e}", file=sys.stderr)
        return []


def query_uvc_cameras():
    """Запрос присутствующих в системе видеокамер через PnP."""
    ps_cmd = (
        'Get-PnpDevice -Class Camera | '
        'Where-Object { $_.Present -eq $true } | '
        'Select-Object Status, FriendlyName, InstanceId | '
        'ConvertTo-Json'
    )
    try:
        proc = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        data = json.loads(proc.stdout)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def query_adb_status():
    """Проверка подключения и авторизации устройства через ADB."""
    if not ADB_BIN.exists():
        return {"status": "ADB_NOT_FOUND", "devices": []}

    try:
        proc = subprocess.run(
            [str(ADB_BIN), "devices", "-l"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
        devices = []
        for line in lines[1:]:  # Пропускаем "List of devices attached"
            parts = line.split()
            if len(parts) >= 2:
                serial = parts[0]
                state = parts[1]  # 'device', 'unauthorized', 'offline'
                info = " ".join(parts[2:])
                devices.append({"serial": serial, "state": state, "info": info})
        return {"status": "OK", "devices": devices}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "devices": []}


def query_scrcpy_cameras():
    """Запрос списка физических и логических камер телефона через scrcpy."""
    if not SCRCPY_BIN.exists():
        return {"status": "SCRCPY_NOT_FOUND", "output": ""}

    try:
        proc = subprocess.run(
            [str(SCRCPY_BIN), "--video-source=camera", "--list-cameras"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=15
        )
        return {
            "status": "OK" if proc.returncode == 0 else "ERROR",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except Exception as e:
        return {"status": "EXCEPTION", "error": str(e)}


def test_msmf_capture(max_indices=4):
    """Проверка доступности видеопотока через OpenCV Media Foundation (CAP_MSMF)."""
    import cv2
    results = []
    for idx in range(max_indices):
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        opened = cap.isOpened()
        if opened:
            ret, frame = cap.read()
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            results.append({
                "index": idx,
                "opened": True,
                "read_success": bool(ret),
                "resolution": f"{int(w)}x{int(h)}",
                "fps": float(fps),
                "latency_sec": round(time.perf_counter() - t0, 3)
            })
        else:
            cap.release()
            results.append({
                "index": idx,
                "opened": False,
                "latency_sec": round(time.perf_counter() - t0, 3)
            })
    return results


def run_full_diagnostic():
    print("=" * 80)
    print(" ДИАГНОСТИКА ПОДКЛЮЧЕНИЯ ТЕЛЕФОНА (SAMSUNG GALAXY S25 ULTRA) ДЛЯ ВКР")
    print("=" * 80)

    # 1. Аппаратная спецификация камер S25 Ultra
    print("\n[1] АППАРАТНАЯ СТРУКТУРА СЕНСОРОВ КАМЕРЫ (SAMSUNG GALAXY S25 ULTRA):")
    print(f"  • Фронтальная камера: СТРОГО 1 ФИЗИЧЕСКИЙ СЕНСОР.")
    f_spec = S25_ULTRA_CAMERA_SPECS["front"][0]
    print(f"    - Разрешение: {f_spec['resolution']}")
    print(f"    - Оптическая система: f/{f_spec['aperture']}, экв. фокусное {f_spec['focal_length_equiv']}")
    print(f"    - Размер пикселя: {f_spec['pixel_size_um']} мкм, матрица: {f_spec['sensor_size']}")
    print(f"    - Примечание: {f_spec['notes']}")
    print(f"  • Тыловые камеры: 4 ФИЗИЧЕСКИХ СЕНСОРА (200 MP Main + 50 MP 5x + 10 MP 3x + 50 MP Ultra-Wide).")

    # 2. PnP устройства Samsung
    print("\n[2] СОСТОЯНИЕ USB PNP-УСТРОЙСТВ (VID 04E8 - SAMSUNG):")
    pnp_list = query_pnp_devices()
    has_samsung = len(pnp_list) > 0
    has_adb_interface = False
    has_mtp_interface = False
    has_uvc_interface = False

    if not has_samsung:
        print("  [!] Устройства Samsung не обнаружены на шине USB.")
        print("      Проверьте физическое подключение USB-C кабеля.")
    else:
        for dev in pnp_list:
            cls = dev.get('Class', '')
            name = dev.get('FriendlyName', '')
            inst = dev.get('InstanceId', '')
            print(f"  • Класс: {cls:<10} | Имя: {name:<35} | ID: {inst}")
            if "MTP" in inst or "WPD" in cls:
                has_mtp_interface = True
            if "ADB" in inst or "AndroidUsbDeviceClass" in cls:
                has_adb_interface = True
            if "Camera" in cls:
                has_uvc_interface = True

    # 3. Диагностика режима
    print("\n[3] АНАЛИЗ ТЕКУЩЕГО РЕЖИМА USB:")
    if has_mtp_interface and not has_adb_interface and not has_uvc_interface:
        print("  [!] ТЕКУЩИЙ РЕЖИМ: MTP / Передача файлов (PID 6860).")
        print("      В режиме MTP телефон монтируется как файловая система WPD (Windows Portable Device).")
        print("      Протокол MTP не имеет конечных точек (endpoints) для изохронной передачи видеопотока.")
        print("      Прямой захват кадров в OpenCV / CV-модели в режиме MTP драйверно НЕВОЗМОЖЕН.")
        print("\n  [+] НЕОБХОДИМЫЕ ДЕЙСТВИЯ ДЛЯ ПОЛУЧЕНИЯ ВИДЕОПОТОКА:")
        print("      ВАРИАНТ А (Рекомендуется для ВКР / многопотокового захвата с минимальной задержкой):")
        print("        1. На телефоне: Настройки -> Сведения о телефоне -> Сведения о ПО -> 7 раз нажать «Номер сборки».")
        print("        2. На телефоне: Настройки -> Параметры разработчика -> Включить «Отладка по USB».")
        print("        3. На экране телефона появится запрос: «Разрешить отладку по USB с этого компьютера?» -> Нажать «Разрешить».")
        print("        4. Это предоставит прямой доступ к Camera2 API через scrcpy/ADB с задержкой 30-45 мс на 60 FPS.")
        print("\n      ВАРИАНТ Б (Режим стандартной UVC Веб-камеры в Android 14/15 One UI):")
        print("        1. Опустить шторку уведомлений на телефоне.")
        print("        2. Нажать на уведомление «Зарядка через USB / Передача файлов».")
        print("        3. Выбрать пункт «Веб-камера» (Webcam).")
        print("        4. Телефон станет доступен как UVC-камера через cv2.VideoCapture(index).")
        print("           (Ограничение варианта Б: UVC-режим передает только 1 поток камеры одновременно).")
    elif has_adb_interface:
        print("  [+] ТЕКУЩИЙ РЕЖИМ: ADB (Отладка по USB активирована).")
    elif has_uvc_interface:
        print("  [+] ТЕКУЩИЙ РЕЖИМ: UVC WebCam (Телефон работает как веб-камера).")

    # 4. Проверка ADB
    print("\n[4] СОСТОЯНИЕ ADB (Android Debug Bridge):")
    adb_res = query_adb_status()
    print(f"  • Статус сервиса ADB: {adb_res['status']}")
    devices = adb_res.get('devices', [])
    if not devices:
        print("  • Подключенные ADB-устройства: ОТСУТСТВУЮТ (требуется включить «Отладка по USB»).")
    else:
        for d in devices:
            print(f"  • Серийный номер: {d['serial']} | Состояние: {d['state']} | Инфо: {d['info']}")
            if d['state'] == 'unauthorized':
                print("    [!] ВНИМАНИЕ: Устройство не авторизовано. Подтвердите запрос на экране телефона!")

    # 5. Если ADB авторизован, запрос списка камер через scrcpy
    if any(d.get('state') == 'device' for d in devices):
        print("\n[5] ОПРОС АППАРАТНЫХ КАМЕР ЧЕРЕЗ SCRCPY (Camera2 API):")
        cam_res = query_scrcpy_cameras()
        if cam_res.get("stdout"):
            print(cam_res["stdout"])
        if cam_res.get("stderr"):
            print(cam_res["stderr"])

    # 6. Проверка UVC камер в системе через MSMF
    print("\n[6] ОПРОС UVC-КАМЕР В СИСТЕМЕ WINDOWS (OpenCV MSMF Backend):")
    msmf_res = test_msmf_capture(max_indices=3)
    for r in msmf_res:
        idx = r['index']
        opened = r['opened']
        lat = r['latency_sec']
        if opened:
            print(f"  • Индекс {idx}: ДОСТУПЕН | Разрешение: {r['resolution']} | FPS: {r['fps']} | Задержка открытия: {lat} с")
        else:
            print(f"  • Индекс {idx}: Недоступен ({lat} с)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_full_diagnostic()
