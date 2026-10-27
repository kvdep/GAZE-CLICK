#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_cv_pipeline.py
Демонстрационный конвейер инференса модели компьютерного зрения (CV)
на реальном видеопотоке с камеры телефона.

Интегрирует:
- Захват кадра через ScrcpyCameraStream или UVCCameraStream.
- Быстрый детектор лиц YuNet (cv2.FaceDetectorYN) с даунскейлом до 320x180 (Проблема 53).
- Замер сквозной задержки прямого прохода и частоты кадров (FPS) через time.perf_counter() (Проблема 47).
- Отрисовка ключевых точек (глаза, нос, углы рта) и ограничивающего прямоугольника лица.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np

# Реконфигурация UTF-8 потоков (Проблема 10/20 SYSTEM_AND_ENVIRONMENT_GUIDE.md)
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Включение DPI Aware (Проблема 48)
import ctypes
import ctypes
import shutil
import tempfile

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

from scrcpy_stream import ScrcpyCameraStream
from uvc_stream import UVCCameraStream

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
YUNET_WEIGHTS_PATH = WORKSPACE_DIR / "mvp" / "models" / "weights" / "face_detection_yunet_2023mar.onnx"


def get_safe_onnx_path(model_path: Path) -> str:
    """
    Разрешает проблему C++ ONNX парсера OpenCV на путях с кириллицей в Windows.
    Стандартный 8.3 short path усекает расширение до .ONN, ломая определение фреймворка.
    Решение: копирование весов во временный файл с явным сохранением расширения '.onnx' в ASCII-пути.
    """
    path_str = str(model_path.resolve())
    try:
        path_str.encode('ascii')
        return path_str
    except UnicodeEncodeError:
        temp_dir = Path(tempfile.gettempdir())
        cached_onnx = temp_dir / f"cached_{model_path.name}"
        if not cached_onnx.exists() or cached_onnx.stat().st_size != model_path.stat().st_size:
            shutil.copyfile(model_path, cached_onnx)
        return str(cached_onnx)


class FastFaceDetector:
    """Обертка над нативным OpenCV YuNet детектором с ускорением 10.9x по Проблеме 53."""

    def __init__(self, model_path: Path, input_size=(320, 180), conf_threshold=0.6, nms_threshold=0.3):
        if not model_path.exists():
            raise FileNotFoundError(f"Файл весов YuNet не найден: {model_path}")
        self.input_w, self.input_h = input_size
        safe_path = get_safe_onnx_path(model_path)
        self.detector = cv2.FaceDetectorYN.create(
            safe_path,
            "",
            (self.input_w, self.input_h),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

    def detect(self, frame_bgr: np.ndarray):
        """
        Выполняет инференс на уменьшенном разрешении (320x180) и пересчитывает координаты в исходный масштаб.
        :return: (faces, t_infer_ms)
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        scale_x = orig_w / float(self.input_w)
        scale_y = orig_h / float(self.input_h)

        t0 = time.perf_counter()
        small_frame = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        _, faces = self.detector.detect(small_frame)
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

        res_faces = []
        if faces is not None:
            for face in faces:
                # Масштабирование координат рамки
                box_x = face[0] * scale_x
                box_y = face[1] * scale_y
                box_w = face[2] * scale_x
                box_h = face[3] * scale_y
                score = face[14]

                # 5 ключевых точек (x, y): правый глаз, левый глаз, нос, правый угол рта, левый угол рта
                landmarks = []
                for idx in range(4, 14, 2):
                    lx = face[idx] * scale_x
                    ly = face[idx + 1] * scale_y
                    landmarks.append((lx, ly))

                res_faces.append({
                    "bbox": [int(box_x), int(box_y), int(box_w), int(box_h)],
                    "score": float(score),
                    "landmarks": landmarks
                })

        return res_faces, t_infer_ms


def draw_face_annotations(frame: np.ndarray, faces, t_infer_ms: float, fps_val: float):
    """Отрисовка аннотаций лица, ориентиров и телеметрии задержки."""
    out = frame.copy()
    for f in faces:
        x, y, w, h = f["bbox"]
        score = f["score"]
        # Рамка лица
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"Face: {score:.2f}", (x, max(15, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5 ключевых ориентиров
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        for (lx, ly), c in zip(f["landmarks"], colors):
            cv2.circle(out, (int(lx), int(ly)), 4, c, -1)

    # Строка телеметрии
    telemetry_str = f"Inference: {t_infer_ms:.2f} ms | Pipeline FPS: {fps_val:.1f} | Faces: {len(faces)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(out, telemetry_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Запуск CV-пайплайна на видеопотоке с камеры телефона.")
    parser.add_argument("--mode", choices=["scrcpy", "uvc", "test_uvc0"], default="scrcpy",
                        help="Режим подключения: 'scrcpy' (ADB), 'uvc' (Samsung Webcam), 'test_uvc0' (локальная веб-камера ноутбука для проверки).")
    parser.add_argument("--camera-id", default="1", help="ID камеры для scrcpy ('1' - фронтальная, '0' - основная).")
    parser.add_argument("--uvc-index", type=int, default=0, help="Индекс устройства для UVC.")
    parser.add_argument("--width", type=int, default=1280, help="Ширина кадра.")
    parser.add_argument("--height", type=int, default=720, help="Высота кадра.")
    parser.add_argument("--fps", type=int, default=60, help="Частота кадров.")
    parser.add_argument("--headless", action="store_true", help="Запуск без графического окна OpenCV GUI (только логирование телеметрии).")
    parser.add_argument("--max-frames", type=int, default=0, help="Лимит кадров (0 - бесконечно до нажатия 'q').")
    args = parser.parse_args()

    print("=" * 80)
    print(" ЗАПУСК КОНВЕЙЕРА КОМПЬЮТЕРНОГО ЗРЕНИЯ (CV INFERENCE PIPELINE)")
    print("=" * 80)
    print(f"Режим: {args.mode}")

    # Инициализация детектора лиц
    print(f"[*] Загрузка YuNet весов: {YUNET_WEIGHTS_PATH}...")
    detector = FastFaceDetector(YUNET_WEIGHTS_PATH)
    print("[+] YuNet инициализирован.")

    # Инициализация видеоисточника
    stream = None
    if args.mode == "scrcpy":
        stream = ScrcpyCameraStream(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
    elif args.mode in ["uvc", "test_uvc0"]:
        idx = 0 if args.mode == "test_uvc0" else args.uvc_index
        stream = UVCCameraStream(
            camera_index=idx,
            width=args.width,
            height=args.height,
            fps=args.fps
        )

    try:
        stream.start(timeout=10.0)
    except Exception as e:
        print(f"[!] Сбой запуска видеоисточника: {e}", file=sys.stderr)
        if args.mode == "scrcpy":
            print("[!] Убедитесь, что телефон подключен с включенной «Отладкой по USB» (запустите device_detector.py).", file=sys.stderr)
        return 1

    print("[+] Видеопоток активен. Начало цикла обработки...")
    frame_idx = 0
    t_start = time.perf_counter()
    prev_time = t_start
    fps_ema = 30.0

    try:
        while True:
            t_loop_start = time.perf_counter()
            ok, frame, t_stamp = stream.get_frame()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # Инференс CV модели
            faces, t_infer_ms = detector.detect(frame)

            t_loop_end = time.perf_counter()
            dt = t_loop_end - prev_time
            prev_time = t_loop_end
            if dt > 0:
                current_fps = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * current_fps

            frame_idx += 1

            if frame_idx % 30 == 0 or args.headless:
                print(f"[Кадр {frame_idx:05d}] Лиц: {len(faces)} | Инференс: {t_infer_ms:.2f} мс | Сквозной FPS: {fps_ema:.1f}")

            if not args.headless:
                annotated = draw_face_annotations(frame, faces, t_infer_ms, fps_ema)
                cv2.imshow("Phone Camera CV Pipeline (Samsung Galaxy S25 Ultra)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' или ESC
                    print("[*] Выход по команде пользователя.")
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[*] Достигнут лимит кадров: {args.max_frames}.")
                break

    except KeyboardInterrupt:
        print("\n[*] Прерывание по сигналу Ctrl+C.")
    finally:
        stream.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    print(f"\n[+] Итоги сессии: обработано {frame_idx} кадров за {total_time:.2f} с (Средний сквозной FPS: {avg_fps:.2f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
