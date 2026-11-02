import os, sys, time
import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

parquet_path = "mvp/experiments/test_unified.parquet"
cache_path = "mvp/experiments/_cache_face_test_unified.npz"
yunet_path = "mvp/models/weights/face_detection_yunet_2023mar.onnx"

if os.path.exists(cache_path):
    print(f"Cache already exists at {cache_path}. Loading existing...")
    c = np.load(cache_path)
    print(f"Loaded {len(c['bboxes'])} bboxes, detected={np.sum(c['detected'])}/{len(c['detected'])}")
    sys.exit(0)

print(f"Reading {parquet_path}...")
df = pd.read_parquet(parquet_path)
total = len(df)
print(f"Total test samples to cache: {total}")

detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config="",
    input_size=(1280, 720),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

bboxes = []
landmarks_list = []
detected = []

t0 = time.time()
for idx, row in df.iterrows():
    img_path = row["image_path"]
    img = cv2.imread(img_path)
    if img is None:
        bboxes.append(np.array([0, 0, 0, 0], dtype=np.int32))
        landmarks_list.append(np.zeros((5, 2), dtype=np.float32))
        detected.append(False)
        continue

    h, w = img.shape[:2]
    scale = 1.0
    if max(w, h) > 1280:
        scale = 1280.0 / max(w, h)
        img_in = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_in = img

    sh, sw = img_in.shape[:2]
    detector.setInputSize((sw, sh))
    _, faces = detector.detect(img_in)

    if faces is not None and len(faces) > 0:
        f = faces[0]
        # Rescale bbox and landmarks back to original image coordinates
        bbox = (f[:4] / scale).astype(np.int32)
        lms = (f[4:14].reshape(5, 2) / scale).astype(np.float32)
        bboxes.append(bbox)
        landmarks_list.append(lms)
        detected.append(True)
    else:
        # Fallback centered face crop
        cx = int(0.15 * w)
        cy = int(0.10 * h)
        cw = int(0.70 * w)
        ch = int(0.80 * h)
        bboxes.append(np.array([cx, cy, cw, ch], dtype=np.int32))
        landmarks_list.append(np.zeros((5, 2), dtype=np.float32))
        detected.append(False)

    if (idx + 1) % 500 == 0 or (idx + 1) == total:
        elapsed = time.time() - t0
        speed = (idx + 1) / elapsed
        print(f"[{idx+1}/{total}] Cached in {elapsed:.1f}s ({speed:.1f} frames/s). Detected: {sum(detected)}/{idx+1}")

bboxes_arr = np.array(bboxes, dtype=np.int32)
landmarks_arr = np.array(landmarks_list, dtype=np.float32)
detected_arr = np.array(detected, dtype=bool)

np.savez_compressed(
    cache_path,
    bboxes=bboxes_arr,
    landmarks=landmarks_arr,
    detected=detected_arr
)
print(f"Successfully saved cache to {cache_path} ({len(bboxes_arr)} frames, {np.sum(detected_arr)} detected faces).")
