import os, glob, time
import cv2
import numpy as np

yunet_path = "mvp/models/weights/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config="",
    input_size=(1280, 720),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

sessions = sorted(glob.glob("mvp/benchmarks/benchmark_test_20261017_*.mp4"))
print(f"Caching face detections for {len(sessions)} Oct 17 sessions...")

for vid_path in sessions:
    base = os.path.splitext(os.path.basename(vid_path))[0]
    session_id = base.replace("benchmark_test_", "")
    cache_file = f"mvp/experiments/_cache_face_{session_id}.npz"
    if os.path.exists(cache_file):
        print(f"Session {session_id} already cached at {cache_file}.")
        continue

    cap = cv2.VideoCapture(vid_path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bboxes = []
    landmarks_list = []
    detected = []

    t0 = time.time()
    for idx in range(fc):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            f = faces[0]
            bbox = f[:4].astype(int)
            lms = f[4:14].reshape(5, 2)
            bboxes.append(bbox)
            landmarks_list.append(lms)
            detected.append(True)
        else:
            bboxes.append(np.array([0, 0, 0, 0], dtype=int))
            landmarks_list.append(np.zeros((5, 2), dtype=np.float32))
            detected.append(False)
    cap.release()
    elapsed = time.time() - t0
    np.savez_compressed(
        cache_file,
        bboxes=np.array(bboxes),
        landmarks=np.array(landmarks_list),
        detected=np.array(detected)
    )
    print(f"Cached {session_id} ({fc} frames, {sum(detected)} detected) in {elapsed:.2f}s -> {cache_file}")

print("Done caching all sessions.")
