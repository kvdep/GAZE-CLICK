import os, sys, cv2, torch
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("mvp"))
from adapters.factory import create_adapter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gazetr = create_adapter("gazetr", device)
puregaze = create_adapter("puregaze", device)

vid_path = "mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.mp4"
cap = cv2.VideoCapture(vid_path)

# Let's inspect 50 frames to find where face is detected
yunet = cv2.FaceDetectorYN.create("mvp/models/weights/face_detection_yunet_2023mar.onnx", "", (1280, 720))

# Sample 10 frames from beginning, middle, and end
step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 20)
results = []
for idx in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: break
    _, faces = yunet.detect(frame)
    if faces is not None and len(faces) > 0:
        bb = tuple(faces[0][:4].astype(int))
        lm = faces[0][4:14].reshape(5, 2)
        gz_res = gazetr.predict_gaze(frame, bb, lm)
        pg_res = puregaze.predict_gaze(frame, bb, lm)
        results.append((idx, gz_res["yaw_deg"], gz_res["pitch_deg"], pg_res["yaw_deg"], pg_res["pitch_deg"]))

cap.release()

print(f"{'Frame':<6} | {'GazeTR Yaw':<12} | {'GazeTR Pitch':<12} | {'PureGaze Yaw':<12} | {'PureGaze Pitch':<12}")
print("-" * 65)
for r in results[:10]:
    print(f"{r[0]:<6} | {r[1]:+10.2f}°  | {r[2]:+10.2f}°    | {r[3]:+10.2f}°    | {r[4]:+10.2f}°")
