import os, sys, time, cv2, torch, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("mvp"))
from adapters.factory import create_adapter
from engine.calibration import GazeCalibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading PureGaze adapter...")
adapter = create_adapter("puregaze", device)

vid_path = "mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.mp4"
csv_path = "mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.csv"
df = pd.read_csv(csv_path)

# Extract YuNet faces for this video
yunet = cv2.FaceDetectorYN.create("mvp/models/weights/face_detection_yunet_2023mar.onnx", "", (1280, 720))
cap = cv2.VideoCapture(vid_path)
fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video frames: {fc}, CSV rows: {len(df)}")

pitches = []
yaws = []

for i in range(len(df)):
    ret, frame = cap.read()
    if not ret:
        break
    _, faces = yunet.detect(frame)
    if faces is not None and len(faces) > 0:
        bb = faces[0][:4].astype(int)
        lm = faces[0][4:14].reshape(5, 2)
    else:
        bb = (0, 0, 0, 0)
        lm = None
    res = adapter.predict_gaze(frame, tuple(bb), lm)
    pitches.append(res["pitch_deg"])
    yaws.append(res["yaw_deg"])

cap.release()

df_eval = df.iloc[:len(pitches)].copy()
df_eval["fixed_pitch_deg"] = pitches
df_eval["fixed_yaw_deg"] = yaws

# Check calibration
calib_recs = df_eval[df_eval["stage"] == "calibration"]
test_recs = df_eval[df_eval["stage"] == "dynamic_test"]

print(f"Calibration rows: {len(calib_recs)}, Test rows: {len(test_recs)}")

samples = []
for p_idx, grp in calib_recs.groupby("point_idx"):
    samples.append({
        "target_x": float(grp["target_gt_x"].iloc[0]),
        "target_y": float(grp["target_gt_y"].iloc[0]),
        "yaw_deg": float(grp["fixed_yaw_deg"].median()),
        "pitch_deg": float(grp["fixed_pitch_deg"].median()),
        "head_yaw_deg": float(grp["head_yaw_deg"].median()),
        "head_pitch_deg": float(grp["head_pitch_deg"].median())
    })

calib = GazeCalibration(screen_w=2880, screen_h=1800, model_name="puregaze")
fit_res = calib.fit(samples, l2_reg=1.0)
print("Calibration Fit result:", fit_res)
print("Fitted weights:\n", calib.weights)

# Dynamic test evaluation
pred_xs, pred_ys = [], []
for _, row in test_recs.iterrows():
    px, py = calib.predict(
        pitch_deg=row["fixed_pitch_deg"],
        yaw_deg=row["fixed_yaw_deg"],
        head_pitch_deg=row["head_pitch_deg"],
        head_yaw_deg=row["head_yaw_deg"]
    )
    pred_xs.append(px)
    pred_ys.append(py)

gt_x = test_recs["target_gt_x"].values
gt_y = test_recs["target_gt_y"].values
px = np.array(pred_xs)
py = np.array(pred_ys)

err_px = np.sqrt((px - gt_x)**2 + (py - gt_y)**2)
err_mm = err_px / 9.536842
CX, CY, D = 1440.0, 900.0, 600.0
PX = 9.536842
vgx, vgy, vgz = (gt_x - CX)/PX, (gt_y - CY)/PX, -D
vpx, vpy, vpz = (px - CX)/PX, (py - CY)/PX, -D
ngt = np.sqrt(vgx**2 + vgy**2 + vgz**2)
npr = np.sqrt(vpx**2 + vpy**2 + vpz**2)
cos = np.clip((vgx*vpx + vgy*vpy + vgz*vpz)/(ngt*npr), -1.0, 1.0)
ang_err = np.degrees(np.arccos(cos))

rx = np.corrcoef(px, gt_x)[0, 1]
ry = np.corrcoef(py, gt_y)[0, 1]
ss_res = np.sum((px - gt_x)**2 + (py - gt_y)**2)
ss_tot = np.sum((gt_x - np.mean(gt_x))**2 + (gt_y - np.mean(gt_y))**2)
r2 = 1.0 - ss_res / ss_tot

print(f"\n==========================================")
print(f"VERIFIED PUREGAZE ON USER SESSION 233626:")
print(f"Mean Angular Error:   {np.mean(ang_err):.2f}°")
print(f"Median Angular Error: {np.median(ang_err):.2f}°")
print(f"Screen Error (mm):    {np.mean(err_mm):.1f} mm")
print(f"Screen Error (px):    {np.mean(err_px):.1f} px")
print(f"Pearson Correlation:  Rx = {rx:+.3f}, Ry = {ry:+.3f}")
print(f"R2 Score:             {r2:+.3f}")
print(f"==========================================")
