import os, sys, cv2, torch, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("mvp"))
from adapters.factory import create_adapter
from engine.calibration import GazeCalibration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
puregaze = create_adapter("puregaze", device)
gazetr = create_adapter("gazetr", device)

# Let's run on puregaze_20261025_233626_ang_err_deg_6.42.csv
# In this CSV, gaze_pitch_deg and gaze_yaw_deg were recorded with the transposed PureGazeAdapter.
# Now PureGazeAdapter returns (yaw_rad, pitch_rad), so let's verify what happens when the corrected angles are calibrated using GazeCalibration!

df = pd.read_csv("mvp/benchmarks/puregaze_20261025_233626_ang_err_deg_6.42.csv")
calib_recs = df[df["stage"] == "calibration"]
test_recs = df[df["stage"] == "dynamic_test"]

samples = []
for p_idx, grp in calib_recs.groupby("point_idx"):
    samples.append({
        "target_x": float(grp["target_gt_x"].iloc[0]),
        "target_y": float(grp["target_gt_y"].iloc[0]),
        # With the fix in PureGazeAdapter: gaze_pitch_deg was angles[0,0] (yaw), gaze_yaw_deg was angles[0,1] (pitch)
        "yaw_deg": float(grp["gaze_pitch_deg"].median()),
        "pitch_deg": float(grp["gaze_yaw_deg"].median()),
        "head_yaw_deg": float(grp["head_yaw_deg"].median()),
        "head_pitch_deg": float(grp["head_pitch_deg"].median())
    })

calib = GazeCalibration(screen_w=2880, screen_h=1800, model_name="puregaze")
calib.fit(samples, l2_reg=1.0)

pred_xs, pred_ys = [], []
for _, row in test_recs.iterrows():
    px, py = calib.predict(
        pitch_deg=row["gaze_yaw_deg"], # corrected pitch
        yaw_deg=row["gaze_pitch_deg"],  # corrected yaw
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

print(f"GazeCalibration evaluation of user's live session:")
print(f"Mean AngErr: {np.mean(ang_err):.2f}°")
print(f"Median AngErr: {np.median(ang_err):.2f}°")
print(f"Rx = {rx:+.3f}, Ry = {ry:+.3f}, R2 = {r2:+.3f}")
print(f"Mean error in mm: {np.mean(err_mm):.1f} mm")
