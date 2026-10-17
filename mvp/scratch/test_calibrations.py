import sys, os, cv2, torch, pandas as pd, numpy as np
sys.path.insert(0, os.path.abspath("mvp"))
from engine.face_pipeline import GazeTrackingPipeline
from engine.calibration import GazeCalibration

df = pd.read_csv("mvp/benchmarks/benchmark_test_20261013_182824.csv")
p8 = GazeTrackingPipeline(checkpoint_path="mvp/experiments/yolo26n_gaze_net_yolo26n/run_008_20261014_024920/checkpoints/best_model.pt")
cap = cv2.VideoCapture("mvp/benchmarks/benchmark_test_20261013_182824.mp4")

screen_w, screen_h = 2880, 1800
phys_w_mm, phys_h_mm, dist_mm = 300.0, 190.0, 600.0
px_per_mm = ((screen_w / phys_w_mm) + (screen_h / phys_h_mm)) / 2.0
dist_px = dist_mm * px_per_mm
cx, cy = screen_w / 2.0, screen_h / 2.0
px_per_mm_x = screen_w / phys_w_mm
px_per_mm_y = screen_h / phys_h_mm

records = []
for i in range(len(df)):
    ret, f = cap.read()
    if not ret: break
    r8 = p8.process_frame(f)
    if r8["face_detected"]:
        records.append({
            "gt_x": df.iloc[i]["target_gt_x"],
            "gt_y": df.iloc[i]["target_gt_y"],
            "yaw": r8["raw_yaw_deg"],
            "pitch": r8["raw_pitch_deg"],
            "phase": df.iloc[i].get("phase", "")
        })
cap.release()
rdf = pd.DataFrame(records)

y_rad_1p = np.radians(rdf["yaw"] - 2.57)
p_rad_1p = np.radians(rdf["pitch"] - (-0.74))
dx_1p = -dist_mm * np.tan(y_rad_1p) * 1.85
dy_1p = -dist_mm * np.tan(p_rad_1p) * 1.65
pred_x_1p = np.clip(cx + dx_1p * px_per_mm_x, 0, screen_w - 1)
pred_y_1p = np.clip(cy + dy_1p * px_per_mm_y, 0, screen_h - 1)

gt_x_arr = rdf["gt_x"].to_numpy()
gt_y_arr = rdf["gt_y"].to_numpy()
pred_x_1p = np.array(pred_x_1p)
pred_y_1p = np.array(pred_y_1p)

diffs_1p = np.stack([pred_x_1p - gt_x_arr, pred_y_1p - gt_y_arr], axis=-1)
err_px_1p = np.linalg.norm(diffs_1p, axis=-1)
err_mm_1p = err_px_1p / px_per_mm

v_gt = np.stack([gt_x_arr - cx, gt_y_arr - cy, -np.full(len(rdf), dist_px)], axis=-1)
v_pred_1p = np.stack([pred_x_1p - cx, pred_y_1p - cy, -np.full(len(rdf), dist_px)], axis=-1)
norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)
norm_pred_1p = v_pred_1p / np.linalg.norm(v_pred_1p, axis=-1, keepdims=True)
dot_1p = np.clip(np.sum(norm_gt * norm_pred_1p, axis=-1), -1.0, 1.0)
ang_1p = np.degrees(np.arccos(dot_1p))

same_x_1p = (np.sign(pred_x_1p - cx) == np.sign(gt_x_arr - cx)).mean() * 100
same_y_1p = (np.sign(pred_y_1p - cy) == np.sign(gt_y_arr - cy)).mean() * 100
corr_x_1p = float(np.corrcoef(pred_x_1p, gt_x_arr)[0, 1])
corr_y_1p = float(np.corrcoef(pred_y_1p, gt_y_arr)[0, 1])

print("=== 1-POINT ZERO-SHOT (CENTER ALIGNED) ===")
print(f"Angular Error Mean: {ang_1p.mean():.2f} deg, Median: {np.median(ang_1p):.2f} deg")
print(f"Screen Error Mean: {err_mm_1p.mean():.1f} mm ({err_px_1p.mean():.1f} px)")
print(f"Corr X: {corr_x_1p:.4f}, Corr Y: {corr_y_1p:.4f}")
print(f"Sign Acc X: {same_x_1p:.1f}%, Sign Acc Y: {same_y_1p:.1f}%")
print(f"Bias X: {diffs_1p[:, 0].mean():.1f} px, Bias Y: {diffs_1p[:, 1].mean():.1f} px")

calib_targets = rdf[["gt_x", "gt_y"]].drop_duplicates()
calib_samples = []
for _, ut in calib_targets.iterrows():
    match = rdf[(abs(rdf["gt_x"] - ut["gt_x"]) < 5) & (abs(rdf["gt_y"] - ut["gt_y"]) < 5)].iloc[0]
    calib_samples.append({"target_x": match["gt_x"], "target_y": match["gt_y"], "pitch_deg": match["pitch"], "yaw_deg": match["yaw"]})

calib = GazeCalibration(screen_w=screen_w, screen_h=screen_h, profile_path="mvp/scratch/temp_calib.json")
calib.fit(calib_samples, l2_reg=0.1)

few_pred_x, few_pred_y = [], []
for _, row in rdf.iterrows():
    px, py = calib.predict(row["pitch"], row["yaw"])
    few_pred_x.append(px); few_pred_y.append(py)
few_pred_x = np.array(few_pred_x); few_pred_y = np.array(few_pred_y)

diffs_fs = np.stack([few_pred_x - rdf["gt_x"], few_pred_y - rdf["gt_y"]], axis=-1)
err_px_fs = np.linalg.norm(diffs_fs, axis=-1)
err_mm_fs = err_px_fs / px_per_mm

v_pred_fs = np.stack([few_pred_x - cx, few_pred_y - cy, -np.full(len(rdf), dist_px)], axis=-1)
norm_pred_fs = v_pred_fs / np.linalg.norm(v_pred_fs, axis=-1, keepdims=True)
dot_fs = np.clip(np.sum(norm_gt * norm_pred_fs, axis=-1), -1.0, 1.0)
ang_fs = np.degrees(np.arccos(dot_fs))

same_x_fs = (np.sign(few_pred_x - cx) == np.sign(gt_x_arr - cx)).mean() * 100
same_y_fs = (np.sign(few_pred_y - cy) == np.sign(gt_y_arr - cy)).mean() * 100
corr_x_fs = float(np.corrcoef(few_pred_x, gt_x_arr)[0, 1])
corr_y_fs = float(np.corrcoef(few_pred_y, gt_y_arr)[0, 1])

print("\n=== 9-POINT FEW-SHOT CALIBRATION ===")
print(f"Angular Error Mean: {ang_fs.mean():.2f} deg, Median: {np.median(ang_fs):.2f} deg")
print(f"Screen Error Mean: {err_mm_fs.mean():.1f} mm ({err_px_fs.mean():.1f} px)")
print(f"Corr X: {corr_x_fs:.4f}, Corr Y: {corr_y_fs:.4f}")
print(f"Sign Acc X: {same_x_fs:.1f}%, Sign Acc Y: {same_y_fs:.1f}%")
print(f"Bias X: {diffs_fs[:, 0].mean():.1f} px, Bias Y: {diffs_fs[:, 1].mean():.1f} px")

X_rows, Y_rows = [], []
for s in calib_samples:
    X_rows.append([1.0, s["yaw_deg"], s["pitch_deg"]])
    Y_rows.append([s["target_x"], s["target_y"]])
X = np.array(X_rows); Y = np.array(Y_rows)
reg = 0.01 * np.eye(3); reg[0, 0] = 0.0
W = np.linalg.solve(X.T @ X + reg, X.T @ Y)

feats = np.stack([np.ones(len(rdf)), rdf["yaw"], rdf["pitch"]], axis=1)
preds_pure = feats @ W
pred_x_pure = np.clip(preds_pure[:, 0], 0, screen_w - 1)
pred_y_pure = np.clip(preds_pure[:, 1], 0, screen_h - 1)

diffs_pure = np.stack([pred_x_pure - gt_x_arr, pred_y_pure - gt_y_arr], axis=-1)
err_px_pure = np.linalg.norm(diffs_pure, axis=-1)
err_mm_pure = err_px_pure / px_per_mm

v_pred_pure = np.stack([pred_x_pure - cx, pred_y_pure - cy, -np.full(len(rdf), dist_px)], axis=-1)
norm_pred_pure = v_pred_pure / np.linalg.norm(v_pred_pure, axis=-1, keepdims=True)
ang_pure = np.degrees(np.arccos(np.clip(np.sum(norm_gt * norm_pred_pure, axis=-1), -1.0, 1.0)))

same_x_pure = (np.sign(pred_x_pure - cx) == np.sign(gt_x_arr - cx)).mean() * 100
same_y_pure = (np.sign(pred_y_pure - cy) == np.sign(gt_y_arr - cy)).mean() * 100
corr_x_pure = float(np.corrcoef(pred_x_pure, gt_x_arr)[0, 1])
corr_y_pure = float(np.corrcoef(pred_y_pure, gt_y_arr)[0, 1])

print("\n=== PURE AFFINE FEW-SHOT (WITHOUT ARTIFICIAL EXPANSION) ===")
print(f"Angular Error Mean: {ang_pure.mean():.2f} deg, Median: {np.median(ang_pure):.2f} deg")
print(f"Screen Error Mean: {err_mm_pure.mean():.1f} mm ({err_px_pure.mean():.1f} px)")
print(f"Corr X: {corr_x_pure:.4f}, Corr Y: {corr_y_pure:.4f}")
print(f"Sign Acc X: {same_x_pure:.1f}%, Sign Acc Y: {same_y_pure:.1f}%")
print(f"Bias X: {diffs_pure[:, 0].mean():.1f} px, Bias Y: {diffs_pure[:, 1].mean():.1f} px")

