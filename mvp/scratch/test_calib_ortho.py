import sys, os, cv2, pandas as pd, numpy as np
sys.path.insert(0, "mvp")
from engine.face_pipeline import GazeTrackingPipeline
df = pd.read_parquet("mvp/benchmarks/benchmark_test_20261014_115554.parquet")
cap = cv2.VideoCapture("mvp/benchmarks/benchmark_test_20261014_115554.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

def test_reg_and_ortho(ckpt_path):
    p = GazeTrackingPipeline(checkpoint_path=ckpt_path)
    records = []
    for idx, f in enumerate(frames):
        row = df.iloc[idx]
        res = p.process_frame(f)
        records.append({
            "target_gt_x": float(row["target_gt_x"]),
            "target_gt_y": float(row["target_gt_y"]),
            "phase": str(row.get("phase", "")),
            "p_deg": res.get("gaze_pitch_deg", 0.0),
            "y_deg": res.get("gaze_yaw_deg", 0.0),
            "hp_deg": res.get("head_pitch_deg", 0.0),
            "hy_deg": res.get("head_yaw_deg", 0.0)
        })
    vdf = pd.DataFrame(records)
    gps = [ph for ph in vdf["phase"].unique() if "Grid Point" in ph]
    cal_pts = []
    for gp in gps:
        sub = vdf[vdf["phase"] == gp]
        cal_pts.append([
            1.0,
            float(sub["y_deg"].median()),
            float(sub["p_deg"].median()),
            float(sub["hy_deg"].median()),
            float(sub["hp_deg"].median()),
            float(sub["target_gt_x"].iloc[0]),
            float(sub["target_gt_y"].iloc[0])
        ])
    C = np.array(cal_pts)
    X5 = C[:, :5]; Y_gt = C[:, 5:]
    circle = vdf[vdf["phase"] == "CircleTrajectory"]
    C_test = np.column_stack([
        np.ones(len(circle)),
        circle["y_deg"].to_numpy(),
        circle["p_deg"].to_numpy(),
        circle["hy_deg"].to_numpy(),
        circle["hp_deg"].to_numpy()
    ])
    gt = circle[["target_gt_x", "target_gt_y"]].to_numpy()
    cx, cy, dist_px = 1440.0, 900.0, 600.0 * (2880 / 301.6)
    v_gt = np.stack([gt[:, 0] - cx, gt[:, 1] - cy, -np.full(len(gt), dist_px)], axis=-1)
    norm_gt = v_gt / np.linalg.norm(v_gt, axis=-1, keepdims=True)
    tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path))) + "/" + os.path.basename(ckpt_path)
    for l2 in [1.0, 10.0, 50.0, 100.0, 200.0]:
        I = np.eye(5); I[0, 0] = 0.0
        W = np.linalg.solve(X5.T @ X5 + l2 * I, X5.T @ Y_gt)
        preds = C_test @ W
        v_p = np.stack([preds[:, 0] - cx, preds[:, 1] - cy, -np.full(len(preds), dist_px)], axis=-1)
        norm_p = v_p / np.linalg.norm(v_p, axis=-1, keepdims=True)
        dot = np.clip(np.sum(norm_gt * norm_p, axis=-1), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        err_mm = np.linalg.norm(preds - gt, axis=-1) / (2880 / 301.6)
        print(f"{tag} Full5 L2={l2:5.1f} | Ang: {np.mean(ang):.2f} deg, mm: {np.mean(err_mm):.1f}, Rx: {np.corrcoef(preds[:,0], gt[:,0])[0,1]:.3f}, Ry: {np.corrcoef(preds[:,1], gt[:,1])[0,1]:.3f}")
    X_x3 = C[:, [0, 1, 3]]
    X_y3 = C[:, [0, 2, 4]]
    I3 = np.eye(3); I3[0, 0] = 0.0
    for l2 in [1.0, 10.0, 50.0, 100.0]:
        Wx = np.linalg.solve(X_x3.T @ X_x3 + l2 * I3, X_x3.T @ Y_gt[:, 0])
        Wy = np.linalg.solve(X_y3.T @ X_y3 + l2 * I3, X_y3.T @ Y_gt[:, 1])
        px = C_test[:, [0, 1, 3]] @ Wx
        py = C_test[:, [0, 2, 4]] @ Wy
        preds = np.column_stack([px, py])
        v_p = np.stack([preds[:, 0] - cx, preds[:, 1] - cy, -np.full(len(preds), dist_px)], axis=-1)
        norm_p = v_p / np.linalg.norm(v_p, axis=-1, keepdims=True)
        dot = np.clip(np.sum(norm_gt * norm_p, axis=-1), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        err_mm = np.linalg.norm(preds - gt, axis=-1) / (2880 / 301.6)
        print(f"{tag} ORTHO L2={l2:5.1f} | Ang: {np.mean(ang):.2f} deg, mm: {np.mean(err_mm):.1f}, Rx: {np.corrcoef(preds[:,0], gt[:,0])[0,1]:.3f}, Ry: {np.corrcoef(preds[:,1], gt[:,1])[0,1]:.3f}")

print("=== RUN_010 ===")
test_reg_and_ortho("mvp/experiments/yolo26n_gaze_net_yolo26n/run_010_20261015_020142/checkpoints/best_model.pt")
print("=== RUN_013_BEST ===")
test_reg_and_ortho("mvp/experiments/yolo26n_gaze_net_yolo26n/run_013_20261016_182944/checkpoints/best_model.pt")
print("=== RUN_013_LATEST ===")
test_reg_and_ortho("mvp/experiments/yolo26n_gaze_net_yolo26n/run_013_20261016_182944/checkpoints/latest_model.pt")
