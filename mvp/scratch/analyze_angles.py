import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2, torch, pandas as pd, numpy as np
from PIL import Image
import torchvision.transforms as TVT
from models.registry import ModelRegistry
import models.yolo26n_gaze_net

v = "mvp/benchmarks/benchmark_test_20261014_115554.mp4"
m = "mvp/benchmarks/benchmark_test_20261014_115554.parquet"
df = pd.read_parquet(m)
model, _ = ModelRegistry.load_checkpoint("mvp/experiments/yolo26n_gaze_net_yolo26n/run_013_20261016_182944/checkpoints/latest_model.pt", target_device="cuda:0")
model.eval()

cap = cv2.VideoCapture(v)
fidx = 0; records = []
transform = TVT.Compose([TVT.Resize((224, 224)), TVT.ToTensor(), TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret or fidx >= len(df): break
        row = df.iloc[fidx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_in = transform(Image.fromarray(rgb)).unsqueeze(0).to("cuda:0")
        out = model(t_in)
        records.append({
            "phase": row["phase"],
            "gt_x": row["target_gt_x"],
            "gt_y": row["target_gt_y"],
            "gaze_pitch_deg": out["gaze_pitch_deg"][0].item(),
            "gaze_yaw_deg": out["gaze_yaw_deg"][0].item(),
            "head_pitch_deg": out["head_pitch_deg"][0].item(),
            "head_yaw_deg": out["head_yaw_deg"][0].item(),
            "gx": out["gaze_vector"][0, 0].item(),
            "gy": out["gaze_vector"][0, 1].item(),
            "gz": out["gaze_vector"][0, 2].item(),
            "hx": out["head_vector"][0, 0].item(),
            "hy": out["head_vector"][0, 1].item(),
            "hz": out["head_vector"][0, 2].item(),
        })
        fidx += 1
cap.release()
tdf = pd.DataFrame(records)

print("=== 9 GRID POINTS (Median values) ===")
gps = [x for x in tdf["phase"].unique() if "Grid Point" in x]
for gp in gps:
    s = tdf[tdf["phase"] == gp]
    tx = s["gt_x"].iloc[0]; ty = s["gt_y"].iloc[0]
    g_pit = s["gaze_pitch_deg"].median()
    h_pit = s["head_pitch_deg"].median()
    gy = s["gy"].median(); hy = s["hy"].median()
    print(f"{gp:25s} | tgt=({tx:.0f}, {ty:.0f}) | g_pit={g_pit:+.2f} deg | h_pit={h_pit:+.2f} deg | gy={gy:+.3f} | hy={hy:+.3f}")

print("\n=== CIRCLE TRAJECTORY CORRELATIONS ===")
circ = tdf[tdf["phase"] == "CircleTrajectory"]
for col in ["gaze_pitch_deg", "head_pitch_deg", "gy", "hy", "gaze_yaw_deg", "head_yaw_deg", "gx", "hx"]:
    cx = np.corrcoef(circ[col], circ["gt_x"])[0, 1]
    cy = np.corrcoef(circ[col], circ["gt_y"])[0, 1]
    print(f"  {col:15s}: corr_x={cx:+.3f}, corr_y={cy:+.3f}")
