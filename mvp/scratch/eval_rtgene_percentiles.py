import sys
sys.path.insert(0, "mvp")
import torch, numpy as np, pandas as pd
from models.registry import ModelRegistry
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedGazeDataset
from torch.utils.data import DataLoader

model, _ = ModelRegistry.load_checkpoint("mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_005_20261005_151240/checkpoints/best_model.pt", target_device="cuda:0")
model.eval()

cfg = DatasetSelectionConfig("rtgene", "DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet", "DATA/rtgene/rtgene_dataset")
df = DataUnifier.load_rtgene(cfg)

for subj in ["p005", "p012"]:
    sub_df = df[df["subject_id"] == subj].reset_index(drop=True)
    ds = UnifiedGazeDataset(sub_df, is_train=False, preload_cache=False)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    errs = []
    pitch_errs = []
    yaw_errs = []
    with torch.no_grad():
        for b in dl:
            imgs = b["image"].cuda()
            out = model(imgs)
            pred_v = out["gaze_vector"]
            pred_v = pred_v / torch.clamp(torch.norm(pred_v, dim=-1, keepdim=True), min=1e-8)
            gt_v = b["gaze_vector"].cuda()
            gt_v = gt_v / torch.clamp(torch.norm(gt_v, dim=-1, keepdim=True), min=1e-8)
            dot = torch.clamp(torch.sum(pred_v * gt_v, dim=-1), -1.0, 1.0)
            deg = torch.acos(dot) * (180.0 / 3.141592653589793)
            errs.extend(deg.cpu().numpy().tolist())
            pred_p = out["pitch_rad"].cpu().numpy() * (180.0 / 3.141592653589793)
            tgt_p = b["gaze_angles"][:, 0].numpy() * (180.0 / 3.141592653589793)
            pred_y = out["yaw_rad"].cpu().numpy() * (180.0 / 3.141592653589793)
            tgt_y = b["gaze_angles"][:, 1].numpy() * (180.0 / 3.141592653589793)
            pitch_errs.extend(np.abs(pred_p - tgt_p).tolist())
            yaw_errs.extend(np.abs(pred_y - tgt_y).tolist())
    errs = np.array(errs)
    p_errs = np.array(pitch_errs)
    y_errs = np.array(yaw_errs)
    print(f"Subject {subj} (N={len(errs)}):")
    print(f"  Angular Error: mean={errs.mean():.2f}°, med={np.median(errs):.2f}°, q25={np.percentile(errs, 25):.2f}°, q75={np.percentile(errs, 75):.2f}°, q90={np.percentile(errs, 90):.2f}°, max={errs.max():.2f}°")
    print(f"  Pitch Error:   mean={p_errs.mean():.2f}°, med={np.median(p_errs):.2f}°, q75={np.percentile(p_errs, 75):.2f}°, max={p_errs.max():.2f}°")
    print(f"  Yaw Error:     mean={y_errs.mean():.2f}°, med={np.median(y_errs):.2f}°, q75={np.percentile(y_errs, 75):.2f}°, max={y_errs.max():.2f}°")
