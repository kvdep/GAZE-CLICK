import sys, math, torch, numpy as np, pandas as pd
sys.path.insert(0, "mvp")
from models.registry import ModelRegistry
from data.unification import DataUnifier, UnifiedDataModule, DatasetSelectionConfig, UnifiedGazeDataset
from torch.utils.data import DataLoader

ckpt_path = "mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_002_20261005_012813/checkpoints/best_model.pt"
model, payload = ModelRegistry.load_checkpoint(ckpt_path, target_device="cuda:0")
model.eval()

configs = [
    DatasetSelectionConfig("columbia_gaze", "DATA/columbia_gaze/processed/columbia_gaze_full.parquet", "DATA/columbia_gaze", max_samples=6000),
    DatasetSelectionConfig("mpiifacegaze", "DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet", "DATA/mpiifacegaze", max_samples=6000),
    DatasetSelectionConfig("rtgene", "DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet", "DATA/rtgene/rtgene_dataset", max_samples=6000)
]
df = DataUnifier.unify(configs)
tr_df, va_df, te_df = UnifiedDataModule.split_by_subjects(df)

def inspect_split(split_df, split_name):
    ds = UnifiedGazeDataset(split_df, is_train=False, preload_cache=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    records = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to("cuda:0")
            out = model(imgs)
            pred_gp = (out["pitch_rad"] * (180.0 / math.pi)).cpu().numpy()
            pred_gy = (out["yaw_rad"] * (180.0 / math.pi)).cpu().numpy()
            pred_hp = (out["head_pitch_rad"] * (180.0 / math.pi)).cpu().numpy()
            pred_hy = (out["head_yaw_rad"] * (180.0 / math.pi)).cpu().numpy()
            gt_g = (batch["gaze_angles"] * (180.0 / math.pi)).numpy()
            gt_h = (batch["head_angles"] * (180.0 / math.pi)).numpy()
            gt_gv = batch["gaze_vector"].to("cuda:0")
            pred_gv = out["gaze_vector"]
            p_n = pred_gv / torch.norm(pred_gv, dim=-1, keepdim=True)
            g_n = gt_gv / torch.norm(gt_gv, dim=-1, keepdim=True)
            ang = (torch.acos(torch.clamp(torch.sum(p_n * g_n, dim=-1), -1.0, 1.0)) * (180.0 / math.pi)).cpu().numpy()
            bs = len(ang)
            for j in range(bs):
                row = split_df.iloc[idx + j]
                records.append({
                    "dataset": row["dataset_id"],
                    "subject": row["subject_id"],
                    "ang_err": ang[j],
                    "gp_err": pred_gp[j] - gt_g[j, 0],
                    "gy_err": pred_gy[j] - gt_g[j, 1],
                    "hp_err": pred_hp[j] - gt_h[j, 0],
                    "hy_err": pred_hy[j] - gt_h[j, 1],
                    "gt_gp": gt_g[j, 0],
                    "gt_gy": gt_g[j, 1],
                    "gt_hp": gt_h[j, 0],
                    "gt_hy": gt_h[j, 1],
                    "pred_gp": pred_gp[j],
                    "pred_gy": pred_gy[j],
                    "pred_hp": pred_hp[j],
                    "pred_hy": pred_hy[j]
                })
            idx += bs
    res = pd.DataFrame(records)
    print(f"=== {split_name.upper()} BREAKDOWN ===")
    for ds_name, grp in res.groupby("dataset"):
        print(f"\nDataset: {ds_name} (N={len(grp)})")
        print(f"  Angular Error: mean={grp['ang_err'].mean():.2f}°, median={grp['ang_err'].median():.2f}°")
        print(f"  Gaze Pitch: Bias={grp['gp_err'].mean():.2f}°, MAE={grp['gp_err'].abs().mean():.2f}°")
        print(f"  Gaze Yaw:   Bias={grp['gy_err'].mean():.2f}°, MAE={grp['gy_err'].abs().mean():.2f}°")
        print(f"  Head Pitch: Bias={grp['hp_err'].mean():.2f}°, MAE={grp['hp_err'].abs().mean():.2f}°")
        print(f"  Head Yaw:   Bias={grp['hy_err'].mean():.2f}°, MAE={grp['hy_err'].abs().mean():.2f}°")
        for subj, sgrp in grp.groupby("subject"):
            print(f"    Subj {subj} (N={len(sgrp)}): AngErr={sgrp['ang_err'].mean():.2f}°, GP_bias={sgrp['gp_err'].mean():.2f}°, GY_bias={sgrp['gy_err'].mean():.2f}°, HP_bias={sgrp['hp_err'].mean():.2f}°, HY_bias={sgrp['hy_err'].mean():.2f}°")

inspect_split(va_df, "Validation")
inspect_split(te_df, "Test")
