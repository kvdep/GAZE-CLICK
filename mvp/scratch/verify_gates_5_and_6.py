import sys
sys.path.insert(0, 'mvp')
import numpy as np, pandas as pd, torch
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedDataModule
from models.base import BaseGazeModel

configs = [
    DatasetSelectionConfig('columbia_gaze', 'DATA/columbia_gaze/processed/columbia_gaze_full.parquet', 'DATA/columbia_gaze', max_samples=6000),
    DatasetSelectionConfig('mpiifacegaze', 'DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet', 'DATA/mpiifacegaze', max_samples=6000),
    DatasetSelectionConfig('rtgene', 'DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet', 'DATA/rtgene/rtgene_dataset', max_samples=6000)
]

unified_df = DataUnifier.unify(configs)
train_df, val_df, test_df = UnifiedDataModule.split_by_subjects(unified_df)

train_subjs = set(train_df['subject_id'].unique())
val_subjs = set(val_df['subject_id'].unique())
test_subjs = set(test_df['subject_id'].unique())

tr_va_overlap = train_subjs.intersection(val_subjs)
tr_te_overlap = train_subjs.intersection(test_subjs)
va_te_overlap = val_subjs.intersection(test_subjs)

gate5_pass = (len(tr_va_overlap) == 0 and len(tr_te_overlap) == 0 and len(va_te_overlap) == 0)
print(f"Gate 5 (Subject Disjoint): Tr-Va={len(tr_va_overlap)}, Tr-Te={len(tr_te_overlap)}, Va-Te={len(va_te_overlap)} | PASS={gate5_pass}")

pitch = torch.tensor(unified_df['gaze_pitch_rad'].to_numpy(dtype=np.float32))
yaw = torch.tensor(unified_df['gaze_yaw_rad'].to_numpy(dtype=np.float32))
recon_v = BaseGazeModel.spherical_angles_to_unit_vector(pitch, yaw)

target_v = torch.tensor(unified_df[['gaze_vec_x', 'gaze_vec_y', 'gaze_vec_z']].to_numpy(dtype=np.float32))
target_norm = target_v / torch.clamp(torch.norm(target_v, p=2, dim=-1, keepdim=True), min=1e-8)

dots = torch.sum(recon_v * target_norm, dim=-1)
clamped_dots = torch.clamp(dots, -1.0, 1.0)
disc_deg = torch.acos(clamped_dots) * (180.0 / np.pi)

max_disc = float(torch.max(disc_deg).item())
avg_disc = float(torch.mean(disc_deg).item())
gate6_pass = (max_disc <= 0.05)
print(f"Gate 6 (Discrepancy): MaxDisc={max_disc:.6f} deg, AvgDisc={avg_disc:.6f} deg | PASS={gate6_pass}")

hpitch = torch.tensor(unified_df['head_pitch_rad'].to_numpy(dtype=np.float32))
hyaw = torch.tensor(unified_df['head_yaw_rad'].to_numpy(dtype=np.float32))
recon_hv = BaseGazeModel.spherical_angles_to_unit_vector(hpitch, hyaw)
target_hv = torch.tensor(unified_df[['head_vec_x', 'head_vec_y', 'head_vec_z']].to_numpy(dtype=np.float32))
target_hnorm = target_hv / torch.clamp(torch.norm(target_hv, p=2, dim=-1, keepdim=True), min=1e-8)
hdots = torch.sum(recon_hv * target_hnorm, dim=-1)
hdisc_deg = torch.acos(torch.clamp(hdots, -1.0, 1.0)) * (180.0 / np.pi)
max_hdisc = float(torch.max(hdisc_deg).item())
print(f"Head Discrepancy: MaxHeadDisc={max_hdisc:.6f} deg")

for ds in unified_df['dataset_id'].unique():
    sub = unified_df[unified_df['dataset_id'] == ds]
    sub_disc = disc_deg[unified_df['dataset_id'] == ds]
    print(f"Dataset [{ds}] (N={len(sub)}): MaxDisc={torch.max(sub_disc).item():.6f} deg")
