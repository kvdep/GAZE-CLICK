import sys
sys.path.insert(0, 'mvp')
import torch, numpy as np, pandas as pd
from models.registry import ModelRegistry
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedGazeDataset
from torch.utils.data import DataLoader

model, _ = ModelRegistry.load_checkpoint('mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_003_20261005_032352/checkpoints/best_model.pt', target_device='cuda:0')
model.eval()
configs = [DatasetSelectionConfig(dataset_name='rtgene', data_file_path='DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet', images_base_path='DATA/rtgene/rtgene_dataset', max_samples=6000)]
df = DataUnifier.load_rtgene(configs[0])
ds = UnifiedGazeDataset(df, (224, 224), False, True)
loader = DataLoader(ds, batch_size=64, shuffle=False)

preds_p, preds_y, gts_p, gts_y, angs = [], [], [], [], []
dev = torch.device('cuda:0')

with torch.no_grad():
    for b in loader:
        out = model(b['image'].to(dev))
        pv = out['gaze_vector']
        gv = b['gaze_vector'].to(dev)
        pvn = pv / torch.norm(pv, dim=-1, keepdim=True)
        gvn = gv / torch.norm(gv, dim=-1, keepdim=True)
        dot = torch.sum(pvn * gvn, dim=-1)
        a = torch.acos(torch.clamp(dot, -1.0, 1.0)) * (180.0 / np.pi)
        angs.append(a.cpu().numpy())
        preds_p.append(out['pitch_rad'].cpu().numpy())
        preds_y.append(out['yaw_rad'].cpu().numpy())
        gts_p.append(b['gaze_angles'][:, 0].numpy())
        gts_y.append(b['gaze_angles'][:, 1].numpy())

df['pred_p_deg'] = np.degrees(np.concatenate(preds_p))
df['pred_y_deg'] = np.degrees(np.concatenate(preds_y))
df['gt_p_deg'] = np.degrees(np.concatenate(gts_p))
df['gt_y_deg'] = np.degrees(np.concatenate(gts_y))
df['ang_err_deg'] = np.concatenate(angs)
df['err_p_deg'] = df['pred_p_deg'] - df['gt_p_deg']
df['err_y_deg'] = df['pred_y_deg'] - df['gt_y_deg']

for s, g in df.groupby('subject_id'):
    print(f"Subject {s} (N={len(g)}): AngErr={g['ang_err_deg'].mean():.2f} (med={g['ang_err_deg'].median():.2f}) | Pitch Err={g['err_p_deg'].mean():+.2f} (MAE={g['err_p_deg'].abs().mean():.2f}) | Yaw Err={g['err_y_deg'].mean():+.2f} (MAE={g['err_y_deg'].abs().mean():.2f})")
