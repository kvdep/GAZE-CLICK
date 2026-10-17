import sys
sys.path.insert(0, 'mvp')
import torch, numpy as np, pandas as pd
from models.registry import ModelRegistry
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedGazeDataset
from torch.utils.data import DataLoader

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_003_20261005_032352/checkpoints/best_model.pt'
model, _ = ModelRegistry.load_checkpoint(ckpt_path, target_device='cuda:0')
model.eval()
cfg = DatasetSelectionConfig(dataset_name='rtgene', data_file_path='DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet', images_base_path='DATA/rtgene/rtgene_dataset', max_samples=6000)
df = DataUnifier.load_rtgene(cfg)
raw = pd.read_parquet(cfg.data_file_path)
raw['sid'] = raw['session'].astype(str) + '_' + raw['seq'].astype(str)
merged = df.merge(raw[['sid', 'gaze_pitch_deg', 'gaze_yaw_deg']], left_on='sample_id', right_on='sid')

ds = UnifiedGazeDataset(df, (224, 224), False, True)
loader = DataLoader(ds, batch_size=64, shuffle=False)
preds_p, preds_y, angs_raw, angs_norm = [], [], [], []
dev = torch.device('cuda:0')

with torch.no_grad():
    for b in loader:
        out = model(b['image'].to(dev))
        preds_p.append(out['pitch_rad'].cpu().numpy())
        preds_y.append(out['yaw_rad'].cpu().numpy())
        pv = out['gaze_vector']
        gv = b['gaze_vector'].to(dev)
        pvn = pv / torch.norm(pv, dim=-1, keepdim=True)
        gvn = gv / torch.norm(gv, dim=-1, keepdim=True)
        a_norm = torch.acos(torch.clamp(torch.sum(pvn * gvn, dim=-1), -1.0, 1.0)) * (180.0 / np.pi)
        angs_norm.append(a_norm.cpu().numpy())

merged['pred_p_deg'] = np.degrees(np.concatenate(preds_p))
merged['pred_y_deg'] = np.degrees(np.concatenate(preds_y))
merged['ang_norm_deg'] = np.concatenate(angs_norm)

pred_v = model.spherical_angles_to_unit_vector(torch.tensor(np.concatenate(preds_p), device=dev), torch.tensor(np.concatenate(preds_y), device=dev))
raw_p_rad = torch.tensor(np.radians(merged['gaze_pitch_deg_y'].to_numpy(dtype=np.float32)), device=dev)
raw_y_rad = torch.tensor(np.radians(merged['gaze_yaw_deg_y'].to_numpy(dtype=np.float32)), device=dev)
raw_v = model.spherical_angles_to_unit_vector(raw_p_rad, raw_y_rad)
a_raw = torch.acos(torch.clamp(torch.sum(pred_v * raw_v, dim=-1), -1.0, 1.0)) * (180.0 / np.pi)
merged['ang_raw_deg'] = a_raw.cpu().numpy()

for s, g in merged.groupby('subject_id'):
    raw_p = g['gaze_pitch_deg_y'].mean()
    norm_p = g['gaze_pitch_deg_x'].mean()
    pred_p = g['pred_p_deg'].mean()
    raw_y = g['gaze_yaw_deg_y'].mean()
    norm_y = g['gaze_yaw_deg_x'].mean()
    pred_y = g['pred_y_deg'].mean()
    e_raw = g['ang_raw_deg'].mean()
    e_norm = g['ang_norm_deg'].mean()
    print(f"Subj {s}: E_raw={e_raw:.2f} deg, E_norm={e_norm:.2f} deg | Pitch [RAW={raw_p:+.2f}, NORM={norm_p:+.2f}, PRED={pred_p:+.2f}] | Yaw [RAW={raw_y:+.2f}, NORM={norm_y:+.2f}, PRED={pred_y:+.2f}]")
