import sys
sys.path.insert(0, 'mvp')
import torch, numpy as np, pandas as pd
from models.registry import ModelRegistry
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedGazeDataset
from torch.utils.data import DataLoader

ckpt = 'mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_003_20261005_032352/checkpoints/best_model.pt'
model, _ = ModelRegistry.load_checkpoint(ckpt, target_device='cuda:0')
model.eval()

cfg = DatasetSelectionConfig('rtgene', 'DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet', 'DATA/rtgene/rtgene_dataset', max_samples=6000)
df = DataUnifier.load_rtgene(cfg)
ds = UnifiedGazeDataset(df, (224, 224), False, True)
loader = DataLoader(ds, batch_size=64, shuffle=False)

dev = torch.device('cuda:0')
hps, hys = [], []

with torch.no_grad():
    for b in loader:
        out = model(b['image'].to(dev))
        hps.append(out['head_pitch_rad'].cpu().numpy())
        hys.append(out['head_yaw_rad'].cpu().numpy())

df['pred_hp_deg'] = np.degrees(np.concatenate(hps))
df['pred_hy_deg'] = np.degrees(np.concatenate(hys))
df['hp_err'] = np.abs(df['pred_hp_deg'] - df['head_pitch_deg'])
df['hy_err'] = np.abs(df['pred_hy_deg'] - df['head_yaw_deg'])

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
print(df.groupby('subject_id')[['hp_err', 'hy_err', 'pred_hp_deg', 'head_pitch_deg', 'pred_hy_deg', 'head_yaw_deg']].mean())
