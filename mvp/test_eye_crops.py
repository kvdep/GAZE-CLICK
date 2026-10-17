import os, sys
sys.path.insert(0, "mvp")
import numpy as np, pandas as pd, cv2
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedGazeDataset

configs = [
    DatasetSelectionConfig("columbia_gaze", "DATA/columbia_gaze/processed/columbia_gaze_full.parquet", "DATA/columbia_gaze", max_samples=5),
    DatasetSelectionConfig("mpiifacegaze", "DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet", "DATA/mpiifacegaze", max_samples=5),
    DatasetSelectionConfig("rtgene", "DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet", "DATA/rtgene/rtgene_dataset", max_samples=5)
]
df = DataUnifier.unify(configs)
ds = UnifiedGazeDataset(df, is_train=False, preload_cache=True)
os.makedirs("mvp/scratch", exist_ok=True)

for i in [0, 5, 10]:
    item = ds[i]
    img = item["image"].numpy().transpose(1, 2, 0)
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr, (27, 44), (91, 108), (0, 255, 0), 2)
    cv2.rectangle(img_bgr, (133, 44), (197, 108), (0, 0, 255), 2)
    ds_name = df.iloc[i]["dataset_id"]
    name = f"mvp/scratch/sample_{ds_name}_{i}.jpg"
    cv2.imwrite(name, img_bgr)
    print("Saved", name)
