import sys
sys.path.insert(0, ".")
import torch, numpy as np, pandas as pd
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedDataModule
from models.yolo26n_deep_peft_gaze_net import YOLO26nDeepPeftGazeNet
from models.yolo26s_gaze_net import YOLO26sGazeNet
from models.yolo26n_gaze_net import YOLO26nGazeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = [
    DatasetSelectionConfig(dataset_name="columbia_gaze", data_file_path="DATA/columbia_gaze/processed/columbia_gaze_full.parquet", images_base_path="DATA/columbia_gaze"),
    DatasetSelectionConfig(dataset_name="mpiifacegaze", data_file_path="DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet", images_base_path="DATA/mpiifacegaze"),
    DatasetSelectionConfig(dataset_name="rtgene", data_file_path="DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet", images_base_path="DATA/rtgene/rtgene_dataset"),
    DatasetSelectionConfig(dataset_name="gaze360", data_file_path="DATA/Gaze360/processed/gaze360_full.parquet", images_base_path="DATA/Gaze360")
]
unified_df = DataUnifier.unify(configs)
train_df, val_df, test_df = UnifiedDataModule.split_by_subjects(unified_df)
_, _, test_loader = UnifiedDataModule.build_dataloaders(train_df, val_df, test_df, batch_size=512)

def evaluate_by_dataset(ckpt_path, model_cls, arch_kwargs):
    m = model_cls(**arch_kwargs).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state["model_state_dict"])
    
    all_errors = []
    all_dataset_ids = []
    
    idx = 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            gaze_targets = batch["gaze_vector"].to(device)
            out = m(imgs)
            pred = out["gaze_vector"]
            
            sim = torch.sum(pred * gaze_targets, dim=-1).clamp(-1.0, 1.0)
            ang_err = torch.acos(sim) * (180.0 / np.pi)
            all_errors.extend(ang_err.cpu().numpy().tolist())
            
            b_size = imgs.size(0)
            batch_ds_ids = test_df["dataset_id"].iloc[idx:idx+b_size].tolist()
            all_dataset_ids.extend(batch_ds_ids)
            idx += b_size
            
    eval_df = pd.DataFrame({"dataset_id": all_dataset_ids, "angular_error": all_errors})
    print(f"=== Results for {ckpt_path} ===")
    print("Overall Mean Angular Error:", eval_df["angular_error"].mean())
    for d_id, grp in eval_df.groupby("dataset_id"):
        mean_v = grp["angular_error"].mean()
        med_v = grp["angular_error"].median()
        count_v = len(grp)
        print(f"  {d_id}: mean={mean_v:.2f} deg, median={med_v:.2f} deg (N={count_v})")

print("Evaluating YOLO26n Deep PEFT...")
evaluate_by_dataset("experiments/yolo26n_deep_peft_gaze_net_yolo26n/run_001_20261103_173055/checkpoints/best_model.pt", YOLO26nDeepPeftGazeNet, {"weights_path": "models/weights/yolo26n.pt"})

print("Evaluating YOLO26s...")
evaluate_by_dataset("experiments/yolo26s_gaze_net_yolo26s/run_001_20261103_075359/checkpoints/best_model.pt", YOLO26sGazeNet, {"weights_path": "models/weights/yolo26s.pt"})

print("Evaluating YOLO26n...")
evaluate_by_dataset("experiments/yolo26n_gaze_net_yolo26n/run_018_20261103_014000/checkpoints/best_model.pt", YOLO26nGazeNet, {"weights_path": "models/weights/yolo26n.pt"})
