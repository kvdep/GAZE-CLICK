import os, sys, time, math, torch, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader
from models.registry import ModelRegistry
from data.unification import DataUnifier, UnifiedDataModule, DatasetSelectionConfig, UnifiedGazeDataset
from utils.logger import metric_log
from utils.memory import MemoryManager, scoped_model_device

def audit_split(df_subset: pd.DataFrame, split_name: str, model: torch.nn.Module, target_device: torch.device) -> Dict[str, Dict[str, float]]:
    if len(df_subset) == 0:
        return {}
    eval_ds = UnifiedGazeDataset(df_subset, image_resolution=(224, 224), is_train=False, preload_cache=True)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    ang_errors = []
    hp_errors = []
    hy_errors = []
    disc_errors = []
    screen_errors = []
    has_screen_list = []
    dataset_ids = df_subset["dataset_id"].to_numpy()
    
    with scoped_model_device(model, target_device) as active_model:
        active_model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                imgs = batch["image"].to(target_device)
                outputs = active_model(imgs)
                gt_gaze_v = batch["gaze_vector"].to(target_device)
                pred_gaze_v = outputs["gaze_vector"]
                pred_gaze_norm = pred_gaze_v / torch.clamp(torch.norm(pred_gaze_v, p=2, dim=-1, keepdim=True), min=1e-8)
                gt_gaze_norm = gt_gaze_v / torch.clamp(torch.norm(gt_gaze_v, p=2, dim=-1, keepdim=True), min=1e-8)
                dot = torch.sum(pred_gaze_norm * gt_gaze_norm, dim=-1)
                ang_err_deg = torch.acos(torch.clamp(dot, -1.0, 1.0)) * (180.0 / math.pi)
                gt_head_ang = batch["head_angles"].to(target_device)
                pred_hp = outputs["head_pitch_rad"]; pred_hy = outputs["head_yaw_rad"]
                hp_mae = torch.abs(pred_hp - gt_head_ang[:, 0]) * (180.0 / math.pi)
                hy_mae = torch.abs(pred_hy - gt_head_ang[:, 1]) * (180.0 / math.pi)
                pred_p = outputs["pitch_rad"]; pred_y = outputs["yaw_rad"]
                reconstructed_v = active_model.spherical_angles_to_unit_vector(pred_p, pred_y)
                recon_dot = torch.sum(pred_gaze_norm * reconstructed_v, dim=-1)
                disc_deg = torch.acos(torch.clamp(recon_dot, -1.0, 1.0)) * (180.0 / math.pi)
                has_scr = batch["has_screen_coords"].to(target_device)
                scr_err = torch.zeros_like(ang_err_deg)
                if "screen_coordinates" in outputs and has_scr.any():
                    pred_scr = outputs["screen_coordinates"][has_scr]
                    gt_scr = batch["screen_coords"].to(target_device)[has_scr]
                    scr_err[has_scr] = torch.norm(pred_scr - gt_scr, p=2, dim=-1)
                ang_errors.append(ang_err_deg.cpu().numpy())
                hp_errors.append(hp_mae.cpu().numpy())
                hy_errors.append(hy_mae.cpu().numpy())
                disc_errors.append(disc_deg.cpu().numpy())
                screen_errors.append(scr_err.cpu().numpy())
                has_screen_list.append(has_scr.cpu().numpy())
                
    ang_arr = np.concatenate(ang_errors)
    hp_arr = np.concatenate(hp_errors)
    hy_arr = np.concatenate(hy_errors)
    disc_arr = np.concatenate(disc_errors)
    screen_arr = np.concatenate(screen_errors)
    has_screen_arr = np.concatenate(has_screen_list)
    
    split_results = {}
    unique_datasets = ["ALL"] + list(df_subset["dataset_id"].unique())
    for ds in unique_datasets:
        mask = np.ones(len(df_subset), dtype=bool) if ds == "ALL" else (dataset_ids == ds)
        n_samples = int(np.sum(mask))
        if n_samples == 0:
            continue
        sub_ang = ang_arr[mask]
        sub_hp = hp_arr[mask]
        sub_hy = hy_arr[mask]
        sub_disc = disc_arr[mask]
        sub_scr = screen_arr[mask]
        sub_has_scr = has_screen_arr[mask]
        scr_count = int(np.sum(sub_has_scr))
        avg_pix = float(np.sum(sub_scr[sub_has_scr]) / scr_count) if scr_count > 0 else 0.0
        avg_mm = avg_pix * 0.2238053 if scr_count > 0 else 0.0
        res = {
            "samples": n_samples,
            "angular_error_deg": float(np.mean(sub_ang)),
            "angular_error_median_deg": float(np.median(sub_ang)),
            "head_pitch_mae_deg": float(np.mean(sub_hp)),
            "head_yaw_mae_deg": float(np.mean(sub_hy)),
            "pixel_mae": avg_pix,
            "mm_mae": avg_mm,
            "max_discrepancy_deg": float(np.max(sub_disc)),
            "avg_discrepancy_deg": float(np.mean(sub_disc))
        }
        metric_log(f"[{split_name.upper()} | {ds}] N={n_samples} | E_ang={res['angular_error_deg']:.2f}° (med={res['angular_error_median_deg']:.2f}°) | Head P={res['head_pitch_mae_deg']:.2f}°, Y={res['head_yaw_mae_deg']:.2f}° | Screen={avg_pix:.1f}px ({avg_mm:.2f}mm) | MaxDisc={res['max_discrepancy_deg']:.4f}°")
        split_results[ds] = res
    del eval_ds
    del eval_loader
    MemoryManager.aggressive_cleanup()
    return split_results

def run_hostile_audit(checkpoint_path: str, active_datasets: Optional[List[str]] = None) -> Dict[str, Any]:
    audit_start_mono = time.monotonic()
    dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_device = torch.device(dev_str)
    metric_log(f"Hostile audit initiated for checkpoint: {checkpoint_path} on {target_device}", device_context="GPU" if target_device.type == "cuda" else "CPU")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    model, payload = ModelRegistry.load_checkpoint(checkpoint_path, target_device="cpu")
    model.eval()
    arch_config = payload.get("architecture_config", {})
    epoch_saved = payload.get("epoch", 0)
    saved_metrics = payload.get("metrics", {})
    metric_log(f"Model restored from epoch {epoch_saved}. Config: {arch_config}. Recorded metrics: {saved_metrics}")
    dataset_options = {
        "columbia_gaze": DatasetSelectionConfig(
            dataset_name="columbia_gaze",
            data_file_path="DATA/columbia_gaze/processed/columbia_gaze_full.parquet",
            images_base_path="DATA/columbia_gaze",
            max_samples=6000
        ),
        "mpiifacegaze": DatasetSelectionConfig(
            dataset_name="mpiifacegaze",
            data_file_path="DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet",
            images_base_path="DATA/mpiifacegaze",
            max_samples=6000
        ),
        "rtgene": DatasetSelectionConfig(
            dataset_name="rtgene",
            data_file_path="DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet",
            images_base_path="DATA/rtgene/rtgene_dataset",
            max_samples=6000
        )
    }
    selected_ds = active_datasets or ["columbia_gaze", "mpiifacegaze", "rtgene"]
    configs = [dataset_options[d] for d in selected_ds if d in dataset_options]
    unified_df = DataUnifier.unify(configs)
    train_df, val_df, test_df = UnifiedDataModule.split_by_subjects(unified_df)
    train_subjs = set(train_df["subject_id"].unique())
    val_subjs = set(val_df["subject_id"].unique())
    test_subjs = set(test_df["subject_id"].unique())
    tr_va_overlap = train_subjs.intersection(val_subjs)
    tr_te_overlap = train_subjs.intersection(test_subjs)
    va_te_overlap = val_subjs.intersection(test_subjs)
    disjoint_pass = (len(tr_va_overlap) == 0 and len(tr_te_overlap) == 0 and len(va_te_overlap) == 0)
    metric_log(f"Subject Disjoint Verification: Tr_inter_Va={len(tr_va_overlap)}, Tr_inter_Te={len(tr_te_overlap)}, Va_inter_Te={len(va_te_overlap)}. Verdict: {'PASS' if disjoint_pass else 'FAIL'}")
    
    results_map: Dict[str, Any] = {"disjoint_pass": disjoint_pass}
    for split_tag, df_split in [("val", val_df), ("test", test_df)]:
        split_res = audit_split(df_split, split_tag, model, target_device)
        for ds_key, vals in split_res.items():
            results_map[f"{split_tag}_{ds_key}"] = vals
            
    t_audit_duration = time.monotonic() - audit_start_mono
    metric_log(f"Hostile audit completed in {t_audit_duration:.2f}s")
    return results_map

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_002_20261005_012813/checkpoints/best_model.pt"
    ds_filter = sys.argv[2:] if len(sys.argv) > 2 else None
    run_hostile_audit(ckpt, active_datasets=ds_filter)
