import os, sys, math, time, cv2, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath("mvp"))
from PIL import Image, ImageDraw, ImageFont
from models.registry import ModelRegistry
from data.unification import DataUnifier, UnifiedDataModule, DatasetSelectionConfig, UnifiedGazeDataset
from utils.memory import MemoryManager, scoped_model_device

def draw_gaze_arrow(img_bgr, center, pitch_deg, yaw_deg, color, length=60, thickness=2):
    p_rad = math.radians(pitch_deg)
    y_rad = math.radians(yaw_deg)
    dx = -length * math.sin(y_rad)
    dy = -length * math.sin(p_rad)
    end_pt = (int(center[0] + dx), int(center[1] + dy))
    cv2.arrowedLine(img_bgr, (int(center[0]), int(center[1])), end_pt, color, thickness, tipLength=0.25)
    return end_pt

def run():
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "mvp/experiments/l2cs_gaze_net_mobilenet_v3_large/run_008_20261006_001852/checkpoints/best_model.pt"
    print(f"Loading checkpoint: {ckpt_path} on {target_device}")
    model, payload = ModelRegistry.load_checkpoint(ckpt_path, target_device="cpu")
    model = model.to(target_device)
    model.eval()

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
    configs = [dataset_options["columbia_gaze"], dataset_options["mpiifacegaze"], dataset_options["rtgene"]]
    unified_df = DataUnifier.unify(configs)
    train_df, val_df, test_df = UnifiedDataModule.split_by_subjects(unified_df)
    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    eval_ds = UnifiedGazeDataset(test_df, image_resolution=(224, 224), is_train=False, preload_cache=False)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0)

    ang_errors = []
    pitch_diffs = []
    yaw_diffs = []
    hp_errors = []
    hy_errors = []
    screen_errors = []
    has_screen_list = []
    gt_gazes = []
    pred_gazes = []

    with torch.no_grad():
        for batch in eval_loader:
            imgs = batch["image"].to(target_device)
            outputs = model(imgs)
            gt_gaze_v = batch["gaze_vector"].to(target_device)
            pred_gaze_v = outputs["gaze_vector"]
            pred_gaze_norm = pred_gaze_v / torch.clamp(torch.norm(pred_gaze_v, p=2, dim=-1, keepdim=True), min=1e-8)
            gt_gaze_norm = gt_gaze_v / torch.clamp(torch.norm(gt_gaze_v, p=2, dim=-1, keepdim=True), min=1e-8)
            dot = torch.sum(pred_gaze_norm * gt_gaze_norm, dim=-1)
            ang_err_deg = torch.acos(torch.clamp(dot, -1.0, 1.0)) * (180.0 / math.pi)

            pred_p = outputs["pitch_rad"] * (180.0 / math.pi)
            pred_y = outputs["yaw_rad"] * (180.0 / math.pi)
            gt_p = batch["gaze_angles"][:, 0].to(target_device) * (180.0 / math.pi)
            gt_y = batch["gaze_angles"][:, 1].to(target_device) * (180.0 / math.pi)

            pitch_diffs.append((pred_p - gt_p).cpu().numpy())
            yaw_diffs.append((pred_y - gt_y).cpu().numpy())
            gt_gazes.append(torch.stack([gt_p, gt_y], dim=-1).cpu().numpy())
            pred_gazes.append(torch.stack([pred_p, pred_y], dim=-1).cpu().numpy())

            gt_head_ang = batch["head_angles"].to(target_device)
            pred_hp = outputs["head_pitch_rad"] * (180.0 / math.pi)
            pred_hy = outputs["head_yaw_rad"] * (180.0 / math.pi)
            gt_hp = gt_head_ang[:, 0] * (180.0 / math.pi)
            gt_hy = gt_head_ang[:, 1] * (180.0 / math.pi)
            hp_mae = torch.abs(pred_hp - gt_hp)
            hy_mae = torch.abs(pred_hy - gt_hy)

            has_scr = batch["has_screen_coords"].to(target_device)
            scr_err = torch.zeros_like(ang_err_deg)
            if "screen_coordinates" in outputs and has_scr.any():
                pred_scr = outputs["screen_coordinates"][has_scr]
                gt_scr = batch["screen_coords"].to(target_device)[has_scr]
                scr_err[has_scr] = torch.norm(pred_scr - gt_scr, p=2, dim=-1)

            ang_errors.append(ang_err_deg.cpu().numpy())
            hp_errors.append(hp_mae.cpu().numpy())
            hy_errors.append(hy_mae.cpu().numpy())
            screen_errors.append(scr_err.cpu().numpy())
            has_screen_list.append(has_scr.cpu().numpy())

    ang_arr = np.concatenate(ang_errors)
    p_diff_arr = np.concatenate(pitch_diffs)
    y_diff_arr = np.concatenate(yaw_diffs)
    hp_arr = np.concatenate(hp_errors)
    hy_arr = np.concatenate(hy_errors)
    screen_arr = np.concatenate(screen_errors)
    has_screen_arr = np.concatenate(has_screen_list)
    dataset_ids = test_df["dataset_id"].to_numpy()

    print("\n================== TEST SET AUDIT RESULTS ==================")
    for ds in ["ALL", "columbia_gaze", "mpiifacegaze", "rtgene"]:
        mask = np.ones(len(test_df), dtype=bool) if ds == "ALL" else (dataset_ids == ds)
        n = int(np.sum(mask))
        if n == 0:
            continue
        sub_ang = ang_arr[mask]
        sub_p_diff = p_diff_arr[mask]
        sub_y_diff = y_diff_arr[mask]
        sub_hp = hp_arr[mask]
        sub_hy = hy_arr[mask]
        sub_scr = screen_arr[mask]
        sub_has_scr = has_screen_arr[mask]
        scr_cnt = int(np.sum(sub_has_scr))
        avg_scr_px = float(np.sum(sub_scr[sub_has_scr]) / scr_cnt) if scr_cnt > 0 else 0.0
        avg_scr_mm = avg_scr_px * 0.2238053 if scr_cnt > 0 else 0.0

        print(f"Dataset: {ds:15s} | N={n:5d}")
        print(f"  Angular Error: Mean = {np.mean(sub_ang):.2f}°, Median = {np.median(sub_ang):.2f}°, std = {np.std(sub_ang):.2f}°")
        print(f"  Pitch Bias = {np.mean(sub_p_diff):+.2f}°, Pitch MAE = {np.mean(np.abs(sub_p_diff)):.2f}°")
        print(f"  Yaw Bias   = {np.mean(sub_y_diff):+.2f}°, Yaw MAE   = {np.mean(np.abs(sub_y_diff)):.2f}°")
        print(f"  Head Pose: Pitch MAE = {np.mean(sub_hp):.2f}°, Yaw MAE = {np.mean(sub_hy):.2f}°")
        if scr_cnt > 0:
            print(f"  Screen Error: {avg_scr_px:.1f} px ({avg_scr_mm:.2f} mm)")

    vis_dir = "mvp/scratch/eval_visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    samples_to_visualize = []
    for ds in ["columbia_gaze", "mpiifacegaze", "rtgene"]:
        sub_indices = np.where(dataset_ids == ds)[0]
        if len(sub_indices) == 0:
            continue
        step = max(1, len(sub_indices) // 4)
        for idx in sub_indices[::step][:4]:
            samples_to_visualize.append(idx)

    print(f"\nGenerating {len(samples_to_visualize)} visualization samples into {vis_dir}...")
    for idx in samples_to_visualize:
        item = eval_ds[idx]
        row = test_df.iloc[idx]
        ds_name = row["dataset_id"]
        subj = row["subject_id"]

        img_tensor = item["image"].unsqueeze(0).to(target_device)
        with torch.no_grad():
            out = model(img_tensor)

        p_gt = float(item["gaze_angles"][0].numpy() * (180.0 / math.pi))
        y_gt = float(item["gaze_angles"][1].numpy() * (180.0 / math.pi))
        p_pred = float(out["pitch_rad"].item() * (180.0 / math.pi))
        y_pred = float(out["yaw_rad"].item() * (180.0 / math.pi))

        hp_gt = float(item["head_angles"][0].numpy() * (180.0 / math.pi))
        hy_gt = float(item["head_angles"][1].numpy() * (180.0 / math.pi))
        hp_pred = float(out["head_pitch_rad"].item() * (180.0 / math.pi))
        hy_pred = float(out["head_yaw_rad"].item() * (180.0 / math.pi))

        ang_err = float(ang_arr[idx])

        im_raw = item["image"].cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        im_denorm = (im_raw * std + mean).clip(0, 1)
        im_bgr = (im_denorm.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2BGR)

        im_large = cv2.resize(im_bgr, (448, 448), interpolation=cv2.INTER_NEAREST)
        center = (224, 224)

        draw_gaze_arrow(im_large, center, p_gt, y_gt, (0, 255, 0), length=100, thickness=3)
        draw_gaze_arrow(im_large, center, p_pred, y_pred, (255, 255, 0), length=100, thickness=3)
        draw_gaze_arrow(im_large, (224, 300), hp_gt, hy_gt, (0, 165, 255), length=70, thickness=2)
        draw_gaze_arrow(im_large, (224, 300), hp_pred, hy_pred, (255, 0, 255), length=70, thickness=2)

        info_panel = np.zeros((200, 448, 3), dtype=np.uint8)
        cv2.putText(info_panel, f"Dataset: {ds_name} | Subj: {subj} | Error: {ang_err:.2f} deg", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(info_panel, f"GT Gaze (GREEN):   P={p_gt:+.1f} deg, Y={y_gt:+.1f} deg", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(info_panel, f"Pred Gaze (CYAN):  P={p_pred:+.1f} deg, Y={y_pred:+.1f} deg", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(info_panel, f"GT Head (ORANGE):  P={hp_gt:+.1f} deg, Y={hy_gt:+.1f} deg", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(info_panel, f"Pred Head (MAG):   P={hp_pred:+.1f} deg, Y={hy_pred:+.1f} deg", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        if bool(row["has_screen_coords"]):
            gt_s_x = float(row["screen_x_px"]); gt_s_y = float(row["screen_y_px"])
            pred_s = out["screen_coordinates"][0].cpu().numpy()
            s_err_px = np.linalg.norm(pred_s - np.array([gt_s_x, gt_s_y]))
            cv2.putText(info_panel, f"Screen: GT=({gt_s_x:.0f},{gt_s_y:.0f}) Pred=({pred_s[0]:.0f},{pred_s[1]:.0f}) Err={s_err_px:.1f}px", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

        combined = np.vstack([im_large, info_panel])
        out_filename = os.path.join(vis_dir, f"test_sample_{idx:04d}_{ds_name}_{subj}.png")
        cv2.imwrite(out_filename, combined)

    print("Completed test audit and visual exports.")

if __name__ == "__main__":
    run()
