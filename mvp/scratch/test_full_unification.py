import sys, numpy as np, pandas as pd
sys.path.insert(0, "mvp")
from data.unification import DataUnifier, DatasetSelectionConfig, UnifiedDataModule

configs = [
    DatasetSelectionConfig(
        dataset_name="columbia_gaze",
        data_file_path="DATA/columbia_gaze/processed/columbia_gaze_full.parquet",
        images_base_path="DATA/columbia_gaze",
        max_samples=6000
    ),
    DatasetSelectionConfig(
        dataset_name="mpiifacegaze",
        data_file_path="DATA/mpiifacegaze/processed/mpiifacegaze_full.parquet",
        images_base_path="DATA/mpiifacegaze",
        max_samples=6000
    ),
    DatasetSelectionConfig(
        dataset_name="rtgene",
        data_file_path="DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet",
        images_base_path="DATA/rtgene/rtgene_dataset",
        max_samples=6000
    )
]

df = DataUnifier.unify(configs)
print(f"Total unified samples: {len(df)}")
for ds, grp in df.groupby("dataset_id"):
    p_rad = grp["gaze_pitch_rad"].to_numpy()
    y_rad = grp["gaze_yaw_rad"].to_numpy()
    v_calc = np.stack([-np.cos(p_rad)*np.sin(y_rad), -np.sin(p_rad), -np.cos(p_rad)*np.cos(y_rad)], axis=1)
    v_stored = grp[["gaze_vec_x", "gaze_vec_y", "gaze_vec_z"]].to_numpy()
    dot = np.sum(v_calc * v_stored, axis=1)
    disc = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    hp_rad = grp["head_pitch_rad"].to_numpy()
    hy_rad = grp["head_yaw_rad"].to_numpy()
    hv_calc = np.stack([-np.cos(hp_rad)*np.sin(hy_rad), -np.sin(hp_rad), -np.cos(hp_rad)*np.cos(hy_rad)], axis=1)
    hv_stored = grp[["head_vec_x", "head_vec_y", "head_vec_z"]].to_numpy()
    hdot = np.sum(hv_calc * hv_stored, axis=1)
    hdisc = np.degrees(np.arccos(np.clip(hdot, -1.0, 1.0)))
    print(f"Dataset {ds} (N={len(grp)}):")
    print(f"  Gaze Disc: max={disc.max():.6f}°, mean={disc.mean():.6e}°")
    print(f"  Head Disc: max={hdisc.max():.6f}°, mean={hdisc.mean():.6e}°")
    print(f"  Gaze Pitch: min={np.degrees(p_rad).min():.1f}°, max={np.degrees(p_rad).max():.1f}°, mean={np.degrees(p_rad).mean():.1f}°")
    print(f"  Gaze Yaw:   min={np.degrees(y_rad).min():.1f}°, max={np.degrees(y_rad).max():.1f}°, mean={np.degrees(y_rad).mean():.1f}°")
    print(f"  Head Pitch: min={np.degrees(hp_rad).min():.1f}°, max={np.degrees(hp_rad).max():.1f}°, mean={np.degrees(hp_rad).mean():.1f}°")
    print(f"  Head Yaw:   min={np.degrees(hy_rad).min():.1f}°, max={np.degrees(hy_rad).max():.1f}°, mean={np.degrees(hy_rad).mean():.1f}°")
