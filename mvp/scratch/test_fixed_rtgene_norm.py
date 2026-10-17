import numpy as np, pandas as pd

raw = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")
tx = raw["face_trans_x_m"].to_numpy(dtype=np.float32)
ty = raw["face_trans_y_m"].to_numpy(dtype=np.float32)
tz = raw["face_trans_z_m"].to_numpy(dtype=np.float32)

for mode, fc in [("CURRENT_WITH_MINUS", np.stack([tx, -ty, tz], axis=1)),
                 ("FIXED_WITH_PLUS", np.stack([tx, ty, tz], axis=1)),
                 ("RAW_NO_NORM", None)]:
    if fc is not None:
        dist = np.linalg.norm(fc, axis=1, keepdims=True)
        z_axis = fc / np.maximum(dist, 1e-6)
        y_cam = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = np.cross(y_cam, z_axis)
        x_axis = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-6)
        R_ray = np.stack([x_axis, y_axis, z_axis], axis=1)
        raw_gp_rad = raw["gaze_pitch_rad"].to_numpy(dtype=np.float32)
        raw_gy_rad = raw["gaze_yaw_rad"].to_numpy(dtype=np.float32)
        g_cam = np.stack([-np.cos(raw_gp_rad) * np.sin(raw_gy_rad), -np.sin(raw_gp_rad), -np.cos(raw_gp_rad) * np.cos(raw_gy_rad)], axis=1)
        g_norm = np.einsum('nij,nj->ni', R_ray, g_cam)
        g_norm = g_norm / np.maximum(np.linalg.norm(g_norm, axis=1, keepdims=True), 1e-6)
        pitch_rad = np.arcsin(np.clip(-g_norm[:, 1], -1.0, 1.0))
        yaw_rad = np.arctan2(-g_norm[:, 0], -g_norm[:, 2])
        p_deg = np.degrees(pitch_rad)
        y_deg = np.degrees(yaw_rad)
    else:
        p_deg = raw["gaze_pitch_deg"].to_numpy()
        y_deg = raw["gaze_yaw_deg"].to_numpy()

    raw["p_" + mode] = p_deg
    raw["y_" + mode] = y_deg

for s, g in raw.groupby("person"):
    print(f"Person {s} ({g['session'].iloc[0]}): Pitch RAW={g['p_RAW_NO_NORM'].mean():+.2f} | CURRENT={g['p_CURRENT_WITH_MINUS'].mean():+.2f} | FIXED={g['p_FIXED_WITH_PLUS'].mean():+.2f}")
