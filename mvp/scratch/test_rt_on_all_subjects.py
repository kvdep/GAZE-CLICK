import numpy as np, pandas as pd, torch

raw = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")
tx = raw["face_trans_x_m"].to_numpy(dtype=np.float32)
ty = raw["face_trans_y_m"].to_numpy(dtype=np.float32)
tz = raw["face_trans_z_m"].to_numpy(dtype=np.float32)

fc = np.stack([tx, -ty, tz], axis=1)
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

g_R = np.einsum('nij,nj->ni', R_ray, g_cam)
g_RT = np.einsum('nji,nj->ni', R_ray, g_cam)

p_R = np.degrees(np.arcsin(np.clip(-g_R[:, 1], -1.0, 1.0)))
p_RT = np.degrees(np.arcsin(np.clip(-g_RT[:, 1], -1.0, 1.0)))
raw_p = raw["gaze_pitch_deg"].to_numpy()

raw["p_R"] = p_R
raw["p_RT"] = p_RT

for s, g in raw.groupby("person"):
    print(f"Subj {s}: RAW={g['gaze_pitch_deg'].mean():+.2f} | R={g['p_R'].mean():+.2f} | R.T={g['p_RT'].mean():+.2f}")
