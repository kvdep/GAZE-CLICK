import numpy as np, pandas as pd

df = pd.read_parquet("DATA/rtgene/rtgene_dataset/processed/rtgene_full_labels.parquet")

tx = df["face_trans_x_m"].to_numpy(dtype=np.float32)
ty = df["face_trans_y_m"].to_numpy(dtype=np.float32)
tz = df["face_trans_z_m"].to_numpy(dtype=np.float32)

fc = np.stack([tx, -ty, tz], axis=1)
dist = np.linalg.norm(fc, axis=1, keepdims=True)
z_axis = fc / np.maximum(dist, 1e-6)

y_cam = np.array([0.0, 1.0, 0.0], dtype=np.float32)
x_axis = np.cross(y_cam, z_axis)
x_axis = x_axis / np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-6)
y_axis = np.cross(z_axis, x_axis)
y_axis = y_axis / np.maximum(np.linalg.norm(y_axis, axis=1, keepdims=True), 1e-6)

R_ray = np.stack([x_axis, y_axis, z_axis], axis=1)

hp_rad = df["head_pitch_rad"].to_numpy(dtype=np.float32)
hy_rad = df["head_yaw_rad"].to_numpy(dtype=np.float32)
h_cam = np.stack([
    -np.cos(hp_rad) * np.sin(hy_rad),
    -np.sin(hp_rad),
    -np.cos(hp_rad) * np.cos(hy_rad)
], axis=1)

h_norm = np.einsum('nij,nj->ni', R_ray, h_cam)
h_norm = h_norm / np.maximum(np.linalg.norm(h_norm, axis=1, keepdims=True), 1e-6)

norm_hp_rad = np.arcsin(np.clip(-h_norm[:, 1], -1.0, 1.0))
norm_hy_rad = np.arctan2(-h_norm[:, 0], -h_norm[:, 2])

h_recon = np.stack([
    -np.cos(norm_hp_rad) * np.sin(norm_hy_rad),
    -np.sin(norm_hp_rad),
    -np.cos(norm_hp_rad) * np.cos(norm_hy_rad)
], axis=1)

h_dot = np.sum(h_norm * h_recon, axis=1)
h_disc_deg = np.degrees(np.arccos(np.clip(h_dot, -1.0, 1.0)))
print(f"Head discrepancy: max={h_disc_deg.max():.6f}°, mean={h_disc_deg.mean():.6e}°")
