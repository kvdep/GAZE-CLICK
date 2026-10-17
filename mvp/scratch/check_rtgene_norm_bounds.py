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

gp_rad = df["gaze_pitch_rad"].to_numpy(dtype=np.float32)
gy_rad = df["gaze_yaw_rad"].to_numpy(dtype=np.float32)
g_cam = np.stack([
    -np.cos(gp_rad) * np.sin(gy_rad),
    -np.sin(gp_rad),
    -np.cos(gp_rad) * np.cos(gy_rad)
], axis=1)

hp_rad = df["head_pitch_rad"].to_numpy(dtype=np.float32)
hy_rad = df["head_yaw_rad"].to_numpy(dtype=np.float32)
h_cam = np.stack([
    -np.cos(hp_rad) * np.sin(hy_rad),
    -np.sin(hp_rad),
    -np.cos(hp_rad) * np.cos(hy_rad)
], axis=1)

g_norm = np.einsum('nij,nj->ni', R_ray, g_cam)
g_norm = g_norm / np.maximum(np.linalg.norm(g_norm, axis=1, keepdims=True), 1e-6)

h_norm = np.einsum('nij,nj->ni', R_ray, h_cam)
h_norm = h_norm / np.maximum(np.linalg.norm(h_norm, axis=1, keepdims=True), 1e-6)

gp_deg = np.degrees(np.arcsin(np.clip(-g_norm[:, 1], -1.0, 1.0)))
gy_deg = np.degrees(np.arctan2(-g_norm[:, 0], -g_norm[:, 2]))
hp_deg = np.degrees(np.arcsin(np.clip(-h_norm[:, 1], -1.0, 1.0)))
hy_deg = np.degrees(np.arctan2(-h_norm[:, 0], -h_norm[:, 2]))

print(f"Normalized Gaze Pitch: min={gp_deg.min():.2f}°, max={gp_deg.max():.2f}°, mean={gp_deg.mean():.2f}°")
print(f"Normalized Gaze Yaw:   min={gy_deg.min():.2f}°, max={gy_deg.max():.2f}°, mean={gy_deg.mean():.2f}°")
print(f"Normalized Head Pitch: min={hp_deg.min():.2f}°, max={hp_deg.max():.2f}°, mean={hp_deg.mean():.2f}°")
print(f"Normalized Head Yaw:   min={hy_deg.min():.2f}°, max={hy_deg.max():.2f}°, mean={hy_deg.mean():.2f}°")

valid = (gp_deg >= -50.0) & (gp_deg <= 50.0) & (gy_deg >= -60.0) & (gy_deg <= 60.0)
print(f"Samples within [-50, 50] x [-60, 60]: {valid.sum()} / {len(valid)} ({valid.mean()*100:.2f}%)")
