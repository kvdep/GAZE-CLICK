import os, sys, time, scipy.io as sio, numpy as np, pandas as pd

def preprocess_gaze360(data_dir: str = r"D:\ПРОФ\ВУЗ\ВКР\DATA\Gaze360", output_parquet: str = r"D:\ПРОФ\ВУЗ\ВКР\DATA\Gaze360\processed\gaze360_full.parquet") -> pd.DataFrame:
    mat_path = os.path.join(data_dir, "metadata.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing metadata.mat at {mat_path}")
    
    t0 = time.time()
    mat = sio.loadmat(mat_path)
    recs = [str(r[0]) for r in mat["recordings"][0]]
    total_samples = mat["frame"].shape[1]
    
    rec_indices = mat["recording"][0]
    frames = mat["frame"][0]
    pids = mat["person_identity"][0]
    splits = mat["split"][0]
    gaze_dir = mat["gaze_dir"]
    
    gx = gaze_dir[:, 0].astype(np.float32)
    gy = gaze_dir[:, 1].astype(np.float32)
    gz = gaze_dir[:, 2].astype(np.float32)
    
    pitch_rad = -np.arcsin(np.clip(gy, -1.0, 1.0)).astype(np.float32)
    yaw_rad = np.arctan2(-gx, -gz).astype(np.float32)
    
    rel_paths = []
    sample_ids = []
    subject_ids = []
    
    for i in range(total_samples):
        rec_name = recs[rec_indices[i]]
        pid = int(pids[i])
        frame = int(frames[i])
        rel_paths.append(os.path.normpath(f"imgs/{rec_name}/head/{pid:06d}/{frame:06d}.jpg"))
        sample_ids.append(f"gaze360_{rec_name}_{pid:04d}_{frame:06d}")
        subject_ids.append(f"gaze360_{pid:04d}")
        
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": subject_ids,
        "recording": [recs[idx] for idx in rec_indices],
        "frame": frames,
        "person_identity": pids,
        "split": splits,
        "rel_path": rel_paths,
        "gaze_vec_x": gx,
        "gaze_vec_y": gy,
        "gaze_vec_z": gz,
        "gaze_pitch_rad": pitch_rad,
        "gaze_yaw_rad": yaw_rad,
        "head_pitch_rad": np.zeros(total_samples, dtype=np.float32),
        "head_yaw_rad": np.zeros(total_samples, dtype=np.float32),
        "head_vec_x": np.zeros(total_samples, dtype=np.float32),
        "head_vec_y": np.zeros(total_samples, dtype=np.float32),
        "head_vec_z": -np.ones(total_samples, dtype=np.float32),
        "screen_x_px": pd.Series(np.nan, index=range(total_samples), dtype=np.float32),
        "screen_y_px": pd.Series(np.nan, index=range(total_samples), dtype=np.float32),
        "has_screen_coords": [False] * total_samples
    })
    
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    dt = time.time() - t0
    print(f"Preprocessed {len(df):,} Gaze360 label records in {dt:.2f}s -> {output_parquet}")
    return df

if __name__ == "__main__":
    preprocess_gaze360()
