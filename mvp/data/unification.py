import os, sys, math, torch, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as TVT

class DatasetSelectionConfig:
    def __init__(self, dataset_name: str, data_file_path: str, images_base_path: str, max_samples: Optional[int] = None) -> None:
        self.dataset_name = dataset_name
        self.data_file_path = data_file_path
        self.images_base_path = images_base_path
        self.max_samples = max_samples

class DataUnifier:
    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.exists(path):
            return path
        cand1 = os.path.normpath(os.path.join("..", path))
        if os.path.exists(cand1):
            return cand1
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cand2 = os.path.normpath(os.path.join(repo_root, path))
        if os.path.exists(cand2):
            return cand2
        return path

    @staticmethod
    def _compute_canonical_vectors(pitch_rad: np.ndarray, yaw_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cos_p = np.cos(pitch_rad); sin_p = np.sin(pitch_rad)
        cos_y = np.cos(yaw_rad); sin_y = np.sin(yaw_rad)
        vx = -cos_p * sin_y
        vy = -sin_p
        vz = -cos_p * cos_y
        norm = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-8)
        return vx / norm, vy / norm, vz / norm

    @staticmethod
    def load_columbia(cfg: DatasetSelectionConfig) -> pd.DataFrame:
        df_path = DataUnifier._resolve_path(cfg.data_file_path)
        base_path = DataUnifier._resolve_path(cfg.images_base_path)
        raw_df = pd.read_parquet(df_path)
        crop_base_candidate = os.path.join(base_path, "crops_224")
        if os.path.exists(crop_base_candidate):
            img_paths = [os.path.normpath(os.path.join(base_path, str(rel).replace("Columbia Gaze Data Set", "crops_224"))) for rel in raw_df["rel_path"]]
        else:
            img_paths = [os.path.normpath(os.path.join(base_path, str(rel))) for rel in raw_df["rel_path"]]
        exists_mask = np.array([os.path.exists(p) for p in img_paths], dtype=bool)
        filtered_df = raw_df[exists_mask].copy().reset_index(drop=True)
        valid_paths = [p for p, ex in zip(img_paths, exists_mask) if ex]
        if cfg.max_samples is not None and cfg.max_samples > 0:
            filtered_df = filtered_df.iloc[:cfg.max_samples].copy().reset_index(drop=True)
            valid_paths = valid_paths[:cfg.max_samples]
        p_rad = filtered_df["gaze_pitch_rad"].to_numpy(dtype=np.float32)
        y_rad = filtered_df["gaze_yaw_rad"].to_numpy(dtype=np.float32)
        hp_rad = filtered_df["head_pitch_rad"].to_numpy(dtype=np.float32)
        hy_rad = filtered_df["head_yaw_rad"].to_numpy(dtype=np.float32)
        gvx, gvy, gvz = DataUnifier._compute_canonical_vectors(p_rad, y_rad)
        hvx, hvy, hvz = DataUnifier._compute_canonical_vectors(hp_rad, hy_rad)
        n = len(filtered_df)
        res = pd.DataFrame({
            "dataset_id": ["columbia_gaze"] * n,
            "subject_id": [f"columbia_{sid}" for sid in filtered_df["subject_id"]],
            "sample_id": [f"columbia_{fn}" for fn in filtered_df["filename"]],
            "image_path": valid_paths,
            "image_exists": [True] * n,
            "gaze_pitch_rad": p_rad,
            "gaze_yaw_rad": y_rad,
            "gaze_vec_x": gvx,
            "gaze_vec_y": gvy,
            "gaze_vec_z": gvz,
            "head_pitch_rad": hp_rad,
            "head_yaw_rad": hy_rad,
            "head_vec_x": hvx,
            "head_vec_y": hvy,
            "head_vec_z": hvz,
            "screen_x_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "screen_y_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "has_screen_coords": [False] * n
        })
        return res

    @staticmethod
    def load_mpiifacegaze(cfg: DatasetSelectionConfig) -> pd.DataFrame:
        df_path = DataUnifier._resolve_path(cfg.data_file_path)
        base_path = DataUnifier._resolve_path(cfg.images_base_path)
        raw_df = pd.read_parquet(df_path)
        img_paths = [os.path.normpath(os.path.join(base_path, str(s), str(ip))) for s, ip in zip(raw_df["subject"], raw_df["image_path"])]
        exists_mask = np.array([os.path.exists(p) for p in img_paths], dtype=bool)
        filtered_df = raw_df[exists_mask].copy().reset_index(drop=True)
        valid_paths = [p for p, ex in zip(img_paths, exists_mask) if ex]
        if cfg.max_samples is not None and cfg.max_samples > 0:
            filtered_df = filtered_df.iloc[:cfg.max_samples].copy().reset_index(drop=True)
            valid_paths = valid_paths[:cfg.max_samples]
        p_rad = filtered_df["gaze_pitch_rad"].to_numpy(dtype=np.float32)
        y_rad = filtered_df["gaze_yaw_rad"].to_numpy(dtype=np.float32)
        hp_rad = filtered_df["head_pitch_rad"].to_numpy(dtype=np.float32)
        hy_rad = filtered_df["head_yaw_rad"].to_numpy(dtype=np.float32)
        gvx, gvy, gvz = DataUnifier._compute_canonical_vectors(p_rad, y_rad)
        hvx, hvy, hvz = DataUnifier._compute_canonical_vectors(hp_rad, hy_rad)
        scr_x = filtered_df["screen_x_px"].to_numpy(dtype=np.float32)
        scr_y = filtered_df["screen_y_px"].to_numpy(dtype=np.float32)
        has_scr = (~np.isnan(scr_x)) & (~np.isnan(scr_y))
        n = len(filtered_df)
        res = pd.DataFrame({
            "dataset_id": ["mpiifacegaze"] * n,
            "subject_id": [f"mpii_{s}" for s in filtered_df["subject"]],
            "sample_id": [f"mpii_{s}_{str(ip).replace('/', '_')}" for s, ip in zip(filtered_df["subject"], filtered_df["image_path"])],
            "image_path": valid_paths,
            "image_exists": [True] * n,
            "gaze_pitch_rad": p_rad,
            "gaze_yaw_rad": y_rad,
            "gaze_vec_x": gvx,
            "gaze_vec_y": gvy,
            "gaze_vec_z": gvz,
            "head_pitch_rad": hp_rad,
            "head_yaw_rad": hy_rad,
            "head_vec_x": hvx,
            "head_vec_y": hvy,
            "head_vec_z": hvz,
            "screen_x_px": scr_x,
            "screen_y_px": scr_y,
            "has_screen_coords": has_scr.tolist()
        })
        return res

    @staticmethod
    def load_rtgene(cfg: DatasetSelectionConfig) -> pd.DataFrame:
        df_path = DataUnifier._resolve_path(cfg.data_file_path)
        base_path = DataUnifier._resolve_path(cfg.images_base_path)
        raw_df = pd.read_parquet(df_path)
        img_paths = [os.path.normpath(os.path.join(base_path, "extracted_images", f"{sess}_glasses", "inpainted", "face", f"face_{int(seq):06d}_rgb.png")) for sess, seq in zip(raw_df["session"], raw_df["seq"])]
        exists_mask = np.array([os.path.exists(p) for p in img_paths], dtype=bool)
        filtered_df = raw_df[exists_mask].copy().reset_index(drop=True)
        valid_paths = [p for p, ex in zip(img_paths, exists_mask) if ex]
        if cfg.max_samples is not None and cfg.max_samples > 0:
            filtered_df = filtered_df.iloc[:cfg.max_samples].copy().reset_index(drop=True)
            valid_paths = valid_paths[:cfg.max_samples]
        p_rad = filtered_df["gaze_pitch_rad"].to_numpy(dtype=np.float32)
        y_rad = filtered_df["gaze_yaw_rad"].to_numpy(dtype=np.float32)
        hp_rad = filtered_df["head_pitch_rad"].to_numpy(dtype=np.float32)
        hy_rad = filtered_df["head_yaw_rad"].to_numpy(dtype=np.float32)
        gvx, gvy, gvz = DataUnifier._compute_canonical_vectors(p_rad, y_rad)
        hvx, hvy, hvz = DataUnifier._compute_canonical_vectors(hp_rad, hy_rad)
        n = len(filtered_df)
        res = pd.DataFrame({
            "dataset_id": ["rtgene"] * n,
            "subject_id": [f"rtgene_{p}" for p in filtered_df["person"]],
            "sample_id": [f"{s}_{seq}" for s, seq in zip(filtered_df["session"], filtered_df["seq"])],
            "image_path": valid_paths,
            "image_exists": [True] * n,
            "gaze_pitch_rad": p_rad,
            "gaze_yaw_rad": y_rad,
            "gaze_vec_x": gvx,
            "gaze_vec_y": gvy,
            "gaze_vec_z": gvz,
            "head_pitch_rad": hp_rad,
            "head_yaw_rad": hy_rad,
            "head_vec_x": hvx,
            "head_vec_y": hvy,
            "head_vec_z": hvz,
            "screen_x_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "screen_y_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "has_screen_coords": [False] * n
        })
        return res

    @staticmethod
    def load_gaze360(cfg: DatasetSelectionConfig) -> pd.DataFrame:
        df_path = DataUnifier._resolve_path(cfg.data_file_path)
        base_path = DataUnifier._resolve_path(cfg.images_base_path)
        raw_df = pd.read_parquet(df_path)
        img_paths = [os.path.normpath(os.path.join(base_path, str(rp))) for rp in raw_df["rel_path"]]
        exists_mask = np.array([os.path.exists(p) for p in img_paths], dtype=bool)
        filtered_df = raw_df[exists_mask].copy().reset_index(drop=True)
        valid_paths = [p for p, ex in zip(img_paths, exists_mask) if ex]
        yaw_limit_rad = 60.0 * (np.pi / 180.0)
        frontal_mask = np.abs(filtered_df["gaze_yaw_rad"].to_numpy()) <= yaw_limit_rad
        filtered_df = filtered_df[frontal_mask].copy().reset_index(drop=True)
        valid_paths = [p for p, f in zip(valid_paths, frontal_mask) if f]
        if cfg.max_samples is not None and cfg.max_samples > 0:
            filtered_df = filtered_df.iloc[:cfg.max_samples].copy().reset_index(drop=True)
            valid_paths = valid_paths[:cfg.max_samples]
        p_rad = filtered_df["gaze_pitch_rad"].to_numpy(dtype=np.float32)
        y_rad = filtered_df["gaze_yaw_rad"].to_numpy(dtype=np.float32)
        hp_rad = filtered_df["head_pitch_rad"].to_numpy(dtype=np.float32)
        hy_rad = filtered_df["head_yaw_rad"].to_numpy(dtype=np.float32)
        gvx = filtered_df["gaze_vec_x"].to_numpy(dtype=np.float32)
        gvy = filtered_df["gaze_vec_y"].to_numpy(dtype=np.float32)
        gvz = filtered_df["gaze_vec_z"].to_numpy(dtype=np.float32)
        hvx = filtered_df["head_vec_x"].to_numpy(dtype=np.float32)
        hvy = filtered_df["head_vec_y"].to_numpy(dtype=np.float32)
        hvz = filtered_df["head_vec_z"].to_numpy(dtype=np.float32)
        n = len(filtered_df)
        res = pd.DataFrame({
            "dataset_id": ["gaze360"] * n,
            "subject_id": filtered_df["subject_id"].tolist(),
            "sample_id": filtered_df["sample_id"].tolist(),
            "image_path": valid_paths,
            "image_exists": [True] * n,
            "gaze_pitch_rad": p_rad,
            "gaze_yaw_rad": y_rad,
            "gaze_vec_x": gvx,
            "gaze_vec_y": gvy,
            "gaze_vec_z": gvz,
            "head_pitch_rad": hp_rad,
            "head_yaw_rad": hy_rad,
            "head_vec_x": hvx,
            "head_vec_y": hvy,
            "head_vec_z": hvz,
            "screen_x_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "screen_y_px": pd.Series(np.nan, index=range(n), dtype=np.float32),
            "has_screen_coords": [False] * n
        })
        return res

    @staticmethod
    def unify(configs: List[DatasetSelectionConfig], output_parquet_path: Optional[str] = None) -> pd.DataFrame:
        dataframes: List[pd.DataFrame] = []
        for cfg in configs:
            name = cfg.dataset_name.lower()
            if name == "columbia_gaze":
                dataframes.append(DataUnifier.load_columbia(cfg))
            elif name == "mpiifacegaze":
                dataframes.append(DataUnifier.load_mpiifacegaze(cfg))
            elif name == "rtgene":
                dataframes.append(DataUnifier.load_rtgene(cfg))
            elif name == "gaze360":
                dataframes.append(DataUnifier.load_gaze360(cfg))
            else:
                raise ValueError(f"Unsupported dataset configuration: {cfg.dataset_name}")
        unified = pd.concat(dataframes, ignore_index=True)
        if output_parquet_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
            unified.to_parquet(output_parquet_path, index=False)
        return unified

class UnifiedGazeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_resolution: Tuple[int, int] = (224, 224), is_train: bool = True, preload_cache: bool = False) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_resolution = image_resolution
        self.is_train = is_train
        self.preload_cache = preload_cache and len(self.df) <= 5000
        self.cached_images: Dict[int, torch.Tensor] = {}
        self.transform = TVT.Compose([
            TVT.Resize(self.image_resolution),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths = self.df["image_path"].tolist()
        gaze_cols = ["gaze_vec_x", "gaze_vec_y", "gaze_vec_z"]
        head_cols = ["head_vec_x", "head_vec_y", "head_vec_z"]
        self.gaze_vectors = torch.tensor(self.df[gaze_cols].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
        self.head_vectors = torch.tensor(self.df[head_cols].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
        self.gaze_angles = torch.tensor(self.df[["gaze_pitch_rad", "gaze_yaw_rad"]].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
        self.head_angles = torch.tensor(self.df[["head_pitch_rad", "head_yaw_rad"]].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)
        self.has_screen = torch.tensor(self.df["has_screen_coords"].to_numpy(dtype=bool, copy=True), dtype=torch.bool)
        scr_np = np.nan_to_num(self.df[["screen_x_px", "screen_y_px"]].to_numpy(dtype=np.float32, copy=True), nan=0.0)
        self.screen_coords = torch.tensor(scr_np, dtype=torch.float32)
        self.dataset_ids = self.df["dataset_id"].tolist()
        self.subject_ids = self.df["subject_id"].tolist()
        if self.preload_cache:
            for i in range(len(self.df)):
                self.cached_images[i] = self._load_tensor(self.image_paths[i])

    def _load_tensor(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file does not exist on disk: {path}")
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.preload_cache and index in self.cached_images:
            image_tensor = self.cached_images[index]
        else:
            image_tensor = self._load_tensor(self.image_paths[index])
        return {
            "image": image_tensor,
            "gaze_vector": self.gaze_vectors[index],
            "head_vector": self.head_vectors[index],
            "gaze_angles": self.gaze_angles[index],
            "head_angles": self.head_angles[index],
            "has_screen_coords": self.has_screen[index],
            "screen_coords": self.screen_coords[index],
            "dataset_id": self.dataset_ids[index],
            "subject_id": self.subject_ids[index]
        }

class UnifiedDataModule:
    @staticmethod
    def split_by_subjects(df: pd.DataFrame, train_ratio: float = 0.75, val_ratio: float = 0.125, test_ratio: float = 0.125) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_dfs: List[pd.DataFrame] = []
        val_dfs: List[pd.DataFrame] = []
        test_dfs: List[pd.DataFrame] = []
        for ds_name, sub_df in df.groupby("dataset_id"):
            unique_subjects = sorted(sub_df["subject_id"].unique())
            n_subjs = len(unique_subjects)
            if ds_name == "mpiifacegaze" and n_subjs == 15:
                tr_subjs = set(unique_subjects[:11])
                va_subjs = set(unique_subjects[11:13])
                te_subjs = set(unique_subjects[13:])
            elif ds_name == "columbia_gaze" and n_subjs == 56:
                tr_subjs = set(unique_subjects[:42])
                va_subjs = set(unique_subjects[42:49])
                te_subjs = set(unique_subjects[49:])
            elif ds_name == "rtgene" and n_subjs == 15:
                tr_subjs = set(unique_subjects[:11])
                va_subjs = set(unique_subjects[11:13])
                te_subjs = set(unique_subjects[13:])
            else:
                n_train = max(1, int(round(n_subjs * train_ratio)))
                n_val = max(1, int(round(n_subjs * val_ratio)))
                tr_subjs = set(unique_subjects[:n_train])
                va_subjs = set(unique_subjects[n_train:n_train + n_val])
                te_subjs = set(unique_subjects[n_train + n_val:])
            train_dfs.append(sub_df[sub_df["subject_id"].isin(tr_subjs)])
            val_dfs.append(sub_df[sub_df["subject_id"].isin(va_subjs)])
            test_dfs.append(sub_df[sub_df["subject_id"].isin(te_subjs)])
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        tr_set = set(train_df["subject_id"].unique())
        va_set = set(val_df["subject_id"].unique())
        te_set = set(test_df["subject_id"].unique())
        assert len(tr_set.intersection(va_set)) == 0, f"Subject leakage train-val: {tr_set.intersection(va_set)}"
        assert len(tr_set.intersection(te_set)) == 0, f"Subject leakage train-test: {tr_set.intersection(te_set)}"
        assert len(va_set.intersection(te_set)) == 0, f"Subject leakage val-test: {va_set.intersection(te_set)}"
        return train_df, val_df, test_df

    @staticmethod
    def build_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = UnifiedGazeDataset(train_df, is_train=True, preload_cache=False)
        val_ds = UnifiedGazeDataset(val_df, is_train=False, preload_cache=False)
        test_ds = UnifiedGazeDataset(test_df, is_train=False, preload_cache=False)
        counts = train_df["dataset_id"].value_counts().to_dict()
        n_classes = len(counts)
        sample_weights = np.zeros(len(train_df), dtype=np.float32)
        ds_series = train_df["dataset_id"].to_numpy()
        for ds_id, cnt in counts.items():
            mask = (ds_series == ds_id)
            sample_weights[mask] = 1.0 / float(n_classes * cnt)
        sampler = WeightedRandomSampler(weights=torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        pin = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
        return train_loader, val_loader, test_loader
