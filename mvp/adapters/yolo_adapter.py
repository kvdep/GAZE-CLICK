import math, time, os, glob
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import cv2

from adapters.base import BaseGazeAdapter

class YOLOAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for YOLO26n GazeNet (Custom Trained Model).
    - Architecture: YOLO26n backbone + Multi-scale FPN + Spatial Soft-Argmax gaze head.
    - Input: 224x224 RGB, ImageNet normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    - Output: Pitch and Yaw angles, 3D gaze vector, head pose vectors, attention variance.
    """
    def __init__(self, name: str = "yolo26n", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        from models.registry import ModelRegistry
        import models.yolo26n_gaze_net

        default_yolo = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt"
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        elif os.path.exists(default_yolo):
            weight_file = default_yolo
        else:
            cands = glob.glob("mvp/experiments/**/best_model.pt", recursive=True)
            if cands:
                weight_file = cands[0]
            else:
                raise FileNotFoundError(f"Local YOLO checkpoint not found at: {default_yolo}")

        model, _ = ModelRegistry.load_checkpoint(weight_file, target_device=str(self.device))
        model.eval()
        self.model = model
        print(f"[YOLOAdapter] Successfully loaded YOLO26n from '{weight_file}' onto {self.device}.")

    def predict_gaze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_5pts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        fx, fy, fw, fh = face_bbox
        cx = int(fx + fw / 2.0); cy = int(fy + fh / 2.0)
        box_side = int(max(fw, fh) * 1.10)
        half_side = box_side // 2
        cx1 = max(0, cx - half_side); cy1 = max(0, cy - half_side)
        cx2 = min(w, cx + half_side); cy2 = min(h, cy + half_side)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return {
                "pitch_deg": 0.0, "yaw_deg": 0.0,
                "pitch_rad": 0.0, "yaw_rad": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0]),
                "head_vector": np.array([0.0, 0.0, -1.0]),
                "head_pitch_deg": 0.0, "head_yaw_deg": 0.0,
                "latency_model_ms": 0.0,
                "attention_variance": 0.040
            }

        crop_res = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_res, cv2.COLOR_BGR2RGB)
        tensor_in = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0 - self.mean) / self.std

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pitch_rad = float(outputs["pitch_rad"].item())
            yaw_rad = float(outputs["yaw_rad"].item())
            pitch_deg = float(outputs["gaze_pitch_deg"].item() if "gaze_pitch_deg" in outputs else math.degrees(pitch_rad))
            yaw_deg = float(outputs["gaze_yaw_deg"].item() if "gaze_yaw_deg" in outputs else math.degrees(yaw_rad))

            hp_deg = float(outputs.get("head_pitch_deg", torch.zeros(1)).item())
            hy_deg = float(outputs.get("head_yaw_deg", torch.zeros(1)).item())
            gaze_v = outputs["gaze_vector"][0].detach().cpu().numpy()
            head_v = outputs.get("head_vector", torch.zeros(1, 3))[0].detach().cpu().numpy()
            var_val = float(outputs.get("attention_variance", torch.tensor([0.020]))[0].item())

        return {
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "gaze_vector": gaze_v,
            "head_vector": head_v,
            "latency_model_ms": latency_ms,
            "attention_variance": var_val
        }
