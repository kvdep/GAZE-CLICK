import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.models as models
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

try:
    from adapters.base import BaseGazeAdapter
except ImportError:
    from mvp.adapters.base import BaseGazeAdapter

class PTGazeETHAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for ptgaze ETH-XGaze ResNet-18 (hysts/ptgaze-eth-xgaze-resnet18).
    - Architecture: Standard ResNet-18 + Linear(512, 2) head.
    - Input: 224x224 RGB float tensor with ImageNet normalization (mean/std).
    - Output: out[:, 0] = pitch (rad), out[:, 1] = yaw (rad) per ETH-XGaze convention.
    """
    def __init__(self, name: str = "ptgaze_eth_xgaze", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        else:
            weight_file = hf_hub_download(
                repo_id="hysts/ptgaze-eth-xgaze-resnet18",
                filename="model.safetensors"
            )

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        st = load_file(weight_file)
        model.load_state_dict(st)
        model.to(self.device).eval()
        self.model = model
        print(f"[PTGazeETHAdapter] Successfully loaded ptgaze ETH-XGaze ResNet-18 from '{os.path.basename(weight_file)}' onto {self.device}.")

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
                "gaze_vector": np.array([0.0, 0.0, -1.0], dtype=np.float32),
                "latency_model_ms": 0.0,
                "attention_variance": 0.040
            }

        crop_res = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_res, cv2.COLOR_BGR2RGB)
        tensor_in = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        tensor_in = (tensor_in - self.mean) / self.std

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # out[0, 0] = pitch (vertical), out[0, 1] = yaw (horizontal)
            pitch_rad = float(out[0, 0].item())
            yaw_rad = float(out[0, 1].item())

        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = math.degrees(yaw_rad)

        gx = -math.cos(pitch_rad) * math.sin(yaw_rad)
        gy = -math.sin(pitch_rad)
        gz = -math.cos(pitch_rad) * math.cos(yaw_rad)

        return {
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "gaze_vector": np.array([gx, gy, gz], dtype=np.float32),
            "latency_model_ms": latency_ms,
            "attention_variance": 0.020
        }
