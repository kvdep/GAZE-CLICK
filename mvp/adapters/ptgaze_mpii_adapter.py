import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.models as models
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

try:
    from adapters.base import BaseGazeAdapter
except ImportError:
    from mvp.adapters.base import BaseGazeAdapter

class ResNetSimpleMPII(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu; self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(50176, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PTGazeMPIIAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for ptgaze MPIIFaceGaze ResNet-Simple (hysts/ptgaze-mpiifacegaze-resnet-simple).
    - Architecture: Truncated ResNet-18 (stages 1-3) + spatial weighting conv + Linear(50176, 2).
    - Input: 224x224 BGR float tensor with ImageNet normalization.
    - Output: out[:, 0] = pitch (rad), out[:, 1] = yaw (rad) per MPIIFaceGaze convention.
    """
    def __init__(self, name: str = "ptgaze_mpiifacegaze", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        else:
            weight_file = hf_hub_download(
                repo_id="hysts/ptgaze-mpiifacegaze-resnet-simple",
                filename="model.safetensors"
            )

        model = ResNetSimpleMPII()
        st = load_file(weight_file)
        remap_st = {}
        for k, v in st.items():
            if k.startswith("feature_extractor."):
                remap_st[k.replace("feature_extractor.", "")] = v
            else:
                remap_st[k] = v
        model.load_state_dict(remap_st)
        model.to(self.device).eval()
        self.model = model
        print(f"[PTGazeMPIIAdapter] Successfully loaded ptgaze MPIIFaceGaze from '{os.path.basename(weight_file)}' onto {self.device}.")

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
        # MPIIFaceGaze expects BGR channel order with ImageNet mean/std
        tensor_in = torch.from_numpy(crop_res).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
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
