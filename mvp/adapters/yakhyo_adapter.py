import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, BasicBlock

try:
    from adapters.base import BaseGazeAdapter
except ImportError:
    from mvp.adapters.base import BaseGazeAdapter

class YakhyoResNet(models.ResNet):
    def __init__(self, block, layers, expansion: int, num_bins: int = 90):
        super().__init__(block, layers)
        self.fc_pitch = nn.Linear(512 * expansion, num_bins)
        self.fc_yaw = nn.Linear(512 * expansion, num_bins)
        del self.fc

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc_yaw(x), self.fc_pitch(x)


class YakhyoAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for yakhyo/gaze-estimation models (ResNet-50 / ResNet-18).
    - Architecture: ResNet backbone + Dual classification heads (90 bins per angle).
    - Continuous angle decoded via Softmax expectation: angle = sum(p * idx) * 4 - 180 (deg).
    - Input: 224x224 RGB float tensor with ImageNet normalization.
    """
    def __init__(self, name: str = "yakhyo_resnet50", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224
        self.num_bins = 90
        self.bin_width = 4.0
        self.angle_offset = 180.0
        self.idx_tensor = torch.arange(self.num_bins, device=self.device, dtype=torch.float32)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        elif "18" in self.name:
            weight_file = "mvp/models/weights/yakhyo_resnet18.pt"
        else:
            weight_file = "mvp/models/weights/yakhyo_resnet50.pt"

        if not os.path.exists(weight_file):
            import urllib.request
            model_type = "resnet18" if "18" in self.name else "resnet50"
            url = f"https://github.com/yakhyo/gaze-estimation/releases/download/weights/{model_type}.pt"
            os.makedirs("mvp/models/weights", exist_ok=True)
            urllib.request.urlretrieve(url, weight_file)

        if "18" in self.name:
            model = YakhyoResNet(BasicBlock, [2, 2, 2, 2], expansion=1, num_bins=self.num_bins)
        else:
            model = YakhyoResNet(Bottleneck, [3, 4, 6, 3], expansion=4, num_bins=self.num_bins)

        sd = torch.load(weight_file, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        model.to(self.device).eval()
        self.model = model
        print(f"[YakhyoAdapter] Successfully loaded Yakhyo {self.name} from '{os.path.basename(weight_file)}' onto {self.device}.")

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
            raw_yaw, raw_pitch = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            prob_yaw = F.softmax(raw_yaw, dim=1)
            prob_pitch = F.softmax(raw_pitch, dim=1)

            pred_yaw_deg = (torch.sum(prob_yaw * self.idx_tensor, dim=1) * self.bin_width - self.angle_offset).item()
            pred_pitch_deg = (torch.sum(prob_pitch * self.idx_tensor, dim=1) * self.bin_width - self.angle_offset).item()

        pitch_deg = float(pred_pitch_deg)
        yaw_deg = float(pred_yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

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
