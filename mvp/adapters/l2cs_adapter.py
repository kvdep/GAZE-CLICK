import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision.models.resnet import Bottleneck

from adapters.base import BaseGazeAdapter

class L2CSNetResNet50(nn.Module):
    def __init__(self, num_bins: int = 90):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(2048, num_bins)
        self.fc_pitch_gaze = nn.Linear(2048, num_bins)
        self.fc_finetune = nn.Linear(2048 + 3, 3)

        idx = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer("bin_centers_deg", idx * 4.0 - 180.0, persistent=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        yaw_logits = self.fc_yaw_gaze(x)
        pitch_logits = self.fc_pitch_gaze(x)
        return yaw_logits, pitch_logits


class L2CSAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for L2CS-Net (Abdelrahman et al. / Py-Feat).
    - Architecture: ResNet-50 + dual 90-bin classification heads.
    - Input: 224x224 RGB, ImageNet-normalized.
    - Outputs: Unswapped heads: fc_yaw_gaze -> Yaw (azimuth), fc_pitch_gaze -> Pitch (elevation).
    """
    def __init__(self, name: str = "l2cs", device: torch.device = torch.device("cpu"), variant: str = "pyfeat"):
        super().__init__(name=name, device=device)
        self.variant = variant
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        model = L2CSNetResNet50(num_bins=90).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        elif self.variant == "diversity":
            from huggingface_hub import hf_hub_download
            weight_file = hf_hub_download(repo_id="akgulburak/gaze-estimation-diversity", filename="L2CSNet/L2CSNet_baseline_1.pkl")
        else:
            from huggingface_hub import hf_hub_download
            weight_file = hf_hub_download(repo_id="py-feat/l2cs", filename="l2cs_gaze360_resnet50.safetensors")

        if weight_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(weight_file)
            model.load_state_dict(sd, strict=False)
        else:
            sd = torch.load(weight_file, map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                sd = sd.get("model_state_dict", sd.get("state_dict", sd))
            model.load_state_dict(sd, strict=False)

        model.eval()
        self.model = model
        print(f"[L2CSAdapter] Successfully initialized {self.name} on {self.device} from '{os.path.basename(weight_file)}'.")

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
                "latency_model_ms": 0.0,
                "attention_variance": 0.040
            }

        crop_res = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_res, cv2.COLOR_BGR2RGB)
        tensor_in = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0 - self.mean) / self.std

        t0 = time.perf_counter()
        with torch.no_grad():
            yaw_logits, pitch_logits = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            yaw_probs = F.softmax(yaw_logits, dim=-1)
            pitch_probs = F.softmax(pitch_logits, dim=-1)
            yaw_deg = float(torch.sum(yaw_probs * self.model.bin_centers_deg, dim=-1).item())
            pitch_deg = float(torch.sum(pitch_probs * self.model.bin_centers_deg, dim=-1).item())

        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        gx = -math.sin(yaw_rad) * math.cos(pitch_rad)
        gy = -math.sin(pitch_rad)
        gz = -math.cos(yaw_rad) * math.cos(pitch_rad)

        return {
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "gaze_vector": np.array([gx, gy, gz], dtype=np.float32),
            "latency_model_ms": latency_ms,
            "attention_variance": 0.020
        }
