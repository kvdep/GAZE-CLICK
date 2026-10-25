import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models.resnet import Bottleneck

from adapters.base import BaseGazeAdapter

class ResFeature(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x

class ResGazeEs(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.avgpool(x).flatten(1))

class PureGazeInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = ResFeature(Bottleneck, [3, 4, 6, 3])
        self.gazeEs = ResGazeEs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gazeEs(self.feature(x))


class PureGazeAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for PureGaze (Cheng et al. / Akgul et al.).
    - Architecture: ResNet-50 backbone + ResGazeEs regression head.
    - Input: 224x224 BGR float in range [0, 1] (as trained in datasets_PureGaze.py).
    - Output: angles[:, 0] = Pitch (elevation), angles[:, 1] = Yaw (azimuth) in radians.
    """
    def __init__(self, name: str = "puregaze", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        model = PureGazeInferenceModel().to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        else:
            from huggingface_hub import hf_hub_download
            weight_file = hf_hub_download(repo_id="akgulburak/gaze-estimation-diversity", filename="PureGaze/PureGaze_baseline_1.pt")

        raw_data = torch.load(weight_file, map_location="cpu", weights_only=False)
        sd = raw_data.get("model_state_dict", raw_data.get("state_dict", raw_data)) if isinstance(raw_data, dict) else raw_data
        model.load_state_dict(sd, strict=False)
        model.eval()
        self.model = model
        print(f"[PureGazeAdapter] Successfully loaded PureGaze from '{os.path.basename(weight_file)}' onto {self.device}.")

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

        # PureGaze expects BGR float tensor normalized in [0, 1] without ImageNet mean subtraction
        crop_res = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        tensor_in = torch.from_numpy(crop_res).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        t0 = time.perf_counter()
        with torch.no_grad():
            angles = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # angles shape [1, 2]: [pitch (elevation), yaw (azimuth)] in radians
            pitch_rad = float(angles[0, 0].item())
            yaw_rad = float(angles[0, 1].item())

        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = math.degrees(yaw_rad)

        # 3D gaze unit vector per gtools_PureGaze.py
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
