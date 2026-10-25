import math, time, os, copy
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

from adapters.base import BaseGazeAdapter

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class GazeTRBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        res = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        return self.relu(out)

class GazeTRResNetBackbone(nn.Module):
    def __init__(self, block, layers, maps=32):
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
        self.conv = nn.Sequential(
            nn.Conv2d(512, maps, 1),
            nn.BatchNorm2d(maps),
            nn.ReLU(inplace=True)
        )
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
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.conv(x)

class GazeTRTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=32, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, src, pos):
        q = k = src + pos.unsqueeze(1).repeat(1, src.size(1), 1)
        src2 = self.self_attn(q, k, value=src)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))

class GazeTRTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers=6, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.norm = norm
    def forward(self, src, pos):
        out = src
        for l in self.layers:
            out = l(out, pos)
        if self.norm is not None:
            out = self.norm(out)
        return out

class GazeTRNet(nn.Module):
    def __init__(self):
        super().__init__()
        maps = 32
        self.base_model = GazeTRResNetBackbone(GazeTRBasicBlock, [2, 2, 2, 2], maps=maps)
        enc_layer = GazeTRTransformerEncoderLayer(maps, 8, 512, 0.1)
        self.encoder = GazeTRTransformerEncoder(enc_layer, 6, nn.LayerNorm(maps))
        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.pos_embedding = nn.Embedding(50, maps)
        self.feed = nn.Linear(maps, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base_model(x)
        b = feat.size(0)
        feat = feat.flatten(2).permute(2, 0, 1)
        cls = self.cls_token.repeat(1, b, 1)
        feat = torch.cat([cls, feat], dim=0)
        pos_idx = torch.arange(50, device=x.device)
        pos = self.pos_embedding(pos_idx)
        feat = self.encoder(feat, pos)
        angles = self.feed(feat[0])
        return angles


class GazeTRAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for GazeTR (Cheng et al. / Akgul et al.).
    - Architecture: ResNet-18 + 6-layer Transformer Encoder.
    - Input: 224x224 BGR float in range [0, 1] (as trained in datasets_GazeTR.py).
    - Output: angles[:, 0] = Yaw (azimuth), angles[:, 1] = Pitch (elevation) in radians.
    """
    def __init__(self, name: str = "gazetr", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        model = GazeTRNet().to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        else:
            from huggingface_hub import hf_hub_download
            weight_file = hf_hub_download(repo_id="akgulburak/gaze-estimation-diversity", filename="GazeTR/GazeTR-baseline-1.pt")

        raw_data = torch.load(weight_file, map_location="cpu", weights_only=False)
        sd = raw_data.get("model_state_dict", raw_data.get("state_dict", raw_data)) if isinstance(raw_data, dict) else raw_data
        model.load_state_dict(sd, strict=False)
        model.eval()
        self.model = model
        print(f"[GazeTRAdapter] Successfully loaded GazeTR from '{os.path.basename(weight_file)}' onto {self.device}.")

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

        # GazeTR expects BGR float tensor normalized in [0, 1] without ImageNet mean subtraction
        crop_res = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        tensor_in = torch.from_numpy(crop_res).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        t0 = time.perf_counter()
        with torch.no_grad():
            angles = self.model(tensor_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # angles shape [1, 2]: [azimuth (yaw), elevation (pitch)] in radians
            yaw_rad = float(angles[0, 0].item())
            pitch_rad = float(angles[0, 1].item())

        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)

        # 3D gaze unit vector per gtools_GazeTR.py
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
