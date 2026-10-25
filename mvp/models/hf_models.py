import os, sys, math, time, copy, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, BasicBlock
from typing import Dict, Any, Tuple, Optional

# ==============================================================================
# 1. L2CS-Net Architecture (ResNet-50 Backbone + Dual 90-bin Classification Heads)
# ==============================================================================
class L2CS(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_bins: int = 90):
        super().__init__()
        self.inplanes = 64
        self.num_bins = num_bins
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 512 * block.expansion
        self.fc_yaw_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_pitch_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_finetune = nn.Linear(feat_dim + 3, 3)

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        # In Ahmednull/L2CS-Net training, the model forward returned (pre_yaw_gaze, pre_pitch_gaze)
        # but in train.py it was unpacked as `pitch, yaw = model(images)`.
        # Therefore, fc_yaw_gaze was trained against pitch labels (vertical),
        # and fc_pitch_gaze was trained against yaw labels (horizontal).
        pitch_logits = self.fc_yaw_gaze(x)
        yaw_logits = self.fc_pitch_gaze(x)
        pitch_probs = F.softmax(pitch_logits, dim=-1)
        yaw_probs = F.softmax(yaw_logits, dim=-1)
        pitch_deg = torch.sum(pitch_probs * self.bin_centers_deg, dim=-1)
        yaw_deg = torch.sum(yaw_probs * self.bin_centers_deg, dim=-1)
        pitch_rad = pitch_deg * (math.pi / 180.0)
        yaw_rad = yaw_deg * (math.pi / 180.0)
        gx = -torch.sin(yaw_rad) * torch.cos(pitch_rad)
        gy = -torch.sin(pitch_rad)
        gz = -torch.cos(yaw_rad) * torch.cos(pitch_rad)
        gaze_vector = torch.stack([gx, gy, gz], dim=-1)
        return {
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "gaze_vector": gaze_vector,
            "attention_variance": torch.tensor([0.020], device=x.device)
        }

# ==============================================================================
# 2. Idiap ResNet-18 Dual-Stream Gaze360 Model (Face + Left Eye + Right Eye)
# ==============================================================================
class IdiapResNet18Gaze360(nn.Module):
    def __init__(self):
        super().__init__()
        r_head = models.resnet18()
        r_eye = models.resnet18()
        self.base_head = nn.Sequential(*list(r_head.children())[:-2])
        self.base_eye = nn.Sequential(*list(r_eye.children())[:-2])
        self.head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )

    def forward(self, head: torch.Tensor, left_eye: torch.Tensor, right_eye: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.base_head(head).mean([2, 3])
        le = self.base_eye(left_eye).mean([2, 3])
        re = self.base_eye(right_eye).mean([2, 3])
        feat = torch.cat([h, le, re], dim=1)
        vec = self.head(feat)
        vec_norm = F.normalize(vec, p=2, dim=-1)
        gx, gy, gz = vec_norm[:, 0], vec_norm[:, 1], vec_norm[:, 2]
        pitch_rad = torch.asin(-torch.clamp(gy, -1.0, 1.0))
        yaw_rad = torch.atan2(-gx, -gz)
        return {
            "gaze_vector": vec_norm,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "pitch_deg": pitch_rad * (180.0 / math.pi),
            "yaw_deg": yaw_rad * (180.0 / math.pi),
            "attention_variance": torch.tensor([0.020], device=head.device)
        }

# ==============================================================================
# 3. GazeTR Model (ResNet-18 Feature Extractor + 6-Layer Transformer Encoder)
# ==============================================================================
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

class GazeTRModel(nn.Module):
    def __init__(self):
        super().__init__()
        maps = 32
        self.base_model = GazeTRResNetBackbone(GazeTRBasicBlock, [2, 2, 2, 2], maps=maps)
        enc_layer = GazeTRTransformerEncoderLayer(maps, 8, 512, 0.1)
        self.encoder = GazeTRTransformerEncoder(enc_layer, 6, nn.LayerNorm(maps))
        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.pos_embedding = nn.Embedding(50, maps)
        self.feed = nn.Linear(maps, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.base_model(x)
        b = feat.size(0)
        feat = feat.flatten(2).permute(2, 0, 1)
        cls = self.cls_token.repeat(1, b, 1)
        feat = torch.cat([cls, feat], dim=0)
        pos_idx = torch.arange(50, device=x.device)
        pos = self.pos_embedding(pos_idx)
        feat = self.encoder(feat, pos)
        angles = self.feed(feat[0])
        # In GazeTR (Cheng et al.), output 0 is yaw, output 1 is pitch
        yaw_rad = angles[:, 0]
        pitch_rad = angles[:, 1]
        gx = -torch.sin(yaw_rad) * torch.cos(pitch_rad)
        gy = -torch.sin(pitch_rad)
        gz = -torch.cos(yaw_rad) * torch.cos(pitch_rad)
        gaze_vector = torch.stack([gx, gy, gz], dim=-1)
        return {
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "pitch_deg": pitch_rad * (180.0 / math.pi),
            "yaw_deg": yaw_rad * (180.0 / math.pi),
            "gaze_vector": gaze_vector,
            "attention_variance": torch.tensor([0.020], device=x.device)
        }

# ==============================================================================
# 4. PureGaze Model (ResNet-50 Feature Extractor + Linear Gaze Head)
# ==============================================================================
class PureGazeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = models.resnet50()
        self.feature.fc = nn.Identity()
        self.gazeEs = nn.Sequential()
        self.gazeEs.add_module("fc", nn.Linear(2048, 2))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.feature(x)
        angles = self.gazeEs(feat)
        # In PureGaze (Cheng et al.), output 0 is yaw, output 1 is pitch
        yaw_rad = angles[:, 0]
        pitch_rad = angles[:, 1]
        gx = -torch.sin(yaw_rad) * torch.cos(pitch_rad)
        gy = -torch.sin(pitch_rad)
        gz = -torch.cos(yaw_rad) * torch.cos(pitch_rad)
        gaze_vector = torch.stack([gx, gy, gz], dim=-1)
        return {
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "pitch_deg": pitch_rad * (180.0 / math.pi),
            "yaw_deg": yaw_rad * (180.0 / math.pi),
            "gaze_vector": gaze_vector,
            "attention_variance": torch.tensor([0.020], device=x.device)
        }

# ==============================================================================
# 5. Model Resolver & Factory
# ==============================================================================
HF_MODEL_CATALOG = {
    "l2cs": {
        "repo_id": "py-feat/l2cs",
        "filename": "l2cs_gaze360_resnet50.safetensors",
        "class": L2CS,
        "type": "single_crop",
        "crop_size": 448,
        "name": "L2CS-Net (py-feat/l2cs)"
    },
    "l2cs_resnet50": {
        "repo_id": "py-feat/l2cs",
        "filename": "l2cs_gaze360_resnet50.safetensors",
        "class": L2CS,
        "type": "single_crop",
        "crop_size": 448,
        "name": "L2CS-Net ResNet-50"
    },
    "pyfeat": {
        "repo_id": "py-feat/l2cs",
        "filename": "l2cs_gaze360_resnet50.safetensors",
        "class": L2CS,
        "type": "single_crop",
        "crop_size": 448,
        "name": "L2CS-Net (py-feat/l2cs)"
    },
    "l2cs_diversity": {
        "repo_id": "akgulburak/gaze-estimation-diversity",
        "filename": "L2CSNet/L2CSNet_baseline_1.pkl",
        "class": L2CS,
        "type": "single_crop",
        "crop_size": 448,
        "name": "L2CSNet (akgulburak/gaze-estimation-diversity)"
    },
    "idiap": {
        "repo_id": "Idiap/resnet18-gaze360",
        "filename": "gaze360_resnet18.pt",
        "class": IdiapResNet18Gaze360,
        "type": "dual_stream",
        "crop_size": 224,
        "name": "ResNet-18 Gaze360 (Idiap)"
    },
    "idiap_resnet18": {
        "repo_id": "Idiap/resnet18-gaze360",
        "filename": "gaze360_resnet18.pt",
        "class": IdiapResNet18Gaze360,
        "type": "dual_stream",
        "crop_size": 224,
        "name": "ResNet-18 Gaze360 (Idiap)"
    },
    "resnet18-gaze360": {
        "repo_id": "Idiap/resnet18-gaze360",
        "filename": "gaze360_resnet18.pt",
        "class": IdiapResNet18Gaze360,
        "type": "dual_stream",
        "crop_size": 224,
        "name": "ResNet-18 Gaze360 (Idiap)"
    },
    "gazetr": {
        "repo_id": "akgulburak/gaze-estimation-diversity",
        "filename": "GazeTR/GazeTR-baseline-1.pt",
        "class": GazeTRModel,
        "type": "single_crop",
        "crop_size": 224,
        "name": "GazeTR (akgulburak/gaze-estimation-diversity)"
    },
    "puregaze": {
        "repo_id": "akgulburak/gaze-estimation-diversity",
        "filename": "PureGaze/PureGaze_baseline_1.pt",
        "class": PureGazeModel,
        "type": "single_crop",
        "crop_size": 224,
        "name": "PureGaze (akgulburak/gaze-estimation-diversity)"
    }
}

def load_gaze_model(
    model_name: str = "l2cs",
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Unified loader for local and Hugging Face gaze models.
    Returns:
        (instantiated_model, metadata_dict)
    """
    model_key = model_name.strip().lower()

    # Check if user specified a direct checkpoint file path
    if checkpoint_path and os.path.exists(checkpoint_path):
        from models.registry import ModelRegistry
        import models.yolo26n_gaze_net
        model, _ = ModelRegistry.load_checkpoint(checkpoint_path, target_device=str(device))
        model.eval()
        return model, {
            "name": f"Local Checkpoint ({os.path.basename(checkpoint_path)})",
            "type": "yolo",
            "checkpoint_path": checkpoint_path
        }

    # Check local YOLO aliases
    if model_key in ("yolo", "yolo26n", "yolo26n_gaze_net"):
        from models.registry import ModelRegistry
        import models.yolo26n_gaze_net
        default_yolo = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt"
        if not os.path.exists(default_yolo):
            import glob
            cands = glob.glob("mvp/experiments/**/best_model.pt", recursive=True)
            if cands:
                default_yolo = cands[0]
        if not os.path.exists(default_yolo):
            raise FileNotFoundError(f"Local YOLO checkpoint not found at: {default_yolo}")
        model, _ = ModelRegistry.load_checkpoint(default_yolo, target_device=str(device))
        model.eval()
        return model, {
            "name": f"YOLO26n GazeNet ({os.path.basename(os.path.dirname(os.path.dirname(default_yolo)))})",
            "type": "yolo",
            "checkpoint_path": default_yolo
        }

    # Check Hugging Face catalog
    if model_key in HF_MODEL_CATALOG:
        cfg = HF_MODEL_CATALOG[model_key]
        from huggingface_hub import hf_hub_download
        print(f"[ModelLoader] Resolving weights for {cfg['name']} from HF repo '{cfg['repo_id']}'...")
        weight_file = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"])
        model_instance = cfg["class"]().to(device)
        
        if weight_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(weight_file)
            model_instance.load_state_dict(sd, strict=False)
        else:
            raw_data = torch.load(weight_file, map_location="cpu", weights_only=False)
            if isinstance(raw_data, dict):
                sd = raw_data.get("model_state_dict", raw_data.get("state_dict", raw_data))
            else:
                sd = raw_data
            if cfg["class"] == PureGazeModel:
                sd = {k: v for k, v in sd.items() if not k.startswith("deconv.")}
            model_instance.load_state_dict(sd, strict=False)
            
        model_instance.eval()
        print(f"[ModelLoader] Successfully loaded {cfg['name']} onto {device}.")
        return model_instance, {
            "name": cfg["name"],
            "type": cfg["type"],
            "checkpoint_path": weight_file
        }

    # If user provided a path directly into model_name:
    if os.path.exists(model_name):
        from models.registry import ModelRegistry
        import models.yolo26n_gaze_net
        model, _ = ModelRegistry.load_checkpoint(model_name, target_device=str(device))
        model.eval()
        return model, {
            "name": f"Custom Model ({os.path.basename(model_name)})",
            "type": "yolo",
            "checkpoint_path": model_name
        }

    valid_names = list(HF_MODEL_CATALOG.keys()) + ["mediapipe", "yolo26n"]
    raise ValueError(
        f"Unknown model name: '{model_name}'. Available models: {sorted(set(valid_names))}"
    )
