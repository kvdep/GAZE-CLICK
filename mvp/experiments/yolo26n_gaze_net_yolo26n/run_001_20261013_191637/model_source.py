import os, math, torch, torch.nn as nn
from typing import Dict, Any, Tuple
from ultralytics import YOLO
from models.base import BaseGazeModel
from models.registry import ModelRegistry

class SpatialAttention2d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = feature_map.shape
        attn_logits = self.conv(feature_map).view(b, h * w)
        attn_weights = torch.softmax(attn_logits, dim=-1).view(b, 1, h, w)
        pooled = torch.sum(feature_map * attn_weights, dim=(2, 3))
        return pooled, attn_weights

@ModelRegistry.register("yolo26n_gaze_net")
class YOLO26nGazeNet(BaseGazeModel):
    def __init__(self, weights_path: str = "mvp/models/weights/yolo26n.pt", pretrained: bool = True, feature_dim: int = 256, dropout: float = 0.2, **kwargs: Any) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.dropout = dropout
        resolved_weights = weights_path
        if not os.path.exists(resolved_weights):
            if os.path.exists("yolo26n.pt"):
                resolved_weights = "yolo26n.pt"
            elif os.path.exists(os.path.join("mvp", "models", "weights", "yolo26n.pt")):
                resolved_weights = os.path.join("mvp", "models", "weights", "yolo26n.pt")
        yolo_wrapper = YOLO(resolved_weights)
        self.backbone = nn.Sequential(*list(yolo_wrapper.model.model[:11]))
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.head_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU()
        )
        self.spatial_attention = SpatialAttention2d(feature_dim)
        self.gaze_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim + 64, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "model_type": "yolo26n_gaze_net",
            "backbone_name": "yolo26n",
            "weights_path": self.weights_path,
            "pretrained": self.pretrained,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout
        }

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(input_tensor)
        head_pooled = torch.flatten(self.head_pool(features), 1)
        raw_head_vec = self.head_fc(head_pooled)
        head_norm = torch.norm(raw_head_vec, p=2, dim=-1, keepdim=True)
        head_unit = raw_head_vec / torch.clamp(head_norm, min=1e-8)
        hp_rad = torch.asin(torch.clamp(-head_unit[:, 1], -0.99999, 0.99999))
        hy_rad = torch.atan2(-head_unit[:, 0], -head_unit[:, 2])
        head_emb = self.head_embed(head_unit.detach())
        gaze_pooled, attn_maps = self.spatial_attention(features)
        gaze_input = torch.cat([gaze_pooled, head_emb], dim=-1)
        raw_gaze_vec = self.gaze_fc(gaze_input)
        gaze_norm = torch.norm(raw_gaze_vec, p=2, dim=-1, keepdim=True)
        gaze_unit = raw_gaze_vec / torch.clamp(gaze_norm, min=1e-8)
        gp_rad = torch.asin(torch.clamp(-gaze_unit[:, 1], -0.99999, 0.99999))
        gy_rad = torch.atan2(-gaze_unit[:, 0], -gaze_unit[:, 2])
        rad_to_deg = 180.0 / 3.141592653589793
        return {
            "gaze_vector": gaze_unit,
            "pitch_rad": gp_rad,
            "yaw_rad": gy_rad,
            "gaze_pitch_deg": gp_rad * rad_to_deg,
            "gaze_yaw_deg": gy_rad * rad_to_deg,
            "head_vector": head_unit,
            "head_pitch_rad": hp_rad,
            "head_yaw_rad": hy_rad,
            "head_pitch_deg": hp_rad * rad_to_deg,
            "head_yaw_deg": hy_rad * rad_to_deg,
            "attention_maps": attn_maps
        }

class ContinuousGazeLoss(nn.Module):
    def __init__(self, angular_weight: float = 1.0, cosine_weight: float = 0.5, head_weight: float = 0.2) -> None:
        super().__init__()
        self.angular_weight = angular_weight
        self.cosine_weight = cosine_weight
        self.head_weight = head_weight
        self.rad_to_deg = 180.0 / 3.141592653589793

    def forward(self, predictions_dict: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_gaze = predictions_dict["gaze_vector"]
        target_gaze = batch_data["gaze_vector"]
        pred_norm = torch.norm(pred_gaze, p=2, dim=-1, keepdim=True)
        target_norm = torch.norm(target_gaze, p=2, dim=-1, keepdim=True)
        pred_u = pred_gaze / torch.clamp(pred_norm, min=1e-8)
        tgt_u = target_gaze / torch.clamp(target_norm, min=1e-8)
        dot_gaze = torch.sum(pred_u * tgt_u, dim=-1)
        clamped_dot = torch.clamp(dot_gaze, -0.99999, 0.99999)
        gaze_ang_deg = torch.mean(torch.acos(clamped_dot) * self.rad_to_deg)
        gaze_cos = torch.mean(1.0 - clamped_dot)
        pred_head = predictions_dict["head_vector"]
        target_head = batch_data.get("head_vector", pred_head)
        p_head_norm = torch.norm(pred_head, p=2, dim=-1, keepdim=True)
        t_head_norm = torch.norm(target_head, p=2, dim=-1, keepdim=True)
        ph_u = pred_head / torch.clamp(p_head_norm, min=1e-8)
        th_u = target_head / torch.clamp(t_head_norm, min=1e-8)
        dot_head = torch.clamp(torch.sum(ph_u * th_u, dim=-1), -0.99999, 0.99999)
        head_ang_deg = torch.mean(torch.acos(dot_head) * self.rad_to_deg)
        total = self.angular_weight * gaze_ang_deg + self.cosine_weight * gaze_cos + self.head_weight * head_ang_deg
        return {
            "total_loss": total,
            "gaze_angular_deg_loss": gaze_ang_deg,
            "gaze_cosine_loss": gaze_cos,
            "head_angular_deg_loss": head_ang_deg
        }
