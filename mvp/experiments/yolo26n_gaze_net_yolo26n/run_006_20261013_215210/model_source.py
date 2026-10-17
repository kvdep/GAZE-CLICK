import os, math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Any, Tuple
from ultralytics import YOLO
from models.base import BaseGazeModel
from models.registry import ModelRegistry

class SpatialAttention2d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.conv(feature_map)
        attended = feature_map * weights
        pooled = torch.flatten(self.pool(attended), 1)
        return pooled, weights

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
        self.backbone_layers = nn.ModuleList(list(yolo_wrapper.model.model[:11]))
        for param in self.backbone_layers.parameters(): param.requires_grad = True
        self.backbone = self.backbone_layers
        self.lat5 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat4 = nn.Conv2d(128, 64, kernel_size=1)
        self.p3_proj = nn.Conv2d(128, 64, kernel_size=1)
        self.eye_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.SiLU()
        )
        self.spatial_attention = SpatialAttention2d(128)
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.head_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU()
        )
        self.gaze_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 + 64, 128),
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
        x = input_tensor
        p3 = None; p4 = None; p5 = None
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            if i == 4:
                p3 = x
            elif i == 6:
                p4 = x
            elif i == 10:
                p5 = x
        head_pooled = torch.flatten(self.head_pool(p5), 1)
        raw_head_vec = self.head_fc(head_pooled)
        head_norm = torch.norm(raw_head_vec, p=2, dim=-1, keepdim=True)
        head_unit = raw_head_vec / torch.clamp(head_norm, min=1e-8)
        hp_rad = torch.asin(torch.clamp(-head_unit[:, 1], -0.99999, 0.99999))
        hy_rad = torch.atan2(-head_unit[:, 0], -head_unit[:, 2])
        head_emb = self.head_embed(head_unit.detach())
        lat5_feat = self.lat5(p5)
        lat5_up = F.interpolate(lat5_feat, size=(14, 14), mode="bilinear", align_corners=False)
        p4_fused = lat5_up + p4
        lat4_feat = self.lat4(p4_fused)
        lat4_up = F.interpolate(lat4_feat, size=(28, 28), mode="bilinear", align_corners=False)
        p3_aligned = self.p3_proj(p3) if p3.shape[1] == 128 else p3
        p3_fused = lat4_up + p3_aligned
        eye_map = self.eye_conv(p3_fused)
        eye_pooled, attn_maps = self.spatial_attention(eye_map)
        gaze_input = torch.cat([eye_pooled, head_emb], dim=-1)
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

class StableVectorGazeLoss(nn.Module):
    def __init__(self, smooth_beta: float = 0.05, scale_smooth: float = 50.0, scale_cos: float = 25.0, scale_head: float = 10.0, **kwargs: Any) -> None:
        super().__init__()
        self.smooth_beta = smooth_beta
        self.scale_smooth = scale_smooth
        self.scale_cos = scale_cos
        self.scale_head = scale_head

    def forward(self, predictions_dict: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        g_pred = predictions_dict["gaze_vector"]
        g_gt = batch_data["gaze_vector"]
        g_pred_norm = g_pred / torch.clamp(torch.norm(g_pred, p=2, dim=-1, keepdim=True), min=1e-8)
        g_gt_norm = g_gt / torch.clamp(torch.norm(g_gt, p=2, dim=-1, keepdim=True), min=1e-8)
        loss_smooth_g = self.scale_smooth * F.smooth_l1_loss(g_pred_norm, g_gt_norm, beta=self.smooth_beta)
        dot_g = torch.sum(g_pred_norm * g_gt_norm, dim=-1)
        loss_cos_g = self.scale_cos * torch.mean(1.0 - torch.clamp(dot_g, -1.0, 1.0))
        h_pred = predictions_dict["head_vector"]
        h_gt = batch_data.get("head_vector", h_pred)
        h_pred_norm = h_pred / torch.clamp(torch.norm(h_pred, p=2, dim=-1, keepdim=True), min=1e-8)
        h_gt_norm = h_gt / torch.clamp(torch.norm(h_gt, p=2, dim=-1, keepdim=True), min=1e-8)
        loss_smooth_h = self.scale_head * F.smooth_l1_loss(h_pred_norm, h_gt_norm, beta=self.smooth_beta)
        total_loss = loss_smooth_g + loss_cos_g + loss_smooth_h
        return {
            "total_loss": total_loss,
            "smooth_gaze_loss": loss_smooth_g,
            "cos_gaze_loss": loss_cos_g,
            "smooth_head_loss": loss_smooth_h
        }

ContinuousGazeLoss = StableVectorGazeLoss
