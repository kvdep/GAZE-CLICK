import os, math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Any, Tuple
from ultralytics import YOLO
from models.base import BaseGazeModel
from models.registry import ModelRegistry

class SpatialSoftArgmax2d(nn.Module):
    def __init__(self, temperature: float = 1.0, height: int = 28, width: int = 28) -> None:
        super().__init__()
        self.temperature = temperature; self.height = height; self.width = width
        pos_y = torch.linspace(-1.0, 1.0, height); pos_x = torch.linspace(-1.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        self.register_buffer("grid_x", grid_x, persistent=False)
        self.register_buffer("grid_y", grid_y, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]; h = z.shape[-2]; w = z.shape[-1]
        if self.grid_x is None or self.grid_x.shape[-2:] != (h, w) or self.grid_x.device != z.device or self.grid_x.dtype != z.dtype:
            pos_y = torch.linspace(-1.0, 1.0, h, device=z.device, dtype=z.dtype)
            pos_x = torch.linspace(-1.0, 1.0, w, device=z.device, dtype=z.dtype)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        else:
            grid_x = self.grid_x; grid_y = self.grid_y
        p = F.softmax((z / self.temperature).reshape(b, h * w), dim=-1).view(b, h, w)
        ux = torch.sum(p * grid_x, dim=(-2, -1))
        uy = torch.sum(p * grid_y, dim=(-2, -1))
        vx = torch.sum(p * (grid_x - ux.view(b, 1, 1)) ** 2, dim=(-2, -1))
        vy = torch.sum(p * (grid_y - uy.view(b, 1, 1)) ** 2, dim=(-2, -1))
        return torch.stack([ux, uy, vx, vy], dim=-1)

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
            nn.Conv2d(66, 128, kernel_size=1),
            nn.SiLU()
        )
        self.attn_conv = nn.Conv2d(128, 1, kernel_size=7, padding=3, bias=True)
        self.soft_argmax = SpatialSoftArgmax2d(temperature=1.0, height=28, width=28)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        self.eye_fc = nn.Sequential(
            nn.Linear(132, 128),
            nn.SiLU()
        )
        self.ln_eye = nn.LayerNorm(128)
        self.head_film = nn.Linear(64, 256)
        nn.init.zeros_(self.head_film.weight); nn.init.zeros_(self.head_film.bias)
        self.gaze_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.gaze_out[-1].bias.data[2] = -2.0
        self.head_fc[-1].bias.data[2] = -2.0
        pos_y = torch.linspace(-1.0, 1.0, 28); pos_x = torch.linspace(-1.0, 1.0, 28)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        self.register_buffer("coord_grid", coord_grid, persistent=False)

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
        b = p3_fused.shape[0]; h = p3_fused.shape[-2]; w = p3_fused.shape[-1]
        if self.coord_grid is None or self.coord_grid.shape[-2:] != (h, w) or self.coord_grid.device != p3_fused.device or self.coord_grid.dtype != p3_fused.dtype:
            pos_y = torch.linspace(-1.0, 1.0, h, device=p3_fused.device, dtype=p3_fused.dtype)
            pos_x = torch.linspace(-1.0, 1.0, w, device=p3_fused.device, dtype=p3_fused.dtype)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        else:
            coords = self.coord_grid.expand(b, -1, -1, -1)
        p3_coord = torch.cat([p3_fused, coords], dim=1)
        eye_map = self.eye_conv(p3_coord)
        z_logits = self.attn_conv(eye_map)
        moments = self.soft_argmax(z_logits)
        attn_weights = torch.sigmoid(z_logits)
        attended = eye_map * attn_weights
        eye_pooled = torch.flatten(self.pool(attended), 1)
        eye_mom = torch.cat([eye_pooled, moments], dim=-1)
        film_out = self.head_film(head_emb)
        gamma, beta = film_out.chunk(2, dim=-1)
        fused = self.ln_eye(self.eye_fc(eye_mom)) * (1.0 + torch.tanh(gamma)) + beta
        raw_gaze_vec = self.gaze_out(fused)
        gaze_norm = torch.norm(raw_gaze_vec, p=2, dim=-1, keepdim=True)
        gaze_unit = raw_gaze_vec / torch.clamp(gaze_norm, min=1e-8)
        gp_rad = torch.asin(torch.clamp(-gaze_unit[:, 1], -0.99999, 0.99999))
        gy_rad = torch.atan2(-gaze_unit[:, 0], -gaze_unit[:, 2])
        rad_to_deg = 180.0 / 3.141592653589793
        pitch_rest_rad = -8.99 * 3.141592653589793 / 180.0
        y_diff = torch.clamp(gy_rad, -1.45, 1.45)
        p_diff = torch.clamp(gp_rad - pitch_rest_rad, -1.45, 1.45)
        x_norm = torch.clamp(0.5 + 0.5 * torch.tan(y_diff) * 1.6, 0.0, 1.0)
        y_norm = torch.clamp(0.5 - 0.5 * torch.tan(p_diff) * 1.6, 0.0, 1.0)
        screen_coords = torch.stack([x_norm * 1440.0, y_norm * 900.0], dim=-1)
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
            "attention_maps": attn_weights,
            "moments": moments,
            "screen_coordinates": screen_coords
        }

class StableVectorGazeLoss(nn.Module):
    def __init__(self, smooth_beta: float = 0.02, **kwargs: Any) -> None:
        super().__init__()
        self.smooth_beta = smooth_beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=smooth_beta)

    def forward(self, predictions_dict: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        g_pred = predictions_dict["gaze_vector"]
        g_gt = batch_data["gaze_vector"]
        h_pred = predictions_dict["head_vector"]
        h_gt = batch_data.get("head_vector", h_pred)
        if h_gt is None: h_gt = h_pred
        l1_diff = self.smooth_l1(g_pred, g_gt)
        loss_aniso_vec = 2.0 * l1_diff[:, 0].mean() + 1.0 * l1_diff[:, 1].mean() + 1.5 * l1_diff[:, 2].mean()
        loss_cos = 1.0 * (1.0 - torch.sum(g_pred * g_gt, dim=-1)).mean()
        loss_h = 0.2 * self.smooth_l1(h_pred, h_gt).mean()
        total_loss = loss_aniso_vec + loss_cos + loss_h
        return {
            "loss": total_loss,
            "loss_aniso_vec": loss_aniso_vec,
            "loss_cos": loss_cos,
            "loss_head": loss_h,
            "total_loss": total_loss
        }

ContinuousGazeLoss = StableVectorGazeLoss
