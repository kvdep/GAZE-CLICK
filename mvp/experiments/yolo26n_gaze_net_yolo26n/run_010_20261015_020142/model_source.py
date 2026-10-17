import os, math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Any, Tuple
from ultralytics import YOLO
from models.base import BaseGazeModel
from models.registry import ModelRegistry

class MultiHeadSpatialSoftArgmax2d(nn.Module):
    def __init__(self, num_heads: int = 2, temperature: float = 1.0, height: int = 28, width: int = 28) -> None:
        super().__init__()
        self.num_heads = num_heads; self.temperature = temperature; self.height = height; self.width = width
        pos_y = torch.linspace(-1.0, 1.0, height); pos_x = torch.linspace(-1.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        self.register_buffer("grid_x", grid_x, persistent=False)
        self.register_buffer("grid_y", grid_y, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]; k = z.shape[1]; h = z.shape[-2]; w = z.shape[-1]
        if self.grid_x is None or self.grid_x.shape[-2:] != (h, w) or self.grid_x.device != z.device or self.grid_x.dtype != z.dtype:
            pos_y = torch.linspace(-1.0, 1.0, h, device=z.device, dtype=z.dtype)
            pos_x = torch.linspace(-1.0, 1.0, w, device=z.device, dtype=z.dtype)
            grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        else:
            grid_x = self.grid_x; grid_y = self.grid_y
        p = F.softmax((z / self.temperature).reshape(b, k, h * w), dim=-1).view(b, k, h, w)
        gx = grid_x.unsqueeze(0).unsqueeze(0); gy = grid_y.unsqueeze(0).unsqueeze(0)
        ux = torch.sum(p * gx, dim=(-2, -1))
        uy = torch.sum(p * gy, dim=(-2, -1))
        vx = torch.sum(p * (gx - ux.view(b, k, 1, 1)) ** 2, dim=(-2, -1))
        vy = torch.sum(p * (gy - uy.view(b, k, 1, 1)) ** 2, dim=(-2, -1))
        return torch.stack([ux, uy, vx, vy], dim=-1).view(b, k * 4)

SpatialSoftArgmax2d = MultiHeadSpatialSoftArgmax2d

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
        self.attn_conv = nn.Conv2d(128, 2, kernel_size=7, padding=3, bias=True)
        self.soft_argmax = MultiHeadSpatialSoftArgmax2d(num_heads=2, temperature=1.0, height=28, width=28)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.head_fc[-1].bias.data[2] = -2.0
        self.eye_fc = nn.Sequential(
            nn.Linear(136, 128),
            nn.SiLU()
        )
        self.ln_eye = nn.LayerNorm(128)
        self.gaze_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        self.gaze_out[-1].bias.data[2] = -2.0
        self.binoc_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        nn.init.zeros_(self.binoc_conv[0].weight)
        nn.init.zeros_(self.binoc_conv[0].bias)
        u7 = torch.linspace(-1.0, 1.0, 7); v7 = torch.linspace(-1.0, 1.0, 7)
        grid_y7, grid_x7 = torch.meshgrid(v7, u7, indexing="ij")
        self.register_buffer("grid_x7", grid_x7, persistent=False)
        self.register_buffer("grid_y7", grid_y7, persistent=False)
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
        attn_pooled = attn_weights.mean(dim=1, keepdim=True)
        attended = eye_map * attn_pooled
        eye_pooled = torch.flatten(self.pool(attended), 1)
        u0_x = torch.clamp(moments[:, 0], -0.78, 0.78)
        u0_y = torch.clamp(moments[:, 1], -0.84, 0.84)
        u1_x = torch.clamp(moments[:, 4], -0.78, 0.78)
        u1_y = torch.clamp(moments[:, 5], -0.84, 0.84)
        sx = 0.22; sy = 0.16
        if self.grid_x7 is None or self.grid_x7.shape != (7, 7) or self.grid_x7.device != eye_map.device or self.grid_x7.dtype != eye_map.dtype:
            u_t = torch.linspace(-1.0, 1.0, 7, device=eye_map.device, dtype=eye_map.dtype)
            v_t = torch.linspace(-1.0, 1.0, 7, device=eye_map.device, dtype=eye_map.dtype)
            gy7, gx7 = torch.meshgrid(v_t, u_t, indexing="ij")
        else:
            gx7 = self.grid_x7; gy7 = self.grid_y7
        g0_x = u0_x.view(b, 1, 1) + sx * gx7.unsqueeze(0)
        g0_y = u0_y.view(b, 1, 1) + sy * gy7.unsqueeze(0)
        grid0 = torch.stack([g0_x, g0_y], dim=-1)
        g1_x = u1_x.view(b, 1, 1) + sx * gx7.unsqueeze(0)
        g1_y = u1_y.view(b, 1, 1) + sy * gy7.unsqueeze(0)
        grid1 = torch.stack([g1_x, g1_y], dim=-1)
        f0 = F.grid_sample(eye_map, grid0, mode="bilinear", padding_mode="border", align_corners=False)
        f1 = F.grid_sample(eye_map, grid1, mode="bilinear", padding_mode="border", align_corners=False)
        f_binoc = torch.cat([f0, f1], dim=1)
        binoc_feat = torch.flatten(self.binoc_conv(f_binoc), 1)
        eye_fused = eye_pooled + binoc_feat
        eye_in = torch.cat([eye_fused, moments], dim=-1)
        eye_feat = self.ln_eye(self.eye_fc(eye_in))
        raw_gaze_eye = self.gaze_out(eye_feat)
        g_eye = raw_gaze_eye / torch.clamp(torch.norm(raw_gaze_eye, p=2, dim=-1, keepdim=True), min=1e-8)
        cy = torch.cos(hy_rad); sy = torch.sin(hy_rad)
        cp = torch.cos(hp_rad); sp = torch.sin(hp_rad)
        g_pred_x = cy * g_eye[:, 0] - sy * sp * g_eye[:, 1] + sy * cp * g_eye[:, 2]
        g_pred_y = cp * g_eye[:, 1] + sp * g_eye[:, 2]
        g_pred_z = -sy * g_eye[:, 0] - cy * sp * g_eye[:, 1] + cy * cp * g_eye[:, 2]
        gaze_unit = torch.stack([g_pred_x, g_pred_y, g_pred_z], dim=-1)
        gaze_unit = gaze_unit / torch.clamp(torch.norm(gaze_unit, p=2, dim=-1, keepdim=True), min=1e-8)
        gp_rad = torch.asin(torch.clamp(-gaze_unit[:, 1], -0.99999, 0.99999))
        gy_rad = torch.atan2(-gaze_unit[:, 0], -gaze_unit[:, 2])
        rad_to_deg = 180.0 / 3.141592653589793
        pitch_rest_rad = -8.99 * 3.141592653589793 / 180.0
        y_diff = torch.clamp(gy_rad, -1.45, 1.45)
        p_diff = torch.clamp(gp_rad - pitch_rest_rad, -1.45, 1.45)
        x_norm = torch.clamp(0.5 + 0.5 * torch.tan(y_diff) * 1.6, 0.0, 1.0)
        y_norm = torch.clamp(0.5 - 0.5 * torch.tan(p_diff) * 1.6, 0.0, 1.0)
        screen_coords = torch.stack([x_norm * 1440.0, y_norm * 900.0], dim=-1)
        var_mean = 0.25 * (moments[:, 2] + moments[:, 3] + moments[:, 6] + moments[:, 7])
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
            "attention_variance": var_mean,
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
        loss_h = 0.5 * self.smooth_l1(h_pred, h_gt).mean()
        moments = predictions_dict.get("moments")
        if moments is not None and moments.shape[-1] >= 8:
            mu_x0 = moments[:, 0]; mu_y0 = moments[:, 1]
            mu_x1 = moments[:, 4]; mu_y1 = moments[:, 5]
            loss_sep = 0.5 * F.relu(0.35 - (mu_x1 - mu_x0)).mean() + 0.1 * ((mu_y0 - mu_y1) ** 2).mean()
            total_loss = loss_aniso_vec + loss_cos + loss_h + loss_sep
        else:
            total_loss = loss_aniso_vec + loss_cos + loss_h
        return {
            "loss": total_loss,
            "loss_aniso_vec": loss_aniso_vec,
            "loss_cos": loss_cos,
            "loss_head": loss_h,
            "total_loss": total_loss
        }

ContinuousGazeLoss = StableVectorGazeLoss
