import math, torch, torch.nn as nn
from typing import Dict, Any, Tuple
from torchvision import models
from models.base import BaseGazeModel
from models.registry import ModelRegistry

class SiameseEyeEncoder(nn.Module):
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x); h = self.block1(h); h = self.block2(h); h = self.pool(h)
        h = torch.flatten(h, 1)
        return self.fc(h)

@ModelRegistry.register("dual_stream_gaze_net")
class DualStreamGazeNet(BaseGazeModel):
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        num_bins: int = 90,
        max_angle_deg: float = 45.0,
        num_head_bins: int = 90,
        max_head_angle_deg: float = 90.0,
        eye_feature_dim: int = 128,
        head_embed_dim: int = 128,
        pretrained_backbone: bool = True,
        include_screen_head: bool = True
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name; self.num_bins = num_bins; self.max_angle_deg = max_angle_deg
        self.num_head_bins = num_head_bins; self.max_head_angle_deg = max_head_angle_deg
        self.eye_feature_dim = eye_feature_dim; self.head_embed_dim = head_embed_dim
        self.pretrained_backbone = pretrained_backbone; self.include_screen_head = include_screen_head
        if backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
            bb = models.mobilenet_v3_large(weights=weights)
            self.face_extractor = bb.features
            self.face_feature_dim = 960
        elif backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            bb = models.mobilenet_v3_small(weights=weights)
            self.face_extractor = bb.features
            self.face_feature_dim = 576
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            bb = models.resnet18(weights=weights)
            self.face_extractor = nn.Sequential(*list(bb.children())[:-2])
            self.face_feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.face_dropout = nn.Dropout(0.3)
        self.gaze_dropout = nn.Dropout(0.3)
        self.head_pitch_linear_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.face_feature_dim, num_head_bins))
        self.head_yaw_linear_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.face_feature_dim, num_head_bins))
        self.head_orientation_embed = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.face_feature_dim, head_embed_dim))
        self.siamese_eye_encoder = SiameseEyeEncoder(out_features=eye_feature_dim)
        combined_gaze_feature_dim = (2 * eye_feature_dim) + head_embed_dim
        self.pitch_linear_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(combined_gaze_feature_dim, num_bins))
        self.yaw_linear_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(combined_gaze_feature_dim, num_bins))
        self.screen_linear_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(combined_gaze_feature_dim, 2)) if include_screen_head else None
        bin_width_deg = (2.0 * max_angle_deg) / float(num_bins)
        bin_centers_deg = torch.arange(num_bins, dtype=torch.float32) * bin_width_deg - max_angle_deg + (bin_width_deg / 2.0)
        self.register_buffer("bin_centers_radians", torch.deg2rad(bin_centers_deg))
        head_bin_width_deg = (2.0 * max_head_angle_deg) / float(num_head_bins)
        head_bin_centers_deg = torch.arange(num_head_bins, dtype=torch.float32) * head_bin_width_deg - max_head_angle_deg + (head_bin_width_deg / 2.0)
        self.register_buffer("head_bin_centers_radians", torch.deg2rad(head_bin_centers_deg))

    def spherical_angles_to_unit_vector(self, pitch_tensor: torch.Tensor, yaw_tensor: torch.Tensor) -> torch.Tensor:
        vx = -torch.cos(pitch_tensor) * torch.sin(yaw_tensor)
        vy = -torch.sin(pitch_tensor)
        vz = -torch.cos(pitch_tensor) * torch.cos(yaw_tensor)
        stacked_vector = torch.stack([vx, vy, vz], dim=-1)
        unit_norm = torch.norm(stacked_vector, p=2, dim=-1, keepdim=True)
        return stacked_vector / torch.clamp(unit_norm, min=1e-8)

    def extract_eye_patches(self, face_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        left_eye = face_image[:, :, 44:108, 27:91]
        right_eye = face_image[:, :, 44:108, 133:197]
        right_eye_flipped = torch.flip(right_eye, dims=[-1])
        return left_eye, right_eye_flipped

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = input_tensor.size(0)
        face_features_map = self.face_extractor(input_tensor)
        pooled_face = self.global_pool(face_features_map)
        flat_face = torch.flatten(pooled_face, 1)
        hp_logits = self.head_pitch_linear_head(flat_face); hy_logits = self.head_yaw_linear_head(flat_face)
        hp_probs = torch.softmax(hp_logits, dim=-1); hy_probs = torch.softmax(hy_logits, dim=-1)
        pred_hp_rad = torch.sum(hp_probs * self.head_bin_centers_radians, dim=-1)
        pred_hy_rad = torch.sum(hy_probs * self.head_bin_centers_radians, dim=-1)
        pred_head_vector = self.spherical_angles_to_unit_vector(pred_hp_rad, pred_hy_rad)
        head_embed = self.head_orientation_embed(flat_face)
        left_eye, right_eye_flipped = self.extract_eye_patches(input_tensor)
        eyes_stacked = torch.cat([left_eye, right_eye_flipped], dim=0)
        eyes_features = self.siamese_eye_encoder(eyes_stacked)
        feat_left = eyes_features[:batch_size]; feat_right = eyes_features[batch_size:]
        fused_eyes = torch.cat([feat_left, feat_right], dim=-1)
        gaze_feature = torch.cat([fused_eyes, head_embed], dim=-1)
        pitch_logits = self.pitch_linear_head(gaze_feature); yaw_logits = self.yaw_linear_head(gaze_feature)
        pitch_probs = torch.softmax(pitch_logits, dim=-1); yaw_probs = torch.softmax(yaw_logits, dim=-1)
        pred_pitch_rad = torch.sum(pitch_probs * self.bin_centers_radians, dim=-1)
        pred_yaw_rad = torch.sum(yaw_probs * self.bin_centers_radians, dim=-1)
        pred_gaze_vector = self.spherical_angles_to_unit_vector(pred_pitch_rad, pred_yaw_rad)
        output_payload = {
            "pitch_logits": pitch_logits, "yaw_logits": yaw_logits,
            "pitch_rad": pred_pitch_rad, "yaw_rad": pred_yaw_rad, "gaze_vector": pred_gaze_vector,
            "head_pitch_logits": hp_logits, "head_yaw_logits": hy_logits,
            "head_pitch_rad": pred_hp_rad, "head_yaw_rad": pred_hy_rad, "head_vector": pred_head_vector
        }
        if self.screen_linear_head is not None:
            output_payload["screen_coordinates"] = self.screen_linear_head(gaze_feature)
        return output_payload

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "model_type": "dual_stream_gaze_net",
            "backbone_name": self.backbone_name,
            "num_bins": self.num_bins,
            "max_angle_deg": self.max_angle_deg,
            "num_head_bins": self.num_head_bins,
            "max_head_angle_deg": self.max_head_angle_deg,
            "eye_feature_dim": self.eye_feature_dim,
            "head_embed_dim": self.head_embed_dim,
            "face_feature_dim": self.face_feature_dim,
            "pretrained_backbone": self.pretrained_backbone,
            "include_screen_head": self.include_screen_head
        }
