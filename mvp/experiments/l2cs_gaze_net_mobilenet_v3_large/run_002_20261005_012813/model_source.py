import numpy as np, torch, torch.nn as nn, torchvision.models as models
from typing import Dict, Any, Tuple
from models.base import BaseGazeModel

class L2CSGazeNet(BaseGazeModel):
    def __init__(self, backbone_name: str = "mobilenet_v3_small", num_bins: int = 90, max_angle_deg: float = 45.0, num_head_bins: int = 90, max_head_angle_deg: float = 90.0, pretrained_backbone: bool = True, include_screen_head: bool = True, use_pose_conditioning: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name; self.num_bins = num_bins; self.max_angle_deg = max_angle_deg
        self.num_head_bins = num_head_bins; self.max_head_angle_deg = max_head_angle_deg
        self.pretrained_backbone = pretrained_backbone; self.include_screen_head = include_screen_head
        self.use_pose_conditioning = use_pose_conditioning
        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            base_model = models.mobilenet_v3_small(weights=weights)
            self.feature_extractor = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dimension = 576
        elif backbone_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
            base_model = models.mobilenet_v3_large(weights=weights)
            self.feature_extractor = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dimension = 960
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dimension = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.head_pitch_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_head_bins))
        self.head_yaw_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_head_bins))
        if self.use_pose_conditioning:
            self.pose_embedding_layer = nn.Sequential(nn.Linear(self.feature_dimension, 128), nn.ReLU(), nn.Dropout(0.1))
            gaze_in_dim = self.feature_dimension + 128
        else:
            self.pose_embedding_layer = None
            gaze_in_dim = self.feature_dimension
        self.pitch_linear_head = nn.Sequential(nn.Linear(gaze_in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_bins))
        self.yaw_linear_head = nn.Sequential(nn.Linear(gaze_in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_bins))
        if self.include_screen_head:
            self.screen_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 128), nn.ReLU(), nn.Linear(128, 2))
        else:
            self.screen_linear_head = None
        bin_step_degrees = (2.0 * self.max_angle_deg) / float(self.num_bins)
        bin_center_degrees = -self.max_angle_deg + (torch.arange(self.num_bins, dtype=torch.float32) + 0.5) * bin_step_degrees
        bin_center_radians = bin_center_degrees * (3.141592653589793 / 180.0)
        self.register_buffer("bin_centers_radians", bin_center_radians)
        self.register_buffer("bin_centers_degrees", bin_center_degrees)
        head_bin_step_deg = (2.0 * self.max_head_angle_deg) / float(self.num_head_bins)
        head_bin_center_deg = -self.max_head_angle_deg + (torch.arange(self.num_head_bins, dtype=torch.float32) + 0.5) * head_bin_step_deg
        head_bin_center_rad = head_bin_center_deg * (3.141592653589793 / 180.0)
        self.register_buffer("head_bin_centers_radians", head_bin_center_rad)
        self.register_buffer("head_bin_centers_degrees", head_bin_center_deg)

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "model_type": "l2cs_gaze_net",
            "backbone_name": self.backbone_name,
            "num_bins": self.num_bins,
            "max_angle_deg": self.max_angle_deg,
            "num_head_bins": self.num_head_bins,
            "max_head_angle_deg": self.max_head_angle_deg,
            "feature_dimension": self.feature_dimension,
            "pretrained_backbone": self.pretrained_backbone,
            "include_screen_head": self.include_screen_head,
            "use_pose_conditioning": self.use_pose_conditioning
        }

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        features_map = self.feature_extractor(input_tensor)
        pooled_features = self.global_pool(features_map)
        flattened_features = torch.flatten(pooled_features, 1)
        head_pitch_logits = self.head_pitch_linear_head(flattened_features)
        head_yaw_logits = self.head_yaw_linear_head(flattened_features)
        head_pitch_probabilities = torch.softmax(head_pitch_logits, dim=-1)
        head_yaw_probabilities = torch.softmax(head_yaw_logits, dim=-1)
        predicted_head_pitch_rad = torch.sum(head_pitch_probabilities * self.head_bin_centers_radians, dim=-1)
        predicted_head_yaw_rad = torch.sum(head_yaw_probabilities * self.head_bin_centers_radians, dim=-1)
        predicted_head_vector = self.spherical_angles_to_unit_vector(predicted_head_pitch_rad, predicted_head_yaw_rad)
        if self.use_pose_conditioning and self.pose_embedding_layer is not None:
            pose_embed = self.pose_embedding_layer(flattened_features)
            gaze_input_features = torch.cat([flattened_features, pose_embed], dim=-1)
        else:
            gaze_input_features = flattened_features
        pitch_logits = self.pitch_linear_head(gaze_input_features)
        yaw_logits = self.yaw_linear_head(gaze_input_features)
        pitch_probabilities = torch.softmax(pitch_logits, dim=-1)
        yaw_probabilities = torch.softmax(yaw_logits, dim=-1)
        predicted_pitch_rad = torch.sum(pitch_probabilities * self.bin_centers_radians, dim=-1)
        predicted_yaw_rad = torch.sum(yaw_probabilities * self.bin_centers_radians, dim=-1)
        predicted_gaze_vector = self.spherical_angles_to_unit_vector(predicted_pitch_rad, predicted_yaw_rad)
        output_payload = {
            "pitch_logits": pitch_logits,
            "yaw_logits": yaw_logits,
            "pitch_rad": predicted_pitch_rad,
            "yaw_rad": predicted_yaw_rad,
            "gaze_vector": predicted_gaze_vector,
            "head_pitch_logits": head_pitch_logits,
            "head_yaw_logits": head_yaw_logits,
            "head_pitch_rad": predicted_head_pitch_rad,
            "head_yaw_rad": predicted_head_yaw_rad,
            "head_vector": predicted_head_vector
        }
        if self.screen_linear_head is not None:
            predicted_screen_coordinates = self.screen_linear_head(flattened_features)
            output_payload["screen_coordinates"] = predicted_screen_coordinates
        return output_payload

class L2CSLoss(nn.Module):
    def __init__(self, num_bins: int = 90, max_angle_deg: float = 45.0, num_head_bins: int = 90, max_head_angle_deg: float = 90.0, regression_weight_beta: float = 0.5, angular_weight_gamma: float = 2.0, angular_deg_lambda: float = 1.0, screen_weight_lambda: float = 1.0, head_weight_lambda: float = 0.5, screen_norm_wh: Tuple[float, float] = (1920.0, 1080.0)) -> None:
        super().__init__()
        self.num_bins = num_bins; self.max_angle_deg = max_angle_deg; self.bin_width_deg = (2.0 * max_angle_deg) / float(num_bins)
        self.num_head_bins = num_head_bins; self.max_head_angle_deg = max_head_angle_deg; self.head_bin_width_deg = (2.0 * max_head_angle_deg) / float(num_head_bins)
        self.beta = regression_weight_beta; self.gamma = angular_weight_gamma; self.angular_deg_lambda = angular_deg_lambda; self.screen_lambda = screen_weight_lambda; self.head_lambda = head_weight_lambda
        self.screen_norm_wh = screen_norm_wh
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="mean", beta=1.0)

    def _discretize_angles_to_bins(self, continuous_angles_rad: torch.Tensor) -> torch.Tensor:
        angles_degrees = continuous_angles_rad * (180.0 / 3.141592653589793)
        bin_indices = torch.floor((angles_degrees + self.max_angle_deg) / self.bin_width_deg).long()
        clamped_bins = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return clamped_bins

    def _discretize_head_angles_to_bins(self, continuous_angles_rad: torch.Tensor) -> torch.Tensor:
        angles_degrees = continuous_angles_rad * (180.0 / 3.141592653589793)
        bin_indices = torch.floor((angles_degrees + self.max_head_angle_deg) / self.head_bin_width_deg).long()
        clamped_bins = torch.clamp(bin_indices, 0, self.num_head_bins - 1)
        return clamped_bins

    def forward(self, predictions_dict: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rad_to_deg = 180.0 / 3.141592653589793
        ground_truth_angles = batch_data["gaze_angles"]
        target_pitch_rad = ground_truth_angles[:, 0]; target_yaw_rad = ground_truth_angles[:, 1]
        target_pitch_bins = self._discretize_angles_to_bins(target_pitch_rad)
        target_yaw_bins = self._discretize_angles_to_bins(target_yaw_rad)
        pitch_classification_loss = self.cross_entropy_loss(predictions_dict["pitch_logits"], target_pitch_bins)
        yaw_classification_loss = self.cross_entropy_loss(predictions_dict["yaw_logits"], target_yaw_bins)
        gaze_classification_loss = pitch_classification_loss + yaw_classification_loss
        pred_pitch_deg = predictions_dict["pitch_rad"] * rad_to_deg; pred_yaw_deg = predictions_dict["yaw_rad"] * rad_to_deg
        target_pitch_deg = target_pitch_rad * rad_to_deg; target_yaw_deg = target_yaw_rad * rad_to_deg
        pitch_reg_loss = self.smooth_l1_loss(pred_pitch_deg, target_pitch_deg)
        yaw_reg_loss = self.smooth_l1_loss(pred_yaw_deg, target_yaw_deg)
        gaze_regression_loss = pitch_reg_loss + yaw_reg_loss
        gaze_l2cs_loss = gaze_classification_loss + 0.5 * gaze_regression_loss

        head_angles = batch_data.get("head_angles", ground_truth_angles)
        target_head_pitch_rad = head_angles[:, 0]; target_head_yaw_rad = head_angles[:, 1]
        target_head_pitch_bins = self._discretize_head_angles_to_bins(target_head_pitch_rad)
        target_head_yaw_bins = self._discretize_head_angles_to_bins(target_head_yaw_rad)
        head_pitch_ce = self.cross_entropy_loss(predictions_dict["head_pitch_logits"], target_head_pitch_bins)
        head_yaw_ce = self.cross_entropy_loss(predictions_dict["head_yaw_logits"], target_head_yaw_bins)
        head_classification_loss = head_pitch_ce + head_yaw_ce
        pred_hp_deg = predictions_dict["head_pitch_rad"] * rad_to_deg; pred_hy_deg = predictions_dict["head_yaw_rad"] * rad_to_deg
        target_hp_deg = target_head_pitch_rad * rad_to_deg; target_hy_deg = target_head_yaw_rad * rad_to_deg
        head_pitch_reg = self.smooth_l1_loss(pred_hp_deg, target_hp_deg)
        head_yaw_reg = self.smooth_l1_loss(pred_hy_deg, target_hy_deg)
        head_regression_loss = head_pitch_reg + head_yaw_reg
        head_l2cs_loss = head_classification_loss + 0.5 * head_regression_loss

        pred_vector = predictions_dict["gaze_vector"]; target_vector = batch_data["gaze_vector"]
        pred_norm = torch.norm(pred_vector, p=2, dim=-1, keepdim=True); target_norm = torch.norm(target_vector, p=2, dim=-1, keepdim=True)
        pred_unit = pred_vector / torch.clamp(pred_norm, min=1e-8); target_unit = target_vector / torch.clamp(target_norm, min=1e-8)
        dot_product = torch.sum(pred_unit * target_unit, dim=-1)
        clamped_dot = torch.clamp(dot_product, -0.99999, 0.99999)
        angular_deg_loss = torch.mean(torch.acos(clamped_dot) * (180.0 / 3.141592653589793))
        cosine_proximity_loss = torch.mean(1.0 - torch.clamp(dot_product, -1.0, 1.0))

        screen_loss = torch.tensor(0.0, device=target_pitch_rad.device)
        if "screen_coordinates" in predictions_dict and "screen_coords" in batch_data:
            has_screen_mask = batch_data.get("has_screen_coords", torch.zeros(len(target_pitch_rad), dtype=torch.bool, device=target_pitch_rad.device))
            if has_screen_mask.any():
                pred_screen = predictions_dict["screen_coordinates"][has_screen_mask]
                target_screen = batch_data["screen_coords"][has_screen_mask]
                norm_factors = torch.tensor(self.screen_norm_wh, dtype=torch.float32, device=pred_screen.device)
                norm_pred = pred_screen / norm_factors
                norm_target = target_screen / norm_factors
                screen_loss = self.smooth_l1_loss(norm_pred, norm_target)

        total_loss = gaze_l2cs_loss + self.head_lambda * head_l2cs_loss + self.angular_deg_lambda * angular_deg_loss + self.gamma * cosine_proximity_loss + self.screen_lambda * screen_loss
        loss_breakdown = {
            "total_loss": total_loss,
            "classification_loss": gaze_classification_loss,
            "regression_loss": gaze_regression_loss,
            "gaze_l2cs_loss": gaze_l2cs_loss,
            "head_l2cs_loss": head_l2cs_loss,
            "angular_deg_loss": angular_deg_loss,
            "cosine_loss": cosine_proximity_loss,
            "screen_loss": screen_loss
        }
        return loss_breakdown
