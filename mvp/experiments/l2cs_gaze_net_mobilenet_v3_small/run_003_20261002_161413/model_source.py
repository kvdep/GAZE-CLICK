import numpy as np, torch, torch.nn as nn, torchvision.models as models
from typing import Dict, Any, Tuple
from models.base import BaseGazeModel

class L2CSGazeNet(BaseGazeModel):
    def __init__(self, backbone_name: str = "mobilenet_v3_small", num_bins: int = 90, max_angle_deg: float = 45.0, pretrained_backbone: bool = False, include_screen_head: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name; self.num_bins = num_bins; self.max_angle_deg = max_angle_deg; self.include_screen_head = include_screen_head
        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
            base_model = models.mobilenet_v3_small(weights=weights)
            self.feature_extractor = base_model.features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dimension = 576
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dimension = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.pitch_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_bins))
        self.yaw_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.num_bins))
        if self.include_screen_head:
            self.screen_linear_head = nn.Sequential(nn.Linear(self.feature_dimension, 128), nn.ReLU(), nn.Linear(128, 2))
        else:
            self.screen_linear_head = None
        bin_step_degrees = (2.0 * self.max_angle_deg) / float(self.num_bins)
        bin_center_degrees = -self.max_angle_deg + (torch.arange(self.num_bins, dtype=torch.float32) + 0.5) * bin_step_degrees
        bin_center_radians = bin_center_degrees * (3.141592653589793 / 180.0)
        self.register_buffer("bin_centers_radians", bin_center_radians)
        self.register_buffer("bin_centers_degrees", bin_center_degrees)

    def get_architecture_config(self) -> Dict[str, Any]:
        return {
            "model_type": "l2cs_gaze_net",
            "backbone_name": self.backbone_name,
            "num_bins": self.num_bins,
            "max_angle_deg": self.max_angle_deg,
            "feature_dimension": self.feature_dimension,
            "include_screen_head": self.include_screen_head
        }

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        features_map = self.feature_extractor(input_tensor)
        pooled_features = self.global_pool(features_map)
        flattened_features = torch.flatten(pooled_features, 1)
        pitch_logits = self.pitch_linear_head(flattened_features)
        yaw_logits = self.yaw_linear_head(flattened_features)
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
            "gaze_vector": predicted_gaze_vector
        }
        if self.screen_linear_head is not None:
            predicted_screen_coordinates = self.screen_linear_head(flattened_features)
            output_payload["screen_coordinates"] = predicted_screen_coordinates
        return output_payload

class L2CSLoss(nn.Module):
    def __init__(self, num_bins: int = 90, max_angle_deg: float = 45.0, regression_weight_beta: float = 1.0, angular_weight_gamma: float = 2.0, screen_weight_lambda: float = 0.01) -> None:
        super().__init__()
        self.num_bins = num_bins; self.max_angle_deg = max_angle_deg; self.bin_width_deg = (2.0 * max_angle_deg) / float(num_bins)
        self.beta = regression_weight_beta; self.gamma = angular_weight_gamma; self.screen_lambda = screen_weight_lambda
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mean_squared_error = nn.MSELoss()

    def _discretize_angles_to_bins(self, continuous_angles_rad: torch.Tensor) -> torch.Tensor:
        angles_degrees = continuous_angles_rad * (180.0 / 3.141592653589793)
        bin_indices = torch.floor((angles_degrees + self.max_angle_deg) / self.bin_width_deg).long()
        clamped_bins = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return clamped_bins

    def forward(self, predictions_dict: Dict[str, torch.Tensor], batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ground_truth_angles = batch_data["gaze_angles"]
        target_pitch_rad = ground_truth_angles[:, 0]; target_yaw_rad = ground_truth_angles[:, 1]
        target_pitch_bins = self._discretize_angles_to_bins(target_pitch_rad)
        target_yaw_bins = self._discretize_angles_to_bins(target_yaw_rad)
        pitch_classification_loss = self.cross_entropy_loss(predictions_dict["pitch_logits"], target_pitch_bins)
        yaw_classification_loss = self.cross_entropy_loss(predictions_dict["yaw_logits"], target_yaw_bins)
        total_classification_loss = pitch_classification_loss + yaw_classification_loss
        pitch_regression_loss = self.mean_squared_error(predictions_dict["pitch_rad"], target_pitch_rad)
        yaw_regression_loss = self.mean_squared_error(predictions_dict["yaw_rad"], target_yaw_rad)
        total_regression_loss = pitch_regression_loss + yaw_regression_loss
        pred_vector = predictions_dict["gaze_vector"]; target_vector = batch_data["gaze_vector"]
        dot_product = torch.sum(pred_vector * target_vector, dim=-1)
        cosine_proximity_loss = torch.mean(1.0 - torch.clamp(dot_product, -1.0, 1.0))
        total_loss = total_classification_loss + self.beta * total_regression_loss + self.gamma * cosine_proximity_loss
        loss_breakdown = {
            "total_loss": total_loss,
            "classification_loss": total_classification_loss,
            "regression_loss": total_regression_loss,
            "cosine_loss": cosine_proximity_loss
        }
        if "screen_coordinates" in predictions_dict and "screen_coords" in batch_data:
            has_screen_mask = batch_data.get("has_screen_coords", torch.zeros(len(target_pitch_rad), dtype=torch.bool, device=total_loss.device))
            if has_screen_mask.any():
                pred_screen = predictions_dict["screen_coordinates"][has_screen_mask]
                target_screen = batch_data["screen_coords"][has_screen_mask]
                screen_loss = torch.mean(torch.sqrt(torch.sum((pred_screen - target_screen) ** 2, dim=-1) + 1e-8))
                total_loss = total_loss + self.screen_lambda * screen_loss
                loss_breakdown["screen_loss"] = screen_loss
                loss_breakdown["total_loss"] = total_loss
        return loss_breakdown
