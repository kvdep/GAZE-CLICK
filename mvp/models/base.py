import abc, torch, torch.nn as nn
from typing import Dict, Any

class BaseGazeModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def get_architecture_config(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    @staticmethod
    def spherical_angles_to_unit_vector(pitch_rad: torch.Tensor, yaw_rad: torch.Tensor) -> torch.Tensor:
        cos_pitch = torch.cos(pitch_rad); sin_pitch = torch.sin(pitch_rad)
        cos_yaw = torch.cos(yaw_rad); sin_yaw = torch.sin(yaw_rad)
        vector_x = -cos_pitch * sin_yaw; vector_y = -sin_pitch; vector_z = -cos_pitch * cos_yaw
        stacked_vector = torch.stack([vector_x, vector_y, vector_z], dim=-1)
        vector_norm = torch.norm(stacked_vector, p=2, dim=-1, keepdim=True)
        return stacked_vector / torch.clamp(vector_norm, min=1e-8)

    @staticmethod
    def compute_angular_error_degrees(predicted_vector: torch.Tensor, ground_truth_vector: torch.Tensor) -> torch.Tensor:
        pred_norm = torch.norm(predicted_vector, p=2, dim=-1, keepdim=True)
        gt_norm = torch.norm(ground_truth_vector, p=2, dim=-1, keepdim=True)
        pred_unit = predicted_vector / torch.clamp(pred_norm, min=1e-8)
        gt_unit = ground_truth_vector / torch.clamp(gt_norm, min=1e-8)
        dot_product = torch.sum(pred_unit * gt_unit, dim=-1)
        clamped_dot = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        angular_radians = torch.acos(clamped_dot)
        return angular_radians * (180.0 / 3.141592653589793)

    @staticmethod
    def compute_pixel_mae(predicted_screen_px: torch.Tensor, ground_truth_screen_px: torch.Tensor, validity_mask: torch.Tensor) -> torch.Tensor:
        if not validity_mask.any():
            return torch.tensor(0.0, device=predicted_screen_px.device)
        differences = predicted_screen_px[validity_mask] - ground_truth_screen_px[validity_mask]
        euclidean_distances = torch.sqrt(torch.sum(differences ** 2, dim=-1) + 1e-8)
        return torch.mean(euclidean_distances)
