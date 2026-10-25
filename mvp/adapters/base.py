import math, time, abc
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

class BaseGazeAdapter(abc.ABC):
    """
    Abstract Base Class for all model-specific gaze estimation adapters.
    Each model adapter encapsulates:
    1. Model-specific weight loading and architecture initialization.
    2. Model-specific face crop extraction, resolution, color space (RGB/BGR), and normalization.
    3. Model-specific forward pass execution and latency tracking.
    4. Model-specific output decoding, spherical coordinate transformations, and coordinate alignment.
    """
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self.model: Optional[nn.Module] = None

    @abc.abstractmethod
    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Download or resolve weights and initialize model in eval mode."""
        pass

    @abc.abstractmethod
    def predict_gaze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_5pts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run inference on a single detected face.
        Args:
            frame_bgr: Full raw camera frame (OpenCV BGR uint8).
            face_bbox: (x_min, y_min, width, height) in frame pixel coordinates.
            landmarks_5pts: (5, 2) array of [right_eye, left_eye, nose, right_mouth, left_mouth].
        Returns:
            Dictionary containing:
                pitch_deg (float): Vertical gaze elevation angle in degrees.
                yaw_deg (float): Horizontal gaze azimuth angle in degrees.
                pitch_rad (float): Vertical angle in radians.
                yaw_rad (float): Horizontal angle in radians.
                gaze_vector (np.ndarray): 3D unit gaze direction vector [x, y, z].
                latency_model_ms (float): Model execution time in milliseconds.
                attention_variance (float): Attention/uncertainty estimate.
        """
        pass
