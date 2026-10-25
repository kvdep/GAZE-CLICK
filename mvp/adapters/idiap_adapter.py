import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.models as models

from adapters.base import BaseGazeAdapter

class IdiapResNet18Gaze360Net(nn.Module):
    """
    Idiap Gaze360 ResNet-18 Dual-Stream Architecture.
    - Stream 1: Head Crop ResNet-18 backbone (512-dim feature)
    - Stream 2: Eye Crop ResNet-18 backbone (shared weights for Left & Right Eye, 512-dim each)
    - Concat: [Head (512), Left Eye (512), Right Eye (512)] -> 1536-dim
    - Linear Head: 1536 -> 512 -> 3 (3D unit gaze direction vector)
    """
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

    def forward(self, head: torch.Tensor, left_eye: torch.Tensor, right_eye: torch.Tensor) -> torch.Tensor:
        h = self.base_head(head).mean([2, 3])
        le = self.base_eye(left_eye).mean([2, 3])
        re = self.base_eye(right_eye).mean([2, 3])
        feat = torch.cat([h, le, re], dim=1)
        return self.head(feat)


class IdiapResNet18Adapter(BaseGazeAdapter):
    """
    Dedicated Adapter for Idiap ResNet-18 Gaze360 (Gupta et al. / Idiap MTGS).
    - Architecture: Dual-Stream ResNet-18 (Head + Left Eye + Right Eye).
    - Input: 224x224 RGB, ImageNet normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    - Output: 3D unit gaze vector [gx, gy, gz], decoded to pitch (elevation) and yaw (azimuth).
    """
    def __init__(self, name: str = "idiap_resnet18", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        model = IdiapResNet18Gaze360Net().to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            weight_file = checkpoint_path
        else:
            from huggingface_hub import hf_hub_download
            weight_file = hf_hub_download(repo_id="Idiap/resnet18-gaze360", filename="gaze360_resnet18.pt")

        raw_data = torch.load(weight_file, map_location="cpu", weights_only=False)
        sd = raw_data.get("model_state_dict", raw_data.get("state_dict", raw_data)) if isinstance(raw_data, dict) else raw_data
        model.load_state_dict(sd, strict=False)
        model.eval()
        self.model = model
        print(f"[IdiapResNet18Adapter] Successfully loaded Idiap ResNet-18 from '{os.path.basename(weight_file)}' onto {self.device}.")

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop_bgr, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        return (tensor - self.mean) / self.std

    def predict_gaze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_5pts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        fx, fy, fw, fh = face_bbox
        cx = int(fx + fw / 2.0); cy = int(fy + fh / 2.0)
        box_side = int(max(fw, fh) * 1.10)
        half_side = box_side // 2
        cx1 = max(0, cx - half_side); cy1 = max(0, cy - half_side)
        cx2 = min(w, cx + half_side); cy2 = min(h, cy + half_side)

        head_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if head_crop.size == 0:
            return {
                "pitch_deg": 0.0, "yaw_deg": 0.0,
                "pitch_rad": 0.0, "yaw_rad": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0]),
                "latency_model_ms": 0.0,
                "attention_variance": 0.040
            }

        # Extract eye crops
        eye_side = int(max(fw, fh) * 0.40)
        half_eye = eye_side // 2
        if landmarks_5pts is not None and len(landmarks_5pts) >= 2:
            # landmarks_5pts: [0]=right eye (image left), [1]=left eye (image right)
            rex, rey = int(landmarks_5pts[0][0]), int(landmarks_5pts[0][1])
            lex, ley = int(landmarks_5pts[1][0]), int(landmarks_5pts[1][1])
        else:
            # Fallback based on face box geometry
            rex, rey = int(cx - fw * 0.20), int(cy - fh * 0.15)
            lex, ley = int(cx + fw * 0.20), int(cy - fh * 0.15)

        rcx1, rcy1 = max(0, rex - half_eye), max(0, rey - half_eye)
        rcx2, rcy2 = min(w, rex + half_eye), min(h, rey + half_eye)
        r_crop = frame_bgr[rcy1:rcy2, rcx1:rcx2]
        if r_crop.size == 0:
            r_crop = head_crop

        lcx1, lcy1 = max(0, lex - half_eye), max(0, ley - half_eye)
        lcx2, lcy2 = min(w, lex + half_eye), min(h, ley + half_eye)
        l_crop = frame_bgr[lcy1:lcy2, lcx1:lcx2]
        if l_crop.size == 0:
            l_crop = head_crop

        head_tensor = self._preprocess_crop(head_crop)
        l_tensor = self._preprocess_crop(l_crop)
        r_tensor = self._preprocess_crop(r_crop)

        t0 = time.perf_counter()
        with torch.no_grad():
            vec = self.model(head_tensor, l_tensor, r_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            vec_norm = F.normalize(vec, p=2, dim=-1)
            gx = float(vec_norm[0, 0].item())
            gy = float(vec_norm[0, 1].item())
            gz = float(vec_norm[0, 2].item())

        pitch_rad = math.asin(-max(-1.0, min(1.0, gy)))
        yaw_rad = math.atan2(-gx, -gz)
        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = math.degrees(yaw_rad)

        return {
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "gaze_vector": np.array([gx, gy, gz], dtype=np.float32),
            "latency_model_ms": latency_ms,
            "attention_variance": 0.020
        }
