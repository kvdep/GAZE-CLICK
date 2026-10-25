import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch

from adapters.base import BaseGazeAdapter

def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    for root in [os.path.abspath("."), os.path.abspath("mvp"), os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]:
        cand1 = os.path.join(root, p)
        if os.path.exists(cand1):
            return os.path.abspath(cand1)
        if p.startswith("mvp/") or p.startswith("mvp\\"):
            cand2 = os.path.join(root, p[4:])
            if os.path.exists(cand2):
                return os.path.abspath(cand2)
    return p

class MediaPipeAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for MediaPipe Face Landmarker & Iris.
    - Architecture: MediaPipe Vision FaceLandmarker (BlazeFace + FaceMesh 478 landmarks + Blendshapes).
    - Features: Extracts geometric iris center offsets (ix, iy), blendshape differential eye movements,
      and 6-DOF head transformation matrix.
    """
    def __init__(self, name: str = "mediapipe", device: torch.device = torch.device("cpu")):
        super().__init__(name=name, device=device)
        self.landmarker = None
        self.frame_count = 0

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        import mediapipe as mp
        task_path = checkpoint_path or "mvp/models/weights/face_landmarker.task"
        resolved = _resolve_path(task_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"MediaPipe task asset not found: {resolved}")
        with open(resolved, "rb") as f:
            task_bytes = f.read()
        base_options = mp.tasks.BaseOptions(model_asset_buffer=task_bytes)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        print(f"[MediaPipeAdapter] Successfully initialized MediaPipe FaceLandmarker from '{resolved}'.")

    def predict_gaze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_5pts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1
        ts = int(self.frame_count * 16.666)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_start = time.perf_counter()
        res = self.landmarker.detect_for_video(mp_img, ts)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if not res.face_landmarks or not res.face_blendshapes or len(res.face_landmarks) == 0:
            return {
                "pitch_deg": 0.0, "yaw_deg": 0.0,
                "pitch_rad": 0.0, "yaw_rad": 0.0,
                "head_pitch_deg": 0.0, "head_yaw_deg": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0]),
                "head_vector": np.array([0.0, 0.0, -1.0]),
                "latency_model_ms": latency_ms,
                "attention_variance": 0.020,
                "ix": 0.5, "iy": -0.18,
                "eye_diff_x": 0.0, "eye_diff_y": 0.0
            }

        lm = res.face_landmarks[0]
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
        r_iris = lm[468]; l_iris = lm[473]
        r_c_in = lm[133]; r_c_out = lm[33]
        l_c_in = lm[362]; l_c_out = lm[263]
        r_w = max(1e-4, math.sqrt((r_c_out.x - r_c_in.x)**2 + (r_c_out.y - r_c_in.y)**2))
        l_w = max(1e-4, math.sqrt((l_c_out.x - l_c_in.x)**2 + (l_c_out.y - l_c_in.y)**2))
        r_ix = (r_iris.x - r_c_out.x) / r_w
        r_iy = (r_iris.y - (r_c_out.y + r_c_in.y) / 2.0) / r_w
        l_ix = (l_iris.x - l_c_in.x) / l_w
        l_iy = (l_iris.y - (l_c_out.y + l_c_in.y) / 2.0) / l_w
        ix = float((r_ix + l_ix) / 2.0)
        iy = float((r_iy + l_iy) / 2.0)
        eye_diff_x = float((bs.get("eyeLookOutLeft", 0.0) + bs.get("eyeLookInRight", 0.0)) - (bs.get("eyeLookInLeft", 0.0) + bs.get("eyeLookOutRight", 0.0)))
        eye_diff_y = float((bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) - (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)))
        mat = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else np.eye(4)
        hp_deg = float(math.degrees(math.asin(-max(-1.0, min(1.0, float(mat[1, 2]))))))
        hy_deg = float(math.degrees(math.atan2(float(mat[0, 2]), float(mat[2, 2]))))

        p_deg = float(-(iy + 0.18) * 75.0 + hp_deg * 0.4)
        y_deg = float(-(ix - 0.50) * 85.0 + hy_deg * 0.4)
        rad_p = math.radians(p_deg)
        rad_y = math.radians(y_deg)
        gz_v = np.array([-math.sin(rad_y), -math.sin(rad_p), -math.cos(rad_y) * math.cos(rad_p)], dtype=np.float32)
        hd_rad_p = math.radians(hp_deg)
        hd_rad_y = math.radians(hy_deg)
        hd_v = np.array([-math.sin(hd_rad_y), -math.sin(hd_rad_p), -math.cos(hd_rad_y) * math.cos(hd_rad_p)], dtype=np.float32)

        return {
            "pitch_deg": p_deg,
            "yaw_deg": y_deg,
            "pitch_rad": rad_p,
            "yaw_rad": rad_y,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "gaze_vector": gz_v,
            "head_vector": hd_v,
            "latency_model_ms": latency_ms,
            "attention_variance": 0.015,
            "ix": ix,
            "iy": iy,
            "eye_diff_x": eye_diff_x,
            "eye_diff_y": eye_diff_y
        }
