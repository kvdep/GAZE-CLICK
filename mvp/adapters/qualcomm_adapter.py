import math, time, os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import onnxruntime as ort

from adapters.base import BaseGazeAdapter

class QualcommEyeGazeAdapter(BaseGazeAdapter):
    """
    Dedicated Adapter for Qualcomm EyeGaze (MobileOne-s0 EyeNet ONNX).
    - Architecture: MobileOne-s0 reparameterized (2.58M parameters).
    - Input: 160x96 grayscale eye crop with histogram equalization, float32 in [0, 1].
    - Output: gaze_pitchyaw [1, 2]: [pitch (rad), yaw (rad)].
    """
    def __init__(self, name: str = "qualcomm_eyegaze", device = None):
        super().__init__(name=name, device=device)
        self.session: Optional[ort.InferenceSession] = None
        self.crop_w = 160
        self.crop_h = 96

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        onnx_path = checkpoint_path or "mvp/models/weights/qualcomm_eyegaze/eyegaze-onnx-float/eyegaze.onnx"
        if not os.path.exists(onnx_path):
            import urllib.request, zipfile, io
            url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/eyegaze/releases/v0.65.0/eyegaze-onnx-float.zip"
            os.makedirs("mvp/models/weights/qualcomm_eyegaze", exist_ok=True)
            data = urllib.request.urlopen(url).read()
            z = zipfile.ZipFile(io.BytesIO(data))
            z.extractall("mvp/models/weights/qualcomm_eyegaze")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[QualcommEyeGazeAdapter] Successfully loaded EyeGaze ONNX from '{os.path.basename(onnx_path)}' with providers {self.session.get_providers()}.")

    def predict_gaze(
        self,
        frame_bgr: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks_5pts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        fx, fy, fw, fh = face_bbox

        # Locate eye center from 5-point landmarks if available, otherwise estimate from face bbox
        if landmarks_5pts is not None and len(landmarks_5pts) >= 2:
            # landmarks: [0: right eye, 1: left eye]
            rex, rey = landmarks_5pts[0]
            lex, ley = landmarks_5pts[1]
            eye_cx = int((rex + lex) / 2.0)
            eye_cy = int((rey + ley) / 2.0)
            eye_w = int(abs(lex - rex) * 1.6)
            eye_h = int(eye_w * 0.6)
        else:
            eye_cx = int(fx + fw / 2.0)
            eye_cy = int(fy + fh * 0.35)
            eye_w = int(fw * 0.7)
            eye_h = int(eye_w * 0.6)

        x1 = max(0, eye_cx - eye_w // 2)
        y1 = max(0, eye_cy - eye_h // 2)
        x2 = min(w, eye_cx + eye_w // 2)
        y2 = min(h, eye_cy + eye_h // 2)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "pitch_deg": 0.0, "yaw_deg": 0.0,
                "pitch_rad": 0.0, "yaw_rad": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0], dtype=np.float32),
                "latency_model_ms": 0.0,
                "attention_variance": 0.040
            }

        # Preprocess per Qualcomm EyeGazeApp: grayscale, resize 160x96, equalizeHist, norm to [0, 1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_res = cv2.resize(gray, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
        gray_eq = cv2.equalizeHist(gray_res)
        tensor_in = gray_eq.astype(np.float32) / 255.0
        tensor_in = np.expand_dims(tensor_in, axis=0) # [1, 96, 160]

        t0 = time.perf_counter()
        # Outputs: [heatmaps, landmarks, gaze_pitchyaw]
        outputs = self.session.run(["gaze_pitchyaw"], {"image": tensor_in})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        gaze = outputs[0][0] # [2] -> [pitch, yaw]
        pitch_rad = float(gaze[0])
        yaw_rad = float(gaze[1])

        pitch_deg = math.degrees(pitch_rad)
        yaw_deg = math.degrees(yaw_rad)

        gx = -math.cos(pitch_rad) * math.sin(yaw_rad)
        gy = -math.sin(pitch_rad)
        gz = -math.cos(pitch_rad) * math.cos(yaw_rad)

        return {
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad,
            "gaze_vector": np.array([gx, gy, gz], dtype=np.float32),
            "latency_model_ms": latency_ms,
            "attention_variance": 0.020
        }
