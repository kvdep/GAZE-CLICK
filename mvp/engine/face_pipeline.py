import os, sys, math, time, cv2, torch, importlib.util, numpy as np
from typing import Dict, Any, Optional, Tuple
from models.registry import ModelRegistry
import models.yolo26n_gaze_net

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

class GazeTrackingPipeline:
    def __init__(self, checkpoint_path: str = "mvp/experiments/yolo26n_gaze_net_yolo26n/run_014_20261017_121921/checkpoints/best_model.pt", yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx", device_str: Optional[str] = None) -> None:
        if device_str is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        checkpoint_path = _resolve_path(checkpoint_path)
        yunet_path = _resolve_path(yunet_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Gaze checkpoint not found: {checkpoint_path}")
        if not os.path.exists(yunet_path):
            raise FileNotFoundError(f"YuNet face model not found: {yunet_path}")
        if "resnet18" in checkpoint_path.lower():
            model_src_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "model_source.py")
            spec = importlib.util.spec_from_file_location("res_model_mod", model_src_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.model = mod.L2CSGazeNet(backbone_name="resnet18")
            chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(chk["model_state_dict"])
        else:
            self.model, _ = ModelRegistry.load_checkpoint(checkpoint_path, target_device="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        self.current_input_size = (320, 320)
        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.ema_pitch = None; self.ema_yaw = None; self.ema_alpha = 0.35
        with torch.no_grad():
            dummy_in = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=self.device)
            _ = self.model(dummy_in)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        det_w = 320
        det_h = int(det_w * (h / max(w, 1)))
        if self.current_input_size != (det_w, det_h):
            self.detector.setInputSize((det_w, det_h))
            self.current_input_size = (det_w, det_h)
        small_frame = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        _, faces = self.detector.detect(small_frame)
        if faces is None or len(faces) == 0:
            return {
                "face_detected": False,
                "latency_model_ms": 0.0,
                "gaze_pitch_deg": 0.0,
                "gaze_yaw_deg": 0.0,
                "head_pitch_deg": 0.0,
                "head_yaw_deg": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0]),
                "head_vector": np.array([0.0, 0.0, -1.0]),
                "face_bbox": None,
                "landmarks": None,
                "attention_variance": 0.040
            }
        scale_x = w / float(det_w)
        scale_y = h / float(det_h)
        best_face = faces[0]
        fx = float(best_face[0] * scale_x)
        fy = float(best_face[1] * scale_y)
        fw = float(best_face[2] * scale_x)
        fh = float(best_face[3] * scale_y)
        score = float(best_face[14])
        r_eye = (float(best_face[4] * scale_x), float(best_face[5] * scale_y))
        l_eye = (float(best_face[6] * scale_x), float(best_face[7] * scale_y))
        nose = (float(best_face[8] * scale_x), float(best_face[9] * scale_y))
        r_mouth = (float(best_face[10] * scale_x), float(best_face[11] * scale_y))
        l_mouth = (float(best_face[12] * scale_x), float(best_face[13] * scale_y))
        cx = int(fx + fw / 2.0); cy = int(fy + fh / 2.0)
        box_side = int(max(fw, fh) * 1.35)
        half_side = box_side // 2
        cx1 = int(max(0, cx - half_side)); cy1 = int(max(0, cy - half_side))
        cx2 = int(min(w, cx + half_side)); cy2 = int(min(h, cy + half_side))
        face_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            return {"face_detected": False, "latency_model_ms": 0.0}
        face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor_in = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        tensor_in = (tensor_in - self.mean_tensor) / self.std_tensor
        t_model_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor_in)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_model_ms = (time.perf_counter() - t_model_start) * 1000.0
        p_deg = float(outputs["pitch_rad"].item() * (180.0 / math.pi))
        y_deg = float(outputs["yaw_rad"].item() * (180.0 / math.pi))
        hp_deg = float(outputs.get("head_pitch_rad", torch.zeros(1)).item() * (180.0 / math.pi))
        hy_deg = float(outputs.get("head_yaw_rad", torch.zeros(1)).item() * (180.0 / math.pi))
        if self.ema_pitch is None:
            self.ema_pitch = p_deg; self.ema_yaw = y_deg
        else:
            self.ema_pitch = self.ema_alpha * p_deg + (1.0 - self.ema_alpha) * self.ema_pitch
            self.ema_yaw = self.ema_alpha * y_deg + (1.0 - self.ema_alpha) * self.ema_yaw
        gaze_v = outputs["gaze_vector"][0].detach().cpu().numpy()
        head_v = outputs.get("head_vector", torch.zeros(1, 3))[0].detach().cpu().numpy()
        var_val = float(outputs["attention_variance"][0].item()) if "attention_variance" in outputs else 0.020
        return {
            "face_detected": True,
            "latency_model_ms": latency_model_ms,
            "gaze_pitch_deg": self.ema_pitch,
            "gaze_yaw_deg": self.ema_yaw,
            "raw_pitch_deg": p_deg,
            "raw_yaw_deg": y_deg,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "gaze_vector": gaze_v,
            "head_vector": head_v,
            "face_bbox": (fx, fy, fw, fh),
            "face_crop_coords": (cx1, cy1, cx2, cy2),
            "face_score": score,
            "attention_variance": var_val,
            "landmarks": {
                "right_eye": r_eye,
                "left_eye": l_eye,
                "nose": nose,
                "right_mouth": r_mouth,
                "left_mouth": l_mouth
            }
        }
