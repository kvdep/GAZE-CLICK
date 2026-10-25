import os, sys, math, time, cv2, torch, importlib.util, numpy as np
from typing import Dict, Any, Optional, Tuple
from models.registry import ModelRegistry
import models.yolo26n_gaze_net
from models.hf_models import load_gaze_model

# 3D canonical coordinates for 5 facial landmarks (right eye, left eye, nose, right mouth, left mouth)
_FACE_3D_PTS = np.array([
    [-38.0, -32.0, -40.0],
    [ 38.0, -32.0, -40.0],
    [  0.0,   0.0,   0.0],
    [-28.0,  42.0, -30.0],
    [ 28.0,  42.0, -30.0]
], dtype=np.float32)

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

def _safe_model_path(p: str) -> str:
    resolved = _resolve_path(p)
    try:
        resolved.encode("ascii")
        return resolved
    except UnicodeEncodeError:
        try:
            rel = os.path.relpath(resolved)
            rel.encode("ascii")
            if os.path.exists(rel):
                return rel
        except Exception:
            pass
        import tempfile, shutil
        t_dir = tempfile.gettempdir()
        t_path = os.path.join(t_dir, os.path.basename(resolved))
        if not os.path.exists(t_path) or os.path.getsize(t_path) != os.path.getsize(resolved):
            shutil.copyfile(resolved, t_path)
        return t_path

class GazeTrackingPipeline:
    def __init__(
        self,
        model_name: str = "l2cs",
        checkpoint_path: Optional[str] = None,
        yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx",
        device_str: Optional[str] = None
    ) -> None:
        if device_str is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        yunet_path = _safe_model_path(yunet_path)
        if not os.path.exists(yunet_path):
            raise FileNotFoundError(f"YuNet face model not found: {yunet_path}")

        # Check for legacy resnet18 checkpoint with model_source.py
        if checkpoint_path and "resnet18" in checkpoint_path.lower() and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "model_source.py")):
            model_src_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "model_source.py")
            spec = importlib.util.spec_from_file_location("res_model_mod", model_src_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.model = mod.L2CSGazeNet(backbone_name="resnet18").to(self.device)
            chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(chk["model_state_dict"])
            self.model.eval()
            self.model_info = {"name": "L2CS ResNet-18 (Legacy)", "type": "single_crop", "checkpoint_path": checkpoint_path}
            self.model_name = self.model_info["name"]
        else:
            self.model, self.model_info = load_gaze_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                device=self.device
            )
            self.model_name = self.model_info["name"]

        self.model_type = self.model_info.get("type", "single_crop")
        self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        self.current_input_size = (320, 320)
        self.mean_tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.ema_pitch = None; self.ema_yaw = None; self.ema_alpha = 0.35

        # Warmup forward pass
        with torch.no_grad():
            dummy_in = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=self.device)
            if self.model_type == "dual_stream":
                _ = self.model(dummy_in, dummy_in, dummy_in)
            else:
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
                "raw_pitch_deg": 0.0,
                "raw_yaw_deg": 0.0,
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
        box_side = int(max(fw, fh) * 1.10)
        half_side = box_side // 2
        cx1 = int(max(0, cx - half_side)); cy1 = int(max(0, cy - half_side))
        cx2 = int(min(w, cx + half_side)); cy2 = int(min(h, cy + half_side))
        face_crop = frame_bgr[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            return {"face_detected": False, "latency_model_ms": 0.0}

        crop_res = int(self.model_info.get("crop_size", 224))
        face_resized = cv2.resize(face_crop, (crop_res, crop_res), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        tensor_in = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        tensor_in = (tensor_in - self.mean_tensor) / self.std_tensor

        t_model_start = time.perf_counter()
        with torch.no_grad():
            if self.model_type == "dual_stream":
                eye_side = int(max(fw, fh) * 0.40)
                half_eye = eye_side // 2
                rex, rey = int(r_eye[0]), int(r_eye[1])
                rcx1, rcy1 = max(0, rex - half_eye), max(0, rey - half_eye)
                rcx2, rcy2 = min(w, rex + half_eye), min(h, rey + half_eye)
                r_crop = frame_bgr[rcy1:rcy2, rcx1:rcx2]
                if r_crop.size == 0:
                    r_crop = face_crop
                r_resized = cv2.resize(r_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                r_tensor = torch.from_numpy(cv2.cvtColor(r_resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                r_tensor = (r_tensor - self.mean_tensor) / self.std_tensor

                lex, ley = int(l_eye[0]), int(l_eye[1])
                lcx1, lcy1 = max(0, lex - half_eye), max(0, ley - half_eye)
                lcx2, lcy2 = min(w, lex + half_eye), min(h, ley + half_eye)
                l_crop = frame_bgr[lcy1:lcy2, lcx1:lcx2]
                if l_crop.size == 0:
                    l_crop = face_crop
                l_resized = cv2.resize(l_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                l_tensor = torch.from_numpy(cv2.cvtColor(l_resized, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                l_tensor = (l_tensor - self.mean_tensor) / self.std_tensor

                outputs = self.model(tensor_in, l_tensor, r_tensor)
            else:
                outputs = self.model(tensor_in)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_model_ms = (time.perf_counter() - t_model_start) * 1000.0

        p_deg = float(outputs["pitch_deg"].item() if "pitch_deg" in outputs else outputs["pitch_rad"].item() * (180.0 / math.pi))
        y_deg = float(outputs["yaw_deg"].item() if "yaw_deg" in outputs else outputs["yaw_rad"].item() * (180.0 / math.pi))

        if "head_pitch_rad" in outputs and "head_yaw_rad" in outputs and outputs["head_pitch_rad"].numel() > 0:
            hp_deg = float(outputs["head_pitch_rad"].item() * (180.0 / math.pi))
            hy_deg = float(outputs["head_yaw_rad"].item() * (180.0 / math.pi))
            head_v = outputs.get("head_vector", torch.zeros(1, 3))[0].detach().cpu().numpy()
        else:
            # Estimate head pose using 5-point solvePnP
            landmarks_2d = np.array([r_eye, l_eye, nose, r_mouth, l_mouth], dtype=np.float32)
            focal_len = float(w)
            cam_matrix = np.array([
                [focal_len, 0.0, w / 2.0],
                [0.0, focal_len, h / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            success, rvec, _ = cv2.solvePnP(_FACE_3D_PTS, landmarks_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if success:
                rot_m, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rot_m[0, 0]**2 + rot_m[1, 0]**2)
                if sy > 1e-6:
                    hp_deg = float(math.degrees(math.atan2(-rot_m[2, 1], rot_m[2, 2])))
                    hy_deg = float(math.degrees(math.atan2(rot_m[2, 0], sy)))
                else:
                    hp_deg = float(math.degrees(math.atan2(-rot_m[1, 2], rot_m[1, 1])))
                    hy_deg = float(math.degrees(math.atan2(rot_m[2, 0], sy)))
                hd_rad_p = math.radians(hp_deg)
                hd_rad_y = math.radians(hy_deg)
                head_v = np.array([-math.sin(hd_rad_y), -math.sin(hd_rad_p), -math.cos(hd_rad_y) * math.cos(hd_rad_p)], dtype=np.float32)
            else:
                hp_deg, hy_deg = 0.0, 0.0
                head_v = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        if self.ema_pitch is None:
            self.ema_pitch = p_deg; self.ema_yaw = y_deg
        else:
            self.ema_pitch = self.ema_alpha * p_deg + (1.0 - self.ema_alpha) * self.ema_pitch
            self.ema_yaw = self.ema_alpha * y_deg + (1.0 - self.ema_alpha) * self.ema_yaw

        gaze_v = outputs["gaze_vector"][0].detach().cpu().numpy()
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

class MediaPipeIrisPipeline:
    def __init__(self, task_path: str = "mvp/models/weights/face_landmarker.task") -> None:
        import mediapipe as mp
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
        self.frame_count = 0
        self.ema_pitch = None
        self.ema_yaw = None
        self.ema_alpha = 0.35
        self.model_name = "MediaPipe FaceLandmarker + Iris"
        self.model_info = {"name": self.model_name, "type": "mediapipe"}

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1
        ts = int(timestamp_ms if timestamp_ms is not None else self.frame_count * 16.666)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_start = time.perf_counter()
        res = self.landmarker.detect_for_video(mp_img, ts)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        if not res.face_landmarks or not res.face_blendshapes or len(res.face_landmarks) == 0:
            return {
                "face_detected": False,
                "latency_model_ms": latency_ms,
                "gaze_pitch_deg": 0.0,
                "gaze_yaw_deg": 0.0,
                "raw_pitch_deg": 0.0,
                "raw_yaw_deg": 0.0,
                "head_pitch_deg": 0.0,
                "head_yaw_deg": 0.0,
                "gaze_vector": np.array([0.0, 0.0, -1.0]),
                "head_vector": np.array([0.0, 0.0, -1.0]),
                "face_bbox": None,
                "face_crop_coords": None,
                "landmarks": None,
                "attention_variance": 0.020,
                "ix": 0.5,
                "iy": -0.18,
                "eye_diff_x": 0.0,
                "eye_diff_y": 0.0
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
        if self.ema_pitch is None:
            self.ema_pitch = p_deg; self.ema_yaw = y_deg
        else:
            self.ema_pitch = self.ema_alpha * p_deg + (1.0 - self.ema_alpha) * self.ema_pitch
            self.ema_yaw = self.ema_alpha * y_deg + (1.0 - self.ema_alpha) * self.ema_yaw
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        min_x = min(xs); max_x = max(xs)
        min_y = min(ys); max_y = max(ys)
        box_w = max_x - min_x; box_h = max_y - min_y
        r_eye_px = (float(r_iris.x * w), float(r_iris.y * h))
        l_eye_px = (float(l_iris.x * w), float(l_iris.y * h))
        nose_px = (float(lm[1].x * w), float(lm[1].y * h))
        r_mouth_px = (float(lm[61].x * w), float(lm[61].y * h))
        l_mouth_px = (float(lm[291].x * w), float(lm[291].y * h))
        rad_p = math.radians(self.ema_pitch)
        rad_y = math.radians(self.ema_yaw)
        gz_v = np.array([-math.sin(rad_y), -math.sin(rad_p), -math.cos(rad_y) * math.cos(rad_p)], dtype=np.float32)
        hd_rad_p = math.radians(hp_deg)
        hd_rad_y = math.radians(hy_deg)
        hd_v = np.array([-math.sin(hd_rad_y), -math.sin(hd_rad_p), -math.cos(hd_rad_y) * math.cos(hd_rad_p)], dtype=np.float32)
        return {
            "face_detected": True,
            "latency_model_ms": latency_ms,
            "gaze_pitch_deg": self.ema_pitch,
            "gaze_yaw_deg": self.ema_yaw,
            "raw_pitch_deg": p_deg,
            "raw_yaw_deg": y_deg,
            "head_pitch_deg": hp_deg,
            "head_yaw_deg": hy_deg,
            "gaze_vector": gz_v,
            "head_vector": hd_v,
            "face_bbox": (min_x, min_y, box_w, box_h),
            "face_crop_coords": (int(max(0, min_x)), int(max(0, min_y)), int(min(w, max_x)), int(min(h, max_y))),
            "face_score": 0.99,
            "attention_variance": 0.015,
            "ix": ix,
            "iy": iy,
            "eye_diff_x": eye_diff_x,
            "eye_diff_y": eye_diff_y,
            "landmarks": {
                "right_eye": r_eye_px,
                "left_eye": l_eye_px,
                "nose": nose_px,
                "right_mouth": r_mouth_px,
                "left_mouth": l_mouth_px
            }
        }

def create_gaze_pipeline(
    pipeline_type: Optional[str] = None,
    model_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    yunet_path: str = "mvp/models/weights/face_detection_yunet_2023mar.onnx",
    task_path: str = "mvp/models/weights/face_landmarker.task",
    device_str: Optional[str] = None
):
    selected_name = model_name or pipeline_type or "l2cs"
    if selected_name.lower() == "mediapipe":
        return MediaPipeIrisPipeline(task_path=task_path)
    return GazeTrackingPipeline(
        model_name=selected_name,
        checkpoint_path=checkpoint_path,
        yunet_path=yunet_path,
        device_str=device_str
    )
