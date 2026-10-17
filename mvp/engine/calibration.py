import os, json, math, numpy as np
from typing import List, Tuple, Dict, Any, Optional

class GazeCalibration:
    def __init__(self, screen_w: int = 2880, screen_h: int = 1800, profile_path: str = "mvp/calibration_profile.json") -> None:
        self.screen_w = screen_w; self.screen_h = screen_h
        self.profile_path = profile_path
        self.is_calibrated = False
        self.weights = None
        self.points_recorded = 0
        self.mean_residual_px = 0.0
        self.r2_score = 0.0
        self.user_distance_mm = 600.0
        self.physical_screen_w_mm = 300.0
        self.physical_screen_h_mm = 190.0
        self.resting_pitch_deg = -3.84
        self.resting_yaw_deg = -6.95
        self.gain_x = 1.85
        self.gain_y = 1.65
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.load_profile()

    def recenter(self, pitch_deg: float, yaw_deg: float, target_x: Optional[float] = None, target_y: Optional[float] = None, head_pitch_deg: float = 0.0, head_yaw_deg: float = 0.0) -> Tuple[float, float]:
        tx = float(target_x if target_x is not None else self.screen_w / 2.0)
        ty = float(target_y if target_y is not None else self.screen_h / 2.0)
        self.offset_x = 0.0; self.offset_y = 0.0
        cur_x, cur_y = self.predict(pitch_deg, yaw_deg, head_pitch_deg=head_pitch_deg, head_yaw_deg=head_yaw_deg)
        self.offset_x = tx - cur_x
        self.offset_y = ty - cur_y
        return self.offset_x, self.offset_y

    def center_at(self, pitch_deg: float, yaw_deg: float) -> None:
        self.resting_pitch_deg = float(pitch_deg)
        self.resting_yaw_deg = float(yaw_deg)

    def set_gain(self, gx: float, gy: float) -> None:
        self.gain_x = max(0.5, min(5.0, gx))
        self.gain_y = max(0.5, min(5.0, gy))

    def _extract_features(self, pitch_deg: float, yaw_deg: float, head_pitch_deg: float = 0.0, head_yaw_deg: float = 0.0) -> np.ndarray:
        return np.array([1.0, yaw_deg, pitch_deg, head_yaw_deg, head_pitch_deg], dtype=np.float64)

    def predict_uncalibrated(self, pitch_deg: float, yaw_deg: float) -> Tuple[float, float]:
        p_rad = math.radians(pitch_deg - self.resting_pitch_deg)
        y_rad = math.radians(yaw_deg - self.resting_yaw_deg)
        center_x = self.screen_w / 2.0; center_y = self.screen_h / 2.0
        px_per_mm_x = self.screen_w / max(self.physical_screen_w_mm, 1.0)
        px_per_mm_y = self.screen_h / max(self.physical_screen_h_mm, 1.0)
        dx_mm = -self.user_distance_mm * math.tan(y_rad) * self.gain_x
        dy_mm = -self.user_distance_mm * math.tan(p_rad) * self.gain_y
        raw_x = center_x + dx_mm * px_per_mm_x + self.offset_x
        raw_y = center_y + dy_mm * px_per_mm_y + self.offset_y
        clamped_x = float(np.clip(raw_x, 0.0, self.screen_w - 1.0))
        clamped_y = float(np.clip(raw_y, 0.0, self.screen_h - 1.0))
        return clamped_x, clamped_y

    def predict(self, pitch_deg: float, yaw_deg: float, head_pitch_deg: float = 0.0, head_yaw_deg: float = 0.0) -> Tuple[float, float]:
        if self.is_calibrated and self.weights is not None:
            if self.weights.shape[0] == 5:
                feat = self._extract_features(pitch_deg, yaw_deg, head_pitch_deg, head_yaw_deg)
            else:
                feat = np.array([1.0, yaw_deg, pitch_deg], dtype=np.float64)
            pred = feat @ self.weights
            pred_x = float(np.clip(pred[0] + self.offset_x, 0.0, self.screen_w - 1.0))
            pred_y = float(np.clip(pred[1] + self.offset_y, 0.0, self.screen_h - 1.0))
            return pred_x, pred_y
        return self.predict_uncalibrated(pitch_deg, yaw_deg)

    def update_baseline(self, cur_x: float, cur_y: float, target_x: float, target_y: float, alpha: float = 0.02) -> Tuple[float, float]:
        err_x = target_x - cur_x
        err_y = target_y - cur_y
        self.offset_x += alpha * err_x
        self.offset_y += alpha * err_y
        return self.offset_x, self.offset_y

    def fit(self, calibration_data: List[Dict[str, Any]], l2_reg: float = 1.0) -> Dict[str, float]:
        if len(calibration_data) < 5:
            return {"status": 0, "error": "Insufficient samples"}
        has_head = all("head_yaw_deg" in s and "head_pitch_deg" in s for s in calibration_data)
        if has_head:
            X_rows = []
            Y_rows = []
            for sample in calibration_data:
                feat = np.array([
                    1.0,
                    float(sample["yaw_deg"]),
                    float(sample["pitch_deg"]),
                    float(sample.get("head_yaw_deg", 0.0)),
                    float(sample.get("head_pitch_deg", 0.0))
                ], dtype=np.float64)
                X_rows.append(feat)
                Y_rows.append([float(sample["target_x"]), float(sample["target_y"])])
            X = np.array(X_rows, dtype=np.float64)
            Y = np.array(Y_rows, dtype=np.float64)
            Xx = X[:, [0, 1, 3]]
            Xy = X[:, [0, 2, 4]]
            reg_x = np.diag([0.0, l2_reg, l2_reg])
            reg_y = np.diag([0.0, l2_reg, l2_reg])
            wx = np.linalg.solve(Xx.T @ Xx + reg_x, Xx.T @ Y[:, 0])
            wy = np.linalg.solve(Xy.T @ Xy + reg_y, Xy.T @ Y[:, 1])
            self.weights = np.zeros((5, 2), dtype=np.float64)
            self.weights[0, 0] = wx[0]
            self.weights[1, 0] = wx[1]
            self.weights[3, 0] = wx[2]
            self.weights[0, 1] = wy[0]
            self.weights[2, 1] = wy[1]
            self.weights[4, 1] = wy[2]
        else:
            X_rows = []; Y_rows = []
            for sample in calibration_data:
                feat = np.array([1.0, sample["yaw_deg"], sample["pitch_deg"]], dtype=np.float64)
                X_rows.append(feat)
                Y_rows.append([sample["target_x"], sample["target_y"]])
            X = np.array(X_rows, dtype=np.float64)
            Y = np.array(Y_rows, dtype=np.float64)
            Xx = X[:, [0, 1]]
            Xy = X[:, [0, 2]]
            reg_x = np.diag([0.0, l2_reg])
            reg_y = np.diag([0.0, l2_reg])
            wx = np.linalg.solve(Xx.T @ Xx + reg_x, Xx.T @ Y[:, 0])
            wy = np.linalg.solve(Xy.T @ Xy + reg_y, Xy.T @ Y[:, 1])
            self.weights = np.zeros((3, 2), dtype=np.float64)
            self.weights[0, 0] = wx[0]
            self.weights[1, 0] = wx[1]
            self.weights[0, 1] = wy[0]
            self.weights[2, 1] = wy[1]
        self.offset_x = 0.0; self.offset_y = 0.0
        Y_pred = X @ self.weights
        residuals = np.linalg.norm(Y - Y_pred, axis=-1)
        self.mean_residual_px = float(np.mean(residuals))
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2) + 1e-8
        self.r2_score = float(1.0 - (ss_res / ss_tot))
        self.is_calibrated = True
        self.points_recorded = len(calibration_data)
        self.save_profile()
        return {
            "status": 1,
            "mean_residual_px": self.mean_residual_px,
            "r2_score": self.r2_score,
            "samples": self.points_recorded
        }

    def save_profile(self) -> None:
        if self.weights is None or not self.profile_path:
            return
        os.makedirs(os.path.dirname(self.profile_path) or ".", exist_ok=True)
        payload = {
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "weights": self.weights.tolist(),
            "mean_residual_px": self.mean_residual_px,
            "r2_score": self.r2_score,
            "points_recorded": self.points_recorded,
            "is_calibrated": self.is_calibrated
        }
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_profile(self) -> bool:
        if not self.profile_path or not os.path.exists(self.profile_path):
            return False
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.weights = np.array(payload["weights"], dtype=np.float64)
            if self.weights.shape[0] not in (3, 5):
                return False
            self.mean_residual_px = payload.get("mean_residual_px", 0.0)
            self.r2_score = payload.get("r2_score", 0.0)
            self.points_recorded = payload.get("points_recorded", 0)
            self.is_calibrated = payload.get("is_calibrated", True)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self.is_calibrated = False; self.weights = None
        self.points_recorded = 0; self.mean_residual_px = 0.0; self.r2_score = 0.0
        self.offset_x = 0.0; self.offset_y = 0.0
        if os.path.exists(self.profile_path):
            try:
                os.remove(self.profile_path)
            except Exception:
                pass
