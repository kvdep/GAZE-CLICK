import ctypes, time, math
from typing import Tuple, Optional, List

class CursorController:
    def __init__(self, screen_w: int = 1920, screen_h: int = 1080, enable_control: bool = False) -> None:
        self.screen_w = screen_w; self.screen_h = screen_h
        self.enable_control = enable_control
        self.user32 = ctypes.windll.user32
        try:
            self.user32.SetProcessDPIAware()
        except Exception:
            pass
        self.last_pos = (screen_w / 2.0, screen_h / 2.0)
        self.dwell_start_time = None
        self.dwell_anchor = None
        self.dwell_radius_px = 35.0
        self.dwell_time_sec = 0.85
        self.last_click_time = 0.0
        self.click_cooldown_sec = 0.6
        self.r_capture = 60.0; self.r_release = 85.0; self.v_break = 350.0
        self.r_capture_base = 60.0; self.r_release_base = 85.0
        self.sigma_ema_sq = 0.020; self.sigma_thresh_sq = 0.040
        self.active_target = None; self.prev_raw_pos = None; self.prev_time = None
        self.registered_targets: List[Tuple[float, float, float]] = []
        self.set_taskbar_targets(screen_w, screen_h)

    def set_screen_resolution(self, w: int, h: int) -> None:
        self.screen_w = w; self.screen_h = h
        self.set_taskbar_targets(w, h)

    def set_enabled(self, enabled: bool) -> None:
        self.enable_control = enabled

    def toggle_control(self) -> bool:
        self.enable_control = not self.enable_control
        return self.enable_control

    def register_target(self, center_x: float, center_y: float, radius: float = 40.0) -> None:
        self.registered_targets.append((float(center_x), float(center_y), float(radius)))

    def set_taskbar_targets(self, screen_w: int, screen_h: int) -> None:
        self.registered_targets.clear()
        center_x = screen_w / 2.0; center_y = screen_h - 40.0
        icon_step = 82.0; icon_radius = 40.0
        for i in range(-12, 13):
            self.register_target(center_x + i * icon_step, center_y, radius=icon_radius)
        for i in range(1, 8):
            self.register_target(i * icon_step, center_y, radius=icon_radius)
        for i in range(1, 8):
            self.register_target(screen_w - i * icon_step, center_y, radius=icon_radius)

    def update_variance(self, var_val: float) -> float:
        self.sigma_ema_sq = 0.15 * max(0.0, float(var_val)) + 0.85 * self.sigma_ema_sq
        ratio = self.sigma_ema_sq / self.sigma_thresh_sq
        scale = 1.0 / (1.0 + ratio * ratio)
        self.r_capture = self.r_capture_base * scale
        self.r_release = self.r_release_base * scale
        return self.r_capture

    def apply_magnetic_attractor(self, raw_x: float, raw_y: float, variance: Optional[float] = None) -> Tuple[float, float]:
        if variance is not None:
            self.update_variance(variance)
        now = time.perf_counter()
        if self.prev_raw_pos is not None and self.prev_time is not None:
            dt = max(1e-4, now - self.prev_time)
            dist_move = math.hypot(raw_x - self.prev_raw_pos[0], raw_y - self.prev_raw_pos[1])
            v = dist_move / dt
        else:
            v = 0.0
        self.prev_raw_pos = (raw_x, raw_y)
        self.prev_time = now
        if v > self.v_break:
            self.active_target = None
            return raw_x, raw_y
        if self.active_target is not None:
            tx, ty, tr = self.active_target
            dist_active = math.hypot(raw_x - tx, raw_y - ty)
            if dist_active > self.r_release:
                self.active_target = None
        if self.active_target is None and len(self.registered_targets) > 0:
            best_target = None; min_dist = float("inf")
            for t in self.registered_targets:
                d = math.hypot(raw_x - t[0], raw_y - t[1])
                if d < self.r_capture and d < min_dist:
                    min_dist = d; best_target = t
            if best_target is not None:
                self.active_target = best_target
        if self.active_target is not None:
            tx, ty, tr = self.active_target
            dist_target = math.hypot(raw_x - tx, raw_y - ty)
            u = min(1.0, dist_target / self.r_release)
            alpha = (1.0 - u) ** 2 * (1.0 + 2.0 * u)
            snapped_x = tx + (1.0 - alpha) * (raw_x - tx)
            snapped_y = ty + (1.0 - alpha) * (raw_y - ty)
            return snapped_x, snapped_y
        return raw_x, raw_y

    def move_cursor(self, target_x: float, target_y: float, variance: Optional[float] = None) -> Tuple[int, int]:
        snapped_x, snapped_y = self.apply_magnetic_attractor(target_x, target_y, variance=variance)
        cx = int(max(0, min(self.screen_w - 1, round(snapped_x))))
        cy = int(max(0, min(self.screen_h - 1, round(snapped_y))))
        self.last_pos = (float(cx), float(cy))
        if self.enable_control:
            self.user32.SetCursorPos(cx, cy)
        return cx, cy

    def update_position(self, target_x: float, target_y: float, variance: Optional[float] = None) -> bool:
        cx, cy = self.move_cursor(target_x, target_y, variance=variance)
        return self.check_dwell_click(float(cx), float(cy))

    def check_dwell_click(self, current_x: float, current_y: float) -> bool:
        now = time.perf_counter()
        if now - self.last_click_time < self.click_cooldown_sec:
            self.dwell_anchor = None; self.dwell_start_time = None
            return False
        if self.dwell_anchor is None:
            self.dwell_anchor = (current_x, current_y)
            self.dwell_start_time = now
            return False
        dist = math.hypot(current_x - self.dwell_anchor[0], current_y - self.dwell_anchor[1])
        if dist > self.dwell_radius_px:
            self.dwell_anchor = (current_x, current_y)
            self.dwell_start_time = now
            return False
        if now - self.dwell_start_time >= self.dwell_time_sec:
            self.trigger_left_click()
            self.last_click_time = now
            self.dwell_anchor = None
            self.dwell_start_time = None
            return True
        return False

    def trigger_left_click(self) -> None:
        if self.enable_control:
            self.user32.mouse_event(0x0002, 0, 0, 0, 0)
            self.user32.mouse_event(0x0004, 0, 0, 0, 0)

    def trigger_right_click(self) -> None:
        if self.enable_control:
            self.user32.mouse_event(0x0008, 0, 0, 0, 0)
            self.user32.mouse_event(0x0010, 0, 0, 0, 0)
