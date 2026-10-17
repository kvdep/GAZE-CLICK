import ctypes, time, math
from typing import Tuple, Optional, List, Dict, Any

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
        self.dwell_radius_px = 40.0
        self.dwell_time_sec = 0.50
        self.last_click_time = 0.0
        self.click_cooldown_sec = 0.6
        self.r_capture = 40.0; self.r_release = 55.0; self.v_break = 350.0
        self.r_capture_base = 40.0; self.r_release_base = 55.0
        self.sigma_ema_sq = 0.020; self.sigma_thresh_sq = 0.040
        self.active_target = None; self.prev_raw_pos = None; self.prev_time = None
        self.registered_targets: List[Tuple[float, float, float]] = []
        self.set_taskbar_targets(screen_w, screen_h)
        self.is_magnifier_active = False
        self.magnifier_cluster: List[Dict[str, Any]] = []
        self.taskbar_entry_time = None
        self.taskbar_entry_pos = None
        self.magnifier_dwell_start = None
        self.magnifier_selected_target = None

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
            dist_move = 0.0
        if v < 150.0 and self.prev_raw_pos is not None:
            gain = 0.35
            filtered_x = self.prev_raw_pos[0] + gain * (raw_x - self.prev_raw_pos[0])
            filtered_y = self.prev_raw_pos[1] + gain * (raw_y - self.prev_raw_pos[1])
        else:
            filtered_x = raw_x
            filtered_y = raw_y
        self.prev_raw_pos = (filtered_x, filtered_y)
        self.prev_time = now
        if v > self.v_break:
            self.active_target = None
            return filtered_x, filtered_y
        if self.active_target is not None:
            tx, ty, tr = self.active_target
            dist_active = math.hypot(filtered_x - tx, filtered_y - ty)
            if dist_active > self.r_release:
                self.active_target = None
        if self.active_target is None and len(self.registered_targets) > 0:
            best_target = None; min_dist = float("inf")
            for t in self.registered_targets:
                d = math.hypot(filtered_x - t[0], filtered_y - t[1])
                if d < self.r_capture and d < min_dist:
                    min_dist = d; best_target = t
            if best_target is not None:
                self.active_target = best_target
        if self.active_target is not None:
            tx, ty, tr = self.active_target
            dist_target = math.hypot(filtered_x - tx, filtered_y - ty)
            u = min(1.0, dist_target / self.r_release)
            alpha = (1.0 - u) ** 2 * (1.0 + 2.0 * u)
            snapped_x = tx + (1.0 - alpha) * (filtered_x - tx)
            snapped_y = ty + (1.0 - alpha) * (filtered_y - ty)
            return snapped_x, snapped_y
        return filtered_x, filtered_y

    def activate_magnifier(self, focus_x: float, base_y: float) -> None:
        self.is_magnifier_active = True
        self.magnifier_cluster.clear()
        self.magnifier_dwell_start = None
        self.magnifier_selected_target = None
        icon_step = 82.0
        center_x = self.screen_w / 2.0
        k = round((focus_x - center_x) / icon_step)
        mag_w = 130.0
        mag_h = 110.0
        mag_y = self.screen_h - 180.0
        indices = [k - 2, k - 1, k, k + 1, k + 2]
        for idx, i in enumerate(indices):
            orig_x = center_x + i * icon_step
            orig_y = base_y
            offset_mag_x = focus_x + (idx - 2) * (mag_w + 12.0)
            self.magnifier_cluster.append({
                "index": i,
                "orig_x": float(orig_x),
                "orig_y": float(orig_y),
                "mag_x": float(offset_mag_x),
                "mag_y": float(mag_y),
                "mag_w": float(mag_w),
                "mag_h": float(mag_h),
                "mag_radius": float(mag_w / 2.0)
            })

    def deactivate_magnifier(self) -> None:
        self.is_magnifier_active = False
        self.magnifier_cluster.clear()
        self.magnifier_dwell_start = None
        self.magnifier_selected_target = None
        self.taskbar_entry_time = None
        self.taskbar_entry_pos = None

    def check_taskbar_magnifier(self, raw_x: float, raw_y: float) -> Tuple[bool, Optional[Tuple[float, float]]]:
        now = time.perf_counter()
        taskbar_top_y = self.screen_h - 110.0
        if not self.is_magnifier_active:
            if raw_y >= taskbar_top_y:
                if self.taskbar_entry_time is None:
                    self.taskbar_entry_time = now
                    self.taskbar_entry_pos = (raw_x, raw_y)
                else:
                    dist = math.hypot(raw_x - self.taskbar_entry_pos[0], raw_y - self.taskbar_entry_pos[1])
                    if dist > 100.0:
                        self.taskbar_entry_time = now
                        self.taskbar_entry_pos = (raw_x, raw_y)
                    elif now - self.taskbar_entry_time >= 0.20:
                        self.activate_magnifier(raw_x, self.screen_h - 40.0)
                        return True, None
            else:
                self.taskbar_entry_time = None
                self.taskbar_entry_pos = None
            return False, None
        else:
            if raw_y < self.screen_h - 320.0 or raw_y > self.screen_h + 50.0:
                self.deactivate_magnifier()
                return False, None
            best_icon = None
            min_dist = float("inf")
            for item in self.magnifier_cluster:
                d = math.hypot(raw_x - item["mag_x"], raw_y - item["mag_y"])
                if d < item["mag_radius"] and d < min_dist:
                    min_dist = d
                    best_icon = item
            if best_icon is not None:
                if self.magnifier_selected_target != best_icon:
                    self.magnifier_selected_target = best_icon
                    self.magnifier_dwell_start = now
                elif self.magnifier_dwell_start is not None and now - self.magnifier_dwell_start >= self.dwell_time_sec:
                    cx = int(best_icon["orig_x"])
                    cy = int(best_icon["orig_y"])
                    self.deactivate_magnifier()
                    if self.enable_control:
                        self.user32.SetCursorPos(cx, cy)
                        self.trigger_left_click()
                    return False, (float(cx), float(cy))
                return True, (best_icon["mag_x"], best_icon["mag_y"])
            else:
                self.magnifier_selected_target = None
                self.magnifier_dwell_start = None
                return True, None

    def move_cursor(self, target_x: float, target_y: float, variance: Optional[float] = None) -> Tuple[int, int]:
        mag_active, mag_pos = self.check_taskbar_magnifier(target_x, target_y)
        if mag_active:
            if mag_pos is not None:
                cx = int(max(0, min(self.screen_w - 1, round(mag_pos[0]))))
                cy = int(max(0, min(self.screen_h - 1, round(mag_pos[1]))))
            else:
                cx = int(max(0, min(self.screen_w - 1, round(target_x))))
                cy = int(max(0, min(self.screen_h - 1, round(target_y))))
            self.last_pos = (float(cx), float(cy))
            if self.enable_control:
                self.user32.SetCursorPos(cx, cy)
            return cx, cy
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
