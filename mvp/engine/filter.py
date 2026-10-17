import math, time
from typing import Tuple

class LowPassFilter:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha; self.hat_x_prev = None

    def filter(self, x: float, alpha: float) -> float:
        if self.hat_x_prev is None:
            self.hat_x_prev = x; return x
        hat_x = alpha * x + (1.0 - alpha) * self.hat_x_prev
        self.hat_x_prev = hat_x
        return hat_x

class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.005, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(); self.dx_filter = LowPassFilter()
        self.t_prev = None; self.x_prev = None

    def _alpha(self, cutoff: float, d_t: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(d_t, 1e-6))

    def filter(self, x: float, timestamp: float) -> float:
        if self.t_prev is None:
            self.t_prev = timestamp; self.x_prev = x
            return self.x_filter.filter(x, 1.0)
        d_t = max(timestamp - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / d_t
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, d_t))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        hat_x = self.x_filter.filter(x, self._alpha(cutoff, d_t))
        self.t_prev = timestamp; self.x_prev = x
        return hat_x

class PointOneEuroFilter:
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.005, d_cutoff: float = 1.0, deadzone_radius_px: float = 8.0) -> None:
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.deadzone_r = deadzone_radius_px
        self.anchor_x = None
        self.anchor_y = None

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[float, float]:
        if self.anchor_x is None or self.anchor_y is None:
            self.anchor_x = x; self.anchor_y = y
        dist = math.hypot(x - self.anchor_x, y - self.anchor_y)
        if dist > self.deadzone_r:
            excess = dist - self.deadzone_r
            self.anchor_x += (x - self.anchor_x) * (excess / dist)
            self.anchor_y += (y - self.anchor_y) * (excess / dist)
        fx = self.filter_x.filter(self.anchor_x, timestamp)
        fy = self.filter_y.filter(self.anchor_y, timestamp)
        return fx, fy

    def reset(self) -> None:
        self.anchor_x = None
        self.anchor_y = None
        self.filter_x = OneEuroFilter(self.filter_x.min_cutoff, self.filter_x.beta, self.filter_x.d_cutoff)
        self.filter_y = OneEuroFilter(self.filter_y.min_cutoff, self.filter_y.beta, self.filter_y.d_cutoff)

    filter_point = filter
