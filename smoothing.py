import math

import numpy as np

import config


class EMAFilter:
    """Exponential moving average filter to reduce cursor jitter."""

    def __init__(self, alpha: float = config.EMA_ALPHA):
        self.alpha = alpha
        self._value: np.ndarray | None = None

    def reset(self) -> None:
        self._value = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        current = np.array(point, dtype=float)
        if self._value is None:
            self._value = current
        else:
            self._value = self.alpha * current + (1 - self.alpha) * self._value
        return float(self._value[0]), float(self._value[1])


class HandDeadzoneFilter:
    """Ignore sub-threshold hand jitter so a still hand does not drift the cursor."""

    def __init__(self, threshold: float = config.HAND_DEADZONE_NORM):
        self.threshold = threshold
        self._locked: tuple[float, float] | None = None

    def reset(self) -> None:
        self._locked = None

    def apply(self, point: tuple[float, float]) -> tuple[float, float]:
        if self._locked is None:
            self._locked = point
            return point

        dx = point[0] - self._locked[0]
        dy = point[1] - self._locked[1]
        if (dx * dx + dy * dy) < self.threshold ** 2:
            return self._locked

        self._locked = point
        return point


def clamp_screen_coords(
    screen_x: float,
    screen_y: float,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> tuple[int, int]:
    if screen_width is None or screen_height is None:
        screen_width, screen_height = config.refresh_screen_size()
    sx = int(np.clip(screen_x, 0, screen_width - 1))
    sy = int(np.clip(screen_y, 0, screen_height - 1))
    return sx, sy


def compute_cal_bounds(
    hits: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Hand-space bounds from 4 calibration points, with margin buffer."""
    margin = config.CALIBRATION_MARGIN_NORM
    xs = [h[0] for h in hits]
    ys = [h[1] for h in hits]
    x1 = min(xs) + margin
    x2 = max(xs) - margin
    y1 = min(ys) + margin
    y2 = max(ys) - margin
    if x2 - x1 < 0.05:
        x1, x2 = min(xs), max(xs)
    if y2 - y1 < 0.05:
        y1, y2 = min(ys), max(ys)
    return x1, y1, x2, y2


def map_hand_region(norm_x: float, norm_y: float) -> tuple[float, float]:
    """Map camera region (normalized) to 0-1 screen space via linear interp."""
    x0 = config.HAND_REGION_X_MIN
    x1 = config.HAND_REGION_X_MAX
    y0 = config.HAND_REGION_Y_MIN
    y1 = config.HAND_REGION_Y_MAX
    if x1 <= x0 or y1 <= y0:
        return max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y))

    x = float(np.interp(norm_x, [x0, x1], [0.0, 1.0]))
    y = float(np.interp(norm_y, [y0, y1], [0.0, 1.0]))
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def map_hand_to_screen_linear(
    norm_x: float,
    norm_y: float,
    cal_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[int, int]:
    """Map hand landmark to screen pixels; clamp so cursor never leaves display."""
    sw, sh = config.refresh_screen_size()

    if cal_bounds is not None:
        x1, y1, x2, y2 = cal_bounds
        mx = float(np.interp(norm_x, [x1, x2], [0.0, 1.0]))
        my = float(np.interp(norm_y, [y1, y2], [0.0, 1.0]))
    else:
        mx, my = map_hand_region(norm_x, norm_y)

    mx = float(np.clip(mx, 0.0, 1.0))
    my = float(np.clip(my, 0.0, 1.0))
    if config.INVERT_CURSOR_X:
        mx = 1.0 - mx

    sx = int(np.interp(mx, [0.0, 1.0], [0, sw - 1]))
    sy = int(np.interp(my, [0.0, 1.0], [0, sh - 1]))
    return clamp_screen_coords(sx, sy, sw, sh)


def hand_region_rect_norm() -> tuple[float, float, float, float]:
    """x_min, y_min, x_max, y_max in normalized camera coordinates."""
    return (
        config.HAND_REGION_X_MIN,
        config.HAND_REGION_Y_MIN,
        config.HAND_REGION_X_MAX,
        config.HAND_REGION_Y_MAX,
    )


def apply_pointer_delta(dx: float, dy: float) -> tuple[float, float]:
    """
    Screen-pixel delta with dead zone + acceleration curve.
    Small movements stay slow; large movements speed up (exponent > 1).
    """
    mag = math.hypot(dx, dy)
    dead = config.MOVE_DEADZONE_PX
    if mag <= dead:
        return 0.0, 0.0

    scale = (mag - dead) / mag
    dx *= scale
    dy *= scale
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0

    exp = max(config.ACCEL_EXPONENT, 1.0)
    ref = max(config.ACCEL_REFERENCE_PX, 1.0)
    out_mag = config.POINTER_GAIN * (mag**exp) / (ref ** max(exp - 1.0, 0.0))

    return dx / mag * out_mag, dy / mag * out_mag


def map_to_screen(
    norm_x: float,
    norm_y: float,
    screen_width: int | None = None,
    screen_height: int | None = None,
    margin: float = config.FRAME_MARGIN,
) -> tuple[int, int]:
    """Map normalized hand position (0–1) to screen coordinates."""
    if screen_width is None or screen_height is None:
        screen_width, screen_height = config.refresh_screen_size()

    if margin <= 0:
        x = max(0.0, min(1.0, norm_x))
        y = max(0.0, min(1.0, norm_y))
    else:
        usable = 1.0 - 2 * margin
        x = (norm_x - margin) / usable
        y = (norm_y - margin) / usable
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

    screen_x = int(x * (screen_width - 1))
    screen_y = int(y * (screen_height - 1))
    return screen_x, screen_y
