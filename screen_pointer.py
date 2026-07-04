import json
import os

import cv2

import config
from index_pointing import is_solo_index_move_pose
from smoothing import (
    EMAFilter,
    HandDeadzoneFilter,
    compute_cal_bounds,
    map_hand_to_screen_linear,
)

_CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

INDEX_PIP = 6
INDEX_TIP = 8

CALIBRATION_LABELS = (
    "1/4: point index at TOP-LEFT of camera frame, press SPACE",
    "2/4: point index at TOP-RIGHT of camera frame, press SPACE",
    "3/4: point index at BOTTOM-RIGHT of camera frame, press SPACE",
    "4/4: point index at BOTTOM-LEFT of camera frame, press SPACE",
)


def _calibration_targets() -> tuple[tuple[int, int], ...]:
    w, h = config.refresh_screen_size()
    return ((0, 0), (w, 0), (w, h), (0, h))


class AimPoint:
    __slots__ = ("norm_x", "norm_y", "screen_x", "screen_y", "valid")

    def __init__(
        self,
        norm_x: float = 0.0,
        norm_y: float = 0.0,
        screen_x: int = 0,
        screen_y: int = 0,
        valid: bool = False,
    ):
        self.norm_x = norm_x
        self.norm_y = norm_y
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.valid = valid


class ScreenPointer:
    """Map index fingertip position to screen coordinates."""

    def __init__(self):
        self.ema = EMAFilter(alpha=config.POINTER_EMA_ALPHA)
        self.deadzone = HandDeadzoneFilter()
        self._cal_bounds: tuple[float, float, float, float] | None = None
        self.calibration_step: int | None = None
        self._calibration_hits: list[tuple[float, float]] = []
        self.load_calibration()

    @staticmethod
    def _is_pointing(landmarks) -> bool:
        return is_solo_index_move_pose(landmarks)

    @staticmethod
    def fingertip_raw(landmarks) -> tuple[float, float] | None:
        if not ScreenPointer._is_pointing(landmarks):
            return None
        tip = landmarks[INDEX_TIP]
        return tip.x, tip.y

    def _hits_to_screen(self, hit_x: float, hit_y: float) -> tuple[int, int]:
        return map_hand_to_screen_linear(hit_x, hit_y, self._cal_bounds)

    def get_aim_point(self, landmarks, smooth: bool = True) -> AimPoint:
        hit = self.fingertip_raw(landmarks)
        if hit is None:
            return AimPoint()

        if smooth:
            hit = self.ema.update(hit)
        else:
            self.ema.update(hit)

        hit = self.deadzone.apply(hit)
        sx, sy = self._hits_to_screen(hit[0], hit[1])
        return AimPoint(hit[0], hit[1], sx, sy, True)

    def reset_smoothing(self) -> None:
        self.ema.reset()
        self.deadzone.reset()

    def is_calibrating(self) -> bool:
        return self.calibration_step is not None

    def start_calibration(self) -> None:
        self.calibration_step = 0
        self._calibration_hits = []
        self._cal_bounds = None

    def cancel_calibration(self) -> None:
        self.calibration_step = None

    def calibration_label(self) -> str | None:
        if self.calibration_step is None:
            return None
        if self.calibration_step < len(CALIBRATION_LABELS):
            return CALIBRATION_LABELS[self.calibration_step]
        return None

    @staticmethod
    def _frame_corners_norm() -> tuple[tuple[float, float], ...]:
        """Full camera frame corners in normalized landmark space."""
        return (
            (config.HAND_REGION_X_MIN, config.HAND_REGION_Y_MIN),
            (config.HAND_REGION_X_MAX, config.HAND_REGION_Y_MIN),
            (config.HAND_REGION_X_MAX, config.HAND_REGION_Y_MAX),
            (config.HAND_REGION_X_MIN, config.HAND_REGION_Y_MAX),
        )

    def confirm_calibration_corner(self, landmarks) -> bool:
        if self.calibration_step is None:
            return False

        hit = self.fingertip_raw(landmarks)
        if hit is None:
            return False

        self._calibration_hits.append(hit)
        self.calibration_step += 1

        if self.calibration_step >= len(_calibration_targets()):
            self._finish_calibration()
            return True

        return False

    def _finish_calibration(self) -> None:
        self._cal_bounds = compute_cal_bounds(self._calibration_hits)
        self.calibration_step = None
        x1, y1, x2, y2 = self._cal_bounds
        sw, sh = config.refresh_screen_size()
        print(
            f"Calibration bounds (hand): x1={x1:.3f}, y1={y1:.3f}, "
            f"x2={x2:.3f}, y2={y2:.3f}"
        )
        print(f"Screen size: {sw} x {sh}")
        self.save_calibration()

    def save_calibration(self) -> None:
        if self._cal_bounds is None:
            return
        x1, y1, x2, y2 = self._cal_bounds
        data = {
            "bounds": {
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            },
            "hits": self._calibration_hits,
        }
        with open(_CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self) -> bool:
        if not os.path.exists(_CALIBRATION_FILE):
            return False
        try:
            with open(_CALIBRATION_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self._calibration_hits = [tuple(h) for h in data.get("hits", [])]
            bounds = data.get("bounds")
            if bounds:
                self._cal_bounds = (
                    float(bounds["x_min"]),
                    float(bounds["y_min"]),
                    float(bounds["x_max"]),
                    float(bounds["y_max"]),
                )
            elif len(self._calibration_hits) >= 4:
                self._cal_bounds = compute_cal_bounds(self._calibration_hits)
            else:
                self._cal_bounds = None
            if self._cal_bounds:
                x1, y1, x2, y2 = self._cal_bounds
                sw, sh = config.refresh_screen_size()
                print(
                    f"Loaded calibration: x1={x1:.3f}, y1={y1:.3f}, "
                    f"x2={x2:.3f}, y2={y2:.3f} | screen {sw}x{sh}"
                )
            return self._cal_bounds is not None
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            self._cal_bounds = None
            return False

    @property
    def is_calibrated(self) -> bool:
        return self._cal_bounds is not None

    def should_show_hand_region(self) -> bool:
        if not config.SHOW_HAND_REGION:
            return False
        if getattr(config, "SHOW_HAND_REGION_ALWAYS", False):
            return True
        return self.is_calibrating() or not self.is_calibrated

    def _hand_region_px(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Pixel corners of the active mapping region (full frame = entire image)."""
        x0 = int(round(config.HAND_REGION_X_MIN * (w - 1)))
        y0 = int(round(config.HAND_REGION_Y_MIN * (h - 1)))
        x1 = int(round(config.HAND_REGION_X_MAX * (w - 1)))
        y1 = int(round(config.HAND_REGION_Y_MAX * (h - 1)))
        return x0, y0, x1, y1

    @staticmethod
    def _is_full_frame_region() -> bool:
        return (
            config.HAND_REGION_X_MIN <= 0.001
            and config.HAND_REGION_Y_MIN <= 0.001
            and config.HAND_REGION_X_MAX >= 0.999
            and config.HAND_REGION_Y_MAX >= 0.999
        )

    def _draw_hand_region(self, frame) -> None:
        if not self.should_show_hand_region():
            return

        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._hand_region_px(w, h)

        if not self._is_full_frame_region():
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 220, 80), 2)

        edge_color = (80, 220, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame, "TOP of monitor (tabs)",
            (x0 + 6, min(y0 + 18, h - 4)), font, 0.42, edge_color, 1,
        )
        cv2.putText(
            frame, "BOTTOM of monitor (taskbar)",
            (x0 + 6, max(y1 - 8, 14)), font, 0.42, edge_color, 1,
        )

        if self.is_calibrating():
            hint = "Point at yellow marks on FRAME corners | index up, SPACE"
        else:
            hint = "Full camera frame maps to full monitor"
        cv2.putText(
            frame, hint,
            (x0 + 4, max(y0 - 8, 16)), font, 0.4, edge_color, 1,
        )

    def _draw_calibration_corner_target(self, frame) -> None:
        if not self.is_calibrating() or self.calibration_step is None:
            return
        if self.calibration_step >= len(_calibration_targets()):
            return

        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._hand_region_px(w, h)
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        tx, ty = corners[self.calibration_step]
        cv2.circle(frame, (tx, ty), 22, (0, 255, 255), 3)
        cv2.circle(frame, (tx, ty), 6, (0, 255, 255), -1)
        cv2.line(frame, (tx - 28, ty), (tx + 28, ty), (0, 255, 255), 2)
        cv2.line(frame, (tx, ty - 28), (tx, ty + 28), (0, 255, 255), 2)

    def draw_overlay(
        self, frame, landmarks, aim: AimPoint | None = None
    ) -> None:
        h, w = frame.shape[:2]
        self._draw_hand_region(frame)
        self._draw_calibration_corner_target(frame)

        pip = landmarks[INDEX_PIP]
        tip = landmarks[INDEX_TIP]
        pip_px = (int(pip.x * w), int(pip.y * h))
        tip_px = (int(tip.x * w), int(tip.y * h))
        cv2.line(frame, pip_px, tip_px, (255, 120, 0), 2)

        if aim is not None and aim.valid:
            cv2.circle(frame, tip_px, 16, (0, 0, 255), 2)
            cv2.circle(frame, tip_px, 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Aim -> ({aim.screen_x}, {aim.screen_y})",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        cal_label = self.calibration_label()
        if cal_label:
            cv2.putText(
                frame,
                cal_label,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            step = self.calibration_step or 0
            cv2.putText(
                frame,
                f"Corner {step + 1}/4 - press SPACE at yellow mark",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
