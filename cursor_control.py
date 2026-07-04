import time

import pyautogui

import config
from smoothing import EMAFilter, apply_pointer_delta, clamp_screen_coords, map_to_screen

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class CursorController:
    def __init__(self):
        self.ema = EMAFilter()
        self._last_click_time = 0.0
        self._last_screen_pos: tuple[int, int] | None = None
        self._last_aim_screen: tuple[int, int] | None = None
        self._relative_armed = False
        self._scroll_hold_dir = 0
        self._last_scroll_time = 0.0

    def _clamp_screen(self, x: int, y: int) -> tuple[int, int]:
        return clamp_screen_coords(x, y)

    def arm_relative(self, aim_screen_x: int, aim_screen_y: int) -> None:
        """Anchor cursor at current OS position — no jump on move start."""
        pos = pyautogui.position()
        self._last_screen_pos = (pos.x, pos.y)
        self._last_aim_screen = (aim_screen_x, aim_screen_y)
        self._relative_armed = True

    def move_relative_screen(self, aim_screen_x: int, aim_screen_y: int) -> None:
        """Move cursor by hand delta so it starts from where it already was."""
        if not self._relative_armed or self._last_aim_screen is None:
            self.arm_relative(aim_screen_x, aim_screen_y)
            return

        dx = aim_screen_x - self._last_aim_screen[0]
        dy = aim_screen_y - self._last_aim_screen[1]
        self._last_aim_screen = (aim_screen_x, aim_screen_y)

        dx, dy = apply_pointer_delta(dx, dy)
        if dx == 0.0 and dy == 0.0:
            return

        if self._last_screen_pos is None:
            pos = pyautogui.position()
            self._last_screen_pos = (pos.x, pos.y)

        new_x, new_y = self._clamp_screen(
            self._last_screen_pos[0] + int(round(dx)),
            self._last_screen_pos[1] + int(round(dy)),
        )

        if new_x == self._last_screen_pos[0] and new_y == self._last_screen_pos[1]:
            return

        pyautogui.moveTo(new_x, new_y, _pause=False)
        self._last_screen_pos = (new_x, new_y)

    def move_screen(self, screen_x: int, screen_y: int) -> None:
        """Move cursor to mapped fingertip position (absolute pointing)."""
        screen_x, screen_y = self._clamp_screen(screen_x, screen_y)

        if self._last_screen_pos is not None:
            dx = screen_x - self._last_screen_pos[0]
            dy = screen_y - self._last_screen_pos[1]
            if (dx * dx + dy * dy) < config.MOVE_DEADZONE_PX ** 2:
                return

        pyautogui.moveTo(screen_x, screen_y, _pause=False)
        self._last_screen_pos = (screen_x, screen_y)

    def move(self, norm_x: float, norm_y: float) -> None:
        smoothed = self.ema.update((norm_x, norm_y))
        screen_x, screen_y = map_to_screen(smoothed[0], smoothed[1])
        self.move_screen(screen_x, screen_y)

    def left_click(self) -> bool:
        now = time.time()
        if now - self._last_click_time < config.CLICK_COOLDOWN:
            return False
        pyautogui.click(button="left", _pause=False)
        self._last_click_time = now
        return True

    def right_click(self) -> bool:
        now = time.time()
        if now - self._last_click_time < config.CLICK_COOLDOWN:
            return False
        pyautogui.click(button="right", _pause=False)
        self._last_click_time = now
        return True

    def middle_click(self) -> bool:
        now = time.time()
        if now - self._last_click_time < config.CLICK_COOLDOWN:
            return False
        pyautogui.click(button="middle", _pause=False)
        self._last_click_time = now
        return True

    def scroll_hold(self, direction: int) -> None:
        """Scroll continuously while thumb stays up or down."""
        now = time.time()
        if direction != self._scroll_hold_dir:
            self._scroll_hold_dir = direction
            self._last_scroll_time = 0.0

        if (now - self._last_scroll_time) < config.SCROLL_HOLD_COOLDOWN:
            return

        self._last_scroll_time = now
        self.scroll_step(direction)

    def scroll_step(self, direction: int) -> None:
        """direction: 1 = scroll up, -1 = scroll down."""
        amount = int(direction * config.SCROLL_STEP * config.SCROLL_SENSITIVITY)
        if amount != 0:
            pyautogui.scroll(amount, _pause=False)

    def reset_scroll(self) -> None:
        self._scroll_hold_dir = 0
        self._last_scroll_time = 0.0

    def pause_move(self) -> None:
        """Stop cursor tracking without moving the mouse."""
        self.ema.reset()
        self._last_screen_pos = None
        self._last_aim_screen = None
        self._relative_armed = False

    def reset(self) -> None:
        self.reset_scroll()
        self.pause_move()

