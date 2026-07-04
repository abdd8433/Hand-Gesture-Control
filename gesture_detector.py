import time
from dataclasses import dataclass
from enum import Enum

import config
from index_pointing import (
    is_index_aiming_forward as _shared_index_aiming_forward,
    is_index_extended,
    is_index_move_pose,
    is_index_pointing_up,
    is_solo_index_move_pose,
)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP = 16
RING_PIP = 14
RING_MCP = 13
PINKY_TIP = 20
PINKY_PIP = 18
PINKY_MCP = 17

FINGERS = (
    ("thumb", THUMB_TIP, THUMB_IP, THUMB_MCP),
    ("index", INDEX_TIP, INDEX_PIP, INDEX_MCP),
    ("middle", MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    ("ring", RING_TIP, RING_PIP, RING_MCP),
    ("pinky", PINKY_TIP, PINKY_PIP, PINKY_MCP),
)

TIP_LANDMARKS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

SELF_TOUCH_PAIRS = (
    (INDEX_TIP, MIDDLE_PIP),
    (INDEX_TIP, MIDDLE_MCP),
    (INDEX_TIP, RING_PIP),
    (MIDDLE_TIP, INDEX_PIP),
    (MIDDLE_TIP, INDEX_MCP),
    (MIDDLE_TIP, RING_PIP),
    (RING_TIP, MIDDLE_PIP),
    (THUMB_TIP, INDEX_MCP),
    (THUMB_TIP, MIDDLE_MCP),
    (PINKY_TIP, RING_PIP),
)


class GestureMode(Enum):
    MOVE = "move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    SCROLL = "scroll"
    FIST = "fist"
    BLOCKED = "blocked"
    NONE = "none"


@dataclass
class GestureResult:
    mode: GestureMode
    notice: str | None = None
    scroll_direction: int | None = None  # 1 = up, -1 = down
    scroll_locked: bool = False


class GestureDetector:
    def __init__(self):
        self._scroll_locked = False
        self._scroll_enter_started_at: float | None = None
        self._ok_exit_frames = 0
        self._scroll_hold_dir = 0
        self._click_active: GestureMode | None = None
        self._pending_mode = GestureMode.NONE
        self._pending_frames = 0
        self._stable_mode = GestureMode.NONE
        self._block_frames = 0

    @property
    def scroll_locked(self) -> bool:
        return self._scroll_locked

    def _reset_confirm_state(self) -> None:
        self._pending_mode = GestureMode.NONE
        self._pending_frames = 0
        self._stable_mode = GestureMode.NONE
        self._click_active = None

    def _reset_scroll_enter_timer(self) -> None:
        self._scroll_enter_started_at = None

    @staticmethod
    def _dist(a, b) -> float:
        return (
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        ) ** 0.5

    @staticmethod
    def _finger_up(landmarks, tip: int, pip: int) -> bool:
        return landmarks[tip].y < landmarks[pip].y

    def _thumb_up(self, landmarks, handedness: str) -> bool:
        if handedness == "Right":
            return landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
        return landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x

    def finger_states(self, landmarks, handedness: str = "Right") -> dict[str, bool]:
        return {
            "thumb": self._thumb_up(landmarks, handedness),
            "index": self._finger_up(landmarks, INDEX_TIP, INDEX_PIP),
            "middle": self._finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP),
            "ring": self._finger_up(landmarks, RING_TIP, RING_PIP),
            "pinky": self._finger_up(landmarks, PINKY_TIP, PINKY_PIP),
        }

    @staticmethod
    def _xy_spread(tip, mcp) -> float:
        return ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5

    @staticmethod
    def _depth_ratio(tip, pip, mcp) -> float:
        depth = mcp.z - tip.z
        spread = GestureDetector._xy_spread(tip, mcp)
        if spread < 0.01:
            return 0.0
        return depth / spread

    def _is_finger_extended(self, landmarks, tip: int, pip: int, mcp: int) -> bool:
        tip_lm = landmarks[tip]
        pip_lm = landmarks[pip]
        mcp_lm = landmarks[mcp]
        if self._xy_spread(tip_lm, mcp_lm) < config.FINGER_EXTENSION_MIN:
            return False
        return (
            self._xy_spread(tip_lm, pip_lm) > 0.02
            or tip_lm.y < pip_lm.y
        )

    def _is_finger_pointing_at_camera(self, tip, pip, mcp) -> bool:
        depth_to_pip = pip.z - tip.z
        depth_to_mcp = mcp.z - tip.z
        if depth_to_pip < config.POINTING_Z_THRESHOLD:
            return False
        if depth_to_mcp < config.POINTING_Z_THRESHOLD:
            return False
        return self._depth_ratio(tip, pip, mcp) >= config.POINTING_MIN_DEPTH_RATIO

    def _extended_fingers(self, landmarks) -> list[str]:
        names = []
        for name, tip_i, pip_i, mcp_i in FINGERS:
            if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                names.append(name)
        return names

    def _other_fingers_down(
        self, landmarks, handedness: str, active_finger: str
    ) -> bool:
        for name, tip_i, pip_i, mcp_i in FINGERS:
            if name == active_finger:
                continue
            if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                return False
        return True

    def _index_aiming_forward(self, landmarks) -> bool:
        return _shared_index_aiming_forward(landmarks)

    def _is_index_aiming_gesture(self, landmarks) -> bool:
        return is_solo_index_move_pose(landmarks)

    def is_index_pointing_pose(
        self, landmarks, handedness: str = "Right"
    ) -> bool:
        """Index finger up alone — only pose that moves the cursor."""
        return is_solo_index_move_pose(landmarks)

    def _is_finger_curled_for_pinch(
        self, landmarks, tip_i: int, pip_i: int, mcp_i: int
    ) -> bool:
        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        if self._xy_spread(tip, pip) > 0.034:
            return False
        if tip.y < pip.y - 0.016:
            return False
        return True

    def _blocks_pinch_gestures(self, landmarks) -> bool:
        if is_solo_index_move_pose(landmarks):
            return True
        if is_index_pointing_up(landmarks):
            return True
        if is_index_extended(landmarks) and not self._is_finger_curled_for_pinch(
            landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP
        ):
            return True
        return False

    def is_three_finger_pinch(self, landmarks, handedness: str = "Right") -> bool:
        """Thumb + index + middle pinched together. Ring and pinky down."""
        if self._blocks_pinch_gestures(landmarks):
            return False

        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        middle = landmarks[MIDDLE_TIP]

        ti = self._dist(thumb, index)
        tm = self._dist(thumb, middle)
        im = self._dist(index, middle)
        if max(ti, tm, im) >= config.THREE_PINCH_THRESHOLD:
            return False

        for tip_i, pip_i, mcp_i in (
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ):
            if not self._is_finger_curled_for_pinch(
                landmarks, tip_i, pip_i, mcp_i
            ):
                return False

        return True

    def is_two_finger_pinch(self, landmarks, handedness: str = "Right") -> bool:
        """Thumb + index only. Middle, ring, pinky clearly curled."""
        if self._blocks_pinch_gestures(landmarks):
            return False

        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        middle = landmarks[MIDDLE_TIP]
        clearance = config.PINCH_THRESHOLD * config.PINCH_CLEARANCE_MULT

        ti = self._dist(thumb, index)
        if ti >= config.PINCH_THRESHOLD:
            return False
        if self._dist(thumb, middle) < clearance:
            return False
        if self._dist(index, middle) < clearance:
            return False

        for tip_i, pip_i, mcp_i in (
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ):
            if not self._is_finger_curled_for_pinch(
                landmarks, tip_i, pip_i, mcp_i
            ):
                return False

        return True

    def is_thumb_ring_pinch(self, landmarks, handedness: str = "Right") -> bool:
        """Thumb + ring only — middle (scroll-wheel) click."""
        if self._blocks_pinch_gestures(landmarks):
            return False
        if self.is_three_finger_pinch(landmarks, handedness):
            return False
        if self.is_two_finger_pinch(landmarks, handedness):
            return False

        thumb = landmarks[THUMB_TIP]
        ring = landmarks[RING_TIP]
        index = landmarks[INDEX_TIP]
        middle = landmarks[MIDDLE_TIP]
        clearance = config.PINCH_THRESHOLD * config.PINCH_CLEARANCE_MULT

        if self._dist(thumb, ring) >= config.PINCH_THRESHOLD:
            return False
        if self._dist(thumb, index) < clearance:
            return False
        if self._dist(thumb, middle) < clearance:
            return False

        for tip_i, pip_i, mcp_i in (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        ):
            if not self._is_finger_curled_for_pinch(
                landmarks, tip_i, pip_i, mcp_i
            ):
                return False

        if not self._is_finger_curled_for_pinch(
            landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP
        ):
            return False
        return True

    def get_pinch_click_mode(
        self, landmarks, handedness: str = "Right"
    ) -> GestureMode | None:
        """Pinch thumb to fingers — reliable clicks that don't block move."""
        if self._blocks_pinch_gestures(landmarks):
            return None
        if self.is_three_finger_pinch(landmarks, handedness):
            return GestureMode.LEFT_CLICK
        if self.is_thumb_ring_pinch(landmarks, handedness):
            return GestureMode.MIDDLE_CLICK
        if self.is_two_finger_pinch(landmarks, handedness):
            return GestureMode.RIGHT_CLICK
        return None

    def is_fist(self, landmarks) -> bool:
        """Closed fist — all fingers curled, nothing extended."""
        if self._extended_fingers(landmarks):
            return False

        wrist = landmarks[WRIST]
        near = 0
        for tip_i in TIP_LANDMARKS:
            tip = landmarks[tip_i]
            if self._dist(tip, wrist) < config.FIST_TIP_TO_WRIST_MAX:
                near += 1
        return near >= 4

    def _hand_scale(self, landmarks) -> float:
        return self._dist(landmarks[WRIST], landmarks[MIDDLE_MCP])

    def _palm_center(self, landmarks) -> tuple[float, float, float]:
        mcps = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
        xs = [landmarks[i].x for i in mcps]
        ys = [landmarks[i].y for i in mcps]
        zs = [landmarks[i].z for i in mcps]
        n = len(mcps)
        return sum(xs) / n, sum(ys) / n, sum(zs) / n

    def _dist_to_palm(self, landmarks, tip_i: int) -> float:
        px, py, pz = self._palm_center(landmarks)
        tip = landmarks[tip_i]
        return (
            (tip.x - px) ** 2 + (tip.y - py) ** 2 + (tip.z - pz) ** 2
        ) ** 0.5

    def _is_index_pointing_context(self, landmarks) -> bool:
        """Index-up poses should not trigger grip/self-touch blocks."""
        return is_index_pointing_up(landmarks) or is_index_extended(landmarks)

    def _thumb_opposing_curled_fingers(self, landmarks) -> int:
        """Thumb pressed against two or more curled fingertips — real grip."""
        thumb = landmarks[THUMB_TIP]
        near = 0
        for tip_i, pip_i, mcp_i in (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ):
            if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                continue
            if self._dist(thumb, landmarks[tip_i]) < config.GRIP_TIP_DISTANCE:
                near += 1
        return near

    def _is_grip_pose(self, landmarks, handedness: str) -> bool:
        if self._is_index_pointing_context(landmarks):
            return False
        if self.get_pinch_click_mode(landmarks, handedness):
            return False
        if self._three_finger_scroll_fingers(landmarks):
            return False

        if self._thumb_opposing_curled_fingers(landmarks) >= 2:
            return True

        close_pairs = 0
        tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
        for i, a in enumerate(tips):
            for b in tips[i + 1 :]:
                if self._dist(landmarks[a], landmarks[b]) < config.GRIP_TIP_DISTANCE:
                    close_pairs += 1
        return close_pairs >= config.GRIP_CLUSTER_PAIRS_MIN

    def _is_self_touch(self, landmarks, handedness: str) -> bool:
        if self._is_index_pointing_context(landmarks):
            return False
        if self.get_pinch_click_mode(landmarks, handedness):
            return False
        if self._three_finger_scroll_fingers(landmarks):
            return False

        index_pointing_up = is_index_pointing_up(landmarks)
        index_joints = (INDEX_PIP, INDEX_MCP, INDEX_TIP)

        for tip_i, joint_i in SELF_TOUCH_PAIRS:
            if index_pointing_up and (
                tip_i == INDEX_TIP
                or joint_i in index_joints
                or tip_i in index_joints
            ):
                continue
            if (
                self._dist(landmarks[tip_i], landmarks[joint_i])
                < config.SELF_TOUCH_DISTANCE
            ):
                return True

        wrist = landmarks[WRIST]
        for tip_i in (INDEX_TIP, MIDDLE_TIP):
            if tip_i == INDEX_TIP and index_pointing_up:
                continue
            if self._dist(landmarks[tip_i], wrist) < 0.11:
                return True

        return False

    def _interaction_block_reason(
        self, landmarks, handedness: str, num_hands: int
    ) -> str | None:
        if num_hands >= 2:
            return "Both hands visible — paused"

        if self._hand_scale(landmarks) > config.HAND_CLOSE_SCALE:
            return "Hand too close — paused"

        if self._is_grip_pose(landmarks, handedness):
            return "Holding something — paused"

        if self._is_self_touch(landmarks, handedness):
            return "Touching yourself — paused"

        return None

    def is_index_pointing_at_camera(
        self, landmarks, handedness: str = "Right"
    ) -> bool:
        return self.is_index_pointing_pose(landmarks, handedness)

    def _is_ambiguous_pinch(self, landmarks, handedness: str) -> bool:
        if self.is_index_pointing_pose(landmarks, handedness):
            return False
        # Pointing: thumb often sits near the index — not an unclear pinch
        if is_solo_index_move_pose(landmarks):
            return False
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        return self._dist(thumb, index) < config.PINCH_THRESHOLD * 1.1

    def _fingers_curled_into_fist(self, landmarks) -> bool:
        """Index, middle, ring, pinky folded — classic thumbs up/down base."""
        for name, tip_i, pip_i, mcp_i in FINGERS:
            if name == "thumb":
                continue
            if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                return False
        return True

    def _thumb_extended(self, landmarks) -> bool:
        tip = landmarks[THUMB_TIP]
        mcp = landmarks[THUMB_MCP]
        return self._xy_spread(tip, mcp) >= config.FINGER_EXTENSION_MIN

    def is_thumbs_up(self, landmarks, handedness: str = "Right") -> bool:
        """Classic thumbs-up pose: fist with thumb raised."""
        if not self._fingers_curled_into_fist(landmarks):
            return False
        if not self._thumb_extended(landmarks):
            return False
        if not self._thumb_up(landmarks, handedness):
            return False

        tip = landmarks[THUMB_TIP]
        ip = landmarks[THUMB_IP]
        wrist = landmarks[WRIST]

        if tip.y > ip.y + config.THUMB_SCROLL_MIN_DY:
            return False

        return tip.y < wrist.y + config.THUMB_UP_WRIST_MARGIN

    def is_thumbs_down(self, landmarks, handedness: str = "Right") -> bool:
        """Classic thumbs-down pose: fist with thumb pointing down."""
        if not self._fingers_curled_into_fist(landmarks):
            return False
        if not self._thumb_extended(landmarks):
            return False

        tip = landmarks[THUMB_TIP]
        ip = landmarks[THUMB_IP]
        wrist = landmarks[WRIST]

        return (
            tip.y > ip.y + config.THUMB_SCROLL_MIN_DY
            and tip.y > wrist.y - config.THUMB_UP_WRIST_MARGIN
        )

    def is_ok_gesture(self, landmarks, handedness: str = "Right") -> bool:
        """OK sign — thumb + index touching, middle/ring/pinky extended."""
        if self.is_index_pointing_pose(landmarks, handedness):
            return False

        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        middle = landmarks[MIDDLE_TIP]

        if self._dist(thumb, index) >= config.OK_GESTURE_MAX_DIST:
            return False
        if self._dist(thumb, middle) < config.PINCH_THRESHOLD * 1.3:
            return False
        if self._dist(index, middle) < config.PINCH_THRESHOLD * 1.3:
            return False

        for tip_i, pip_i, mcp_i in (
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ):
            if not self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                return False

        return True

    def _is_finger_raised_for_scroll(
        self, landmarks, tip_i: int, pip_i: int, mcp_i: int
    ) -> bool:
        """Relaxed extension — works when palm faces the camera."""
        if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
            return True

        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        mcp = landmarks[mcp_i]
        if self._xy_spread(tip, mcp) < config.FINGER_EXTENSION_MIN * 0.75:
            return False
        return tip.y < pip.y - 0.006 or self._xy_spread(tip, pip) > 0.018

    def _is_scroll_enter_finger_up(
        self, landmarks, tip_i: int, pip_i: int, mcp_i: int
    ) -> bool:
        """Strict raised finger — all 3 required for scroll enter."""
        tip = landmarks[tip_i]
        pip = landmarks[pip_i]
        mcp = landmarks[mcp_i]
        raise_score = pip.y - tip.y
        span = self._xy_spread(tip, pip)
        tip_mcp = self._xy_spread(tip, mcp)

        if tip_mcp < config.FINGER_EXTENSION_MIN * 0.88:
            return False
        if tip.y >= pip.y + 0.006:
            return False
        if raise_score < 0.014 and span < 0.026:
            return False
        return raise_score >= 0.014 or span >= 0.028

    def _strict_three_finger_up(self, landmarks) -> bool:
        """Exactly index + middle + ring up; not a solo index move."""
        if is_solo_index_move_pose(landmarks):
            return False

        finger_triplets = (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
        )
        for tip_i, pip_i, mcp_i in finger_triplets:
            if not self._is_scroll_enter_finger_up(
                landmarks, tip_i, pip_i, mcp_i
            ):
                return False

        if self._is_scroll_enter_finger_up(
            landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP
        ):
            return False
        return True

    def _three_finger_scroll_fingers(self, landmarks) -> bool:
        """Index + middle + ring raised; pinky tucked."""
        finger_triplets = (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
        )
        for tip_i, pip_i, mcp_i in finger_triplets:
            if not self._is_finger_raised_for_scroll(
                landmarks, tip_i, pip_i, mcp_i
            ):
                return False

        if self._is_finger_raised_for_scroll(
            landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP
        ):
            return False
        return True

    def _thumb_folded_for_scroll_enter(
        self, landmarks, handedness: str
    ) -> bool:
        """Thumb tucked across palm — not sticking out like thumbs up/down."""
        if self.is_thumbs_up(landmarks, handedness):
            return False
        if self.is_thumbs_down(landmarks, handedness):
            return False

        thumb_tip = landmarks[THUMB_TIP]
        if self._dist(thumb_tip, landmarks[INDEX_MCP]) < 0.16:
            return True
        if self._dist(thumb_tip, landmarks[PINKY_MCP]) < 0.15:
            return True
        if self._dist(thumb_tip, landmarks[MIDDLE_MCP]) < 0.17:
            return True
        return not self._thumb_extended(landmarks)

    def is_scroll_enter_gesture(
        self, landmarks, handedness: str = "Right"
    ) -> bool:
        """Exactly index + middle + ring up for 2s — not solo index."""
        if is_solo_index_move_pose(landmarks):
            return False
        if self.get_pinch_click_mode(landmarks, handedness):
            return False
        if not self._strict_three_finger_up(landmarks):
            return False
        if not self._thumb_folded_for_scroll_enter(landmarks, handedness):
            return False

        index_tip = landmarks[INDEX_TIP]
        middle_tip = landmarks[MIDDLE_TIP]
        ring_tip = landmarks[RING_TIP]
        if self._dist(index_tip, middle_tip) < config.THREE_FINGER_MIN_SPREAD:
            return False
        if self._dist(middle_tip, ring_tip) < config.THREE_FINGER_MIN_SPREAD:
            return False

        return True

    def _other_fingers_curled_for_thumb_scroll(self, landmarks) -> bool:
        """Allow a loose fist in scroll mode — block multi-finger or index point."""
        extended: list[str] = []
        for name, tip_i, pip_i, mcp_i in FINGERS:
            if name == "thumb":
                continue
            if self._is_finger_extended(landmarks, tip_i, pip_i, mcp_i):
                extended.append(name)

        if len(extended) >= 2:
            return False
        if "index" in extended and is_index_pointing_up(landmarks):
            return False
        return True

    def _thumb_scroll_score(self, landmarks) -> float:
        """Positive = thumb up, negative = thumb down."""
        tip = landmarks[THUMB_TIP]
        ip = landmarks[THUMB_IP]
        mcp = landmarks[THUMB_MCP]
        wrist = landmarks[WRIST]

        if self._xy_spread(tip, mcp) < config.SCROLL_THUMB_MIN_SPREAD:
            return 0.0

        score = 0.0
        score += (ip.y - tip.y) * 5.0
        score += (wrist.y - tip.y) * 2.5
        score += (mcp.z - tip.z) * 1.8
        return score

    def is_scroll_thumb_up(
        self, landmarks, handedness: str = "Right"
    ) -> bool:
        """Thumbs up tuned for scroll mode (palm-facing camera)."""
        if self.is_ok_gesture(landmarks, handedness):
            return False
        if not self._other_fingers_curled_for_thumb_scroll(landmarks):
            return False

        tip = landmarks[THUMB_TIP]
        ip = landmarks[THUMB_IP]
        wrist = landmarks[WRIST]

        if tip.y > ip.y + config.THUMB_SCROLL_MIN_DY:
            return False
        if tip.y > wrist.y + config.THUMB_UP_WRIST_MARGIN:
            return False

        score = self._thumb_scroll_score(landmarks)
        if score < config.THUMB_SCROLL_UP_SCORE:
            return False

        if self._thumb_up(landmarks, handedness):
            return True
        if tip.y < wrist.y:
            return True
        return score >= config.THUMB_SCROLL_UP_SCORE * 1.35

    def is_scroll_thumb_down(
        self, landmarks, handedness: str = "Right"
    ) -> bool:
        """Thumbs down tuned for scroll mode."""
        if self.is_ok_gesture(landmarks, handedness):
            return False
        if not self._other_fingers_curled_for_thumb_scroll(landmarks):
            return False

        tip = landmarks[THUMB_TIP]
        ip = landmarks[THUMB_IP]
        wrist = landmarks[WRIST]

        if tip.y <= ip.y + config.THUMB_SCROLL_MIN_DY:
            return False
        if tip.y <= wrist.y - config.THUMB_UP_WRIST_MARGIN * 0.5:
            return False

        score = self._thumb_scroll_score(landmarks)
        return score <= -config.THUMB_SCROLL_DOWN_SCORE

    def _finger_scroll_in_lock(
        self, landmarks, handedness: str = "Right"
    ) -> int | None:
        """Scroll mode: 1 finger up = scroll up, 2 fingers up = scroll down."""
        if self.is_ok_gesture(landmarks, handedness):
            return None
        if not self._thumb_folded_for_scroll_enter(landmarks, handedness):
            return None

        finger_triplets = (
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        )
        up = tuple(
            self._is_scroll_enter_finger_up(landmarks, tip_i, pip_i, mcp_i)
            for tip_i, pip_i, mcp_i in finger_triplets
        )
        index_up, middle_up, ring_up, pinky_up = up

        if pinky_up or ring_up:
            return None
        if index_up and middle_up:
            return -1
        if index_up and not middle_up:
            return 1
        return None

    def _thumb_scroll_in_lock(
        self, landmarks, handedness: str = "Right"
    ) -> int | None:
        """Thumbs up/down — scroll-mode detection only."""
        if self.is_ok_gesture(landmarks, handedness):
            return None

        down = self.is_scroll_thumb_down(landmarks, handedness)
        up = self.is_scroll_thumb_up(landmarks, handedness)

        if down and not up:
            return -1
        if up and not down:
            return 1
        return None

    def _raw_mode(self, landmarks, handedness: str) -> GestureMode:
        if self.is_scroll_enter_gesture(landmarks, handedness):
            return GestureMode.NONE
        if is_solo_index_move_pose(landmarks):
            return GestureMode.MOVE
        if self.is_fist(landmarks):
            return GestureMode.FIST
        return GestureMode.NONE

    def _build_notice(
        self, landmarks, handedness: str, raw_mode: GestureMode
    ) -> str | None:
        if raw_mode != GestureMode.NONE:
            return None

        if self.is_fist(landmarks):
            return "Fist detected - cursor paused"

        if is_solo_index_move_pose(landmarks):
            return None

        if is_index_extended(landmarks) and not is_index_pointing_up(landmarks):
            return "Raise index finger straight up to move"

        if is_index_move_pose(landmarks):
            return "Raise only your index finger to move"

        extended = self._extended_fingers(landmarks)

        if len(extended) >= 2:
            return "Unclear gesture - show one clear pose only"

        if any(self.finger_states(landmarks, handedness).values()):
            return "Hold index+middle+ring up 2s to enter scroll mode"

        return "Raise index finger up to move"

    def _confirm_mode(self, raw_mode: GestureMode) -> GestureMode:
        if raw_mode == self._pending_mode:
            self._pending_frames += 1
        else:
            self._pending_mode = raw_mode
            self._pending_frames = 1

        if raw_mode in (
            GestureMode.LEFT_CLICK,
            GestureMode.RIGHT_CLICK,
            GestureMode.MIDDLE_CLICK,
            GestureMode.MOVE,
        ):
            return raw_mode

        required = (
            config.MOVE_CONFIRM_FRAMES
            if raw_mode == GestureMode.MOVE
            else config.GESTURE_CONFIRM_FRAMES
        )
        if self._pending_frames >= required:
            self._stable_mode = raw_mode
        elif raw_mode in (GestureMode.NONE, GestureMode.FIST):
            self._stable_mode = raw_mode

        return self._stable_mode

    def _detect_scroll_locked(
        self, landmarks, handedness: str
    ) -> GestureResult:
        notice = "Scroll mode — 1 finger up=scroll up, 2 fingers up=scroll down, OK sign=exit"

        if self.is_ok_gesture(landmarks, handedness):
            self._ok_exit_frames += 1
        else:
            self._ok_exit_frames = 0

        if self._ok_exit_frames >= config.SCROLL_LOCK_EXIT_FRAMES:
            self._scroll_locked = False
            self._scroll_hold_dir = 0
            self._ok_exit_frames = 0
            self._reset_confirm_state()
            return GestureResult(
                GestureMode.NONE,
                "Scroll mode off",
                scroll_locked=False,
            )

        thumb_dir = self._finger_scroll_in_lock(landmarks, handedness)
        if thumb_dir:
            self._scroll_hold_dir = thumb_dir
            return GestureResult(
                GestureMode.SCROLL,
                notice,
                scroll_direction=thumb_dir,
                scroll_locked=True,
            )

        self._scroll_hold_dir = 0
        return GestureResult(
            GestureMode.SCROLL,
            notice,
            scroll_locked=True,
        )

    def _detect_normal(
        self, landmarks, handedness: str, num_hands: int
    ) -> GestureResult:
        block_reason = self._interaction_block_reason(
            landmarks, handedness, num_hands
        )
        if block_reason:
            self._block_frames += 1
        else:
            self._block_frames = 0

        if self._block_frames >= config.BLOCK_CONFIRM_FRAMES:
            self._reset_scroll_enter_timer()
            self._click_active = None
            return GestureResult(GestureMode.BLOCKED, block_reason)

        raw = self._raw_mode(landmarks, handedness)
        notice = self._build_notice(landmarks, handedness, raw)

        scroll_enter = self.is_scroll_enter_gesture(landmarks, handedness)
        if scroll_enter:
            if self._scroll_enter_started_at is None:
                self._scroll_enter_started_at = time.time()
        else:
            self._reset_scroll_enter_timer()

        if self._scroll_enter_started_at is not None:
            held = time.time() - self._scroll_enter_started_at
            if held >= config.SCROLL_LOCK_ENTER_HOLD_SEC:
                self._scroll_locked = True
                self._reset_scroll_enter_timer()
                self._scroll_hold_dir = 0
                self._reset_confirm_state()
                return GestureResult(
                    GestureMode.SCROLL,
                    "Scroll mode on — 1 finger up=scroll up, 2 fingers up=scroll down, OK sign=exit",
                    scroll_locked=True,
                )

        if raw == GestureMode.MOVE and not scroll_enter:
            self._reset_scroll_enter_timer()
            stable = self._confirm_mode(raw)
            if stable == GestureMode.MOVE and not self.is_fist(landmarks):
                return GestureResult(GestureMode.MOVE, notice)
            return GestureResult(GestureMode.NONE, notice)

        if scroll_enter:
            held = time.time() - self._scroll_enter_started_at
            remaining = max(0.0, config.SCROLL_LOCK_ENTER_HOLD_SEC - held)
            return GestureResult(
                GestureMode.NONE,
                f"Hold 3 fingers up {remaining:.1f}s to enter scroll mode",
            )

        if raw == GestureMode.FIST:
            self._reset_scroll_enter_timer()
            return GestureResult(
                GestureMode.FIST,
                "Fist detected - cursor paused",
            )

        stable = self._confirm_mode(raw)
        if stable == GestureMode.NONE:
            return GestureResult(GestureMode.NONE, notice)

        return GestureResult(GestureMode.NONE, notice)

    def detect(
        self, landmarks, handedness: str = "Right", num_hands: int = 1
    ) -> GestureResult:
        if self._scroll_locked:
            return self._detect_scroll_locked(landmarks, handedness)

        return self._detect_normal(landmarks, handedness, num_hands)

    def reset_state(self, keep_scroll_mode: bool = False) -> None:
        if not keep_scroll_mode:
            self._scroll_locked = False
            self._scroll_hold_dir = 0
        self._reset_scroll_enter_timer()
        self._ok_exit_frames = 0
        self._click_active = None
        self._pending_mode = GestureMode.NONE
        self._pending_frames = 0
        self._stable_mode = GestureMode.NONE
        self._block_frames = 0

    def get_cursor_point(self, landmarks) -> tuple[float, float]:
        tip = landmarks[INDEX_TIP]
        return tip.x, tip.y
