"""Shared index-finger pointing detection for gestures and cursor aim."""

import config

INDEX_PIP = 6
INDEX_TIP = 8
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

_OTHER_FINGERS = (
    (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    (RING_TIP, RING_PIP, RING_MCP),
    (PINKY_TIP, PINKY_PIP, PINKY_MCP),
)

_OTHER_TIPS = (MIDDLE_TIP, RING_TIP, PINKY_TIP)


def _xy_spread(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _finger_raise_score(landmarks, tip_i: int, pip_i: int) -> float:
    """Larger = finger tip higher in the frame."""
    return landmarks[pip_i].y - landmarks[tip_i].y


def is_index_extended(landmarks) -> bool:
    tip = landmarks[INDEX_TIP]
    pip = landmarks[INDEX_PIP]
    mcp = landmarks[INDEX_MCP]
    tip_pip = _xy_spread(tip, pip)
    tip_mcp = _xy_spread(tip, mcp)
    min_ext = config.FINGER_EXTENSION_MIN * 0.8

    if tip_mcp < min_ext:
        return False
    if tip_pip > 0.014:
        return True
    if tip.y < pip.y + 0.012:
        return True
    return tip_mcp >= config.FINGER_EXTENSION_MIN * 0.95


def is_index_pointing_up(landmarks) -> bool:
    """
    Index raised from fist while palm faces camera.
    Uses relative finger height — works without depth/forward checks.
    """
    if not is_index_extended(landmarks):
        return False

    tip = landmarks[INDEX_TIP]
    pip = landmarks[INDEX_PIP]
    index_raise = _finger_raise_score(landmarks, INDEX_TIP, INDEX_PIP)
    index_span = _xy_spread(tip, pip)

    other_raises = [
        _finger_raise_score(landmarks, tip_i, pip_i)
        for tip_i, pip_i, _mcp_i in _OTHER_FINGERS
    ]
    max_other_raise = max(other_raises) if other_raises else 0.0

    # Index tip should be the highest fingertip in the image.
    highest_other_y = min(landmarks[t].y for t in _OTHER_TIPS)
    index_is_tallest = tip.y <= highest_other_y + 0.014

    index_dominant = (
        index_raise >= max_other_raise - 0.01
        or tip.y <= highest_other_y + 0.008
    )

    has_extension = (
        index_raise > -0.02
        or index_span > 0.016
        or tip.y < pip.y + 0.015
    )

    return index_is_tallest and index_dominant and has_extension


def is_index_aiming_forward(landmarks) -> bool:
    return is_index_pointing_up(landmarks)


def is_finger_clearly_raised(
    landmarks, tip_i: int, pip_i: int, mcp_i: int
) -> bool:
    """Another finger unmistakably up — blocks solo index move."""
    tip = landmarks[tip_i]
    pip = landmarks[pip_i]
    raise_score = _finger_raise_score(landmarks, tip_i, pip_i)
    span = _xy_spread(tip, pip)

    if raise_score < 0.02:
        return False
    if span < 0.028:
        return False
    if tip.y >= pip.y - 0.012:
        return False
    return True


def is_finger_pointing_up(
    landmarks, tip_i: int, pip_i: int, mcp_i: int, *, relaxed: bool = False
) -> bool:
    if tip_i == INDEX_TIP:
        return is_index_pointing_up(landmarks)
    return is_finger_clearly_raised(landmarks, tip_i, pip_i, mcp_i)


def count_other_fingers_clearly_raised(landmarks) -> int:
    return sum(
        1
        for tip_i, pip_i, mcp_i in _OTHER_FINGERS
        if is_finger_clearly_raised(landmarks, tip_i, pip_i, mcp_i)
    )


def is_solo_index_move_pose(landmarks) -> bool:
    if not is_index_move_pose(landmarks):
        return False
    return count_other_fingers_clearly_raised(landmarks) == 0


def is_index_move_pose(landmarks) -> bool:
    return is_index_pointing_up(landmarks)
