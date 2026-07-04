import pyautogui

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Screen (refreshed at runtime for accurate full-screen mapping)
def refresh_screen_size() -> tuple[int, int]:
    return pyautogui.size()


SCREEN_WIDTH, SCREEN_HEIGHT = refresh_screen_size()

# Coordinate mapping — hand region in camera view maps to full screen
FRAME_MARGIN = 0.0
# Inset applied to calibrated hand range (normalized 0-1) so edges stay on-screen
CALIBRATION_MARGIN_NORM = 0.012
# Normalized camera bounds (0–1). Full frame = maximum reach to screen edges.
HAND_REGION_X_MIN = 0.0
HAND_REGION_X_MAX = 1.0
HAND_REGION_Y_MIN = 0.0
HAND_REGION_Y_MAX = 1.0
SHOW_HAND_REGION = True
# Keep green box visible after calibration so you know where screen edges are
SHOW_HAND_REGION_ALWAYS = True

# absolute = cursor goes where fingertip points; relative = drag from current spot
ABSOLUTE_POINTER = True

# Smoothing & move stability
EMA_ALPHA = 0.22
POINTER_GAIN = 1.0
HAND_DEADZONE_NORM = 0.004
MOVE_DEADZONE_PX = 3
ACCEL_EXPONENT = 1.6
ACCEL_REFERENCE_PX = 30
MOVE_CONFIRM_FRAMES = 1

# Pinch clicks
PINCH_THRESHOLD = 0.050
THREE_PINCH_THRESHOLD = 0.065
PINCH_CLEARANCE_MULT = 1.55
CLICK_COOLDOWN = 0.25
GESTURE_CONFIRM_FRAMES = 3

# Voice clicks — say: "alpha"=left, "bravo"=middle, "charlie"=right
VOICE_CMD_LEFT = "alpha"
VOICE_CMD_MIDDLE = "bravo"
VOICE_CMD_RIGHT = "charlie"
VOICE_CLICKS_ENABLED = True
VOICE_LANGUAGE = "en-US"
VOICE_USE_VOSK = True  # fast offline recognition (~100ms)
VOICE_GOOGLE_FALLBACK = True  # backup if local misses
VOICE_GOOGLE_SHOW_ALL = False  # slower; only used on fallback retry
VOICE_PHRASE_TIME_LIMIT = 1.2
VOICE_BLOCK_MS = 20
VOICE_MIN_ENERGY = 40
VOICE_ENERGY_MULTIPLIER = 2.2
VOICE_SILENCE_END_MS = 140
VOICE_MIN_PHRASE_SEC = 0.10
VOICE_MAX_PHRASE_SEC = 0.85
VOICE_REPEAT_COOLDOWN = 0.25
VOICE_MIC_INDEX = None  # None = default mic; set to device index if needed
VOICE_DEBUG = False

# Scroll lock — three fingers up enters; thumbs up/down scroll; OK sign exits
SCROLL_SENSITIVITY = 14
SCROLL_STEP = 5
SCROLL_HOLD_COOLDOWN = 0.03
SCROLL_LOCK_ENTER_HOLD_SEC = 2.0
SCROLL_LOCK_EXIT_FRAMES = 3
OK_GESTURE_MAX_DIST = 0.048
THUMB_SCROLL_MIN_DY = 0.015
THUMB_UP_WRIST_MARGIN = 0.05
THUMB_SCROLL_UP_SCORE = 0.045
THUMB_SCROLL_DOWN_SCORE = 0.05
SCROLL_THUMB_MIN_SPREAD = 0.026
THREE_FINGER_MIN_SPREAD = 0.012
VERTICAL_FINGER_MIN_DY = 0.012
FINGER_EXTENSION_MIN = 0.028

# Index fingertip pointing & cursor aim
POINTING_RAY_MIN_DZ = 0.002
POINTER_EMA_ALPHA = 0.32

# Pointing-at-camera detection (index finger aimed at screen)
POINTING_Z_THRESHOLD = 0.016
POINTING_MIN_DEPTH_RATIO = 0.95

# Fist — pauses cursor (no movement)
FIST_TIP_TO_WRIST_MAX = 0.16

# MediaPipe
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.65

# Pause when holding/touching something, self-contact, or both hands visible
BLOCK_CONFIRM_FRAMES = 2
GRIP_TIP_DISTANCE = 0.042
SELF_TOUCH_DISTANCE = 0.048
HAND_CLOSE_SCALE = 0.30
PALM_GRIP_DISTANCE = 0.07
GRIP_CLUSTER_PAIRS_MIN = 5

# Display (resizable Windows window, not fullscreen)
SHOW_PREVIEW = True
PREVIEW_FULLSCREEN = False
PREVIEW_WINDOW_WIDTH = 960
PREVIEW_WINDOW_HEIGHT = 720
WINDOW_NAME = "Hand Gesture Cursor"

# Mirror webcam horizontally (selfie view)
FLIP_FRAME = True
# Flip left/right hand label if it still looks reversed on your camera
INVERT_HANDEDNESS = False
# Flip horizontal cursor direction (uncalibrated mode only; calibrated auto-corrects)
INVERT_CURSOR_X = False
