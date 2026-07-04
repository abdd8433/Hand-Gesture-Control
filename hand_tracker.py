import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

import config

THUMB_TIP = 4
INDEX_MCP = 5

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "hand_landmarker.task"
)


class HandTracker:
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter_ns()

    def process(self, frame_bgr) -> HandLandmarkerResult | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = (time.perf_counter_ns() - self._start_time) // 1_000_000
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame_bgr, result: HandLandmarkerResult | None) -> None:
        if not result or not result.hand_landmarks:
            return
        for hand_landmarks in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

    @staticmethod
    def infer_handedness(landmarks) -> str:
        """Physical left/right hand from thumb vs index layout."""
        if landmarks[THUMB_TIP].x < landmarks[INDEX_MCP].x:
            label = "Right"
        else:
            label = "Left"
        if config.INVERT_HANDEDNESS:
            label = "Left" if label == "Right" else "Right"
        return label

    @staticmethod
    def hand_count(result: HandLandmarkerResult | None) -> int:
        if not result or not result.hand_landmarks:
            return 0
        return len(result.hand_landmarks)

    def get_primary_hand(self, result: HandLandmarkerResult | None):
        if not result or not result.hand_landmarks:
            return None, None

        landmarks = result.hand_landmarks[0]
        handedness = self.infer_handedness(landmarks)
        return landmarks, handedness

    def close(self) -> None:
        self.landmarker.close()
