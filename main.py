import cv2

import config
from cursor_control import CursorController
from gesture_detector import GestureMode, GestureDetector
from hand_tracker import HandTracker
from screen_pointer import ScreenPointer
from voice_control import VoiceClickController


def draw_hud(
    frame,
    mode: GestureMode,
    handedness: str | None,
    notice: str | None = None,
    scroll_direction: int | None = None,
    scroll_locked: bool = False,
    calibrated: bool = False,
    calibrating: bool = False,
    voice_active: bool = False,
    voice_heard: str | None = None,
    voice_status: str | None = None,
    voice_error: str | None = None,
) -> None:
    if scroll_locked:
        text = "Mode: scroll"
        if handedness:
            text += f" | Hand: {handedness}"
        text += " | Other gestures off"
        if scroll_direction and scroll_direction > 0:
            text += " | Scrolling up"
        elif scroll_direction and scroll_direction < 0:
            text += " | Scrolling down"
    else:
        text = f"Mode: {mode.value}"
        if handedness:
            text += f" | Hand: {handedness}"

        if calibrating:
            text += " | CALIBRATING (frame corners = monitor corners)"
        elif calibrated:
            text += " | Top of box=tabs | Bottom=taskbar"
        else:
            text += " | Hand zone -> screen (C=calibrate)"

        if mode == GestureMode.SCROLL and scroll_direction:
            if scroll_direction > 0:
                text += " | 1 finger up — scrolling up"
            else:
                text += " | 2 fingers up — scrolling down"
        elif mode == GestureMode.FIST:
            text += " | Cursor paused"
        elif mode == GestureMode.BLOCKED:
            text += " | Paused"

    if voice_active and not scroll_locked:
        text += " | Voice on"

    if voice_error and not scroll_locked:
        text += f" | Voice err: {voice_error[:40]}"

    if scroll_locked and not notice:
        notice = "Move & clicks off — 1 finger up / 2 fingers up to scroll, OK sign to exit"

    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    if voice_heard and not calibrating:
        cv2.putText(
            frame,
            f"Heard: {voice_heard}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 180, 0),
            2,
        )

    if voice_status and voice_active and not calibrating and not scroll_locked:
        is_click = "CLICK" in voice_status.upper()
        cv2.putText(
            frame,
            voice_status[:70],
            (10, 118 if voice_heard else 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 if is_click else 0.45,
            (0, 255, 120) if is_click else (180, 180, 255),
            2 if is_click else 1,
        )

    if notice and not calibrating:
        y = 148 if (voice_heard or voice_status) else 60
        cv2.putText(
            frame,
            notice,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )

    footer = (
        "Scroll mode: 1 finger up=scroll up, 2 fingers up=scroll down | OK sign=exit"
        if scroll_locked
        else "Q=quit C=calibrate | Index up=move | Voice: alpha/bravo/charlie | 3 up 2s=scroll"
    )
    cv2.putText(
        frame,
        footer,
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX in config.py.")

    tracker = HandTracker()
    gestures = GestureDetector()
    cursor = CursorController()
    pointer = ScreenPointer()
    voice = VoiceClickController()
    prev_mode = GestureMode.NONE

    print("Hand Gesture Cursor Control")
    print("Raise index finger up (other fingers down) to move cursor")
    print("Press C to calibrate (hold index up at each screen corner, SPACE to confirm)")
    print("Closed fist = cursor paused")
    print("Hold index+middle+ring up for 2 seconds to enter scroll mode (OK sign to exit)")
    print("In scroll mode: 1 finger up = scroll up | 2 fingers up = scroll down")
    if voice.start():
        from voice_control import mic_device_label

        print(
            f"Voice clicks: say '{config.VOICE_CMD_LEFT}' = left | "
            f"'{config.VOICE_CMD_MIDDLE}' = middle | "
            f"'{config.VOICE_CMD_RIGHT}' = right"
        )
        print(f"Voice mic: {mic_device_label()} (set VOICE_MIC_INDEX in config.py to change)")
        if config.VOICE_USE_VOSK:
            print("Voice: fast local mode (first run downloads ~40 MB model)")
        elif config.VOICE_GOOGLE_FALLBACK:
            print("Voice needs internet — watch HUD for 'Heard:' feedback")
        try:
            import sounddevice as sd

            print("Input devices:")
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    mark = " <-- active" if (
                        config.VOICE_MIC_INDEX is None
                        and dev.get("name") == mic_device_label()
                    ) or config.VOICE_MIC_INDEX == i else ""
                    print(f"  [{i}] {dev['name']}{mark}")
        except Exception:
            pass
    elif voice.error:
        print(f"Voice clicks unavailable: {voice.error}")

    if not pointer.is_calibrated:
        print("No calibration file found — accuracy improves after pressing C")

    window_name = config.WINDOW_NAME
    if config.SHOW_PREVIEW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if config.PREVIEW_FULLSCREEN:
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.resizeWindow(
                window_name,
                config.PREVIEW_WINDOW_WIDTH,
                config.PREVIEW_WINDOW_HEIGHT,
            )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1) if config.FLIP_FRAME else frame
            result = tracker.process(frame)
            hand_data = tracker.get_primary_hand(result)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if voice.enabled and not gestures.scroll_locked:
                voice_click = voice.poll_click()
                if voice_click == "left":
                    cursor.left_click()
                elif voice_click == "right":
                    cursor.right_click()
                elif voice_click == "middle":
                    cursor.middle_click()

            if key == ord("c") and not gestures.scroll_locked:
                pointer.start_calibration()
                print("Calibration started — point at each corner, press SPACE")

            notice = None
            scroll_direction = None
            scroll_locked = False
            mode = GestureMode.NONE
            handedness = None
            aim = None

            if hand_data[0] is None:
                keep_scroll = gestures.scroll_locked
                cursor.reset_scroll()
                gestures.reset_state(keep_scroll_mode=keep_scroll)
                pointer.reset_smoothing()
                if keep_scroll:
                    mode = GestureMode.SCROLL
                    scroll_locked = True
                    notice = "Scroll mode — 1 finger up / 2 fingers up to scroll, OK sign to exit"
                else:
                    cursor.reset()
                    prev_mode = GestureMode.NONE
            else:
                landmarks, handedness = hand_data
                aim = pointer.get_aim_point(landmarks)

                if pointer.is_calibrating():
                    if key == ord(" "):
                        if pointer.confirm_calibration_corner(landmarks):
                            print("Calibration saved — 3D aim mapped to your screen")
                        elif pointer.calibration_step is not None:
                            print(
                                f"Corner {pointer.calibration_step}/4 captured"
                            )
                        else:
                            print("Could not detect index up — raise index finger")
                    mode = GestureMode.NONE
                else:
                    num_hands = HandTracker.hand_count(result)
                    gesture = gestures.detect(landmarks, handedness, num_hands)
                    mode = gesture.mode
                    notice = gesture.notice
                    scroll_direction = gesture.scroll_direction
                    scroll_locked = gesture.scroll_locked

                if scroll_locked:
                    cursor.pause_move()
                    pointer.reset_smoothing()

                if mode != prev_mode:
                    if not scroll_locked:
                        if prev_mode == GestureMode.SCROLL and mode != GestureMode.SCROLL:
                            cursor.reset_scroll()
                        if mode == GestureMode.SCROLL and prev_mode != GestureMode.SCROLL:
                            cursor.reset_scroll()
                            cursor.pause_move()
                            pointer.reset_smoothing()
                        if prev_mode == GestureMode.MOVE and mode != GestureMode.MOVE:
                            pointer.reset_smoothing()
                        if mode in (
                            GestureMode.FIST,
                            GestureMode.BLOCKED,
                            GestureMode.NONE,
                            GestureMode.SCROLL,
                        ):
                            cursor.pause_move()
                            if mode != GestureMode.MOVE:
                                pointer.reset_smoothing()
                        if mode == GestureMode.MOVE and prev_mode != GestureMode.MOVE:
                            cursor.pause_move()
                    prev_mode = mode

                if (
                    mode == GestureMode.MOVE
                    and aim
                    and aim.valid
                    and not scroll_locked
                ):
                    if config.ABSOLUTE_POINTER:
                        cursor.move_screen(aim.screen_x, aim.screen_y)
                    else:
                        cursor.move_relative_screen(aim.screen_x, aim.screen_y)
                elif mode == GestureMode.SCROLL and scroll_direction:
                    cursor.scroll_hold(scroll_direction)
                elif scroll_locked:
                    cursor.reset_scroll()

                if config.SHOW_PREVIEW and not scroll_locked:
                    pointer.draw_overlay(frame, landmarks, aim)

            draw_hud(
                frame,
                mode,
                handedness,
                notice,
                scroll_direction,
                scroll_locked,
                calibrated=pointer.is_calibrated,
                calibrating=pointer.is_calibrating(),
                voice_active=voice.enabled,
                voice_heard=voice.last_heard,
                voice_status=voice.status,
                voice_error=voice.error,
            )

            if config.SHOW_PREVIEW:
                if hand_data[0] is not None:
                    tracker.draw_landmarks(frame, result)
                display = cv2.resize(
                    frame,
                    (config.PREVIEW_WINDOW_WIDTH, config.PREVIEW_WINDOW_HEIGHT),
                )
                cv2.imshow(window_name, display)
    finally:
        voice.stop()
        cursor.reset()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
