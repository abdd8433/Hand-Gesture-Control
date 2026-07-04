"""Voice commands for mouse clicks (replaces pinch gestures)."""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import config

ClickType = Literal["left", "right", "middle"]

_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "vosk-model-small-en-us-0.15"

# (phrase variants, click type) — NATO words: distinct and easy for STT
_COMMANDS: tuple[tuple[tuple[str, ...], ClickType], ...] = (
    (
        (
            config.VOICE_CMD_RIGHT,
            "charley",
            "charly",
            "charlie",
        ),
        "right",
    ),
    (
        (
            config.VOICE_CMD_MIDDLE,
            "brava",
            "bravo",
        ),
        "middle",
    ),
    (
        (
            config.VOICE_CMD_LEFT,
            "alfa",
            "alpha",
        ),
        "left",
    ),
)


def voice_command_hint() -> str:
    return f"{config.VOICE_CMD_LEFT} / {config.VOICE_CMD_MIDDLE} / {config.VOICE_CMD_RIGHT}"


def mic_device_label(device_index: int | None = None) -> str:
    import sounddevice as sd

    if device_index is None:
        device_index = config.VOICE_MIC_INDEX
    try:
        if device_index is None:
            info = sd.query_devices(kind="input")
        else:
            info = sd.query_devices(device_index)
        return str(info.get("name", info))
    except Exception as exc:
        return f"unknown ({exc})"


class _VoskEngine:
    """Fast offline recognizer limited to our three command words."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recognizer = None
        self._ready = False
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def ensure_loaded(self) -> bool:
        if self._ready:
            return True
        if not config.VOICE_USE_VOSK:
            return False

        with self._lock:
            if self._ready:
                return True
            try:
                from vosk import KaldiRecognizer, Model, SetLogLevel

                SetLogLevel(-1)
                model_path = self._ensure_model()
                if model_path is None:
                    return False
                model = Model(str(model_path))
                grammar = json.dumps(
                    [
                        config.VOICE_CMD_LEFT,
                        config.VOICE_CMD_MIDDLE,
                        config.VOICE_CMD_RIGHT,
                        "[unk]",
                    ]
                )
                self._recognizer = KaldiRecognizer(model, 16000, grammar)
                self._ready = True
                return True
            except ImportError:
                self._error = "vosk not installed"
                return False
            except Exception as exc:
                self._error = str(exc)
                return False

    def _ensure_model(self) -> Path | None:
        if (_MODEL_DIR / "am").exists() or (_MODEL_DIR / "graph").exists():
            return _MODEL_DIR

        zip_path = _MODEL_DIR.with_suffix(".zip")
        _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading fast voice model (one-time, ~40 MB)...")
        try:
            urlretrieve(_MODEL_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(_MODEL_DIR.parent)
            zip_path.unlink(missing_ok=True)
            # Zip extracts to vosk-model-small-en-us-0.15 folder
            if not (_MODEL_DIR / "am").exists() and not (_MODEL_DIR / "graph").exists():
                for child in _MODEL_DIR.parent.iterdir():
                    if child.is_dir() and "vosk-model" in child.name:
                        if child != _MODEL_DIR:
                            child.rename(_MODEL_DIR)
                        break
            return _MODEL_DIR if (_MODEL_DIR / "am").exists() or (_MODEL_DIR / "graph").exists() else None
        except Exception as exc:
            self._error = f"Model download failed: {exc}"
            zip_path.unlink(missing_ok=True)
            return None

    def transcribe(self, audio_bytes: bytes) -> str | None:
        if not self.ensure_loaded() or self._recognizer is None:
            return None
        with self._lock:
            self._recognizer.Reset()
            self._recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(self._recognizer.Result())
            text = (result.get("text") or "").strip()
            if text and text != "[unk]":
                return text
            partial = json.loads(self._recognizer.FinalResult())
            text = (partial.get("text") or "").strip()
            if text and text != "[unk]":
                return text
        return None


_VOSK = _VoskEngine()


class _SpeechSegmenter:
    """Detect short speech bursts — tuned for single-word commands."""

    def __init__(self, sample_rate: int) -> None:
        self._block_ms = config.VOICE_BLOCK_MS
        self._block_samples = max(1, int(sample_rate * self._block_ms / 1000))
        self._silence_blocks_end = max(
            1, int(config.VOICE_SILENCE_END_MS / self._block_ms)
        )
        self._max_blocks = max(
            3, int(config.VOICE_MAX_PHRASE_SEC * 1000 / self._block_ms)
        )
        self._min_blocks = max(
            1, int(config.VOICE_MIN_PHRASE_SEC * 1000 / self._block_ms)
        )

        self._noise_floor = 80.0
        self._calibrated = False
        self._calibration_blocks = 0
        self._calibration_sum = 0.0
        self._calibration_target = max(3, int(150 / self._block_ms))

        self._in_speech = False
        self._speech_blocks: list = []
        self._silence_blocks = 0

    def _energy(self, block) -> float:
        import numpy as np

        return float(np.abs(block.astype(np.float32)).mean())

    def _threshold_now(self) -> float:
        return max(
            config.VOICE_MIN_ENERGY,
            self._noise_floor * config.VOICE_ENERGY_MULTIPLIER,
        )

    def feed(self, block):
        import numpy as np

        energy = self._energy(block)

        if not self._calibrated:
            self._calibration_sum += energy
            self._calibration_blocks += 1
            if self._calibration_blocks >= self._calibration_target:
                self._noise_floor = self._calibration_sum / self._calibration_blocks
                self._calibrated = True
            return None

        threshold = self._threshold_now()

        if energy > threshold:
            self._in_speech = True
            self._speech_blocks.append(block)
            self._silence_blocks = 0
            if len(self._speech_blocks) >= self._max_blocks:
                return self._flush(np)
            return None

        if not self._in_speech:
            self._noise_floor = self._noise_floor * 0.92 + energy * 0.08
            return None

        self._speech_blocks.append(block)
        self._silence_blocks += 1
        if self._silence_blocks >= self._silence_blocks_end:
            return self._flush(np)
        return None

    def _flush(self, np):
        if len(self._speech_blocks) < self._min_blocks:
            self._reset()
            return None
        audio = np.concatenate(self._speech_blocks, axis=0)
        self._reset()
        return audio

    def _reset(self) -> None:
        self._in_speech = False
        self._speech_blocks = []
        self._silence_blocks = 0


class VoiceClickController:
    """Background microphone listener for click voice commands."""

    def __init__(self) -> None:
        self._queue: queue.Queue[ClickType] = queue.Queue()
        self._recognize_q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._recognize_thread: threading.Thread | None = None
        self._enabled = False
        self._last_heard: str | None = None
        self._last_click: ClickType | None = None
        self._last_click_at = 0.0
        self._error: str | None = None
        self._status = "starting"
        self._use_vosk = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_heard(self) -> str | None:
        return self._last_heard

    @property
    def last_click(self) -> ClickType | None:
        return self._last_click

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> bool:
        if not config.VOICE_CLICKS_ENABLED:
            self._error = "Voice clicks disabled in config"
            return False

        if config.VOICE_USE_VOSK:
            threading.Thread(target=_VOSK.ensure_loaded, daemon=True).start()

        try:
            import speech_recognition as sr  # noqa: F401
        except ImportError:
            if not config.VOICE_USE_VOSK:
                self._error = (
                    "Install SpeechRecognition: pip install SpeechRecognition"
                )
                return False

        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._recognize_thread = threading.Thread(
            target=self._recognize_worker, daemon=True
        )
        self._thread.start()
        self._recognize_thread.start()

        if not self._ready.wait(timeout=4.0):
            self._error = self._error or "Microphone did not start in time"
            self._enabled = False
            return False

        if self._error and not self._enabled:
            return False

        self._use_vosk = _VOSK.ready
        self._enabled = True
        return True

    def stop(self) -> None:
        self._stop.set()
        self._enabled = False

    def poll_click(self) -> ClickType | None:
        try:
            click = self._queue.get_nowait()
            self._last_click = click
            self._last_click_at = time.time()
            return click
        except queue.Empty:
            return None

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def _tokens(self, normalized: str) -> set[str]:
        return {t for t in normalized.split() if t}

    def _match_command(self, text: str) -> ClickType | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        self._last_heard = normalized
        tokens = self._tokens(normalized)

        for phrases, click in _COMMANDS:
            for phrase in phrases:
                phrase_norm = self._normalize(phrase)
                if not phrase_norm:
                    continue
                if " " in phrase_norm:
                    if phrase_norm in normalized:
                        return click
                elif phrase_norm in tokens or normalized == phrase_norm:
                    return click
                elif len(phrase_norm) >= 4 and phrase_norm in normalized:
                    return click
        return None

    def _prepare_audio(self, audio_np):
        import numpy as np

        audio_np = self._trim_silence(audio_np)
        if audio_np is None or audio_np.size == 0:
            return None

        samples = audio_np.astype(np.float32)
        peak = float(np.max(np.abs(samples)))
        if peak < 60:
            return None
        target = 6000.0
        if peak < target:
            samples = samples * (target / peak)
        samples = np.clip(samples, -32767, 32767)
        return samples.astype(np.int16)

    def _trim_silence(self, audio_np):
        import numpy as np

        if audio_np.size == 0:
            return None

        samples = audio_np.astype(np.float32).flatten()
        window = max(80, int(16000 * 0.02))
        threshold = max(40.0, float(np.mean(np.abs(samples))) * 0.35)

        start = 0
        for i in range(0, len(samples) - window, window // 2):
            if float(np.mean(np.abs(samples[i : i + window]))) > threshold:
                start = i
                break

        end = len(samples)
        for i in range(len(samples) - window, 0, -window // 2):
            if float(np.mean(np.abs(samples[i : i + window]))) > threshold:
                end = min(len(samples), i + window)
                break

        trimmed = samples[start:end]
        if trimmed.size < int(16000 * 0.08):
            return None
        return trimmed.reshape(-1, 1).astype(np.int16)

    def _listen_loop(self) -> None:
        try:
            if not self._try_pyaudio_loop():
                self._sounddevice_loop()
        except Exception as exc:
            self._error = f"Voice listener crashed: {exc}"
            self._status = "error"
            self._enabled = False
            self._ready.set()

    def _try_pyaudio_loop(self) -> bool:
        import speech_recognition as sr

        try:
            microphone = sr.Microphone(device_index=config.VOICE_MIC_INDEX)
        except (OSError, AttributeError, ImportError, AssertionError):
            return False

        recognizer = sr.Recognizer()
        try:
            with microphone as source:
                self._status = f"listening — say {voice_command_hint()}"
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.25)
                except Exception:
                    pass
                self._ready.set()

                while not self._stop.is_set():
                    try:
                        audio = recognizer.listen(
                            source,
                            timeout=0.5,
                            phrase_time_limit=config.VOICE_PHRASE_TIME_LIMIT,
                        )
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as exc:
                        self._error = str(exc)
                        continue

                    self._enqueue_recognition(audio.get_raw_data(), 16000)
        except OSError as exc:
            self._error = f"Microphone unavailable: {exc}"
            self._status = "error"
            self._ready.set()
            return True

        return True

    def _sounddevice_loop(self) -> None:
        import numpy as np
        import sounddevice as sd

        rate = 16000
        block_samples = max(160, int(rate * config.VOICE_BLOCK_MS / 1000))
        segmenter = _SpeechSegmenter(rate)
        audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)

        def callback(indata, _frames, _time, status) -> None:
            if status:
                self._error = str(status)
            try:
                audio_q.put_nowait(indata.copy())
            except queue.Full:
                pass

        try:
            stream = sd.InputStream(
                samplerate=rate,
                channels=1,
                dtype="int16",
                device=config.VOICE_MIC_INDEX,
                blocksize=block_samples,
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            self._error = f"Microphone unavailable: {exc}"
            self._status = "error"
            self._ready.set()
            return

        mode = "fast local" if config.VOICE_USE_VOSK else "online"
        self._status = f"listening ({mode}) — say {voice_command_hint()}"
        self._ready.set()

        while not self._stop.is_set():
            try:
                block = audio_q.get(timeout=0.15)
            except queue.Empty:
                continue

            phrase = segmenter.feed(block)
            if phrase is None:
                continue

            prepared = self._prepare_audio(phrase)
            if prepared is None:
                continue

            self._status = "processing…"
            self._enqueue_recognition(prepared.tobytes(), rate)

        stream.stop()
        stream.close()

    def _enqueue_recognition(self, audio_bytes: bytes, sample_rate: int = 16000) -> None:
        item = (audio_bytes, sample_rate, time.time())
        while True:
            try:
                self._recognize_q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._recognize_q.get_nowait()
                except queue.Empty:
                    return

    def _recognize_worker(self) -> None:
        while not self._stop.is_set():
            try:
                audio_bytes, _sample_rate, _ts = self._recognize_q.get(timeout=0.15)
            except queue.Empty:
                continue

            self._recognize_and_queue(audio_bytes)

            if self._enabled and self._status == "processing…":
                mode = "fast local" if self._use_vosk else "online"
                self._status = f"listening ({mode}) — say {voice_command_hint()}"

    def _transcripts_from_google(self, audio_bytes: bytes, sample_rate: int) -> list[str]:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        audio = sr.AudioData(audio_bytes, sample_rate, 2)
        transcripts: list[str] = []

        try:
            text = recognizer.recognize_google(
                audio, language=config.VOICE_LANGUAGE
            )
            if text.strip():
                transcripts.append(text.strip())
                return transcripts
        except sr.UnknownValueError:
            pass
        except sr.RequestError as exc:
            self._error = f"Speech service error: {exc}"
            self._status = "error (need internet)"
            return transcripts

        if not config.VOICE_GOOGLE_SHOW_ALL:
            return transcripts

        try:
            result = recognizer.recognize_google(
                audio,
                language=config.VOICE_LANGUAGE,
                show_all=True,
            )
            if isinstance(result, dict):
                for alt in result.get("alternative", []):
                    text = alt.get("transcript", "").strip()
                    if text:
                        transcripts.append(text)
        except (sr.UnknownValueError, sr.RequestError):
            pass

        return transcripts

    def _recognize_and_queue(self, audio_bytes: bytes) -> None:
        transcripts: list[str] = []
        t0 = time.time()

        if config.VOICE_USE_VOSK:
            local = _VOSK.transcribe(audio_bytes)
            if local:
                transcripts.append(local)
                self._use_vosk = True

        if not transcripts and config.VOICE_GOOGLE_FALLBACK:
            transcripts = self._transcripts_from_google(audio_bytes, 16000)

        if not transcripts:
            self._status = "didn't catch that — try again"
            return

        if config.VOICE_DEBUG:
            ms = int((time.time() - t0) * 1000)
            print(f"[voice] {ms}ms heard: {transcripts}")

        click = self._match_command(transcripts[0])
        if not click:
            for text in transcripts[1:]:
                click = self._match_command(text)
                if click:
                    break

        if not click:
            self._status = f"heard '{self._last_heard}' — not a command"
            return

        now = time.time()
        if (
            click == self._last_click
            and now - self._last_click_at < config.VOICE_REPEAT_COOLDOWN
        ):
            return

        self._queue.put(click)
        self._last_click = click
        self._last_click_at = now
        label = {"left": "LEFT", "right": "RIGHT", "middle": "MIDDLE"}[click]
        self._status = f"{label} CLICK"
