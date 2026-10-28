"""Capture lifecycle and per-session state shared by independent request channels."""

import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.config.live import MAXIMUM_SESSIONS, SESSION_ID_PATTERN, SESSION_IDLE_SECONDS
from src.core.audio.realtime import MultilingualTranscriber, StreamingTranscriber
from src.core.tracking.realtime import RealtimeTracker
from src.core.tracking.voice import SpeechBuffer
from src.core.vision.live import LiveFaceDetector


class RecognitionBusyError(RuntimeError):
    """A bounded worker is occupied; callers retain audio and retry."""


@dataclass
class LiveSession:
    lock: threading.RLock = field(default_factory=threading.RLock)
    vision_lock: threading.Lock = field(default_factory=threading.Lock)
    tracker: RealtimeTracker = field(default_factory=RealtimeTracker)
    detector: LiveFaceDetector | None = None
    audio_lock: threading.Lock = field(default_factory=threading.Lock)
    speech_models: dict[str, object] = field(default_factory=dict)
    transcriber: StreamingTranscriber | MultilingualTranscriber | None = None
    audio_finished: bool = False
    received_until: float = 0.0
    languages: tuple[str, ...] = ()
    last_job_id: str | None = None
    last_result: dict | None = None
    touched_at: float = field(default_factory=time.monotonic)
    active_requests: int = 0
    speech_buffer: SpeechBuffer = field(default_factory=SpeechBuffer)
    voice_job_active: bool = False
    closed: bool = False

    def close(self) -> None:
        self.closed = True
        if self.detector is not None:
            self.detector.close()
            self.detector = None


class SessionRegistry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.sessions: dict[str, LiveSession] = {}

    @staticmethod
    def validate(session_id: str) -> None:
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            raise ValueError("invalid capture session")

    @contextmanager
    def use(self, session_id: str):
        self.validate(session_id)
        with self.lock:
            self._expire_idle()
            session = self.sessions.get(session_id)
            if session is None:
                if len(self.sessions) >= MAXIMUM_SESSIONS:
                    raise RecognitionBusyError("too many capture sessions")
                session = LiveSession()
                self.sessions[session_id] = session
            session.active_requests += 1
            session.touched_at = time.monotonic()
        try:
            yield session
        finally:
            with self.lock:
                session.active_requests -= 1
                session.touched_at = time.monotonic()

    def end(self, session_id: str) -> None:
        self.validate(session_id)
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            if session.active_requests:
                raise RecognitionBusyError("capture requests are still finishing")
            session.close()
            del self.sessions[session_id]

    def close(self) -> None:
        with self.lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()

    def _expire_idle(self) -> None:
        now = time.monotonic()
        expired = [
            identity
            for identity, session in self.sessions.items()
            if not session.active_requests
            and now - session.touched_at > SESSION_IDLE_SECONDS
        ]
        for identity in expired:
            self.sessions.pop(identity).close()
