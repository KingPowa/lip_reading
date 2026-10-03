"""Run the local browser app that captions speakers from a camera and microphone."""

import argparse
import json
import logging
import math
import re
import ssl
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from .config import FACE_TRACK_IOU_THRESHOLD, LIGHT_ASD_FILENAME, MODEL_DIRECTORY
from .contracts import FaceBox, FaceTrack
from .light_asd import warm_up_activity_model
from .live_tracking import LiveTracker
from .live_view import build_page
from .main import recognize
from .media import MediaError, extract_audio, inspect_media, normalize_camera_recording
from .person_segmentation import warm_up_person_model
from .render import track_payloads
from .speech import (
    NORMALIZED_AUDIO_SAMPLE_RATE,
    load_faster_whisper_model,
    warm_up_speaker_model,
)
from .vision import intersection_over_union

LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CHUNK_SECONDS = 4
DEFAULT_LIVE_WHISPER_MODEL = "base"
DEFAULT_WARMUP_LANGUAGE = "en"
MINIMUM_CHUNK_SECONDS = 1
MAXIMUM_CHUNK_SECONDS = 10
MAXIMUM_UPLOAD_BYTES = 100 * 1024 * 1024
RECOGNITION_PATH = "/recognize"
TRACKING_PATH = "/track"
END_SESSION_PATH = "/end"
LIVE_CONTEXT_SECONDS = 3.0
LIVE_COMMIT_DELAY_SECONDS = 1.0
MAXIMUM_PROMPT_CHARACTERS = 240
MAXIMUM_SESSIONS = 16
SESSION_IDLE_SECONDS = 900
MAXIMUM_FRAME_BYTES = 2 * 1024 * 1024
MAXIMUM_SESSION_SECONDS = 24 * 60 * 60
# GCD ignores a thread count of one on macOS; zero disables its parallel pool.
LIVE_OPENCV_THREADS = 0


@dataclass
class LiveSession:
    voice_centroids: list = field(default_factory=list)
    voice_labels: dict[str, str] = field(default_factory=dict)
    track_labels: dict[str, str] = field(default_factory=dict)
    latest_face_boxes: dict[str, FaceBox] = field(default_factory=dict)
    next_face_number: int = 1
    committed_until: float = 0.0
    prompt: str = ""
    language: str | None = None
    last_job_id: str | None = None
    last_result: dict[str, object] | None = None
    touched_at: float = field(default_factory=time.monotonic)
    tracker: LiveTracker | None = None


class RecognitionBusyError(RuntimeError):
    """Raised when the client should retry a retained recording."""


class RecognitionService:
    """Serialize model work while browser requests may arrive concurrently."""

    def __init__(
        self,
        whisper_model: str,
        language: str | None,
        model_directory: Path,
        activity_device: str = "auto",
    ) -> None:
        self.whisper_model = whisper_model
        self.language = language
        self.model_directory = model_directory
        self.activity_device = activity_device
        if self.activity_device == "auto" and torch.cuda.is_available():
            self.activity_device = "cuda"
        if self.activity_device == "auto" and torch.backends.mps.is_available():
            self.activity_device = "mps"
        if self.activity_device == "auto":
            self.activity_device = "cpu"
        self.lock = threading.Lock()
        self.transcription_model: object | None = None
        self.sessions: dict[str, LiveSession] = {}
        self.sessions_lock = threading.Lock()

    def session(self, session_id: str) -> LiveSession:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid capture session")
        with self.sessions_lock:
            now = time.monotonic()
            expired = [
                key
                for key, session in self.sessions.items()
                if now - session.touched_at > SESSION_IDLE_SECONDS
            ]
            for key in expired:
                session = self.sessions.pop(key)
                if session.tracker is not None:
                    session.tracker.close()
            if session_id not in self.sessions:
                if len(self.sessions) >= MAXIMUM_SESSIONS:
                    raise RecognitionBusyError(
                        "too many capture sessions; retry after an idle session expires"
                    )
                self.sessions[session_id] = LiveSession()
            session = self.sessions[session_id]
            session.touched_at = now
            return session

    def recognize_clip(
        self,
        content: bytes,
        language: str | None = None,
        session_id: str = "legacy",
        capture_offset: float = 0.0,
        final: bool = True,
        job_id: str | None = None,
    ) -> dict[str, object]:
        """Recognize one self-contained browser recording and return overlay data."""
        if not self.lock.acquire(blocking=False):
            raise RecognitionBusyError(
                "recognizer is busy; recording retained for retry"
            )
        try:
            session = self.session(session_id)
            if job_id is not None and job_id == session.last_job_id:
                return session.last_result
            effective_language = language or self.language
            if (
                session.last_result is not None
                and effective_language != session.language
            ):
                raise ValueError("start a new capture session to change language")
            result = self._recognize_clip(
                content, effective_language, session, capture_offset, final
            )
            result["sessionId"] = session_id
            result["jobId"] = job_id
            session.language = effective_language
            session.last_job_id = job_id
            session.last_result = result
            session.touched_at = time.monotonic()
            return result
        finally:
            self.lock.release()

    def _recognize_clip(
        self,
        content: bytes,
        language: str | None,
        session: LiveSession,
        capture_offset: float,
        final: bool,
    ) -> dict[str, object]:
        started_at = time.perf_counter()
        timings: dict[str, float] = {}
        seek_seconds = max(
            0.0, session.committed_until - capture_offset - LIVE_CONTEXT_SECONDS
        )
        clip_offset = capture_offset + seek_seconds
        with tempfile.TemporaryDirectory(prefix="live-speaker-") as temporary_directory:
            recording_path = Path(temporary_directory) / "recording.capture"
            normalized_path = Path(temporary_directory) / "recording.mp4"
            audio_path = Path(temporary_directory) / "audio.wav"
            recording_path.write_bytes(content)
            normalize_camera_recording(recording_path, normalized_path, seek_seconds)
            timings["normalize"] = time.perf_counter() - started_at
            stage_started = time.perf_counter()
            extract_audio(normalized_path, audio_path)
            timings["audio"] = time.perf_counter() - stage_started
            normalized_info = inspect_media(normalized_path)
            capture_end = clip_offset + normalized_info.duration
            commit_until = (
                capture_end
                if final
                else max(
                    session.committed_until, capture_end - LIVE_COMMIT_DELAY_SECONDS
                )
            )
            transformer = partial(self._stabilize_tracks, session=session)
            if session.tracker is not None:
                transformer = session.tracker.association_callback(clip_offset)
            info, tracks, events, _, words = recognize(
                normalized_path,
                audio_path,
                self.whisper_model,
                language or self.language,
                self.model_directory,
                self._full_transcription_model(),
                speaker_centroids=session.voice_centroids,
                voice_labels=session.voice_labels,
                track_labels=session.track_labels,
                live=True,
                track_id_transformer=transformer,
                initial_prompt=session.prompt or None,
                minimum_word_timestamp=session.committed_until - clip_offset,
                maximum_word_timestamp=float("inf")
                if final
                else commit_until - clip_offset,
                timings=timings,
                activity_device=self.activity_device,
            )
        events = [
            replace(event, start=event.start + clip_offset, end=event.end + clip_offset)
            for event in events
        ]
        deferred_words = (
            []
            if final
            else [word for word in words if word.end > commit_until - clip_offset]
        )
        provisional_text = " ".join(word.text for word in deferred_words)
        if deferred_words:
            commit_until = min(
                commit_until, clip_offset + min(word.start for word in deferred_words)
            )
        session.prompt = " ".join([session.prompt, *(event.text for event in events)])[
            -MAXIMUM_PROMPT_CHARACTERS:
        ]
        session.committed_until = max(session.committed_until, commit_until)
        timings["total"] = time.perf_counter() - started_at
        LOGGER.info(
            "recognized capture %.2f–%.2fs; committed %.2fs; %d tracks, %d events; device=%s timings=%s",
            clip_offset,
            capture_end,
            session.committed_until,
            len(tracks),
            len(events),
            self.activity_device,
            {name: round(seconds, 3) for name, seconds in timings.items()},
        )
        return {
            "duration": info.duration,
            "width": info.width,
            "height": info.height,
            "tracks": track_payloads(tracks, info.width, info.height),
            "events": [event.as_json() for event in events],
            "provisionalText": provisional_text,
            "captureStart": clip_offset,
            "captureEnd": capture_end,
            "committedUntil": session.committed_until,
            "timings": timings,
        }

    def track_frame(
        self, content: bytes, session_id: str, timestamp: float
    ) -> dict[str, object]:
        session = self.session(session_id)
        with self.sessions_lock:
            if session.tracker is None:
                session.tracker = LiveTracker(self.model_directory)
        result = session.tracker.track(content, timestamp)
        return result | {"sessionId": session_id, "timestamp": timestamp}

    def close(self) -> None:
        for session in self.sessions.values():
            if session.tracker is not None:
                session.tracker.close()

    def close_session(self, session_id: str) -> None:
        """Release an ended capture after its outstanding recognition completes."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid capture session")
        with self.lock, self.sessions_lock:
            session = self.sessions.pop(session_id, None)
            if session is not None and session.tracker is not None:
                session.tracker.close()

    def _full_transcription_model(self) -> object:
        if self.transcription_model is None:
            self.transcription_model = load_faster_whisper_model(self.whisper_model)
        return self.transcription_model

    def warm_up(self) -> None:
        """Load reusable models before camera capture begins."""
        with self.lock:
            self._full_transcription_model().transcribe(
                np.zeros(NORMALIZED_AUDIO_SAMPLE_RATE, dtype=np.float32),
                language=self.language or DEFAULT_WARMUP_LANGUAGE,
                fp16=False,
                word_timestamps=True,
                temperature=0.0,
                verbose=None,
            )
            try:
                warm_up_activity_model(
                    self.model_directory / LIGHT_ASD_FILENAME, self.activity_device
                )
            except (RuntimeError, NotImplementedError):
                if self.activity_device == "cpu":
                    raise
                LOGGER.exception(
                    "activity backend %s failed warm-up; using CPU",
                    self.activity_device,
                )
                self.activity_device = "cpu"
                warm_up_activity_model(
                    self.model_directory / LIGHT_ASD_FILENAME, self.activity_device
                )
            warm_up_speaker_model(self.model_directory)
            warm_up_person_model(self.model_directory)
            cv2.setNumThreads(LIVE_OPENCV_THREADS)
            LOGGER.info(
                "live inference: Light-ASD=%s, Torch threads=%d, OpenCV threads=%d",
                self.activity_device,
                torch.get_num_threads(),
                cv2.getNumThreads(),
            )
            LOGGER.info(
                "live transcription: faster-whisper %s on CPU INT8", self.whisper_model
            )

    def _stabilize_tracks(
        self, tracks: list[FaceTrack], session: LiveSession | None = None
    ) -> None:
        session = session or self.session("legacy")
        available_boxes = dict(session.latest_face_boxes)
        current_boxes: dict[str, FaceBox] = {}
        for track in tracks:
            last_box = track.boxes[-1]
            track_id = max(
                available_boxes,
                key=lambda candidate: intersection_over_union(
                    available_boxes[candidate], last_box
                ),
                default=None,
            )
            if (
                track_id is None
                or intersection_over_union(available_boxes[track_id], last_box)
                < FACE_TRACK_IOU_THRESHOLD
            ):
                track_id = f"live-face-{session.next_face_number}"
                session.next_face_number += 1
            else:
                del available_boxes[track_id]
            track.track_id = track_id
            current_boxes[track_id] = last_box
        session.latest_face_boxes = current_boxes


class LiveServer(ThreadingHTTPServer):
    """HTTP server carrying the shared recognizer and page configuration."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: RecognitionService,
        chunk_milliseconds: int,
    ) -> None:
        super().__init__(address, LiveRequestHandler)
        self.service = service
        self.page = build_page(chunk_milliseconds, service.language)


class LiveRequestHandler(BaseHTTPRequestHandler):
    """Serve the camera page and accept same-origin camera recordings."""

    server: LiveServer

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = self.server.page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in (RECOGNITION_PATH, TRACKING_PATH, END_SESSION_PATH):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
            session_id = self.headers.get("X-Capture-Session", "legacy")
            if path == END_SESSION_PATH:
                if content_length != 0:
                    raise ValueError("ending a capture session requires an empty body")
                self.server.service.close_session(session_id)
                self._send_json(HTTPStatus.OK, {"sessionId": session_id})
                return
            maximum_bytes = (
                MAXIMUM_FRAME_BYTES if path == TRACKING_PATH else MAXIMUM_UPLOAD_BYTES
            )
            if not 0 < content_length <= maximum_bytes:
                raise ValueError(
                    f"recording must be between 1 byte and {MAXIMUM_UPLOAD_BYTES // (1024 * 1024)} MB"
                )
            content = self.rfile.read(content_length)
            if len(content) != content_length:
                raise ValueError("incomplete recording upload")
            capture_offset = float(self.headers.get("X-Capture-Offset", "0"))
            if (
                not math.isfinite(capture_offset)
                or not 0 <= capture_offset <= MAXIMUM_SESSION_SECONDS
            ):
                raise ValueError("invalid capture timestamp")
            language = self.headers.get("X-Whisper-Language")
            if language and not re.fullmatch(r"[A-Za-z-]{2,16}", language):
                raise ValueError("invalid transcription language")
            if path == TRACKING_PATH:
                result = self.server.service.track_frame(
                    content, session_id, capture_offset
                )
            else:
                result = self.server.service.recognize_clip(
                    content,
                    language,
                    session_id,
                    capture_offset,
                    self.headers.get("X-Capture-Final", "true") == "true",
                    self.headers.get("X-Capture-Job"),
                )
        except RecognitionBusyError as error:
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
            return
        except (MediaError, RuntimeError, ValueError) as error:
            LOGGER.exception("live recognition failed")
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(HTTPStatus.OK, result)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *arguments: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % arguments)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line for the local camera application."""
    parser = argparse.ArgumentParser(
        description="Run local camera active-speaker captions in a browser"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Bind address; use 0.0.0.0 for another device on your network",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP(S) port")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help="Seconds per recognized clip",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_LIVE_WHISPER_MODEL,
        help="Local Whisper model name",
    )
    parser.add_argument(
        "--language",
        help="ISO language code for Whisper; choose it to avoid short-clip misdetection",
    )
    parser.add_argument(
        "--activity-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Light-ASD backend; auto uses an available GPU",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=MODEL_DIRECTORY,
        help="Local model download directory",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        help="TLS certificate required for camera access from another device",
    )
    parser.add_argument("--key", type=Path, help="TLS private key")
    return parser


def serve(arguments: argparse.Namespace) -> None:
    """Start the camera page until interrupted."""
    if not MINIMUM_CHUNK_SECONDS <= arguments.chunk_seconds <= MAXIMUM_CHUNK_SECONDS:
        raise ValueError(
            f"--chunk-seconds must be {MINIMUM_CHUNK_SECONDS} through {MAXIMUM_CHUNK_SECONDS}"
        )
    if bool(arguments.cert) != bool(arguments.key):
        raise ValueError("--cert and --key must be supplied together")
    service = RecognitionService(
        arguments.whisper_model,
        arguments.language,
        arguments.model_directory,
        arguments.activity_device,
    )
    service.warm_up()
    server = LiveServer(
        (arguments.host, arguments.port),
        service,
        arguments.chunk_seconds * 1000,
    )
    scheme = "https" if arguments.cert else "http"
    if arguments.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(arguments.cert, arguments.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    LOGGER.info("open %s://%s:%d", scheme, arguments.host, arguments.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


def main() -> None:
    """Parse arguments and run the local live-speaker application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        serve(arguments)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
