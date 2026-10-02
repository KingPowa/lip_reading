"""Run the local browser app that captions speakers from a camera and microphone."""

import argparse
import json
import logging
import ssl
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .config import FACE_TRACK_IOU_THRESHOLD, MODEL_DIRECTORY
from .contracts import FaceBox, FaceTrack
from .live_view import build_page
from .main import recognize
from .media import MediaError, extract_audio, normalize_camera_recording
from .render import track_payloads
from .speech import load_transcription_model
from .vision import intersection_over_union


LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CHUNK_SECONDS = 2
DEFAULT_LIVE_WHISPER_MODEL = "base"
MINIMUM_CHUNK_SECONDS = 1
MAXIMUM_CHUNK_SECONDS = 10
MAXIMUM_UPLOAD_BYTES = 100 * 1024 * 1024
RECOGNITION_PATH = "/recognize"


class RecognitionBusyError(RuntimeError):
    """Raised when a newer live segment would otherwise wait behind stale work."""


class RecognitionService:
    """Serialize model work while browser requests may arrive concurrently."""

    def __init__(
        self,
        whisper_model: str,
        language: str | None,
        model_directory: Path,
    ) -> None:
        self.whisper_model = whisper_model
        self.language = language
        self.model_directory = model_directory
        self.lock = threading.Lock()
        self.transcription_model: object | None = None
        self.track_labels: dict[str, str] = {}
        self.latest_face_boxes: dict[str, FaceBox] = {}
        self.next_face_number = 1

    def recognize_clip(self, content: bytes, language: str | None = None) -> dict[str, object]:
        """Recognize one self-contained browser recording and return overlay data."""
        if not self.lock.acquire(blocking=False):
            raise RecognitionBusyError("recognizer is busy; keeping the live transcript current")
        try:
            return self._recognize_clip(content, language)
        finally:
            self.lock.release()

    def _recognize_clip(self, content: bytes, language: str | None) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="live-speaker-") as temporary_directory:
            recording_path = Path(temporary_directory) / "recording.capture"
            normalized_path = Path(temporary_directory) / "recording.mp4"
            audio_path = Path(temporary_directory) / "audio.wav"
            recording_path.write_bytes(content)
            normalize_camera_recording(recording_path, normalized_path)
            extract_audio(normalized_path, audio_path)
            info, tracks, events, _, _ = recognize(
                normalized_path,
                audio_path,
                self.whisper_model,
                language or self.language,
                self.model_directory,
                self._full_transcription_model(),
                track_labels=self.track_labels,
                live=True,
                track_id_transformer=self._stabilize_tracks,
            )
        return {
            "duration": info.duration,
            "width": info.width,
            "height": info.height,
            "tracks": track_payloads(tracks, info.width, info.height),
            "events": [event.as_json() for event in events],
        }

    def _full_transcription_model(self) -> object:
        if self.transcription_model is None:
            self.transcription_model = load_transcription_model(
                self.whisper_model, self.model_directory / "whisper"
            )
        return self.transcription_model

    def warm_up(self) -> None:
        """Load the transcription model before camera capture begins."""
        with self.lock:
            self._full_transcription_model()

    def _stabilize_tracks(self, tracks: list[FaceTrack]) -> None:
        available_boxes = dict(self.latest_face_boxes)
        current_boxes: dict[str, FaceBox] = {}
        for track in tracks:
            last_box = track.boxes[-1]
            track_id = max(
                available_boxes,
                key=lambda candidate: intersection_over_union(available_boxes[candidate], last_box),
                default=None,
            )
            if track_id is None or intersection_over_union(available_boxes[track_id], last_box) < FACE_TRACK_IOU_THRESHOLD:
                track_id = f"live-face-{self.next_face_number}"
                self.next_face_number += 1
            else:
                del available_boxes[track_id]
            track.track_id = track_id
            current_boxes[track_id] = last_box
        self.latest_face_boxes = current_boxes

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
        if path != RECOGNITION_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
            if not 0 < content_length <= MAXIMUM_UPLOAD_BYTES:
                raise ValueError(f"recording must be between 1 byte and {MAXIMUM_UPLOAD_BYTES // (1024 * 1024)} MB")
            content = self.rfile.read(content_length)
            result = self.server.service.recognize_clip(content, self.headers.get("X-Whisper-Language"))
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
    parser = argparse.ArgumentParser(description="Run local camera active-speaker captions in a browser")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address; use 0.0.0.0 for another device on your network")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP(S) port")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS, help="Seconds per recognized clip")
    parser.add_argument("--whisper-model", default=DEFAULT_LIVE_WHISPER_MODEL, help="Local Whisper model name")
    parser.add_argument("--language", help="ISO language code for Whisper; choose it to avoid short-clip misdetection")
    parser.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY, help="Local model download directory")
    parser.add_argument("--cert", type=Path, help="TLS certificate required for camera access from another device")
    parser.add_argument("--key", type=Path, help="TLS private key")
    return parser


def serve(arguments: argparse.Namespace) -> None:
    """Start the camera page until interrupted."""
    if not MINIMUM_CHUNK_SECONDS <= arguments.chunk_seconds <= MAXIMUM_CHUNK_SECONDS:
        raise ValueError(f"--chunk-seconds must be {MINIMUM_CHUNK_SECONDS} through {MAXIMUM_CHUNK_SECONDS}")
    if bool(arguments.cert) != bool(arguments.key):
        raise ValueError("--cert and --key must be supplied together")
    service = RecognitionService(
        arguments.whisper_model,
        arguments.language,
        arguments.model_directory,
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


def main() -> None:
    """Parse arguments and run the local live-speaker application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        serve(arguments)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
