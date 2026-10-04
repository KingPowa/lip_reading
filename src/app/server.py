"""HTTP transport for the independent PCM, activity, and camera channels."""

import json
import logging
import math
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlparse

import cv2

from src.app.rendering.live import build_page
from src.app.service import RecognitionService
from src.app.sessions import RecognitionBusyError
from src.config.live import (
    ACTIVITY_PATH,
    AUDIO_PATH,
    AUDIO_SAMPLE_RATE,
    AUDIO_WORKLET_PATH,
    CLIENT_PATH,
    END_SESSION_PATH,
    LANGUAGE_PATTERN,
    MAXIMUM_AUDIO_ACTIVITY_SECONDS,
    MAXIMUM_FRAME_BYTES,
    MAXIMUM_SESSION_SECONDS,
    MAXIMUM_TRANSCRIPTION_SECONDS,
    PCM_SAMPLE_BYTES,
    RENDERER_PATH,
    START_SESSION_PATH,
    TRACKING_PATH,
)

LOGGER = logging.getLogger(__name__)
STATIC_ASSETS = {
    "/audio-buffer.js": "audio-buffer.js",
    AUDIO_WORKLET_PATH: "audio-worklet.js",
    CLIENT_PATH: "client.js",
    RENDERER_PATH: "renderer.js",
}


class LiveServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: RecognitionService,
        transcription_interval: float,
    ) -> None:
        super().__init__(address, LiveRequestHandler)
        self.service = service
        self.page = build_page(round(transcription_interval * 1000), service.language)


class LiveRequestHandler(BaseHTTPRequestHandler):
    server: LiveServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_body(self.server.page.encode(), "text/html; charset=utf-8")
            return
        filename = STATIC_ASSETS.get(path)
        if filename is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = files("src.app").joinpath("static", filename).read_bytes()
        self._send_body(body, "text/javascript; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in (
            AUDIO_PATH,
            ACTIVITY_PATH,
            TRACKING_PATH,
            END_SESSION_PATH,
            START_SESSION_PATH,
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
            session_id = self.headers.get("X-Capture-Session", "")
            if path == START_SESSION_PATH:
                if content_length != 0:
                    raise ValueError("starting a session requires an empty body")
                language = self._language()
                result = self.server.service.start_session(session_id, language)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == END_SESSION_PATH:
                if content_length != 0:
                    raise ValueError("ending a session requires an empty body")
                self.server.service.close_session(session_id)
                self._send_json(HTTPStatus.OK, {"sessionId": session_id})
                return
            maximum_seconds = (
                MAXIMUM_AUDIO_ACTIVITY_SECONDS
                if path == ACTIVITY_PATH
                else MAXIMUM_TRANSCRIPTION_SECONDS
            )
            maximum_bytes = (
                MAXIMUM_FRAME_BYTES
                if path == TRACKING_PATH
                else round(maximum_seconds * AUDIO_SAMPLE_RATE) * PCM_SAMPLE_BYTES
            )
            final_flush = (
                path == AUDIO_PATH and self.headers.get("X-Capture-Final") == "true"
            )
            minimum_bytes = 0 if final_flush else 1
            if not minimum_bytes <= content_length <= maximum_bytes:
                raise ValueError(
                    f"request body must be between 1 and {maximum_bytes} bytes"
                )
            content = self.rfile.read(content_length)
            if len(content) != content_length:
                raise ValueError("incomplete upload")
            timestamp = float(self.headers.get("X-Capture-Offset", "nan"))
            if (
                not math.isfinite(timestamp)
                or not 0 <= timestamp <= MAXIMUM_SESSION_SECONDS
            ):
                raise ValueError("invalid capture timestamp")
            result = self._dispatch(path, content, session_id, timestamp)
        except RecognitionBusyError as error:
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
            return
        except (RuntimeError, ValueError, cv2.error) as error:
            LOGGER.warning("live request failed: %s", error)
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(HTTPStatus.OK, result)

    def _dispatch(
        self, path: str, content: bytes, session_id: str, timestamp: float
    ) -> dict:
        if path == TRACKING_PATH:
            return self.server.service.track_frame(content, session_id, timestamp)
        if path == ACTIVITY_PATH:
            return self.server.service.update_activity(content, session_id, timestamp)
        language = self._language()
        job_id = self.headers.get("X-Capture-Job", "")
        if not re.fullmatch(r"[0-9]{1,12}", job_id):
            raise ValueError("invalid audio job identifier")
        final = self.headers.get("X-Capture-Final", "false")
        if final not in ("true", "false"):
            raise ValueError("invalid audio final flag")
        return self.server.service.recognize_audio(
            content, language, session_id, timestamp, final == "true", job_id
        )

    def _language(self) -> str | None:
        language = self.headers.get("X-Whisper-Language") or None
        if language and not re.fullmatch(LANGUAGE_PATTERN, language):
            raise ValueError("invalid transcription language")
        return language

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self._send_body(
            json.dumps(payload, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def _send_body(
        self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.debug("client disconnected")

    def log_message(self, format_string: str, *arguments: object) -> None:
        LOGGER.debug("%s - %s", self.address_string(), format_string % arguments)
