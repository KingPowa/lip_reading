"""Coordinate independent audio and camera channels; adapt domain output for UI."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.app.rendering.payloads import caption_payload, face_payload, speaker_payload
from src.app.sessions import LiveSession, RecognitionBusyError, SessionRegistry
from src.config.live import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_LANGUAGE,
    LIVE_OPENCV_THREADS,
    MAXIMUM_AUDIO_ACTIVITY_SECONDS,
)
from src.config.streaming import SAMPLE_TIMESTAMP_TOLERANCE
from src.core.audio.models import SpeechModelStore
from src.core.audio.language import (
    LanguageIdentifier,
    load_language_identifier,
    parse_languages,
)
from src.core.audio.realtime import (
    MultilingualTranscriber,
    StreamingTranscriber,
    decode_pcm,
)
from src.core.audio.speech import embed_samples, warm_up_speaker_model
from src.core.vision.live import LiveFaceDetector

LOGGER = logging.getLogger(__name__)


class RecognitionService:
    def __init__(
        self,
        language: str | None,
        model_directory: Path,
        speech_model: Path | None = None,
    ) -> None:
        self.languages = parse_languages(language, speech_model is not None)
        self.language = ",".join(self.languages) or None
        self.custom_model = speech_model is not None
        self.model_directory = model_directory
        self.registry = SessionRegistry()
        self.models = SpeechModelStore(model_directory, speech_model)
        self.identifier: LanguageIdentifier | None = None
        self.identifier_lock = threading.Lock()
        self.identifier_attempted = False
        self.speaker_classifier: object | None = None
        self.voice_executor: ThreadPoolExecutor | None = None

    def warm_up(self) -> None:
        cv2.setNumThreads(LIVE_OPENCV_THREADS)
        languages = self.languages or (DEFAULT_LANGUAGE,)
        for language in languages:
            StreamingTranscriber(self.models.load(language), 0).accept(
                np.zeros(AUDIO_SAMPLE_RATE, dtype="<i2").tobytes(), True
            )
        self._load_identifier(languages)
        self._warm_up_voice()
        detector = None
        try:
            detector = LiveFaceDetector(self.model_directory)
            image = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))[
                1
            ].tobytes()
            detector.detect(image, 0.0)
        except (RuntimeError, cv2.error) as error:
            LOGGER.warning(
                "camera warm-up failed; microphone captions remain available: %s", error
            )
        finally:
            if detector is not None:
                detector.close()

    def _warm_up_voice(self) -> None:
        try:
            self.enable_voice(warm_up_speaker_model(self.model_directory))
        except (RuntimeError, ImportError, OSError, ValueError) as error:
            LOGGER.warning(
                "voice identity unavailable; speakers use faces only: %s", error
            )

    def enable_voice(self, classifier: object) -> None:
        self.speaker_classifier = classifier
        if self.voice_executor is None:
            self.voice_executor = ThreadPoolExecutor(max_workers=1)

    def _load_identifier(self, languages: tuple[str, ...]) -> None:
        """Load the shared language-ID model once, only for multilingual use."""
        if len(languages) < 2:
            return
        with self.identifier_lock:
            if not self.identifier_attempted:
                self.identifier_attempted = True
                self.identifier = load_language_identifier(self.model_directory)

    def _session_languages(self, language: str | None) -> tuple[str, ...]:
        requested = parse_languages(language, self.custom_model)
        return requested or self.languages or (DEFAULT_LANGUAGE,)

    def start_session(self, session_id: str, language: str | None) -> dict:
        languages = self._session_languages(language)
        with self.registry.use(session_id) as session:
            models = {name: self.models.load(name) for name in languages}
            self._load_identifier(languages)
            with session.lock:
                if session.languages and session.languages != languages:
                    raise ValueError("start a new capture session to change language")
                session.languages = languages
                session.speech_models = models
        return {
            "sessionId": session_id,
            "languages": list(languages),
            "language": languages[0],
        }

    def update_activity(
        self, content: bytes, session_id: str, timestamp: float
    ) -> dict:
        samples = decode_pcm(content, MAXIMUM_AUDIO_ACTIVITY_SECONDS)
        with self.registry.use(session_id) as session, session.lock:
            estimate = session.tracker.update_audio(samples, timestamp)
            self._buffer_voice(session_id, session, samples, timestamp)
            return {"sessionId": session_id, "speaker": speaker_payload(estimate)}

    def _buffer_voice(
        self, session_id: str, session: LiveSession, samples: np.ndarray, timestamp: float
    ) -> None:
        executor = self.voice_executor
        if executor is None:
            return
        session.speech_buffer.add(samples, timestamp, session.tracker.speaking)
        if session.voice_job_active or not session.speech_buffer.ready():
            return
        window = session.speech_buffer.take()
        session.voice_job_active = True
        try:
            executor.submit(self._embed_voice, session_id, session, *window)
        except RuntimeError:
            session.voice_job_active = False

    def _embed_voice(
        self,
        session_id: str,
        session: LiveSession,
        samples: np.ndarray,
        start: float,
        end: float,
    ) -> None:
        embedding = None
        try:
            embedding = embed_samples(self.speaker_classifier, samples)
        except Exception as error:  # noqa: BLE001 - voice must never break captions
            LOGGER.warning("voice embedding failed; voice identity disabled: %s", error)
            self.voice_executor = None
        with session.lock:
            session.voice_job_active = False
            if embedding is not None and not session.closed:
                session.tracker.update_voice(embedding, start, end)

    def track_frame(self, content: bytes, session_id: str, timestamp: float) -> dict:
        started = time.perf_counter()
        with self.registry.use(session_id) as session, session.vision_lock:
            if session.detector is None:
                session.detector = LiveFaceDetector(self.model_directory)
            width, height, observations = session.detector.detect(content, timestamp)
            with session.lock:
                estimate = session.tracker.update_faces(observations, timestamp)
                faces = list(session.tracker.current_faces)
        return {
            "sessionId": session_id,
            "timestamp": timestamp,
            "width": width,
            "height": height,
            "faces": [face_payload(face) for face in faces],
            "speaker": speaker_payload(estimate),
            "timings": {"vision": time.perf_counter() - started},
        }

    def recognize_audio(
        self,
        content: bytes,
        language: str | None,
        session_id: str,
        capture_offset: float,
        final: bool,
        job_id: str,
    ) -> dict:
        if content:
            samples = decode_pcm(content)
        elif not final:
            raise ValueError("empty audio requires a final flush")
        else:
            samples = np.empty(0, dtype=np.float32)
        started = time.perf_counter()
        with self.registry.use(session_id) as session:
            if not session.audio_lock.acquire(blocking=False):
                raise RecognitionBusyError(
                    "this session is decoding audio; retain PCM and retry"
                )
            try:
                if job_id == session.last_job_id:
                    return session.last_result
                if not session.speech_models:
                    raise ValueError(
                        "prepare the speech language with /start before capture"
                    )
                requested = parse_languages(language, self.custom_model)
                if requested and requested != session.languages:
                    raise ValueError("start a new capture session to change language")
                if session.audio_finished:
                    raise ValueError("audio stream has already finished")
                if session.transcriber is None:
                    session.transcriber = self._build_transcriber(
                        session, capture_offset
                    )
                    session.received_until = capture_offset
                if (
                    abs(capture_offset - session.received_until)
                    > SAMPLE_TIMESTAMP_TOLERANCE
                ):
                    raise ValueError(
                        "audio packets must be contiguous and in capture order"
                    )
                update = session.transcriber.accept(content, final)
                capture_end = capture_offset + len(samples) / AUDIO_SAMPLE_RATE
                with session.lock:
                    events = session.tracker.caption_events(update.words)
                    session.received_until = capture_end
                    session.audio_finished = final
                    session.last_job_id = job_id
                    result = {
                        "sessionId": session_id,
                        "jobId": job_id,
                        "captureStart": capture_offset,
                        "captureEnd": capture_end,
                        "acknowledgedUntil": capture_end,
                        "events": [caption_payload(event) for event in events],
                        "provisionalText": update.partial_text,
                        "language": update.language or session.languages[0],
                        "timings": {"transcription": time.perf_counter() - started},
                    }
                    session.last_result = result
                return result
            finally:
                session.audio_lock.release()

    def _build_transcriber(
        self, session: LiveSession, origin: float
    ) -> StreamingTranscriber | MultilingualTranscriber:
        if len(session.languages) == 1:
            return StreamingTranscriber(
                session.speech_models[session.languages[0]], origin
            )
        identify = self.identifier.probabilities if self.identifier else None
        return MultilingualTranscriber(session.speech_models, origin, identify)

    def close_session(self, session_id: str) -> None:
        self.registry.end(session_id)

    def close(self) -> None:
        if self.voice_executor is not None:
            self.voice_executor.shutdown(wait=False, cancel_futures=True)
        self.registry.close()
