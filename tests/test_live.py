import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.app.live import build_parser
from src.app.rendering.live import build_page
from src.app.service import RecognitionService
from src.app.sessions import RecognitionBusyError, SessionRegistry
from src.config.live import MAXIMUM_SESSIONS
from src.core.audio.realtime import TranscriptionUpdate
from src.core.contracts import FaceBox, Word


class LiveServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = RecognitionService("it", Path("models"))
        self.service.models = Mock()
        self.service.start_session("camera", "it")
        self.decoder = Mock()
        self.decoder.accept.return_value = TranscriptionUpdate(
            [Word("ciao", 0.1, 0.4, 0.95)], ""
        )
        with self.service.registry.use("camera") as session:
            session.transcriber = self.decoder
        self.pcm = np.full(1600, 1600, dtype="<i2").tobytes()
        self.addCleanup(self.service.close)

    def recognize(self, **options):
        defaults = {
            "language": "it",
            "session_id": "camera",
            "capture_offset": 0.0,
            "final": False,
            "job_id": "1",
        }
        return self.service.recognize_audio(self.pcm, **(defaults | options))

    def test_audio_transcribes_without_camera_or_activity_requests(self):
        result = self.recognize()
        self.assertEqual(result["events"][0]["text"], "ciao")
        self.assertEqual(result["events"][0]["state"], "uncertain")
        self.decoder.accept.assert_called_once_with(self.pcm, False)
        self.assertNotIn("tracks", result)

    def test_audio_activity_and_vision_finish_while_transcription_is_blocked(self):
        started, release = threading.Event(), threading.Event()

        def slow_decode(*arguments):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test decoder timeout")
            return TranscriptionUpdate([], "ciao")

        self.decoder.accept.side_effect = slow_decode
        thread = threading.Thread(target=self.recognize)
        thread.start()
        self.assertTrue(started.wait(1))
        try:
            activity = self.service.update_activity(self.pcm, "camera", 0.1)
            with self.service.registry.use("camera") as session:
                session.detector = Mock()
                session.detector.detect.return_value = (
                    640,
                    360,
                    [(FaceBox(0.1, 10, 10, 20, 20), 0.1)],
                )
            geometry = self.service.track_frame(b"image", "camera", 0.1)
            self.assertIn("speaker", activity)
            self.assertEqual(len(geometry["faces"]), 1)
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_transcription_finishes_while_camera_inference_is_blocked(self):
        started, release = threading.Event(), threading.Event()

        def slow_camera(*arguments):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test camera timeout")
            return 640, 360, []

        with self.service.registry.use("camera") as session:
            session.detector = Mock()
            session.detector.detect.side_effect = slow_camera
        thread = threading.Thread(
            target=self.service.track_frame, args=(b"image", "camera", 0.1)
        )
        thread.start()
        self.assertTrue(started.wait(1))
        try:
            result = self.recognize()
            self.assertEqual(result["events"][0]["text"], "ciao")
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())

    def test_busy_decoder_does_not_enqueue_stale_audio(self):
        with self.service.registry.use("camera") as session:
            session.audio_lock.acquire()
            try:
                with self.assertRaises(RecognitionBusyError):
                    self.recognize()
            finally:
                session.audio_lock.release()

    def test_retry_does_not_feed_same_pcm_twice(self):
        first = self.recognize()
        self.assertEqual(self.recognize(), first)
        self.assertEqual(self.decoder.accept.call_count, 1)

    def test_partial_text_is_delivered_before_final_word_timestamps(self):
        self.decoder.accept.return_value = TranscriptionUpdate([], "ciao")
        result = self.recognize()
        self.assertEqual(result["events"], [])
        self.assertEqual(result["provisionalText"], "ciao")
        self.assertEqual(result["acknowledgedUntil"], 0.1)
        self.decoder.accept.return_value = TranscriptionUpdate(
            [Word("ciao", 0.0, 0.1, 0.9)], ""
        )
        result = self.service.recognize_audio(b"", "it", "camera", 0.1, True, "2")
        self.assertEqual(result["events"][0]["text"], "ciao")
        self.decoder.accept.assert_called_with(b"", True)

    def test_audio_gaps_or_overlapping_replays_are_rejected(self):
        self.recognize()
        for offset in (0.0, 0.2):
            with self.assertRaises(ValueError):
                self.recognize(capture_offset=offset, job_id="2")
        self.recognize(capture_offset=0.1, job_id="2")
        self.assertEqual(self.decoder.accept.call_count, 2)

    def test_language_is_fixed_per_session(self):
        with self.assertRaises(ValueError):
            self.recognize(language="en")
        with self.assertRaises(ValueError):
            self.service.start_session("camera", "en")

    def test_audio_must_be_prepared_before_capture(self):
        with self.assertRaises(ValueError):
            self.recognize(session_id="unprepared")

    def test_ended_audio_cannot_accept_more_packets(self):
        self.recognize(final=True)
        with self.assertRaises(ValueError):
            self.recognize(capture_offset=0.1, job_id="2")

    def test_end_waits_for_active_requests_and_releases_detector(self):
        with self.service.registry.use("camera") as session:
            detector = Mock()
            session.detector = detector
            with self.assertRaises(RecognitionBusyError):
                self.service.close_session("camera")
        self.service.close_session("camera")
        self.service.close_session("camera")
        detector.close.assert_called_once()

    def test_capacity_and_invalid_identifiers(self):
        registry = SessionRegistry()
        for index in range(MAXIMUM_SESSIONS):
            with registry.use(f"camera-{index}"):
                pass
        with self.assertRaises(RecognitionBusyError), registry.use("extra"):
            pass
        registry.end("camera-0")
        with registry.use("extra"):
            pass
        with self.assertRaises(ValueError), registry.use("../camera"):
            pass
        registry.close()

    def test_startup_warms_only_the_streaming_decoder_and_face_models(self):
        with (
            patch("src.app.service.StreamingTranscriber") as decoder,
            patch("src.app.service.LiveFaceDetector") as detector,
        ):
            self.service.warm_up()
        decoder.return_value.accept.assert_called_once()
        detector.return_value.detect.assert_called_once()
        detector.return_value.close.assert_called_once()

    def test_face_models_use_configured_model_directory(self):
        service = RecognitionService("it", Path("custom-models"))
        service.models = Mock()
        self.addCleanup(service.close)
        with (
            patch("src.app.service.StreamingTranscriber"),
            patch("src.app.service.LiveFaceDetector") as detector,
        ):
            service.warm_up()
            detector.assert_called_with(Path("custom-models"))
            detector.return_value.detect.return_value = (640, 480, [])
            service.track_frame(b"image", "custom-camera", 0.1)
            self.assertEqual(detector.call_count, 2)
            detector.assert_called_with(Path("custom-models"))

    def test_frame_payload_restores_identity_without_exposing_features(self):
        first_features = np.array([1.0, 0.0], dtype=np.float32)
        second_features = np.array([0.0, 1.0], dtype=np.float32)
        with self.service.registry.use("camera") as session:
            session.detector = Mock()
            session.detector.detect.side_effect = [
                (640, 480, [(FaceBox(0.1, 20, 10, 40, 40), 0.02, first_features)]),
                (640, 480, []),
                (640, 480, [(FaceBox(0.3, 20, 10, 40, 40), 0.02, second_features)]),
                (640, 480, [(FaceBox(10, 300, 10, 40, 40), 0.02, first_features)]),
            ]
        first = self.service.track_frame(b"image", "camera", 0.1)["faces"][0]
        self.service.update_activity(self.pcm, "camera", 0.2)
        missing = self.service.track_frame(b"image", "camera", 0.2)
        self.assertEqual(missing["faces"], [])
        self.assertEqual(missing["speaker"]["state"], "off_screen")
        second = self.service.track_frame(b"image", "camera", 0.3)["faces"][0]
        self.assertNotEqual(second["trackId"], first["trackId"])
        returned = self.service.track_frame(b"image", "camera", 10)["faces"][0]
        self.assertEqual(returned["trackId"], first["trackId"])
        self.assertNotIn("features", returned)

    def test_camera_warmup_failure_does_not_stop_microphone_startup(self):
        with (
            patch("src.app.service.StreamingTranscriber"),
            patch(
                "src.app.service.LiveFaceDetector",
                side_effect=RuntimeError("camera unavailable"),
            ),
            self.assertLogs("src.app.service", level="WARNING"),
        ):
            self.service.warm_up()
        self.assertEqual(self.recognize()["events"][0]["text"], "ciao")

    def test_page_and_cli_use_short_independent_capture(self):
        page = build_page(100, "it")
        self.assertIn('"language": "it"', page)
        self.assertIn("/audio-buffer.js", page)
        self.assertIn('id="speaker-state"', page)
        self.assertNotIn("MediaRecorder", page)
        self.assertEqual(build_parser().parse_args([]).chunk_seconds, 0.1)
        self.assertNotIn("</script><evil>", build_page(100, "</script><evil>"))
