import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.contracts import CaptionEvent, FaceBox, FaceTrack, Word
from src.live import (
    MAXIMUM_SESSIONS,
    RecognitionBusyError,
    RecognitionService,
    build_parser,
)
from src.live_view import build_page
from src.media import MediaInfo


class LiveViewerTest(unittest.TestCase):
    def test_page_has_local_capture_current_highlights_and_collapsible_transcript(
        self,
    ) -> None:
        page = build_page(4_000)
        self.assertIn('id="camera"', page)
        self.assertIn('id="faces"', page)
        self.assertIn("SVG_NAMESPACE", page)
        self.assertIn("person.contour", page)
        self.assertIn("event.color", page)
        self.assertIn("line.style.color", page)
        self.assertIn('id="transcript-panel"', page)
        self.assertIn('classList.toggle("collapsed")', page)
        self.assertNotIn('id="result"', page)
        self.assertNotIn("playbackQueue", page)
        self.assertIn("activeEvents", page)
        self.assertIn("X-Whisper-Language", page)
        self.assertNotIn("SpeechRecognition", page)
        self.assertNotIn("number !== chunkNumber", page)
        self.assertIn("pendingClips", page)
        self.assertIn("navigator.language", page)

    def test_server_language_sets_the_browser_transcript_language(self) -> None:
        page = build_page(2_000, "it")
        self.assertIn('"language": "it"', page)
        self.assertIn("languageSelect.value = configuration.language", page)

    def test_chunk_duration_has_a_safe_range(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).chunk_seconds, 4)

    def test_camera_recording_is_normalized_before_recognition(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        expected_info = MediaInfo(1.0, 640, 360, 25.0, True, True)
        expected_tracks = [
            FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 100, 50, 40, 40)])
        ]
        expected_events = [
            CaptionEvent(0.0, 1.0, "Person 1", "face-1", "hello", 0.9, "visible")
        ]
        with (
            patch("src.live.normalize_camera_recording") as normalize,
            patch("src.live.extract_audio"),
            patch("src.live.inspect_media", return_value=expected_info),
            patch("src.live.load_faster_whisper_model", return_value=object()),
            patch(
                "src.live.recognize",
                return_value=(expected_info, expected_tracks, expected_events, [], []),
            ) as recognize,
        ):
            result = service.recognize_clip(b"camera recording")

        normalized_path = normalize.call_args.args[1]
        self.assertEqual(normalized_path.name, "recording.mp4")
        self.assertEqual(recognize.call_args.args[0], normalized_path)
        self.assertEqual(result["events"][0]["text"], "hello")
        self.assertEqual(result["events"][0]["color"], "#38bdf8")
        self.assertEqual(result["tracks"][0]["boxes"][0]["bodyBox"]["width"], 120)
        self.assertTrue(recognize.call_args.kwargs["live"])
        self.assertIs(
            recognize.call_args.kwargs["track_labels"],
            service.session("legacy").track_labels,
        )

    def test_busy_recognizer_does_not_queue_a_stale_camera_segment(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        service.lock.acquire()
        try:
            with self.assertRaises(RecognitionBusyError):
                service.recognize_clip(b"camera recording")
        finally:
            service.lock.release()

    def test_session_identity_and_transcript_context_are_isolated(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        first = service.session("first")
        second = service.session("second")
        first.track_labels["face"] = "Person 1"
        first.prompt = "Previous speech"
        self.assertEqual(second.track_labels, {})
        self.assertEqual(second.prompt, "")
        self.assertIs(service.session("first"), first)

    def test_retried_job_returns_same_result_without_repeating_inference(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        with patch.object(
            service, "_recognize_clip", return_value={"events": []}
        ) as recognizer:
            first = service.recognize_clip(
                b"recording", session_id="camera", job_id="1"
            )
            second = service.recognize_clip(
                b"recording", session_id="camera", job_id="1"
            )
        self.assertEqual(first, second)
        self.assertEqual(recognizer.call_count, 1)

    def test_word_crossing_commit_boundary_remains_available_for_next_update(
        self,
    ) -> None:
        service = RecognitionService("base", "it", Path("models"))
        info = MediaInfo(4.0, 640, 360, 25.0, True, True)
        words = [Word("boundary", 2.8, 3.4, 0.9)]
        with (
            patch("src.live.normalize_camera_recording"),
            patch("src.live.extract_audio"),
            patch("src.live.inspect_media", return_value=info),
            patch("src.live.load_faster_whisper_model", return_value=object()),
            patch("src.live.recognize", return_value=(info, [], [], [], words)),
        ):
            result = service.recognize_clip(
                b"recording", session_id="camera", final=False
            )
        self.assertEqual(result["provisionalText"], "boundary")
        self.assertEqual(result["committedUntil"], 2.8)

    def test_ended_sessions_release_trackers_and_capacity_for_restarting(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        tracker = Mock()
        service.session("camera").tracker = tracker
        service.close_session("camera")
        service.close_session("camera")
        tracker.close.assert_called_once()
        self.assertNotIn("camera", service.sessions)
        for number in range(MAXIMUM_SESSIONS + 1):
            session_id = f"camera-{number}"
            service.session(session_id)
            service.close_session(session_id)
        self.assertEqual(service.sessions, {})

    def test_ending_a_session_rejects_invalid_identifiers(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        with self.assertRaises(ValueError):
            service.close_session("../camera")

    def test_warm_up_loads_the_transcription_model_before_capture(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        with (
            patch("src.live.load_faster_whisper_model", return_value=Mock()) as loader,
            patch("src.live.warm_up_activity_model") as activity_loader,
            patch("src.live.warm_up_speaker_model") as speaker_loader,
            patch("src.live.warm_up_person_model") as person_loader,
        ):
            service.warm_up()
        self.assertIsNotNone(service.transcription_model)
        loader.assert_called_once()
        loader.return_value.transcribe.assert_called_once()
        activity_loader.assert_called_once()
        speaker_loader.assert_called_once()
        person_loader.assert_called_once_with(Path("models"))

    def test_activity_backend_failure_falls_back_before_capture(self) -> None:
        service = RecognitionService(
            "base", None, Path("models"), activity_device="mps"
        )
        with (
            patch("src.live.load_faster_whisper_model", return_value=Mock()),
            patch(
                "src.live.warm_up_activity_model",
                side_effect=[RuntimeError("unsupported GPU operation"), None],
            ) as activity_loader,
            patch("src.live.warm_up_speaker_model"),
            patch("src.live.warm_up_person_model"),
            self.assertLogs("src.live", level="ERROR"),
        ):
            service.warm_up()
        self.assertEqual(service.activity_device, "cpu")
        self.assertEqual(
            [call.args[1] for call in activity_loader.call_args_list], ["mps", "cpu"]
        )

    def test_face_tracks_keep_their_live_identity_between_windows(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        first_track = FaceTrack("face-1", 0.0, 1.0, [FaceBox(1.0, 100, 50, 40, 40)])
        next_track = FaceTrack("face-1", 0.0, 1.0, [FaceBox(1.0, 102, 50, 40, 40)])
        service._stabilize_tracks([first_track])
        service._stabilize_tracks([next_track])
        self.assertEqual(
            [first_track.track_id, next_track.track_id], ["live-face-1", "live-face-1"]
        )


if __name__ == "__main__":
    unittest.main()
