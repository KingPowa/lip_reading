import unittest
from pathlib import Path
from unittest.mock import patch

from src.contracts import CaptionEvent, FaceBox, FaceTrack
from src.live import RecognitionBusyError, RecognitionService, build_parser
from src.live_view import build_page
from src.media import MediaInfo


class LiveViewerTest(unittest.TestCase):
    def test_page_has_camera_capture_colored_body_highlight_and_collapsible_transcript(self) -> None:
        page = build_page(4_000)
        self.assertIn('id="camera"', page)
        self.assertIn('id="faces"', page)
        self.assertIn('face.title', page)
        self.assertIn("bodyBox", page)
        self.assertIn("event.color", page)
        self.assertIn("line.style.color", page)
        self.assertIn('id="transcript-panel"', page)
        self.assertIn('classList.toggle("collapsed")', page)
        self.assertNotIn('id="result"', page)
        self.assertNotIn("playbackQueue", page)
        self.assertIn("activeEvents", page)
        self.assertIn('X-Whisper-Language', page)
        self.assertIn("SpeechRecognition", page)
        self.assertIn("use Chromium for immediate browser transcription", page)
        self.assertIn("number !== chunkNumber", page)
        self.assertIn("navigator.language", page)

    def test_server_language_sets_the_browser_transcript_language(self) -> None:
        page = build_page(2_000, "it")
        self.assertIn('"language": "it"', page)
        self.assertIn("languageSelect.value = configuration.language", page)

    def test_chunk_duration_has_a_safe_range(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).chunk_seconds, 2)

    def test_camera_recording_is_normalized_before_recognition(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        expected_info = MediaInfo(1.0, 640, 360, 25.0, True, True)
        expected_tracks = [FaceTrack("face-1", 0.0, 1.0, [FaceBox(0.0, 100, 50, 40, 40)])]
        expected_events = [CaptionEvent(0.0, 1.0, "Person 1", "face-1", "hello", 0.9, "visible")]
        with (
            patch("src.live.normalize_camera_recording") as normalize,
            patch("src.live.extract_audio"),
            patch("src.live.load_transcription_model", return_value=object()),
            patch("src.live.recognize", return_value=(expected_info, expected_tracks, expected_events, [], [])) as recognize,
        ):
            result = service.recognize_clip(b"camera recording")

        normalized_path = normalize.call_args.args[1]
        self.assertEqual(normalized_path.name, "recording.mp4")
        self.assertEqual(recognize.call_args.args[0], normalized_path)
        self.assertEqual(result["events"][0]["text"], "hello")
        self.assertEqual(result["events"][0]["color"], "#38bdf8")
        self.assertEqual(result["tracks"][0]["boxes"][0]["bodyBox"]["width"], 120)
        self.assertTrue(recognize.call_args.kwargs["live"])
        self.assertIs(recognize.call_args.kwargs["track_labels"], service.track_labels)

    def test_busy_recognizer_does_not_queue_a_stale_camera_segment(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        service.lock.acquire()
        try:
            with self.assertRaises(RecognitionBusyError):
                service.recognize_clip(b"camera recording")
        finally:
            service.lock.release()

    def test_warm_up_loads_the_transcription_model_before_capture(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        with patch("src.live.load_transcription_model", return_value=object()) as loader:
            service.warm_up()
        self.assertIsNotNone(service.transcription_model)
        loader.assert_called_once()

    def test_face_tracks_keep_their_live_identity_between_windows(self) -> None:
        service = RecognitionService("base", None, Path("models"))
        first_track = FaceTrack("face-1", 0.0, 1.0, [FaceBox(1.0, 100, 50, 40, 40)])
        next_track = FaceTrack("face-1", 0.0, 1.0, [FaceBox(1.0, 102, 50, 40, 40)])
        service._stabilize_tracks([first_track])
        service._stabilize_tracks([next_track])
        self.assertEqual([first_track.track_id, next_track.track_id], ["live-face-1", "live-face-1"])


if __name__ == "__main__":
    unittest.main()
