import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config.live import GEOMETRY_EXPIRY_SECONDS, IDENTITY_ABSENCE_SECONDS
from src.core.contracts import FaceBox
from src.core.tracking.realtime import RealtimeTracker
from src.core.vision.live import LiveFaceDetector


class RealtimeTrackingTest(unittest.TestCase):
    def setUp(self):
        self.tracker = RealtimeTracker()
        self.speech = np.full(1600, 0.05, dtype=np.float32)

    def faces(self, timestamp, aperture, second_aperture=0.0):
        return self.tracker.update_faces(
            [
                (FaceBox(timestamp, 20, 10, 40, 40), aperture),
                (FaceBox(timestamp, 100, 10, 40, 40), second_aperture),
            ],
            timestamp,
        )

    def test_audio_and_mouth_motion_identify_person_without_transcription(self):
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture)
        self.assertEqual(
            (estimate.state, estimate.speaker, estimate.face_track),
            ("visible", "Person 1", "face-1"),
        )
        self.assertEqual(self.tracker.speaker_at(0.3).face_track, "face-1")

    def test_mouth_motion_without_audio_does_not_highlight_a_speaker(self):
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            estimate = self.faces(timestamp, aperture)
        self.assertEqual(estimate.state, "silent")

    def test_voice_without_visible_mouth_motion_is_off_screen(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.assertEqual(self.faces(0.1, 0.0).state, "off_screen")
        self.assertEqual(self.tracker.update_faces([], 0.2).state, "off_screen")

    def test_competing_mouths_are_uncertain(self):
        for timestamp, aperture in [(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)]:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture, aperture)
        self.assertEqual(estimate.state, "uncertain")
        self.assertIsNone(estimate.face_track)

    def test_audio_silence_releases_highlight_and_stale_geometry_is_uncertain(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.faces(0.1, 0.05)
        estimate = self.tracker.update_audio(np.zeros(1600), 0.4)
        self.assertEqual(estimate.state, "silent")
        estimate = self.tracker.update_audio(
            self.speech, 0.1 + GEOMETRY_EXPIRY_SECONDS + 0.01
        )
        self.assertEqual(estimate.state, "uncertain")

    def test_face_miss_retains_only_identity_and_expiry_creates_new_person(self):
        self.faces(0.1, 0.02)
        self.tracker.update_faces([], 0.2)
        self.assertEqual(self.tracker.current_faces, [])
        self.faces(0.3, 0.05)
        self.assertEqual(self.tracker.current_faces[0].track_id, "face-1")
        self.faces(0.3 + IDENTITY_ABSENCE_SECONDS + 0.01, 0.03)
        self.assertNotEqual(self.tracker.current_faces[0].track_id, "face-1")

    def test_missing_landmarks_do_not_create_artificial_mouth_motion(self):
        self.tracker.update_audio(self.speech, 0.1)
        self.faces(0.1, 0.08)
        self.faces(0.2, 0.08)
        self.tracker.update_faces([(FaceBox(0.3, 20, 10, 40, 40), None)], 0.3)
        self.assertEqual(self.tracker.current_faces[0].mouth_motion, 0.0)
        estimate = self.faces(0.4, 0.08)
        self.assertNotEqual(estimate.state, "visible")

    def test_out_of_order_frames_do_not_change_identity(self):
        self.faces(1.0, 0.02)
        with self.assertRaises(ValueError):
            self.faces(0.5, 0.09)
        self.assertEqual(self.tracker.latest_vision, 1.0)

    def test_oversized_image_is_rejected_before_opencv_allocation(self):
        content = BytesIO()
        Image.new("RGB", (641, 480)).save(content, format="JPEG")
        with (
            patch("src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"),
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ),
            patch("src.core.vision.live.cv2.imdecode") as decode,
        ):
            detector = LiveFaceDetector()
            with self.assertRaises(ValueError):
                detector.detect(content.getvalue(), 0)
        decode.assert_not_called()
