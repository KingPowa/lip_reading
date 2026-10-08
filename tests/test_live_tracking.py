import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.config.live import GEOMETRY_EXPIRY_SECONDS, IDENTITY_ABSENCE_SECONDS
from src.config.models import MODEL_DIRECTORY
from src.core.contracts import FaceBox
from src.core.tracking.realtime import RealtimeTracker
from src.core.vision.live import LiveFaceDetector, face_roll_degrees


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

    def speak(self, steps, second_aperture=0.0):
        estimate = None
        for timestamp, aperture in steps:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.faces(timestamp, aperture, second_aperture)
        return estimate

    def test_unmeasurable_mouth_while_speaking_is_not_off_screen(self):
        self.speak([(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)])
        self.tracker.update_audio(self.speech, 0.7)
        estimate = self.tracker.update_faces(
            [(FaceBox(0.7, 20, 10, 40, 40), None)], 0.7
        )
        self.assertEqual((estimate.state, estimate.face_track), ("visible", "face-1"))
        fresh = RealtimeTracker()
        fresh.update_audio(self.speech, 0.1)
        estimate = fresh.update_faces([(FaceBox(0.1, 20, 10, 40, 40), None)], 0.1)
        self.assertEqual(estimate.state, "uncertain")

    def test_one_dropped_mesh_frame_does_not_split_speaker(self):
        apertures = [0.02, 0.09, 0.03, 0.08, None, 0.03, 0.09, 0.02]
        states = []
        for index, aperture in enumerate(apertures):
            timestamp = round(0.1 * (index + 1), 1)
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [(FaceBox(timestamp, 20, 10, 40, 40), aperture)], timestamp
            )
            states.append((estimate.state, estimate.face_track))
        self.assertEqual(set(states[2:]), {("visible", "face-1")})

    def test_brief_challenger_does_not_take_over_but_sustained_one_does(self):
        self.speak([(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)])
        self.assertEqual(self.tracker.active_identity, "face-1")
        flip = [(0.4, 0.05), (0.5, 0.05), (0.6, 0.05)]
        for timestamp, first in flip:
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [
                    (FaceBox(timestamp, 20, 10, 40, 40), first),
                    (FaceBox(timestamp, 100, 10, 40, 40), 0.02 + 0.2 * (timestamp > 0.4)),
                ],
                timestamp,
            )
        self.assertEqual(estimate.face_track, "face-1")
        for timestamp in (0.7, 0.8, 0.9, 1.0):
            self.tracker.update_audio(self.speech, timestamp)
            estimate = self.tracker.update_faces(
                [
                    (FaceBox(timestamp, 20, 10, 40, 40), 0.05),
                    (FaceBox(timestamp, 100, 10, 40, 40), 0.02 + 0.2 * (timestamp % 0.2 > 0.1)),
                ],
                timestamp,
            )
        self.assertEqual(estimate.face_track, "face-2")

    def test_measured_quiet_faces_release_to_off_screen_after_hold(self):
        self.speak([(0.1, 0.02), (0.2, 0.09), (0.3, 0.03)])
        for timestamp in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            estimate = self.speak([(timestamp, 0.05)])
        self.assertEqual(estimate.state, "off_screen")

    def test_oversized_image_is_rejected_before_opencv_allocation(self):
        content = BytesIO()
        Image.new("RGB", (641, 480)).save(content, format="JPEG")
        with (
            patch("src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"),
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ),
            patch("src.core.vision.live.cv2.imdecode") as decode,
            patch("src.core.vision.live.face_identity_encoder"),
        ):
            detector = LiveFaceDetector()
            with self.assertRaises(ValueError):
                detector.detect(content.getvalue(), 0)
        decode.assert_not_called()


class LiveFaceFeaturesTest(unittest.TestCase):
    def setUp(self):
        content = BytesIO()
        Image.new("RGB", (320, 240)).save(content, format="JPEG")
        self.content = content.getvalue()
        self.detection = SimpleNamespace(
            location_data=SimpleNamespace(
                relative_bounding_box=SimpleNamespace(
                    xmin=0.2, ymin=0.2, width=0.3, height=0.4
                )
            )
        )

    def test_tilted_face_is_rotated_upright_before_mesh_and_encoder(self):
        tilted = SimpleNamespace(
            location_data=SimpleNamespace(
                relative_bounding_box=self.detection.location_data.relative_bounding_box,
                relative_keypoints=[
                    SimpleNamespace(x=0.3, y=0.3),
                    SimpleNamespace(x=0.4, y=0.4),
                ],
            )
        )
        self.assertAlmostEqual(face_roll_degrees(tilted, 320, 240), 36.87, places=1)
        crops = []
        for detection in (self.detection, tilted):
            with (
                patch(
                    "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
                ) as mesh,
                patch(
                    "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
                ) as detector_class,
                patch("src.core.vision.live.face_identity_encoder"),
            ):
                detector_class.return_value.process.return_value.detections = [
                    detection
                ]
                mesh.return_value.process.return_value.multi_face_landmarks = []
                detector = LiveFaceDetector()
                self.addCleanup(detector.close)
                image = Image.effect_noise((320, 240), 60).convert("RGB")
                content = BytesIO()
                image.save(content, format="PNG")
                detector.detect(content.getvalue(), 0.1)
                crops.append(mesh.return_value.process.call_args.args[0])
        self.assertEqual(crops[0].shape, crops[1].shape)
        self.assertFalse(np.array_equal(crops[0], crops[1]))

    def test_current_frame_passes_crop_identity_features_to_tracker(self):
        features = np.ones(128, dtype=np.float32)
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        with (
            patch(
                "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
            ) as mesh,
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ) as detection,
            patch("src.core.vision.live.face_identity_encoder", create=True) as encoder,
        ):
            detection.return_value.process.return_value.detections = [self.detection]
            mesh.return_value.process.return_value.multi_face_landmarks = [
                SimpleNamespace(landmark=landmarks)
            ]
            encoder.return_value.encode.return_value = features
            detector = LiveFaceDetector()
            self.addCleanup(detector.close)
            width, height, observations = detector.detect(self.content, 0.1)
            self.assertEqual((width, height), (320, 240))
            self.assertIs(observations[0].features, features)
            encoder.assert_called_once_with(MODEL_DIRECTORY)
            crop, extracted_landmarks = encoder.return_value.encode.call_args.args
            self.assertGreater(crop.size, 0)
            self.assertIs(extracted_landmarks, landmarks)

    def test_missing_mesh_keeps_face_box_without_identity_or_mouth_features(self):
        with (
            patch(
                "src.core.vision.live.mediapipe.solutions.face_mesh.FaceMesh"
            ) as mesh,
            patch(
                "src.core.vision.live.mediapipe.solutions.face_detection.FaceDetection"
            ) as detection,
            patch("src.core.vision.live.face_identity_encoder", create=True) as encoder,
        ):
            detection.return_value.process.return_value.detections = [self.detection]
            mesh.return_value.process.return_value.multi_face_landmarks = []
            detector = LiveFaceDetector()
            self.addCleanup(detector.close)
            _, _, observations = detector.detect(self.content, 0.1)
            self.assertIsNone(observations[0].features)
            self.assertIsNone(observations[0].aperture)
            encoder.return_value.encode.assert_not_called()
