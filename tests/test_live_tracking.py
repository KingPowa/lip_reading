import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from src.contracts import FaceBox, FaceTrack
from src.live_tracking import IDENTITY_ABSENCE_SECONDS, LiveTracker


def detection(x, y, width, height):
    return SimpleNamespace(
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(
                xmin=x,
                ymin=y,
                width=width,
                height=height,
            )
        )
    )


class LiveTrackingTest(unittest.TestCase):
    def setUp(self):
        self.detector = Mock()
        self.segmenter = Mock()
        self.detector.process.return_value = SimpleNamespace(
            detections=[detection(0.2, 0.1, 0.2, 0.2)]
        )
        self.mask = np.zeros((100, 100), dtype=np.float32)
        self.mask[5:90, 15:45] = 1
        self.segmenter.process.return_value = [self.mask]
        self.face_patch = patch(
            "src.live_tracking.mp.solutions.face_detection.FaceDetection",
            return_value=self.detector,
        )
        self.mask_patch = patch(
            "src.live_tracking.person_segmenter",
            return_value=self.segmenter,
        )
        self.face_factory = self.face_patch.start()
        self.mask_factory = self.mask_patch.start()
        self.addCleanup(self.face_patch.stop)
        self.addCleanup(self.mask_patch.stop)
        self.tracker = LiveTracker()
        self.addCleanup(self.tracker.close)
        self.image_bytes = cv2.imencode(
            ".jpg", np.zeros((100, 100, 3), dtype=np.uint8)
        )[1].tobytes()

    def test_current_geometry_reuses_models_and_identity(self):
        first = self.tracker.track(self.image_bytes, 0)
        self.detector.process.return_value = SimpleNamespace(
            detections=[detection(0.22, 0.1, 0.2, 0.2)]
        )
        second = self.tracker.track(self.image_bytes, 0.2)
        self.assertEqual(first["faces"][0]["trackId"], second["faces"][0]["trackId"])
        self.assertEqual(second["faces"][0]["box"]["x"], 22)
        contour = second["faces"][0]["contour"]
        self.assertEqual(min(point[0] for point in contour), 15)
        self.assertEqual(max(point[1] for point in contour), 89)
        self.face_factory.assert_called_once()
        self.mask_factory.assert_called_once()

    def test_shared_person_mask_does_not_claim_either_face(self):
        self.detector.process.return_value = SimpleNamespace(
            detections=[
                detection(0.2, 0.1, 0.1, 0.2),
                detection(0.32, 0.1, 0.1, 0.2),
            ]
        )
        result = self.tracker.track(self.image_bytes, 0)
        self.assertEqual([face["contour"] for face in result["faces"]], [[], []])

    def test_partial_face_mask_cannot_claim_a_person_silhouette(self):
        self.mask[:] = 0
        self.mask[20:90, 15:45] = 1
        result = self.tracker.track(self.image_bytes, 0)
        self.assertEqual(result["faces"][0]["contour"], [])

    def test_touching_person_instances_have_individual_contours(self):
        second_mask = np.zeros_like(self.mask)
        second_mask[5:90, 45:90] = 1
        self.segmenter.process.return_value = [self.mask, second_mask]
        self.detector.process.return_value = SimpleNamespace(
            detections=[
                detection(0.2, 0.1, 0.2, 0.2),
                detection(0.65, 0.1, 0.2, 0.2),
            ]
        )
        result = self.tracker.track(self.image_bytes, 0)
        self.assertEqual(len(result["faces"]), 2)
        self.assertTrue(all(face["contour"] for face in result["faces"]))
        self.assertLess(max(point[0] for point in result["faces"][0]["contour"]), 60)

    def test_absent_face_disappears_and_expired_identity_is_not_reused(self):
        first_id = self.tracker.track(self.image_bytes, 0)["faces"][0]["trackId"]
        self.detector.process.return_value = SimpleNamespace(detections=[])
        self.assertEqual(self.tracker.track(self.image_bytes, 1)["faces"], [])
        self.detector.process.return_value = SimpleNamespace(
            detections=[detection(0.2, 0.1, 0.2, 0.2)]
        )
        self.assertNotEqual(
            self.tracker.track(self.image_bytes, IDENTITY_ABSENCE_SECONDS + 1)["faces"][
                0
            ]["trackId"],
            first_id,
        )

    def test_brief_detector_miss_retains_identity_without_showing_stale_geometry(self):
        first_id = self.tracker.track(self.image_bytes, 0)["faces"][0]["trackId"]
        self.detector.process.return_value = SimpleNamespace(detections=[])
        self.assertEqual(self.tracker.track(self.image_bytes, 3)["faces"], [])
        self.detector.process.return_value = SimpleNamespace(
            detections=[detection(0.2, 0.1, 0.2, 0.2)]
        )
        self.assertEqual(
            self.tracker.track(self.image_bytes, 4)["faces"][0]["trackId"], first_id
        )

    def test_sequential_fragments_share_history_identity_but_overlapping_faces_do_not(
        self,
    ):
        historical_id = self.tracker.track(self.image_bytes, 10)["faces"][0]["trackId"]
        self.tracker.track(self.image_bytes, 11)
        self.tracker.track(self.image_bytes, 12)
        tracks = [
            FaceTrack(
                "first",
                0,
                0.8,
                [FaceBox(0, 20, 10, 20, 20), FaceBox(0.8, 20, 10, 20, 20)],
            ),
            FaceTrack(
                "second", 1, 2, [FaceBox(1, 20, 10, 20, 20), FaceBox(2, 20, 10, 20, 20)]
            ),
            FaceTrack(
                "competing",
                1,
                2,
                [FaceBox(1, 20, 10, 20, 20), FaceBox(2, 20, 10, 20, 20)],
            ),
        ]
        self.tracker.associate_tracks(tracks, 10)
        self.assertEqual(
            [track.track_id for track in tracks[:2]], [historical_id, historical_id]
        )
        self.assertNotEqual(tracks[2].track_id, historical_id)

    def test_delayed_recognition_matches_capture_history_not_current_position(self):
        historical_id = self.tracker.track(self.image_bytes, 10)["faces"][0]["trackId"]
        self.tracker.track(self.image_bytes, 11)
        self.detector.process.return_value = SimpleNamespace(
            detections=[detection(0.7, 0.1, 0.2, 0.2)]
        )
        current_id = self.tracker.track(self.image_bytes, 15)["faces"][0]["trackId"]
        tracks = [
            FaceTrack(
                "face-1", 0, 1, [FaceBox(0, 20, 10, 20, 20), FaceBox(1, 20, 10, 20, 20)]
            )
        ]
        self.tracker.associate_tracks(tracks, 10)
        self.assertEqual(tracks[0].track_id, historical_id)
        self.assertNotEqual(tracks[0].track_id, current_id)

    def test_unmatched_recognition_cannot_inherit_current_face_id(self):
        current_id = self.tracker.track(self.image_bytes, 10)["faces"][0]["trackId"]
        tracks = [FaceTrack("face-1", 0, 1, [FaceBox(0, 70, 10, 20, 20)])]
        self.tracker.associate_tracks(tracks, 10)
        self.assertNotEqual(tracks[0].track_id, current_id)

    def test_association_snapshot_survives_live_history_expiry(self):
        historical_id = self.tracker.track(self.image_bytes, 10)["faces"][0]["trackId"]
        self.tracker.track(self.image_bytes, 11)
        associate = self.tracker.association_callback(10)
        self.tracker.track(self.image_bytes, 102)
        tracks = [
            FaceTrack(
                "face-1", 0, 1, [FaceBox(0, 20, 10, 20, 20), FaceBox(1, 20, 10, 20, 20)]
            )
        ]
        associate(tracks)
        self.assertEqual(tracks[0].track_id, historical_id)
        current_tracks = [FaceTrack("face-2", 0, 1, list(tracks[0].boxes))]
        self.tracker.associate_tracks(current_tracks, 10)
        self.assertNotEqual(current_tracks[0].track_id, historical_id)

    def test_history_expires_and_invalid_frames_do_not_corrupt_state(self):
        self.tracker.track(self.image_bytes, 0)
        self.tracker.track(self.image_bytes, 91)
        tracks = [FaceTrack("face-1", 0, 1, [FaceBox(0, 20, 10, 20, 20)])]
        self.tracker.associate_tracks(tracks, 0)
        self.assertNotEqual(tracks[0].track_id, "live-face-1")
        for image_bytes, timestamp in [
            (b"invalid", 92),
            (self.image_bytes, float("nan")),
            (self.image_bytes, 1),
        ]:
            with self.assertRaises(ValueError):
                self.tracker.track(image_bytes, timestamp)

    def test_oversized_image_header_is_rejected_before_opencv_allocation(self):
        content = BytesIO()
        Image.new("RGB", (1281, 720)).save(content, format="JPEG")
        with patch("src.live_tracking.cv2.imdecode") as decode:
            with self.assertRaises(ValueError):
                self.tracker.track(content.getvalue(), 0)
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
