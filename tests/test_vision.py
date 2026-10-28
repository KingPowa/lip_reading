import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.core.contracts import FaceBox
from src.core.media import VideoFrame
from src.core.vision import faces as vision
from src.core.vision.faces import _track_detections, analyze_faces


class VisionTest(unittest.TestCase):
    def test_identity_features_are_reused_only_for_one_stable_recent_box(self) -> None:
        from src.core.vision.live import identity_feature_time

        box = FaceBox(1.0, 100, 100, 100, 100)
        previous = [(FaceBox(0.9, 102, 100, 100, 100), 0.5)]
        self.assertEqual(identity_feature_time(box, previous, 1.2), 0.5)
        self.assertIsNone(identity_feature_time(box, previous, 1.5))
        self.assertIsNone(identity_feature_time(FaceBox(1.0, 300, 100, 100, 100), previous, 1.2))
        self.assertIsNone(identity_feature_time(box, [], 1.2))
        twin = (FaceBox(0.9, 98, 100, 100, 100), 0.5)
        self.assertIsNone(identity_feature_time(box, previous + [twin], 1.2))

    def test_live_detector_reuses_model_and_tracks_clipped_boxes_without_downloads(
        self,
    ) -> None:
        relative_box = SimpleNamespace(xmin=-0.1, ymin=0.25, width=0.6, height=0.5)
        detection = SimpleNamespace(
            location_data=SimpleNamespace(relative_bounding_box=relative_box)
        )
        detector = Mock()
        detector.process.return_value = SimpleNamespace(detections=[detection])
        frames = [
            VideoFrame(index / 25, np.zeros((40, 40, 3), dtype=np.uint8))
            for index in range(13)
        ]
        with (
            patch(
                "src.core.vision.faces.mp.solutions.face_detection.FaceDetection",
                return_value=detector,
            ) as factory,
            patch("src.core.vision.faces._download_model_if_missing") as download,
            patch("src.core.vision.faces._cascade_detections") as cascade,
        ):
            first = analyze_faces(
                frames, Path("unused.task"), 2, 0.0, use_face_detector=True
            )
            second = analyze_faces(
                frames, Path("unused.task"), 2, 0.0, use_face_detector=True
            )
        vision._full_range_detector.cache_clear()
        factory.assert_called_once_with(model_selection=1, min_detection_confidence=0.5)
        download.assert_not_called()
        cascade.assert_not_called()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].boxes[0], FaceBox(0, 0, 10, 20, 20))
        self.assertEqual(
            [box.timestamp for box in first[0].boxes], [0.0, 0.2, 0.4, 0.48]
        )

    def test_live_detector_does_not_replace_absent_faces_with_cascade_candidates(
        self,
    ) -> None:
        frames = [VideoFrame(0, np.zeros((40, 40, 3), dtype=np.uint8))]
        detector = Mock()
        detector.process.return_value = SimpleNamespace(detections=None)
        with (
            patch(
                "src.core.vision.faces.mp.solutions.face_detection.FaceDetection",
                return_value=detector,
            ),
            patch("src.core.vision.faces._cascade_detections") as cascade,
        ):
            tracks = analyze_faces(
                frames, Path("unused.task"), 1, 0.0, use_face_detector=True
            )
        vision._full_range_detector.cache_clear()
        self.assertEqual(tracks, [])
        cascade.assert_not_called()

    def test_detector_samples_frames_separately_and_covers_clip_end(self) -> None:
        frames = [
            VideoFrame(index / 25, np.zeros((40, 40, 3), dtype=np.uint8))
            for index in range(13)
        ]

        def detect(sampled_frames, _model_path):
            return [
                [FaceBox(frame.timestamp, 10, 10, 20, 20)] for frame in sampled_frames
            ]

        with (
            patch("src.core.vision.faces._download_model_if_missing"),
            patch(
                "src.core.vision.faces._landmarker_detections", side_effect=detect
            ) as detector,
        ):
            tracks = analyze_faces(frames, Path("face_landmarker.task"), 2, 0.0)

        self.assertEqual(
            [frame.timestamp for frame in detector.call_args.args[0]],
            [0.0, 0.2, 0.4, 0.48],
        )
        self.assertEqual(tracks[0].end, frames[-1].timestamp)

    def test_unstable_mediapipe_result_uses_stable_cascade_tracks(self) -> None:
        frames = [
            VideoFrame(index / 25, np.zeros((40, 40, 3), dtype=np.uint8))
            for index in range(76)
        ]

        def cascade(frame):
            return [FaceBox(frame.timestamp, 10, 10, 20, 20)]

        with (
            patch("src.core.vision.faces._download_model_if_missing"),
            patch(
                "src.core.vision.faces._landmarker_detections",
                return_value=[[FaceBox(0.0, 10, 10, 20, 20)]] + [[]] * 15,
            ),
            patch("src.core.vision.faces._cascade_detections", side_effect=cascade),
        ):
            tracks = analyze_faces(frames, Path("face_landmarker.task"))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].boxes), 16)

    def test_live_windows_accept_short_face_tracks(self) -> None:
        tracks = _track_detections(
            [
                [FaceBox(0.0, 10, 10, 20, 20)],
                [FaceBox(0.2, 10, 10, 20, 20)],
            ],
            minimum_detections=2,
            minimum_duration_seconds=0.0,
        )
        self.assertEqual(len(tracks), 1)


if __name__ == "__main__":
    unittest.main()
