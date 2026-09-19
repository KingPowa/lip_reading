import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.contracts import FaceBox
from src.media import VideoFrame
from src.vision import analyze_faces


class VisionTest(unittest.TestCase):
    def test_unstable_mediapipe_result_uses_stable_cascade_tracks(self) -> None:
        frames = [VideoFrame(index * 0.2, np.zeros((40, 40, 3), dtype=np.uint8)) for index in range(16)]

        def cascade(frame):
            return [FaceBox(frame.timestamp, 10, 10, 20, 20)]

        with (
            patch("src.vision._download_model_if_missing"),
            patch("src.vision._landmarker_detections", return_value=[[FaceBox(0.0, 10, 10, 20, 20)]] + [[]] * 15),
            patch("src.vision._cascade_detections", side_effect=cascade),
        ):
            tracks = analyze_faces(frames, Path("face_landmarker.task"))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].boxes), len(frames))


if __name__ == "__main__":
    unittest.main()
