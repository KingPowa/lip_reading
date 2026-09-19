import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config import MAXIMUM_ACTIVITY_TRACKS
from src.contracts import FaceActivity, FaceBox, FaceTrack
from src.light_asd import score_face_activity


class LightAsdTest(unittest.TestCase):
    def test_every_selected_track_is_scored_for_its_own_timeline(self) -> None:
        tracks = [
            FaceTrack(f"face-{index}", 0.0, 2.0, [FaceBox(0.0, 0, 0, index + 1, index + 1)])
            for index in range(MAXIMUM_ACTIVITY_TRACKS + 1)
        ]

        def score_track(_model, _audio, _frames, track):
            return [FaceActivity(track.track_id, 0.0, 2.0, 0.9)]

        with (
            patch("src.light_asd._load_model", return_value=object()),
            patch("src.light_asd._audio_features", return_value=np.empty((1, 13))),
            patch("src.light_asd._score_track", side_effect=score_track) as scorer,
        ):
            activities = score_face_activity(Path("audio.wav"), [], tracks, Path("model.pt"))

        self.assertEqual(scorer.call_count, MAXIMUM_ACTIVITY_TRACKS)
        self.assertEqual(len(activities), MAXIMUM_ACTIVITY_TRACKS)


if __name__ == "__main__":
    unittest.main()
