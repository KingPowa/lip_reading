import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.contracts import FaceActivity, FaceBox, FaceTrack
from src.light_asd import _score_track, score_face_activity
from src.media import VideoFrame

FACE_TRACK_COUNT = 9


class LightAsdTest(unittest.TestCase):
    def test_live_activity_window_scores_only_new_speech_on_original_timestamps(self):
        tracks = [
            FaceTrack("old", 0.0, 1.0, [FaceBox(0, 0, 0, 10, 10)]),
            FaceTrack(
                "current",
                0.0,
                6.0,
                [FaceBox(0, 0, 0, 10, 10), FaceBox(6, 0, 0, 10, 10)],
            ),
            FaceTrack("later", 7.0, 8.0, [FaceBox(7, 0, 0, 10, 10)]),
        ]
        with (
            patch("src.light_asd._load_model", return_value=object()) as loader,
            patch("src.light_asd._audio_features", return_value=np.empty((1, 13))),
            patch("src.light_asd._score_track", return_value=[]) as scorer,
        ):
            score_face_activity(
                Path("audio.wav"), [], tracks, Path("model.pt"), "mps", 3.0, 5.0
            )
        loader.assert_called_once_with(Path("model.pt"), "mps")
        scored_track = scorer.call_args.args[-1]
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(
            (scored_track.track_id, scored_track.start, scored_track.end),
            ("current", 3.0, 5.0),
        )
        self.assertEqual(tracks[1].start, 0.0)
        self.assertIs(scored_track.boxes, tracks[1].boxes)

    def test_activity_tensors_follow_model_device_and_scores_keep_frame_timestamps(
        self,
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, audio, visual):
                self.audio_device = audio.device
                self.visual_device = visual.device
                return torch.tensor([[0.0, 2.0]]).repeat(visual.shape[1], 1)

        model = Model()
        track = FaceTrack(
            "face", 0.0, 0.04, [FaceBox(0.0, 1, 1, 8, 8), FaceBox(0.04, 1, 1, 8, 8)]
        )
        frames = [
            VideoFrame(0.0, np.zeros((10, 10, 3), dtype=np.uint8)),
            VideoFrame(0.04, np.zeros((10, 10, 3), dtype=np.uint8)),
        ]
        activities = _score_track(model, np.zeros((8, 13)), frames, track)
        self.assertEqual(model.audio_device, model.weight.device)
        self.assertEqual(model.visual_device, model.weight.device)
        self.assertEqual([activity.start for activity in activities], [0.0, 0.04])
        self.assertTrue(
            all(activity.speaking_probability > 0.8 for activity in activities)
        )

    def test_every_track_is_scored_for_its_own_timeline(self) -> None:
        tracks = [
            FaceTrack(
                f"face-{index}", 0.0, 2.0, [FaceBox(0.0, 0, 0, index + 1, index + 1)]
            )
            for index in range(FACE_TRACK_COUNT)
        ]

        def score_track(_model, _audio, _frames, track):
            return [FaceActivity(track.track_id, 0.0, 2.0, 0.9)]

        with (
            patch("src.light_asd._load_model", return_value=object()),
            patch("src.light_asd._audio_features", return_value=np.empty((1, 13))),
            patch("src.light_asd._score_track", side_effect=score_track) as scorer,
        ):
            activities = score_face_activity(
                Path("audio.wav"), [], tracks, Path("model.pt")
            )

        self.assertEqual(scorer.call_count, len(tracks))
        self.assertEqual(len(activities), len(tracks))

    def test_later_shots_are_scored_even_when_earlier_faces_are_larger(self) -> None:
        tracks = [
            FaceTrack(
                f"face-{index}",
                float(index),
                float(index + 1),
                [FaceBox(float(index), 0, 0, 20 - index, 20 - index)],
            )
            for index in range(FACE_TRACK_COUNT)
        ]

        with (
            patch("src.light_asd._load_model", return_value=object()),
            patch("src.light_asd._audio_features", return_value=np.empty((1, 13))),
            patch("src.light_asd._score_track", return_value=[]) as scorer,
        ):
            score_face_activity(Path("audio.wav"), [], tracks, Path("model.pt"))

        self.assertEqual(
            [call.args[-1].track_id for call in scorer.call_args_list],
            [track.track_id for track in tracks],
        )


if __name__ == "__main__":
    unittest.main()
