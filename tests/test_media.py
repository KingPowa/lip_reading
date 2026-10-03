import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.media import MediaError, _run, normalize_camera_recording, read_video_frames


class MediaTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "posix_spawn"), "requires POSIX spawn")
    def test_media_process_avoids_fork_handlers_in_the_threaded_server(self):
        with patch(
            "subprocess._fork_exec",
            side_effect=AssertionError("fork can deadlock with OpenBLAS"),
        ):
            result = _run(["ffprobe", "-version"], timeout_seconds=5)
        self.assertEqual(result.returncode, 0)
        self.assertIn("ffprobe version", result.stdout)

    def test_live_recordings_reject_large_dimensions_before_decoding(self) -> None:
        dimensions = MagicMock(
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]})
        )
        with patch("src.media._run", return_value=dimensions) as runner:
            with self.assertRaises(MediaError):
                normalize_camera_recording(
                    Path("recording.webm"), Path("normalized.mp4")
                )
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[0][0], "ffprobe")

    def test_live_recording_duration_includes_already_processed_audio(self) -> None:
        dimensions = MagicMock(
            stdout=json.dumps({"streams": [{"width": 640, "height": 360}]})
        )
        duration = MagicMock(stdout=json.dumps({"format": {"duration": "23"}}))
        with patch("src.media._run", side_effect=[dimensions, MagicMock(), duration]):
            with self.assertRaises(MediaError):
                normalize_camera_recording(
                    Path("recording.webm"), Path("normalized.mp4"), 10.0
                )

    def test_live_recording_without_input_duration_is_accepted_within_limit(
        self,
    ) -> None:
        dimensions = MagicMock(
            stdout=json.dumps({"streams": [{"width": 1280, "height": 720}]})
        )
        duration = MagicMock(stdout=json.dumps({"format": {"duration": "20"}}))
        with patch(
            "src.media._run", side_effect=[dimensions, MagicMock(), duration]
        ) as runner:
            result = normalize_camera_recording(
                Path("recording.webm"), Path("normalized.mp4"), 10.0
            )
        self.assertEqual(result, Path("normalized.mp4"))
        command = runner.call_args_list[1].args[0]
        self.assertEqual(command[command.index("-t") + 1], "23.0")
        self.assertEqual(command[command.index("-r") + 1], "25")
        self.assertEqual(
            [call.kwargs["timeout_seconds"] for call in runner.call_args_list],
            [20, 60, 20],
        )

    def test_media_timeout_is_reported_as_a_media_error(self) -> None:
        with patch(
            "src.media.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 60),
        ):
            with self.assertRaises(MediaError):
                _run(["ffmpeg"], timeout_seconds=60)

    def test_invalid_recording_start_is_rejected_before_ffmpeg(self) -> None:
        with patch("src.media._run") as runner:
            for start_seconds in (-1.0, float("nan"), float("inf")):
                with (
                    self.subTest(start_seconds=start_seconds),
                    self.assertRaises(MediaError),
                ):
                    normalize_camera_recording(
                        Path("recording.webm"), Path("normalized.mp4"), start_seconds
                    )
        runner.assert_not_called()

    def test_decoder_retains_consecutive_images_for_mouth_motion(self) -> None:
        images = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(11)]
        timestamps = [index / 25 for index in range(len(images))]
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.side_effect = [(True, image) for image in images] + [(False, None)]

        with (
            patch("src.media.video_presentation_timestamps", return_value=timestamps),
            patch("src.media.cv2.VideoCapture", return_value=capture),
        ):
            frames = read_video_frames(Path("recording.mp4"))

        self.assertEqual([frame.timestamp for frame in frames], timestamps)
        self.assertEqual(
            [int(frame.image[0, 0, 0]) for frame in frames], list(range(len(images)))
        )
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
