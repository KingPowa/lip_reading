import unittest

from src.contracts import CaptionEvent, FaceBox, FaceTrack
from src.render import _active_tracks, _caption_font, _caption_lines, track_payloads


class RenderTest(unittest.TestCase):
    def test_caption_lines_fit_the_available_pixels(self) -> None:
        maximum_width = 240
        lines = _caption_lines("Person 1: Perché questa didascalia resta nel video", maximum_width)
        self.assertTrue(lines)
        self.assertTrue(all(_caption_font().getlength(line) <= maximum_width for line in lines))

    def test_active_tracks_draws_one_box_per_person(self) -> None:
        events = [
            CaptionEvent(0.0, 1.0, "Person 1", "face-1", "first", 0.7, "visible"),
            CaptionEvent(0.0, 1.0, "Person 1", "face-2", "second", 0.9, "visible"),
        ]
        self.assertEqual(set(_active_tracks(events)), {"face-2"})

    def test_track_payload_contains_a_clipped_upper_body_box(self) -> None:
        track = FaceTrack("face-1", 0.0, 0.0, [FaceBox(0.0, 80, 60, 40, 40)])
        payload = track_payloads([track], 160, 180)
        self.assertEqual(payload[0]["boxes"][0]["bodyBox"], {"timestamp": 0.0, "x": 40, "y": 47, "width": 120, "height": 133})


if __name__ == "__main__":
    unittest.main()
