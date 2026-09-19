import unittest

from src.contracts import CaptionEvent
from src.render import _active_tracks, _caption_font, _caption_lines


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


if __name__ == "__main__":
    unittest.main()
