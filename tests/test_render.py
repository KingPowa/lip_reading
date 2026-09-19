import unittest

from src.render import _caption_font, _caption_lines


class RenderTest(unittest.TestCase):
    def test_caption_lines_fit_the_available_pixels(self) -> None:
        maximum_width = 240
        lines = _caption_lines("Person 1: Perché questa didascalia resta nel video", maximum_width)
        self.assertTrue(lines)
        self.assertTrue(all(_caption_font().getlength(line) <= maximum_width for line in lines))


if __name__ == "__main__":
    unittest.main()
