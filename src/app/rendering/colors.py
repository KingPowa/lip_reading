"""Consistent, high-contrast colours for anonymous speaker labels."""

from src.config.rendering import NEUTRAL_COLOR, SPEAKER_COLORS


def speaker_color(speaker: str) -> str:
    """Return the stable display colour assigned to a ``Person N`` label."""
    prefix, separator, number = speaker.rpartition(" ")
    if prefix != "Person" or not separator or not number.isdecimal():
        return NEUTRAL_COLOR
    return SPEAKER_COLORS[(int(number) - 1) % len(SPEAKER_COLORS)]


def speaker_color_bgr(speaker: str) -> tuple[int, int, int]:
    """Return the speaker display colour in OpenCV's BGR component order."""
    color = speaker_color(speaker).removeprefix("#")
    return tuple(int(color[index : index + 2], 16) for index in (4, 2, 0))
