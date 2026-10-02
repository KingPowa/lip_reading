"""Consistent, high-contrast colours for anonymous speaker labels."""

SPEAKER_COLORS = (
    "#38bdf8",
    "#fb7185",
    "#a3e635",
    "#fbbf24",
    "#c084fc",
    "#2dd4bf",
    "#fb923c",
    "#f472b6",
)
NEUTRAL_COLOR = "#e5e7eb"


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
