from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """A detected face rectangle at one presentation timestamp."""

    timestamp: float
    x: int
    y: int
    width: int
    height: int


class FaceObservation(NamedTuple):
    box: FaceBox
    aperture: float | None
    features: np.ndarray | None = None


@dataclass(frozen=True)
class LiveFace:
    track_id: str
    box: FaceBox
    mouth_motion: float


@dataclass
class FaceTrack:
    """One temporarily stable face identity within a video."""

    track_id: str
    start: float
    end: float
    boxes: list[FaceBox]


@dataclass(frozen=True)
class SpeakerTurn:
    """An anonymous voice interval with a confidence score."""

    speaker_id: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class Word:
    """A transcribed word and its time range."""

    text: str
    start: float
    end: float
    confidence: float


@dataclass(frozen=True)
class FaceActivity:
    """An active-speaker score for a visible face during an interval."""

    track_id: str
    start: float
    end: float
    speaking_probability: float


@dataclass(frozen=True)
class CaptionEvent:
    """A rendered, speaker-attributed caption interval."""

    start: float
    end: float
    speaker: str
    face_track: str | None
    text: str
    confidence: float
    state: str

    def as_json(self) -> dict[str, Any]:
        """Return the public JSON representation of this caption event.

        Returns:
            A dictionary using the documented camel-case ``faceTrack`` key.
        """
        data = asdict(self)
        data["faceTrack"] = data.pop("face_track")
        return data
