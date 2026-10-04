"""Add display colours at the presentation boundary."""

from dataclasses import asdict

from src.app.rendering.colors import speaker_color
from src.core.contracts import CaptionEvent, LiveFace
from src.core.tracking.realtime import SpeakerEstimate


def caption_payload(event: CaptionEvent) -> dict:
    return event.as_json() | {"color": speaker_color(event.speaker)}


def speaker_payload(estimate: SpeakerEstimate) -> dict:
    return estimate.as_json() | {"color": speaker_color(estimate.speaker)}


def face_payload(face: LiveFace) -> dict:
    return {
        "trackId": face.track_id,
        "box": asdict(face.box),
        "mouthMotion": face.mouth_motion,
    }
