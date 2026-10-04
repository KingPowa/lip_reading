"""Timestamped audio activity, box continuity, and cheap speaker disambiguation."""

from collections import deque
from dataclasses import asdict, dataclass, replace
from math import sqrt

import numpy as np

from src.config.live import (
    GEOMETRY_EXPIRY_SECONDS,
    HISTORY_SECONDS,
    IDENTITY_ABSENCE_SECONDS,
    MAXIMUM_HISTORY_ITEMS,
    MINIMUM_MOUTH_OBSERVATIONS,
    MINIMUM_SPEECH_ENERGY,
    MOUTH_ASSIGNMENT_MARGIN,
    MOUTH_HISTORY_SECONDS,
    MOUTH_MOTION_THRESHOLD,
    OFF_SCREEN_SPEAKER,
    SPEAKER_CONTINUITY_BONUS,
    SPEECH_RELEASE_SECONDS,
    UNCERTAIN_SPEAKER,
)
from src.config.vision import FACE_TRACK_IOU_THRESHOLD
from src.core.contracts import CaptionEvent, FaceBox, LiveFace, Word
from src.core.tracking.association import merge_word_events
from src.core.vision.faces import intersection_over_union


@dataclass(frozen=True)
class SpeakerEstimate:
    timestamp: float
    speaker: str
    face_track: str | None
    state: str
    confidence: float
    revision: int = 0

    def as_json(self) -> dict:
        payload = asdict(self)
        payload["faceTrack"] = payload.pop("face_track")
        return payload


class RealtimeTracker:
    """Owned by one session; callers synchronize brief state updates."""

    def __init__(self) -> None:
        self.boxes: dict[str, FaceBox] = {}
        self.mouths: dict[str, deque[tuple[float, float]]] = {}
        self.current_faces: list[LiveFace] = []
        self.speaker_history: deque[SpeakerEstimate] = deque(
            maxlen=MAXIMUM_HISTORY_ITEMS
        )
        self.latest_vision = -1.0
        self.latest_audio = -1.0
        self.last_speech = float("-inf")
        self.next_identity = 1
        self.revision = 0
        self.active_identity: str | None = None

    def update_audio(self, samples: np.ndarray, timestamp: float) -> SpeakerEstimate:
        if timestamp < self.latest_audio:
            raise ValueError("audio activity timestamp is out of order")
        energy = float(np.sqrt(np.mean(samples * samples)))
        speaking = energy >= MINIMUM_SPEECH_ENERGY
        if speaking:
            self.last_speech = timestamp
        self.latest_audio = timestamp
        return self._record_estimate(max(timestamp, self.latest_vision))

    def update_faces(
        self, observations: list[tuple[FaceBox, float | None]], timestamp: float
    ) -> SpeakerEstimate:
        if timestamp < self.latest_vision:
            raise ValueError("frame timestamp is out of order")
        self.boxes = {
            identity: box
            for identity, box in self.boxes.items()
            if timestamp - box.timestamp <= IDENTITY_ABSENCE_SECONDS
        }
        self.mouths = {
            identity: values
            for identity, values in self.mouths.items()
            if identity in self.boxes
        }
        available = set(self.boxes)
        faces = []
        for box, aperture in observations:
            identity = max(
                available,
                key=lambda candidate: intersection_over_union(
                    self.boxes[candidate], box
                ),
                default=None,
            )
            matched = (
                identity is not None
                and intersection_over_union(self.boxes[identity], box)
                >= FACE_TRACK_IOU_THRESHOLD
            )
            if matched:
                available.remove(identity)
            else:
                identity = f"face-{self.next_identity}"
                self.next_identity += 1
            self.boxes[identity] = box
            history = self.mouths.setdefault(
                identity, deque(maxlen=MAXIMUM_HISTORY_ITEMS)
            )
            if aperture is not None:
                history.append((timestamp, aperture))
            while history and timestamp - history[0][0] > MOUTH_HISTORY_SECONDS:
                history.popleft()
            motion = 0.0
            if aperture is not None and len(history) >= MINIMUM_MOUTH_OBSERVATIONS:
                motion = max(value for _, value in history) - min(
                    value for _, value in history
                )
            faces.append(LiveFace(identity, box, motion))
        self.current_faces = faces
        self.latest_vision = timestamp
        return self._record_estimate(max(timestamp, self.latest_audio))

    def estimate(self, timestamp: float) -> SpeakerEstimate:
        silent = timestamp - self.last_speech > SPEECH_RELEASE_SECONDS
        if silent:
            return SpeakerEstimate(timestamp, "", None, "silent", 0.0)
        vision_missing = (
            self.latest_vision < 0
            or timestamp - self.latest_vision > GEOMETRY_EXPIRY_SECONDS
        )
        if vision_missing:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        candidates = [
            face
            for face in self.current_faces
            if face.mouth_motion >= MOUTH_MOTION_THRESHOLD
        ]
        ranked = sorted(candidates, key=self._candidate_score, reverse=True)
        if not ranked or self._candidate_score(ranked[0]) < MOUTH_MOTION_THRESHOLD:
            return SpeakerEstimate(
                timestamp, OFF_SCREEN_SPEAKER, None, "off_screen", 0.0
            )
        selected = ranked[0]
        runner_up_motion = self._candidate_score(ranked[1]) if len(ranked) > 1 else 0.0
        margin = self._candidate_score(selected) - runner_up_motion
        if margin < MOUTH_ASSIGNMENT_MARGIN:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        identity = selected.track_id
        label = f"Person {identity.rsplit('-', 1)[1]}"
        confidence = min(1.0, margin / MOUTH_MOTION_THRESHOLD)
        return SpeakerEstimate(timestamp, label, identity, "visible", confidence)

    def speaker_at(self, timestamp: float) -> SpeakerEstimate:
        candidates = [
            item
            for item in self.speaker_history
            if abs(item.timestamp - timestamp) <= GEOMETRY_EXPIRY_SECONDS
            and item.state != "silent"
        ]
        if not candidates:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        return min(
            candidates,
            key=lambda item: (abs(item.timestamp - timestamp), -item.revision),
        )

    def caption_events(self, words: list[Word]) -> list[CaptionEvent]:
        events = []
        for word in words:
            estimate = self.speaker_at((word.start + word.end) / 2)
            events.append(
                CaptionEvent(
                    word.start,
                    word.end,
                    estimate.speaker,
                    estimate.face_track,
                    word.text,
                    word.confidence,
                    estimate.state,
                )
            )
        return merge_word_events(events)

    def _candidate_score(self, face: LiveFace) -> float:
        bonus = (
            SPEAKER_CONTINUITY_BONUS if face.track_id == self.active_identity else 0.0
        )
        maximum_face_area = max(
            candidate.box.width * candidate.box.height
            for candidate in self.current_faces
        )
        relative_face_size = sqrt(face.box.width * face.box.height / maximum_face_area)
        return face.mouth_motion * relative_face_size + bonus

    def _record_estimate(self, timestamp: float) -> SpeakerEstimate:
        self.revision += 1
        estimate = replace(self.estimate(timestamp), revision=self.revision)
        if self.speaker_history and self.speaker_history[-1].timestamp == timestamp:
            self.speaker_history.pop()
        self.speaker_history.append(estimate)
        self.active_identity = estimate.face_track
        while (
            self.speaker_history
            and timestamp - self.speaker_history[0].timestamp > HISTORY_SECONDS
        ):
            self.speaker_history.popleft()
        return estimate
