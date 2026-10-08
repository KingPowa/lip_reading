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
    SPEAKER_HOLD_SECONDS,
    SPEAKER_SWITCH_SECONDS,
    SPEECH_RELEASE_SECONDS,
    UNCERTAIN_SPEAKER,
)
from src.core.contracts import CaptionEvent, FaceBox, FaceObservation, LiveFace, Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.identity import FaceIdentityGallery


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
        self.identities = FaceIdentityGallery()
        self.revision = 0
        self.active_identity: str | None = None
        self.unmeasured: set[str] = set()
        self.held: SpeakerEstimate | None = None
        self.challenger: tuple[str, float] | None = None

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
        self, observations: list[FaceObservation], timestamp: float
    ) -> SpeakerEstimate:
        if timestamp < self.latest_vision:
            raise ValueError("frame timestamp is out of order")
        self.boxes = {
            identity: box
            for identity, box in self.boxes.items()
            if timestamp - box.timestamp <= IDENTITY_ABSENCE_SECONDS
        }
        observations = [FaceObservation(*item) for item in observations]
        identities = self.identities.identify(observations, self.boxes)
        faces = []
        self.unmeasured = set()
        for identity, observation in zip(identities, observations):
            box, aperture, _ = observation
            self.boxes[identity] = box
            history = self.mouths.setdefault(
                identity, deque(maxlen=MAXIMUM_HISTORY_ITEMS)
            )
            if aperture is not None:
                history.append((timestamp, aperture))
            while history and timestamp - history[0][0] > MOUTH_HISTORY_SECONDS:
                history.popleft()
            motion = 0.0
            if len(history) >= MINIMUM_MOUTH_OBSERVATIONS:
                motion = max(value for _, value in history) - min(
                    value for _, value in history
                )
            elif aperture is None:
                self.unmeasured.add(identity)
            faces.append(LiveFace(identity, box, motion))
        self.mouths = {face.track_id: self.mouths[face.track_id] for face in faces}
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
            return self._without_measured_motion(timestamp)
        selected = ranked[0]
        runner_up_motion = self._candidate_score(ranked[1]) if len(ranked) > 1 else 0.0
        margin = self._candidate_score(selected) - runner_up_motion
        if margin < MOUTH_ASSIGNMENT_MARGIN:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        confidence = min(1.0, margin / MOUTH_MOTION_THRESHOLD)
        return self._visible_estimate(timestamp, selected.track_id, confidence)

    @staticmethod
    def _visible_estimate(
        timestamp: float, identity: str, confidence: float
    ) -> SpeakerEstimate:
        label = f"Person {identity.rsplit('-', 1)[1]}"
        return SpeakerEstimate(timestamp, label, identity, "visible", confidence)

    def _without_measured_motion(self, timestamp: float) -> SpeakerEstimate:
        if not self.unmeasured:
            return SpeakerEstimate(
                timestamp, OFF_SCREEN_SPEAKER, None, "off_screen", 0.0
            )
        if self.active_identity in self.unmeasured:
            return self._visible_estimate(timestamp, self.active_identity, 0.0)
        return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)

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

    def _stabilized(self, raw: SpeakerEstimate) -> SpeakerEstimate:
        held = self.held
        timestamp = raw.timestamp
        holding = (
            held is not None
            and timestamp - held.timestamp <= SPEAKER_HOLD_SECONDS
            and timestamp - self.latest_vision <= GEOMETRY_EXPIRY_SECONDS
            and any(face.track_id == held.face_track for face in self.current_faces)
        )
        if raw.state == "silent" or not holding:
            self.challenger = None
            self.held = raw if raw.state == "visible" else None
            return raw
        if raw.state != "visible":
            return replace(held, timestamp=timestamp)
        if raw.face_track == held.face_track:
            self.challenger = None
            self.held = raw
            return raw
        if self.challenger is None or self.challenger[0] != raw.face_track:
            self.challenger = (raw.face_track, timestamp)
        if timestamp - self.challenger[1] < SPEAKER_SWITCH_SECONDS:
            return replace(held, timestamp=timestamp)
        self.challenger = None
        self.held = raw
        return raw

    def _record_estimate(self, timestamp: float) -> SpeakerEstimate:
        self.revision += 1
        estimate = replace(
            self._stabilized(self.estimate(timestamp)), revision=self.revision
        )
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
