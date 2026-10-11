"""Timestamped audio activity, box continuity, and cheap speaker disambiguation."""

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from math import sqrt

import numpy as np

from src.config.live import (
    GEOMETRY_EXPIRY_SECONDS,
    HISTORY_SECONDS,
    IDENTITY_ABSENCE_SECONDS,
    MAXIMUM_HISTORY_ITEMS,
    MAXIMUM_TENTATIVE_GAP_FRAMES,
    MINIMUM_FACE_TRACK_FRAMES,
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
    UNMEASURED_CARRYOVER_SECONDS,
)
from src.config.tracking import (
    MAXIMUM_VOICE_TIMELINE,
    VOICE_BIND_DOMINANT_FRACTION,
    VOICE_BIND_MIN_CONFIDENCE,
    VOICE_BIND_MIN_SAMPLES,
    VOICE_FALLBACK_CONFIDENCE,
    VOICE_MATCH_BONUS,
    VOICE_MERGE_MIN_CONFIRMATIONS,
    VOICE_VALIDITY_SECONDS,
)
from src.core.contracts import CaptionEvent, FaceBox, FaceObservation, LiveFace, Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.identity import FaceIdentityGallery
from src.core.tracking.voice import VoiceGallery


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
        self.unmeasured: dict[str, float] = {}
        self.sightings: dict[str, int] = {}
        self.labels: dict[str, int] = {}
        self.label_count = 0
        self.gaps: dict[str, int] = {}
        self.promoted_at: dict[str, float] = {}
        self.voices = VoiceGallery()
        self.voice_timeline: deque[tuple[float, float, str]] = deque(
            maxlen=MAXIMUM_VOICE_TIMELINE
        )
        self.clear_speakers: deque[tuple[float, str]] = deque(
            maxlen=MAXIMUM_HISTORY_ITEMS
        )
        self.merge_votes: tuple[str, str, int] | None = None
        self.merge_count = 0
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
        self._confirm(identities, timestamp)
        faces = []
        unmeasured = {}
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
                unmeasured[identity] = self.unmeasured.get(identity, timestamp)
            if identity in self.labels:
                faces.append(LiveFace(identity, box, motion))
        self.unmeasured = unmeasured
        self.mouths = {identity: self.mouths[identity] for identity in identities}
        self.current_faces = faces
        self._forget_stale_labels()
        self.latest_vision = timestamp
        return self._record_estimate(max(timestamp, self.latest_audio))

    def _confirm(self, identities: list[str], timestamp: float) -> None:
        for identity in identities:
            self.sightings[identity] = self.sightings.get(identity, 0) + 1
            self.gaps.pop(identity, None)
            if identity not in self.labels and (
                self.sightings[identity] >= MINIMUM_FACE_TRACK_FRAMES
            ):
                self.label_count += 1
                self.labels[identity] = self.label_count
                self.promoted_at[identity] = timestamp
        for identity in set(self.sightings) - set(identities):
            self.gaps[identity] = self.gaps.get(identity, 0) + 1
            if identity in self.labels or self.gaps[identity] > MAXIMUM_TENTATIVE_GAP_FRAMES:
                self._forget_tentative(identity)

    def _forget_tentative(self, identity: str) -> None:
        self.sightings.pop(identity, None)
        self.gaps.pop(identity, None)
        if identity not in self.labels:
            self.boxes.pop(identity, None)
            self.identities.forget(identity)

    def _forget_stale_labels(self) -> None:
        self.labels = {
            identity: number
            for identity, number in self.labels.items()
            if identity in self.boxes
            or identity in self.identities.features
            or identity in self.voices.bound_faces()
        }
        self.promoted_at = {
            identity: when
            for identity, when in self.promoted_at.items()
            if identity in self.labels
        }

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
        voice_face = self._voice_face(timestamp)

        def score(face: LiveFace) -> float:
            return self._candidate_score(face, voice_face)

        ranked = sorted(candidates, key=score, reverse=True)
        if not ranked or score(ranked[0]) < MOUTH_MOTION_THRESHOLD:
            return self._without_measured_motion(timestamp, voice_face)
        selected = ranked[0]
        runner_up_motion = score(ranked[1]) if len(ranked) > 1 else 0.0
        margin = score(selected) - runner_up_motion
        if margin < MOUTH_ASSIGNMENT_MARGIN:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        confidence = min(1.0, margin / MOUTH_MOTION_THRESHOLD)
        return self._visible_estimate(timestamp, selected.track_id, confidence)

    def _visible_estimate(
        self, timestamp: float, identity: str, confidence: float
    ) -> SpeakerEstimate:
        label = f"Person {self.labels[identity]}"
        return SpeakerEstimate(timestamp, label, identity, "visible", confidence)

    def _without_measured_motion(
        self, timestamp: float, voice_face: str | None
    ) -> SpeakerEstimate:
        by_voice = self._voice_estimate(timestamp, voice_face)
        if by_voice is not None:
            return by_voice
        if not self.unmeasured:
            return SpeakerEstimate(
                timestamp, OFF_SCREEN_SPEAKER, None, "off_screen", 0.0
            )
        if self.active_identity in self.unmeasured and not self._carry_expired(
            timestamp
        ):
            return self._visible_estimate(timestamp, self.active_identity, 0.0)
        return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)

    def _voice_estimate(
        self, timestamp: float, face: str | None
    ) -> SpeakerEstimate | None:
        if face is None:
            return None
        if any(item.track_id == face for item in self.current_faces):
            if face in self.unmeasured:
                return self._visible_estimate(timestamp, face, VOICE_FALLBACK_CONFIDENCE)
            return None
        if self.unmeasured or face not in self.labels:
            return None
        label = f"Person {self.labels[face]}"
        return SpeakerEstimate(
            timestamp, label, None, "off_screen", VOICE_FALLBACK_CONFIDENCE
        )

    def _voice_face(self, timestamp: float) -> str | None:
        for start, end, voice in reversed(self.voice_timeline):
            if start <= timestamp:
                valid = timestamp - end <= VOICE_VALIDITY_SECONDS
                return self.voices.face_for(voice) if valid else None
        return None

    def _carry_expired(self, timestamp: float) -> bool:
        since = self.unmeasured.get(self.active_identity)
        return since is not None and timestamp - since > UNMEASURED_CARRYOVER_SECONDS

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

    def _candidate_score(self, face: LiveFace, voice_face: str | None) -> float:
        bonus = (
            SPEAKER_CONTINUITY_BONUS if face.track_id == self.active_identity else 0.0
        )
        if face.track_id == voice_face:
            bonus += VOICE_MATCH_BONUS
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
            and not self._carry_expired(timestamp)
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
        raw = self.estimate(timestamp)
        self._note_clear_speaker(raw)
        estimate = replace(self._stabilized(raw), revision=self.revision)
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

    def _note_clear_speaker(self, raw: SpeakerEstimate) -> None:
        moving = [
            face for face in self.current_faces
            if face.mouth_motion >= MOUTH_MOTION_THRESHOLD
        ]
        if (
            raw.state == "visible"
            and raw.confidence >= VOICE_BIND_MIN_CONFIDENCE
            and [face.track_id for face in moving] == [raw.face_track]
        ):
            self.clear_speakers.append((raw.timestamp, raw.face_track))

    def update_voice(self, embedding: np.ndarray, start: float, end: float) -> str | None:
        voice = self.voices.assign(embedding)
        if voice is None:
            return None
        self.voice_timeline.append((start, end, voice))
        if not self._merge_into_absent(voice):
            self._bind_voice(voice, start, end)
        return voice

    def _bind_voice(self, voice: str, start: float, end: float) -> None:
        faces = [face for when, face in self.clear_speakers if start <= when <= end]
        if len(faces) < VOICE_BIND_MIN_SAMPLES:
            return
        face, count = Counter(faces).most_common(1)[0]
        if count >= VOICE_BIND_DOMINANT_FRACTION * len(faces):
            self.voices.bind(voice, face)

    def _merge_into_absent(self, voice: str) -> bool:
        old = self.voices.face_for(voice)
        new = self._merge_candidate(old)
        if new is None:
            self.merge_votes = None
            return False
        previous = self.merge_votes
        votes = previous[2] + 1 if previous and previous[:2] == (old, new) else 1
        self.merge_votes = (old, new, votes)
        if votes >= VOICE_MERGE_MIN_CONFIRMATIONS:
            self.merge_votes = None
            self._merge_identity(new, old)
        return True

    def _merge_candidate(self, old: str | None) -> str | None:
        visible = {face.track_id for face in self.current_faces}
        if old is None or old not in self.labels or old in visible | set(self.sightings):
            return None
        last_seen = self.boxes[old].timestamp if old in self.boxes else float("-inf")
        newcomers = [
            face
            for face in self.current_faces
            if self.promoted_at.get(face.track_id, float("-inf")) > last_seen
            and self.voices.voice_for(face.track_id) is None
        ]
        if len(newcomers) != 1 or newcomers[0].mouth_motion < MOUTH_MOTION_THRESHOLD:
            return None
        return newcomers[0].track_id

    def _merge_identity(self, new: str, old: str) -> None:
        self.merge_count += 1
        self.identities.merge(new, old)
        self.boxes[old] = self.boxes.pop(new)
        self.mouths[old] = self.mouths.pop(new)
        self.current_faces = [
            replace(face, track_id=old) if face.track_id == new else face
            for face in self.current_faces
        ]
        if new in self.unmeasured:
            self.unmeasured[old] = self.unmeasured.pop(new)
        for registry in (self.sightings, self.gaps, self.promoted_at):
            registry.pop(new, None)
        if self.labels.pop(new, None) == self.label_count:
            self.label_count -= 1
        self.clear_speakers = deque(
            ((when, old if face == new else face) for when, face in self.clear_speakers),
            maxlen=MAXIMUM_HISTORY_ITEMS,
        )
        self.speaker_history = deque(
            (self._relabeled(item, new, old) for item in self.speaker_history),
            maxlen=MAXIMUM_HISTORY_ITEMS,
        )
        if self.active_identity == new:
            self.active_identity = old

    def _relabeled(self, item: SpeakerEstimate, new: str, old: str) -> SpeakerEstimate:
        if item.face_track != new:
            return item
        return replace(item, speaker=f"Person {self.labels[old]}", face_track=old)
