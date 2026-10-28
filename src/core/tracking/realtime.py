"""Timestamped audio activity, box continuity, and cheap speaker disambiguation."""

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from math import exp, sqrt

import numpy as np

from src.config.live import (
    EVIDENCE_ASSIGNMENT_MARGIN,
    EVIDENCE_CONFIDENCE_SCALE,
    GEOMETRY_EXPIRY_SECONDS,
    HISTORY_SECONDS,
    IDENTITY_ABSENCE_SECONDS,
    MAXIMUM_HISTORY_ITEMS,
    MAXIMUM_TENTATIVE_GAP_FRAMES,
    MINIMUM_FACE_TRACK_FRAMES,
    MINIMUM_MOUTH_OBSERVATIONS,
    MAXIMUM_MOUTH_BASELINE,
    MINIMUM_FACE_SIZE_WEIGHT,
    MOUTH_BASELINE_RATE,
    MOUTH_HISTORY_SECONDS,
    MOUTH_MOTION_THRESHOLD,
    OFF_SCREEN_SPEAKER,
    RELATIVE_MOUTH_MARGIN,
    SPEAKER_CONTINUITY_BONUS,
    SPEAKER_HOLD_SECONDS,
    SPEAKER_SWITCH_SECONDS,
    SILENT_EVIDENCE_DECAY_SECONDS,
    SPEAKING_EVIDENCE_SECONDS,
    SPEAKING_EVIDENCE_THRESHOLD,
    SPEECH_RELEASE_SECONDS,
    TRACKING_INTERVAL_MILLISECONDS,
    UNCERTAIN_SPEAKER,
    UNMEASURED_CARRYOVER_SECONDS,
)
from src.config.tracking import (
    CLEAR_SPEAKER_RATIO,
    MAXIMUM_VOICE_TIMELINE,
    VOICE_BIND_DOMINANT_FRACTION,
    VOICE_BIND_MIN_SAMPLES,
    VOICE_FALLBACK_CONFIDENCE,
    VOICE_MATCH_BONUS,
    VOICE_MERGE_MIN_CONFIRMATIONS,
    VOICE_STICKY_MAXIMUM_SECONDS,
    VOICE_TURN_LEAD,
    VOICE_TURN_MIN_SECONDS,
    VOICE_TURN_SINGLE_VOICE_SHARE,
    VOICE_VALIDITY_SECONDS,
)
from src.core.audio.activity import SpeechActivityGate
from src.core.contracts import CaptionEvent, FaceBox, FaceObservation, LiveFace, Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.identity import FaceIdentityGallery
from src.core.tracking.voice import VoiceGallery

TURN_MAXIMUM_STEP_SECONDS = 0.2
TURN_NAMED_SHARE = 0.3
TURN_SPLIT_SECONDS = 1.5
TURN_CONTINUITY_SECONDS = 2.0


@dataclass(frozen=True)
class SpeakerEstimate:
    timestamp: float
    speaker: str
    face_track: str | None
    state: str
    confidence: float
    revision: int = 0
    voiced: bool = False  # set by the voice-majority override, never by mouth motion
    raw_face: str | None = None  # mouth-motion pick behind a voiced estimate

    def as_json(self) -> dict:
        payload = asdict(self)
        del payload["voiced"], payload["raw_face"]
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
        self.speech_started = float("-inf")
        self.speaking = False
        self.gate = SpeechActivityGate()
        self.mouth_baselines: dict[str, float] = {}
        self.speaking_evidence: dict[str, float] = {}
        self.evidence_times: dict[str, float] = {}
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
        self.last_event: CaptionEvent | None = None
        self.voice_turn_votes: dict[str, Counter[str]] = {}
        self.challenger: tuple[str, float] | None = None

    def update_audio(self, samples: np.ndarray, timestamp: float) -> SpeakerEstimate:
        if timestamp < self.latest_audio:
            raise ValueError("audio activity timestamp is out of order")
        speaking = self.gate.update(samples)
        if speaking:
            if not self.speaking:
                self.speech_started = timestamp
            self.last_speech = timestamp
        self.speaking = speaking
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
                raw = max(value for _, value in history) - min(
                    value for _, value in history
                )
                motion = self._excess_motion(identity, raw, timestamp)
            elif aperture is None:
                unmeasured[identity] = self.unmeasured.get(identity, timestamp)
            if identity in self.labels:
                self._update_evidence(identity, motion, timestamp)
                faces.append(LiveFace(identity, box, motion))
        self.unmeasured = unmeasured
        self.mouths = {identity: self.mouths[identity] for identity in identities}
        self.mouth_baselines = {
            identity: value
            for identity, value in self.mouth_baselines.items()
            if identity in self.mouths
        }
        self.speaking_evidence = {
            identity: value
            for identity, value in self.speaking_evidence.items()
            if identity in self.mouths
        }
        self.evidence_times = {
            identity: value
            for identity, value in self.evidence_times.items()
            if identity in self.mouths
        }
        self.current_faces = faces
        self._forget_stale_labels()
        self.latest_vision = timestamp
        return self._record_estimate(max(timestamp, self.latest_audio))

    def _excess_motion(self, identity: str, raw: float, timestamp: float) -> float:
        """Motion above this face's landmark jitter, learned only during silence."""
        baseline = self.mouth_baselines.get(identity, 0.0)
        if timestamp - self.last_speech > SPEECH_RELEASE_SECONDS:
            baseline += MOUTH_BASELINE_RATE * (raw - baseline)
            baseline = min(MAXIMUM_MOUTH_BASELINE, baseline)
            self.mouth_baselines[identity] = baseline
        return max(0.0, raw - baseline)

    def _update_evidence(self, identity: str, motion: float, timestamp: float) -> None:
        """Integrate mouth motion over seconds while speech is present."""
        previous = self.evidence_times.get(identity)
        step = (
            TRACKING_INTERVAL_MILLISECONDS / 1000
            if previous is None
            else min(1.0, max(0.0, timestamp - previous))
        )
        evidence = self.speaking_evidence.get(identity, 0.0)
        if timestamp - self.last_speech <= SPEECH_RELEASE_SECONDS:
            alpha = 1 - exp(-step / SPEAKING_EVIDENCE_SECONDS)
            evidence += alpha * (motion - evidence)
        else:
            evidence *= exp(-step / SILENT_EVIDENCE_DECAY_SECONDS)
        self.speaking_evidence[identity] = evidence
        self.evidence_times[identity] = timestamp

    def _evidence(self, face: LiveFace) -> float:
        return self.speaking_evidence.get(face.track_id, 0.0)

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
            if self._evidence(face) >= SPEAKING_EVIDENCE_THRESHOLD
            and face.track_id not in self.unmeasured
        ]
        voice_face = self._voice_face(timestamp)

        def score(face: LiveFace) -> float:
            return self._candidate_score(face, voice_face)

        ranked = sorted(candidates, key=score, reverse=True)
        if not ranked or score(ranked[0]) < SPEAKING_EVIDENCE_THRESHOLD:
            return self._without_measured_motion(timestamp, voice_face)
        selected = ranked[0]
        runner_up_motion = score(ranked[1]) if len(ranked) > 1 else 0.0
        margin = score(selected) - runner_up_motion
        required = min(
            EVIDENCE_ASSIGNMENT_MARGIN, RELATIVE_MOUTH_MARGIN * score(selected)
        )
        if margin < required:
            return SpeakerEstimate(timestamp, UNCERTAIN_SPEAKER, None, "uncertain", 0.0)
        confidence = min(1.0, margin / EVIDENCE_CONFIDENCE_SCALE)
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
        if face not in self.labels:
            return None
        label = f"Person {self.labels[face]}"
        return SpeakerEstimate(
            timestamp, label, None, "off_screen", VOICE_FALLBACK_CONFIDENCE
        )

    def _current_voice(self, timestamp: float) -> str | None:
        for start, end, voice in reversed(self.voice_timeline):
            if start <= timestamp:
                valid = timestamp - end <= VOICE_VALIDITY_SECONDS or (
                    self.speech_started <= end
                    and timestamp - end <= VOICE_STICKY_MAXIMUM_SECONDS
                )
                return voice if valid else None
        return None

    def _voice_face(self, timestamp: float) -> str | None:
        voice = self._current_voice(timestamp)
        return self.voices.face_for(voice) if voice else None

    def _voice_leader(self, voice: str) -> str | None:
        """The labelled face that clearly dominates this voice's mouth-motion votes."""
        tally = self.voice_turn_votes.get(voice, Counter())
        ranked = tally.most_common(2) + [(None, 0.0), (None, 0.0)]
        (lead, seconds), (_, runner_up) = ranked[0], ranked[1]
        if (
            lead in self.labels
            and sum(tally.values()) >= VOICE_TURN_MIN_SECONDS
            and seconds >= VOICE_TURN_LEAD * runner_up
        ):
            return lead
        return None

    def _voiced(self, raw: SpeakerEstimate) -> SpeakerEstimate:
        """Follow the current voice's leading face instead of frame-level mouth jitter."""
        voice = self._current_voice(raw.timestamp) if raw.state != "silent" else None
        face = self._voice_leader(voice) if voice else None
        if face is None:
            return raw
        fresh = raw.timestamp - self.latest_vision <= GEOMETRY_EXPIRY_SECONDS
        if fresh and any(item.track_id == face for item in self.current_faces):
            confidence = raw.confidence if raw.face_track == face else 0.0
            estimate = self._visible_estimate(
                raw.timestamp, face, max(confidence, VOICE_FALLBACK_CONFIDENCE)
            )
        else:
            estimate = SpeakerEstimate(
                raw.timestamp, f"Person {self.labels[face]}", None, "off_screen",
                VOICE_FALLBACK_CONFIDENCE,
            )
        raw_face = raw.face_track if raw.state == "visible" else None
        return replace(estimate, voiced=True, raw_face=raw_face)

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
        if not words:
            return []
        window = [
            item
            for item in self.speaker_history
            if words[0].start <= item.timestamp <= words[-1].end
            and item.state != "silent"
        ]
        # Duration of each estimate: gap to the next one, capped so pauses don't dominate.
        weights = [
            min(nxt.timestamp - item.timestamp, TURN_MAXIMUM_STEP_SECONDS)
            for item, nxt in zip(window, window[1:])
        ] + [TURN_MAXIMUM_STEP_SECONDS]
        votes: Counter[str] = Counter()
        latest: dict[str, SpeakerEstimate] = {}
        for item, weight in zip(window, weights):
            votes[item.speaker] += weight
            latest[item.speaker] = item
        total = sum(votes.values())
        # Listeners' mouths flicker, so a named person holding a fair share beats any
        # generic label even when the generic label has more raw votes.
        named = [s for s in votes if self._named(latest[s])]
        best = max(named, key=votes.__getitem__, default=None)
        if best is not None and votes[best] >= TURN_NAMED_SHARE * total:
            turn = latest[best]
        elif (
            self.last_event is not None
            and self._named(self.last_event)
            and words[0].start - self.last_event.end < TURN_CONTINUITY_SECONDS
        ):
            previous = self.last_event
            turn = SpeakerEstimate(
                words[0].start,
                previous.speaker,
                previous.face_track,
                previous.state,
                previous.confidence,
            )
        elif votes:
            turn = latest[votes.most_common(1)[0][0]]
        else:
            turn = self.speaker_at((words[0].start + words[-1].end) / 2)
        voiced = self._voice_turn(words, window, weights)
        if voiced is not None:
            turn = voiced
        # A different named person only takes words after a clear uninterrupted stretch.
        stretches = []
        run: list[tuple[SpeakerEstimate, float]] = []
        for item, weight in [*zip(window, weights), (None, 0.0)]:
            if run and (item is None or item.speaker != run[0][0].speaker):
                length = run[-1][0].timestamp - run[0][0].timestamp + run[-1][1]
                if voiced is None and self._named(run[0][0]) and length >= TURN_SPLIT_SECONDS:
                    stretches.append((run[0][0].timestamp, run[-1][0].timestamp, run[-1][0]))
                run = []
            if item is not None:
                run.append((item, weight))
        events = []
        for word in words:
            middle = (word.start + word.end) / 2
            estimate = next(
                (e for start, end, e in stretches if start <= middle <= end
                 and e.speaker != turn.speaker),
                turn,
            )
            events.append(
                CaptionEvent(
                    word.start,
                    word.end,
                    estimate.speaker,
                    estimate.face_track,
                    word.text,
                    word.confidence,
                    estimate.state,
                    word.language,
                )
            )
        merged = merge_word_events(events)
        self.last_event = merged[-1]
        return merged

    def _voice_turn(self, words, window, weights) -> SpeakerEstimate | None:
        start, end = words[0].start, words[-1].end
        overlaps: Counter[str] = Counter()
        for voice_start, voice_end, voice in self.voice_timeline:
            overlaps[voice] += max(0.0, min(end, voice_end) - max(start, voice_start))
        total = sum(overlaps.values())
        if total <= 0:
            return None
        # One person's voice may be split over several voice ids, so each overlapping
        # voice learns this utterance's face votes in proportion to its overlap and then
        # votes for its own leading face.
        faces: Counter[str] = Counter()
        for voice, covered in overlaps.items():
            tally = self.voice_turn_votes.setdefault(voice, Counter())
            for item, weight in zip(window, weights):
                # Mouth-motion evidence only, so a voiced override never votes for itself.
                face = item.raw_face if item.voiced else (
                    item.face_track if item.state == "visible" else None
                )
                if face:
                    tally[face] += weight * covered / total
            if lead := self._voice_leader(voice):
                faces[lead] += covered
        face, covered = faces.most_common(1)[0] if faces else (None, 0.0)
        if covered < VOICE_TURN_SINGLE_VOICE_SHARE * total:
            return None
        label = f"Person {self.labels[face]}"
        seen = any(f.track_id == face for f in self.current_faces) or (
            face in self.boxes and end - self.boxes[face].timestamp <= GEOMETRY_EXPIRY_SECONDS
        )
        return SpeakerEstimate(
            start, label, face if seen else None, "visible" if seen else "off_screen", VOICE_FALLBACK_CONFIDENCE
        )

    @staticmethod
    def _named(item: SpeakerEstimate | CaptionEvent) -> bool:
        return item.state in ("visible", "off_screen") and item.speaker not in (
            UNCERTAIN_SPEAKER,
            OFF_SCREEN_SPEAKER,
        )

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
        relative_face_size = max(
            MINIMUM_FACE_SIZE_WEIGHT,
            sqrt(face.box.width * face.box.height / maximum_face_area),
        )
        return self._evidence(face) * relative_face_size + bonus

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
        self._note_clear_speaker(raw)  # raw mouth motion only: binding evidence stays independent of voice
        estimate = replace(self._stabilized(self._voiced(raw)), revision=self.revision)
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
        if raw.state != "visible":
            return
        selected = self.speaking_evidence.get(raw.face_track, 0.0)
        others = [
            self._evidence(face)
            for face in self.current_faces
            if face.track_id != raw.face_track
        ]
        if selected >= SPEAKING_EVIDENCE_THRESHOLD and selected >= (
            CLEAR_SPEAKER_RATIO * max(others, default=0.0)
        ):
            self.clear_speakers.append((raw.timestamp, raw.face_track))

    def update_voice(self, embedding: np.ndarray, start: float, end: float) -> str | None:
        voice = self.voices.assign(embedding)
        if voice is None:
            voice = self.voices.identify(embedding)
            if voice is None:
                return None
            self.voice_timeline.append((start, end, voice))
        else:
            self.voice_timeline.append((start, end, voice))
            if not self._merge_into_absent(voice):
                self._bind_voice(voice, start, end)
        self._relabel_off_screen(voice, start, end)
        return voice

    def _relabel_off_screen(self, voice: str, start: float, end: float) -> None:
        """Name earlier generic off-screen history once the voice is known."""
        face = self.voices.face_for(voice)
        if face not in self.labels or any(
            item.track_id == face for item in self.current_faces
        ):
            return
        speaker = f"Person {self.labels[face]}"
        self.speaker_history = deque(
            (
                self._named_off_screen(item, speaker)
                if start <= item.timestamp <= end
                and item.state == "off_screen"
                and item.speaker == OFF_SCREEN_SPEAKER
                else item
                for item in self.speaker_history
            ),
            maxlen=MAXIMUM_HISTORY_ITEMS,
        )

    def _named_off_screen(self, item: SpeakerEstimate, speaker: str) -> SpeakerEstimate:
        self.revision += 1
        return replace(
            item,
            speaker=speaker,
            confidence=VOICE_FALLBACK_CONFIDENCE,
            revision=self.revision,
        )

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
        if new in self.mouth_baselines:
            self.mouth_baselines[old] = self.mouth_baselines.pop(new)
        for registry in (self.speaking_evidence, self.evidence_times):
            if new in registry:
                registry[old] = registry.pop(new)
        self.current_faces = [
            replace(face, track_id=old) if face.track_id == new else face
            for face in self.current_faces
        ]
        for tally in self.voice_turn_votes.values():
            tally[old] += tally.pop(new, 0.0)
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
        if item.raw_face == new:
            item = replace(item, raw_face=old)
        if item.face_track != new:
            return item
        return replace(item, speaker=f"Person {self.labels[old]}", face_track=old)
