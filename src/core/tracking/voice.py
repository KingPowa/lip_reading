"""Session-local voice centroids, face bindings, and live speech windows."""

from collections import Counter, deque

import numpy as np

from src.config.live import (
    AUDIO_SAMPLE_RATE,
    LIVE_VOICE_WINDOW_SECONDS,
    MAXIMUM_VOICE_BUFFER_SECONDS,
    VOICE_GAP_RESET_SECONDS,
)
from src.config.tracking import (
    MAXIMUM_VOICE_WEIGHT_COUNT,
    MAXIMUM_VOICES,
    MINIMUM_FEATURE_NORM,
    VOICE_BIND_DOMINANCE_RATIO,
    VOICE_BIND_MIN_CONFIRMATIONS,
    VOICE_IDENTIFY_MARGIN,
    VOICE_IDENTIFY_THRESHOLD,
    VOICE_MATCH_MARGIN,
    VOICE_MATCH_THRESHOLD,
    VOICE_NEW_THRESHOLD,
)


class VoiceGallery:
    """Anonymous voices and how often each coincided with one clear visible face."""

    def __init__(self) -> None:
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.evidence: dict[str, Counter] = {}
        self.next_voice = 1

    def assign(self, embedding: np.ndarray) -> str | None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        magnitude = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or magnitude < MINIMUM_FEATURE_NORM:
            return None
        vector = vector / magnitude
        ranked = sorted(
            (
                (float(np.dot(vector, centroid)), voice)
                for voice, centroid in self.centroids.items()
                if centroid.shape == vector.shape
            ),
            reverse=True,
        )
        if ranked and ranked[0][0] >= VOICE_MATCH_THRESHOLD:
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if ranked[0][0] - runner_up < VOICE_MATCH_MARGIN:
                return None
            self._update(ranked[0][1], vector)
            return ranked[0][1]
        if ranked and ranked[0][0] >= VOICE_NEW_THRESHOLD:
            return None
        if len(self.centroids) >= MAXIMUM_VOICES:
            return None
        voice = f"voice-{self.next_voice}"
        self.next_voice += 1
        self.centroids[voice] = vector
        self.counts[voice] = 1
        self.evidence[voice] = Counter()
        return voice

    def identify(self, embedding: np.ndarray) -> str | None:
        """Read-only: the clearly best face-bound voice, never updating anything."""
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        magnitude = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or magnitude < MINIMUM_FEATURE_NORM:
            return None
        vector = vector / magnitude
        ranked = sorted(
            (
                (float(np.dot(vector, centroid)), voice)
                for voice, centroid in self.centroids.items()
                if centroid.shape == vector.shape
            ),
            reverse=True,
        )
        bound = [item for item in ranked if self.face_for(item[1]) is not None]
        if not bound or bound[0][0] < VOICE_IDENTIFY_THRESHOLD:
            return None
        best = bound[0]
        others = [score for score, voice in ranked if voice != best[1]]
        lead = best[0] - (others[0] if others else 0.0)
        return best[1] if lead >= VOICE_IDENTIFY_MARGIN else None

    def bind(self, voice: str, face: str) -> None:
        if voice in self.evidence:
            self.evidence[voice][face] += 1

    def face_for(self, voice: str) -> str | None:
        ranked = self.evidence.get(voice, Counter()).most_common(2)
        if not ranked or ranked[0][1] < VOICE_BIND_MIN_CONFIRMATIONS:
            return None
        rival = ranked[1][1] if len(ranked) > 1 else 0
        if ranked[0][1] < VOICE_BIND_DOMINANCE_RATIO * rival:
            return None
        return ranked[0][0]

    def voice_for(self, face: str) -> str | None:
        for voice in self.evidence:
            if self.face_for(voice) == face:
                return voice
        return None

    def bound_faces(self) -> set[str]:
        faces = (self.face_for(voice) for voice in self.evidence)
        return {face for face in faces if face is not None}

    def _update(self, voice: str, vector: np.ndarray) -> None:
        self.counts[voice] = min(self.counts[voice] + 1, MAXIMUM_VOICE_WEIGHT_COUNT)
        weight = 1.0 / self.counts[voice]
        mean = (1.0 - weight) * self.centroids[voice] + weight * vector
        self.centroids[voice] = mean / np.linalg.norm(mean)


class SpeechBuffer:
    """Collect only speech samples into bounded windows for one embedding job."""

    def __init__(self) -> None:
        self.chunks: deque[tuple[float, float, np.ndarray]] = deque()

    def add(self, samples: np.ndarray, timestamp: float, speaking: bool) -> None:
        if not speaking:
            return
        start = timestamp - len(samples) / AUDIO_SAMPLE_RATE
        if self.chunks and start - self.chunks[-1][1] > VOICE_GAP_RESET_SECONDS:
            self.chunks.clear()
        self.chunks.append((start, timestamp, np.asarray(samples, dtype=np.float32)))
        while self._seconds() > MAXIMUM_VOICE_BUFFER_SECONDS:
            self.chunks.popleft()

    def ready(self) -> bool:
        return self._seconds() >= LIVE_VOICE_WINDOW_SECONDS - 1e-6

    def take(self) -> tuple[np.ndarray, float, float]:
        chunks = list(self.chunks)
        self.chunks.clear()
        samples = np.concatenate([chunk for _, _, chunk in chunks])
        return samples, chunks[0][0], chunks[-1][1]

    def _seconds(self) -> float:
        return sum(len(chunk) for _, _, chunk in self.chunks) / AUDIO_SAMPLE_RATE
