"""Session-local face features and one-to-one identity association."""

from collections import OrderedDict, deque

import numpy as np

from src.config.tracking import (
    FACE_IDENTITY_ASSIGNMENT_MARGIN,
    FACE_IDENTITY_CONTINUITY_FLOOR,
    FACE_IDENTITY_SIMILARITY_THRESHOLD,
    MAXIMUM_IDENTITY_FEATURES,
    MAXIMUM_SESSION_IDENTITIES,
    MINIMUM_FEATURE_NORM,
)
from src.config.vision import FACE_TRACK_IOU_THRESHOLD
from src.core.contracts import FaceBox, FaceObservation
from src.core.vision.faces import intersection_over_union


class FaceIdentityGallery:
    def __init__(self) -> None:
        self.features: OrderedDict[str, deque[np.ndarray]] = OrderedDict()
        self.next_identity = 1

    def identify(
        self, observations: list[FaceObservation], boxes: dict[str, FaceBox]
    ) -> list[str]:
        features = [normalized_features(item.features) for item in observations]
        matches: dict[int, str] = {}
        available = set(self.features) | set(boxes)
        proposals = []
        for index, descriptor in enumerate(features):
            ranked = self._ranked_matches(descriptor)
            if not ranked:
                continue
            score, identity = ranked[0]
            runner_up_score = ranked[1][0] if len(ranked) > 1 else 0.0
            distinct = score - runner_up_score >= FACE_IDENTITY_ASSIGNMENT_MARGIN
            if score >= FACE_IDENTITY_SIMILARITY_THRESHOLD and distinct:
                proposals.append((score, index, identity))

        # Reserve feature matches before a face without landmarks can claim its box.
        for _, index, identity in sorted(proposals, reverse=True):
            if identity not in available:
                continue
            matches[index] = identity
            available.remove(identity)

        identities = []
        for index, observation in enumerate(observations):
            identity = matches.get(index)
            if identity is None:
                identity = self._geometry_match(
                    observation, features[index], boxes, available, observations
                )
            if identity is None:
                identity = f"face-{self.next_identity}"
                self.next_identity += 1
            available.discard(identity)
            identities.append(identity)

        for identity, descriptor in zip(identities, features):
            self._remember(identity, descriptor)
        self._limit_history(set(identities))
        return identities

    def _ranked_matches(self, descriptor: np.ndarray | None) -> list[tuple[float, str]]:
        if descriptor is None:
            return []
        scores = []
        for identity, examples in self.features.items():
            compatible = [item for item in examples if item.shape == descriptor.shape]
            if not compatible:
                continue
            score = max(float(np.dot(descriptor, item)) for item in compatible)
            scores.append((score, identity))
        return sorted(scores, reverse=True)

    def _geometry_match(
        self,
        observation: FaceObservation,
        descriptor: np.ndarray | None,
        boxes: dict[str, FaceBox],
        available: set[str],
        observations: list[FaceObservation],
    ) -> str | None:
        candidates = [
            identity
            for identity in available
            if identity in boxes
            and self._continuous(identity, descriptor, boxes[identity], observations)
        ]
        identity = max(
            candidates,
            key=lambda candidate: intersection_over_union(
                boxes[candidate], observation.box
            ),
            default=None,
        )
        if identity is None:
            return None
        overlap = intersection_over_union(boxes[identity], observation.box)
        return identity if overlap >= FACE_TRACK_IOU_THRESHOLD else None

    def _continuous(
        self,
        identity: str,
        descriptor: np.ndarray | None,
        previous: FaceBox,
        observations: list[FaceObservation],
    ) -> bool:
        if descriptor is None or identity not in self.features:
            return True
        scores = [
            float(np.dot(descriptor, item))
            for item in self.features[identity]
            if item.shape == descriptor.shape
        ]
        if not scores or max(scores) < FACE_IDENTITY_CONTINUITY_FLOOR:
            return False
        contenders = sum(
            intersection_over_union(previous, item.box) >= FACE_TRACK_IOU_THRESHOLD
            for item in observations
        )
        return contenders == 1

    def _remember(self, identity: str, descriptor: np.ndarray | None) -> None:
        if descriptor is None:
            return
        examples = self.features.setdefault(
            identity, deque(maxlen=MAXIMUM_IDENTITY_FEATURES)
        )
        examples.append(descriptor)
        self.features.move_to_end(identity)

    def _limit_history(self, visible: set[str]) -> None:
        dormant = [identity for identity in self.features if identity not in visible]
        for identity in dormant:
            if len(self.features) <= MAXIMUM_SESSION_IDENTITIES:
                return
            del self.features[identity]


def normalized_features(features: np.ndarray | None) -> np.ndarray | None:
    if features is None:
        return None
    descriptor = np.asarray(features, dtype=np.float32).reshape(-1)
    if not descriptor.size or not np.isfinite(descriptor).all():
        return None
    magnitude = float(np.linalg.norm(descriptor))
    if not np.isfinite(magnitude) or magnitude < MINIMUM_FEATURE_NORM:
        return None
    return descriptor / magnitude
