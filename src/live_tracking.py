"""Current-frame face identities and conservative person-mask association."""

import math
import threading
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from types import MappingProxyType

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from .config import FACE_TRACK_IOU_THRESHOLD, MODEL_DIRECTORY
from .contracts import FaceBox, FaceTrack
from .person_segmentation import person_segmenter
from .vision import (
    FACE_DETECTION_CONFIDENCE,
    FACE_DETECTION_MODEL,
    detection_box,
    intersection_over_union,
)

HISTORY_SECONDS = 90
MAXIMUM_HISTORY_FRAMES = 400
IDENTITY_ABSENCE_SECONDS = 5
HISTORY_MATCH_SECONDS = 0.6
MINIMUM_FACE_MASK_COVERAGE = 0.60
FACE_MASK_ASSIGNMENT_MARGIN = 0.15
CONTOUR_SIMPLIFICATION_RATIO = 0.002
MAXIMUM_IMAGE_WIDTH = 1280
MAXIMUM_IMAGE_HEIGHT = 720
MAXIMUM_IMAGE_PIXELS = MAXIMUM_IMAGE_WIDTH * MAXIMUM_IMAGE_HEIGHT


class LiveTracker:
    """Keep visual tracking independent of delayed speech model work."""

    def __init__(self, model_directory: Path = MODEL_DIRECTORY) -> None:
        self.model_directory = model_directory
        self.lock = threading.Lock()
        self.detector = None
        self.segmenter = None
        self.previous_boxes: dict[str, FaceBox] = {}
        self.history: deque[tuple[float, dict[str, FaceBox]]] = deque(
            maxlen=MAXIMUM_HISTORY_FRAMES
        )
        self.next_face_number = 1
        self.latest_timestamp = -1.0

    def track(self, image_bytes: bytes, timestamp: float) -> dict[str, object]:
        """Return current pixel rectangles and unambiguous person contours."""
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("frame timestamp must be finite and nonnegative")
        if not image_bytes:
            raise ValueError("frame image is empty")
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                width, height = source.size
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("frame must be a decodable image") from error
        if width * height > MAXIMUM_IMAGE_PIXELS:
            raise ValueError("frame resolution is too large")
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("frame must be a decodable image")
        height, width = image.shape[:2]
        if width * height > MAXIMUM_IMAGE_PIXELS:
            raise ValueError("frame resolution is too large")
        with self.lock:
            if timestamp < self.latest_timestamp:
                raise ValueError("frame timestamp precedes the latest tracked frame")
            self._load_models()
            color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.detector.process(color_image)
            boxes = [
                detection_box(
                    detection.location_data.relative_bounding_box,
                    timestamp,
                    width,
                    height,
                )
                for detection in result.detections or []
            ]
            boxes = [box for box in boxes if box.width > 0 and box.height > 0]
            current_boxes = self._identify_faces(boxes, timestamp)
            contours = [[] for _ in boxes]
            if boxes:
                masks = self.segmenter.process(image)
                contours = _person_contours(masks, boxes)
            self.history.append((timestamp, dict(current_boxes)))
            while self.history and timestamp - self.history[0][0] > HISTORY_SECONDS:
                self.history.popleft()
            self.latest_timestamp = timestamp
            return {
                "timestamp": timestamp,
                "width": width,
                "height": height,
                "faces": [
                    {
                        "trackId": track_id,
                        "box": {
                            "x": box.x,
                            "y": box.y,
                            "width": box.width,
                            "height": box.height,
                        },
                        "contour": contour,
                    }
                    for (track_id, box), contour in zip(current_boxes.items(), contours)
                ],
            }

    def association_callback(
        self, capture_offset: float
    ) -> Callable[[list[FaceTrack]], None]:
        """Freeze capture history before slow recognition can outlive its retention."""
        if not math.isfinite(capture_offset) or capture_offset < 0:
            raise ValueError("capture offset must be finite and nonnegative")
        with self.lock:
            history = tuple(
                (timestamp, MappingProxyType(dict(boxes)))
                for timestamp, boxes in self.history
            )
        return lambda tracks: self.associate_tracks(
            tracks, capture_offset, history=history
        )

    def associate_tracks(
        self,
        tracks: list[FaceTrack],
        capture_offset: float,
        *,
        history: Sequence[tuple[float, Mapping[str, FaceBox]]] | None = None,
    ) -> None:
        """Map recording-relative tracks to the matching capture-time identities."""
        if not math.isfinite(capture_offset) or capture_offset < 0:
            raise ValueError("capture offset must be finite and nonnegative")
        with self.lock:
            capture_history = self.history if history is None else history
            available_ids = {
                track_id for _, boxes in capture_history for track_id in boxes
            }
            assigned_intervals: dict[str, list[tuple[float, float]]] = {}
            for track in tracks:
                scores = self._history_scores(track, capture_offset, capture_history)
                candidates = {
                    candidate
                    for candidate in available_ids
                    if not any(
                        track.start <= end and track.end >= start
                        for start, end in assigned_intervals.get(candidate, [])
                    )
                }
                track_id = max(
                    candidates,
                    key=lambda candidate: scores.get(candidate, 0.0),
                    default=None,
                )
                if (
                    track_id is None
                    or scores.get(track_id, 0.0) < FACE_TRACK_IOU_THRESHOLD
                ):
                    track.track_id = self._new_identity()
                    continue
                assigned_intervals.setdefault(track_id, []).append(
                    (track.start, track.end)
                )
                track.track_id = track_id

    def close(self) -> None:
        """Release native MediaPipe graphs when the capture session ends."""
        with self.lock:
            if self.detector is not None:
                self.detector.close()
            self.detector = None
            self.segmenter = None
            self.previous_boxes.clear()
            self.history.clear()

    def _load_models(self) -> None:
        if self.detector is None:
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=FACE_DETECTION_MODEL,
                min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            )
        if self.segmenter is None:
            self.segmenter = person_segmenter(self.model_directory)

    def _new_identity(self) -> str:
        track_id = f"live-face-{self.next_face_number}"
        self.next_face_number += 1
        return track_id

    def _identify_faces(
        self, boxes: list[FaceBox], timestamp: float
    ) -> dict[str, FaceBox]:
        # ponytail: overlap continuity; appearance embeddings are needed for crossings and lookalike replacements.
        self.previous_boxes = {
            track_id: box
            for track_id, box in self.previous_boxes.items()
            if timestamp - box.timestamp <= IDENTITY_ABSENCE_SECONDS
        }
        available_ids = set(self.previous_boxes)
        current_boxes = {}
        for box in boxes:
            track_id = max(
                available_ids,
                key=lambda candidate: intersection_over_union(
                    self.previous_boxes[candidate], box
                ),
                default=None,
            )
            matched = track_id is not None and (
                intersection_over_union(self.previous_boxes[track_id], box)
                >= FACE_TRACK_IOU_THRESHOLD
            )
            if not matched:
                track_id = self._new_identity()
            else:
                available_ids.remove(track_id)
            current_boxes[track_id] = box
        self.previous_boxes.update(current_boxes)
        return current_boxes

    def _history_scores(
        self,
        track: FaceTrack,
        capture_offset: float,
        history: Sequence[tuple[float, Mapping[str, FaceBox]]],
    ) -> dict[str, float]:
        if not track.boxes or not history:
            return {}
        sample_indices = sorted({0, len(track.boxes) // 2, len(track.boxes) - 1})
        scores: Counter[str] = Counter()
        for sample_index in sample_indices:
            sample = track.boxes[sample_index]
            timestamp = capture_offset + sample.timestamp
            frame_timestamp, boxes = min(
                history, key=lambda frame: abs(frame[0] - timestamp)
            )
            if abs(frame_timestamp - timestamp) > HISTORY_MATCH_SECONDS:
                continue
            for track_id, box in boxes.items():
                scores[track_id] += intersection_over_union(sample, box) / len(
                    sample_indices
                )
        return dict(scores)


def _person_contours(
    masks: list[np.ndarray], boxes: list[FaceBox]
) -> list[list[list[int]]]:
    if not masks:
        return [[] for _ in boxes]
    # ponytail: masks remain probabilistic under occlusion; uncertain face matches keep only their face outline.
    assignments = []
    for box in boxes:
        scores = []
        for mask in masks:
            face_mask = mask[box.y : box.y + box.height, box.x : box.x + box.width]
            center_inside = bool(mask[box.y + box.height // 2, box.x + box.width // 2])
            coverage = np.count_nonzero(face_mask) / face_mask.size
            scores.append(coverage if center_inside else 0.0)
        ranked = np.argsort(scores)[::-1]
        best = int(ranked[0])
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        confident = scores[best] >= MINIMUM_FACE_MASK_COVERAGE
        distinct = scores[best] - second_score >= FACE_MASK_ASSIGNMENT_MARGIN
        assignments.append(best if confident and distinct else None)
    assignment_counts = Counter(assignments)
    contours = []
    for assignment in assignments:
        if assignment is None or assignment_counts[assignment] != 1:
            contours.append([])
            continue
        found, _ = cv2.findContours(
            masks[assignment].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(found, key=cv2.contourArea)
        contour = cv2.approxPolyDP(
            contour, cv2.arcLength(contour, True) * CONTOUR_SIMPLIFICATION_RATIO, True
        )
        contours.append(contour.reshape(-1, 2).tolist())
    return contours
