"""Face tracking and visual active-speaker evidence."""

import logging
import urllib.request
from bisect import bisect_left
from pathlib import Path

import cv2
import mediapipe as mp

from .config import (
    FACE_LANDMARKER_URL,
    FACE_TRACK_IOU_THRESHOLD,
    MINIMUM_TRACK_DETECTIONS,
    MINIMUM_TRACK_DURATION_SECONDS,
    MISSING_FACE_LIMIT,
)
from .contracts import FaceBox, FaceTrack
from .media import VideoFrame


LOGGER = logging.getLogger(__name__)
DEFAULT_FRAME_DURATION = 0.04
CASCADE_SCALE_FACTOR = 1.05
CASCADE_MINIMUM_NEIGHBORS = 2
CASCADE_MINIMUM_SIZE = (30, 30)


def analyze_faces(frames: list[VideoFrame], model_path: Path) -> list[FaceTrack]:
    """Track MediaPipe faces on the input presentation timeline.

    Args:
        frames: Decoded frames with original presentation timestamps.
        model_path: Local MediaPipe Face Landmarker task model path.

    Returns:
        Temporally tracked visible faces.

    Raises:
        RuntimeError: If the face model cannot be obtained or initialized.
    """
    if not frames:
        return []
    try:
        _download_model_if_missing(model_path)
        detections = _landmarker_detections(frames, model_path)
    except RuntimeError as error:
        LOGGER.warning("MediaPipe face landmarker is unavailable; using OpenCV cascades: %s", error)
    else:
        tracks = _track_detections(detections)
        if tracks:
            return tracks
        LOGGER.warning("MediaPipe face landmarker found no stable tracks; using OpenCV cascades")
    detections = [_cascade_detections(frame) for frame in frames]
    return _track_detections(detections)


def _landmarker_detections(frames: list[VideoFrame], model_path: Path) -> list[list[FaceBox]]:
    landmarker = _create_landmarker(model_path)
    with landmarker:
        return [
            _face_detections(
                landmarker.detect_for_video(_media_image(frame.image), _timestamp_ms(frame.timestamp, frame_index)),
                frame,
            )
            for frame_index, frame in enumerate(frames)
        ]


def _cascade_detections(frame: VideoFrame) -> list[FaceBox]:
    grayscale = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    cascade_paths = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + "haarcascade_profileface.xml",
    )
    boxes = [
        FaceBox(frame.timestamp, int(x), int(y), int(width), int(height))
        for cascade_path in cascade_paths
        for x, y, width, height in cv2.CascadeClassifier(cascade_path).detectMultiScale(
            grayscale,
            scaleFactor=CASCADE_SCALE_FACTOR,
            minNeighbors=CASCADE_MINIMUM_NEIGHBORS,
            minSize=CASCADE_MINIMUM_SIZE,
        )
    ]
    selected: list[FaceBox] = []
    for box in sorted(boxes, key=lambda candidate: candidate.width * candidate.height, reverse=True):
        if all(intersection_over_union(box, existing) < FACE_TRACK_IOU_THRESHOLD for existing in selected):
            selected.append(box)
    return selected


def _track_detections(detections_by_frame: list[list[FaceBox]], missing_face_limit: int = MISSING_FACE_LIMIT) -> list[FaceTrack]:
    tracks: list[FaceTrack] = []
    active_tracks: dict[str, tuple[FaceTrack, int]] = {}
    previous_boxes: dict[str, FaceBox] = {}
    next_track_number = 1
    for detections in detections_by_frame:
        unmatched_ids = set(active_tracks)
        for box in detections:
            track_id = _matching_track_id(box, unmatched_ids, previous_boxes)
            if track_id is None:
                track_id = f"face-{next_track_number}"
                next_track_number += 1
                active_tracks[track_id] = (FaceTrack(track_id, box.timestamp, box.timestamp, [box]), 0)
            else:
                track, _ = active_tracks[track_id]
                track.boxes.append(box)
                track.end = box.timestamp
                active_tracks[track_id] = (track, 0)
                unmatched_ids.remove(track_id)
            previous_boxes[track_id] = box
        _expire_missing_tracks(active_tracks, previous_boxes, unmatched_ids, tracks, missing_face_limit)
    tracks.extend(track for track, _ in active_tracks.values())
    tracks = [track for track in tracks if _is_stable_track(track)]
    LOGGER.info("tracked %d face(s)", len(tracks))
    return tracks


def _is_stable_track(track: FaceTrack) -> bool:
    return len(track.boxes) >= MINIMUM_TRACK_DETECTIONS and track.end - track.start >= MINIMUM_TRACK_DURATION_SECONDS


def intersection_over_union(first: FaceBox, second: FaceBox) -> float:
    """Calculate the overlap ratio between two face rectangles.

    Args:
        first: First face rectangle.
        second: Second face rectangle.

    Returns:
        Intersection-over-union in the inclusive range zero through one.
    """
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def box_at(track: FaceTrack, timestamp: float) -> FaceBox | None:
    """Interpolate a face rectangle at a frame presentation timestamp.

    Args:
        track: Face track containing timestamped rectangles.
        timestamp: Requested presentation timestamp.

    Returns:
        An interpolated rectangle, or ``None`` when the track is not visible.
    """
    if not track.boxes or timestamp < track.start or timestamp > track.end + DEFAULT_FRAME_DURATION:
        return None
    index = bisect_left(track.boxes, timestamp, key=lambda box: box.timestamp)
    before = track.boxes[max(0, index - 1)]
    after = track.boxes[min(index, len(track.boxes) - 1)]
    if after.timestamp == before.timestamp:
        return before
    fraction = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
    return FaceBox(
        timestamp,
        round(before.x + (after.x - before.x) * fraction),
        round(before.y + (after.y - before.y) * fraction),
        round(before.width + (after.width - before.width) * fraction),
        round(before.height + (after.height - before.height) * fraction),
    )


def _download_model_if_missing(model_path: Path) -> None:
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading MediaPipe Face Landmarker model")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    except OSError as error:
        raise RuntimeError(f"cannot download Face Landmarker model: {error}") from error


def _create_landmarker(model_path: Path):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=10,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _media_image(image: object) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _timestamp_ms(timestamp: float, frame_index: int) -> int:
    return max(frame_index, round(timestamp * 1000))


def _face_detections(result: object, frame: VideoFrame) -> list[FaceBox]:
    height, width = frame.image.shape[:2]
    return [
        _landmarks_box(landmarks, frame.timestamp, width, height)
        for landmarks in result.face_landmarks
    ]


def _landmarks_box(landmarks: list[object], timestamp: float, width: int, height: int) -> FaceBox:
    coordinates_x = [landmark.x for landmark in landmarks]
    coordinates_y = [landmark.y for landmark in landmarks]
    left = max(0, round(min(coordinates_x) * width))
    top = max(0, round(min(coordinates_y) * height))
    right = min(width, round(max(coordinates_x) * width))
    bottom = min(height, round(max(coordinates_y) * height))
    return FaceBox(timestamp, left, top, max(1, right - left), max(1, bottom - top))


def _matching_track_id(box: FaceBox, unmatched_ids: set[str], previous_boxes: dict[str, FaceBox]) -> str | None:
    track_id = max(
        unmatched_ids,
        key=lambda candidate: intersection_over_union(previous_boxes[candidate], box),
        default=None,
    )
    if track_id is None or intersection_over_union(previous_boxes[track_id], box) < FACE_TRACK_IOU_THRESHOLD:
        return None
    return track_id


def _expire_missing_tracks(
    active_tracks: dict[str, tuple[FaceTrack, int]],
    previous_boxes: dict[str, FaceBox],
    unmatched_ids: set[str],
    tracks: list[FaceTrack],
    missing_face_limit: int,
) -> None:
    for track_id in unmatched_ids:
        track, missing_frames = active_tracks[track_id]
        if missing_frames + 1 > missing_face_limit:
            tracks.append(track)
            del active_tracks[track_id]
            del previous_boxes[track_id]
        else:
            active_tracks[track_id] = (track, missing_frames + 1)
