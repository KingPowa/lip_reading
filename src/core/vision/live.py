"""Small current-frame face boxes and normalized mouth aperture measurements."""

import math
import threading
from io import BytesIO
from pathlib import Path

import cv2
import mediapipe
import numpy as np
from PIL import Image

from src.config.live import (
    FACE_CROP_PADDING_RATIO,
    LEFT_EYE_KEYPOINT,
    LIVE_FACE_DETECTION_CONFIDENCE,
    MAXIMUM_FRAME_HEIGHT,
    MAXIMUM_FRAME_PIXELS,
    MAXIMUM_FRAME_WIDTH,
    MAXIMUM_VISIBLE_FACES,
    MINIMUM_LANDMARK_DISTANCE,
    MOUTH_LEFT_LANDMARK,
    MOUTH_LOWER_LANDMARK,
    MOUTH_RIGHT_LANDMARK,
    MOUTH_UPPER_LANDMARK,
    RIGHT_EYE_KEYPOINT,
    ROLL_CORRECTION_THRESHOLD_DEGREES,
)
from src.config.models import MODEL_DIRECTORY
from src.config.vision import FACE_DETECTION_MODEL
from src.core.contracts import FaceBox, FaceObservation
from src.core.vision.face_identity import face_identity_encoder
from src.core.vision.faces import detection_box


class LiveFaceDetector:
    """Reuse face detection, mouth landmarks, and a local identity encoder."""

    def __init__(self, model_directory: Path = MODEL_DIRECTORY) -> None:
        self.lock = threading.Lock()
        self.identity_encoder = face_identity_encoder(model_directory)
        self.detector = mediapipe.solutions.face_detection.FaceDetection(
            model_selection=FACE_DETECTION_MODEL,
            min_detection_confidence=LIVE_FACE_DETECTION_CONFIDENCE,
        )
        self.model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
        )

    def detect(
        self, content: bytes, timestamp: float
    ) -> tuple[int, int, list[FaceObservation]]:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("frame timestamp must be finite and nonnegative")
        try:
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("frame must be a decodable image") from error
        if (
            width > MAXIMUM_FRAME_WIDTH
            or height > MAXIMUM_FRAME_HEIGHT
            or width * height > MAXIMUM_FRAME_PIXELS
        ):
            raise ValueError("frame exceeds the live resolution budget")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("frame must be a decodable image")
        observations = []
        with self.lock:
            detections = (
                self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).detections
                or []
            )
            ranked = sorted(
                detections,
                key=lambda detection: (
                    detection.location_data.relative_bounding_box.width
                    * detection.location_data.relative_bounding_box.height
                ),
                reverse=True,
            )
            for detection in ranked[:MAXIMUM_VISIBLE_FACES]:
                box = detection_box(
                    detection.location_data.relative_bounding_box,
                    timestamp,
                    width,
                    height,
                )
                padding = round(max(box.width, box.height) * FACE_CROP_PADDING_RATIO)
                left, top = max(0, box.x - padding), max(0, box.y - padding)
                right, bottom = (
                    min(width, box.x + box.width + padding),
                    min(height, box.y + box.height + padding),
                )
                crop = image[top:bottom, left:right]
                if not crop.size:
                    continue
                crop = upright_crop(crop, detection, width, height)
                result = self.model.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                aperture = None
                features = None
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    _, aperture = face_observation(
                        landmarks,
                        timestamp,
                        right - left,
                        bottom - top,
                    )
                    features = self.identity_encoder.encode(crop, landmarks)
                observations.append(FaceObservation(box, aperture, features))
        return width, height, observations

    def close(self) -> None:
        with self.lock:
            self.model.close()
            self.detector.close()


def face_roll_degrees(detection, width: int, height: int) -> float:
    keypoints = getattr(detection.location_data, "relative_keypoints", ())
    if len(keypoints) <= max(RIGHT_EYE_KEYPOINT, LEFT_EYE_KEYPOINT):
        return 0.0
    right_eye, left_eye = keypoints[RIGHT_EYE_KEYPOINT], keypoints[LEFT_EYE_KEYPOINT]
    return math.degrees(
        math.atan2((left_eye.y - right_eye.y) * height, (left_eye.x - right_eye.x) * width)
    )


def upright_crop(crop: np.ndarray, detection, width: int, height: int) -> np.ndarray:
    roll = face_roll_degrees(detection, width, height)
    if not math.isfinite(roll) or abs(roll) < ROLL_CORRECTION_THRESHOLD_DEGREES:
        return crop
    crop_height, crop_width = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), roll, 1.0)
    return cv2.warpAffine(
        crop, matrix, (crop_width, crop_height), borderMode=cv2.BORDER_REPLICATE
    )


def face_observation(
    landmarks: list, timestamp: float, width: int, height: int
) -> tuple[FaceBox, float]:
    coordinates = np.array([(point.x * width, point.y * height) for point in landmarks])
    left, top = np.maximum(coordinates.min(axis=0), 0)
    right, bottom = np.minimum(coordinates.max(axis=0), (width, height))
    mouth_width = max(
        MINIMUM_LANDMARK_DISTANCE,
        np.linalg.norm(
            coordinates[MOUTH_LEFT_LANDMARK] - coordinates[MOUTH_RIGHT_LANDMARK]
        ),
    )
    aperture = (
        np.linalg.norm(
            coordinates[MOUTH_UPPER_LANDMARK] - coordinates[MOUTH_LOWER_LANDMARK]
        )
        / mouth_width
    )
    box = FaceBox(
        timestamp,
        round(left),
        round(top),
        max(1, round(right - left)),
        max(1, round(bottom - top)),
    )
    return box, float(aperture)
