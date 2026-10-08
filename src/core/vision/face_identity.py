"""Align face crops and extract local SFace identity embeddings."""

import hashlib
import tempfile
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.config.models import (
    FACE_IDENTITY_MODEL_FILENAME,
    FACE_IDENTITY_MODEL_SHA256,
    FACE_IDENTITY_MODEL_URL,
)
from src.config.vision import (
    DOWNLOAD_CHUNK_BYTES,
    FACE_IDENTITY_EYE_LANDMARKS,
    FACE_IDENTITY_FEATURE_COUNT,
    FACE_IDENTITY_MOUTH_LANDMARKS,
    FACE_IDENTITY_NOSE_LANDMARK,
    MINIMUM_IDENTITY_EYE_DISTANCE_PIXELS,
    MINIMUM_IDENTITY_FEATURE_NORM,
    MODEL_DOWNLOAD_TIMEOUT_SECONDS,
)

MODEL_LOADING_LOCK = threading.Lock()
FACE_IDENTITY_ENCODERS: dict[Path, "FaceIdentityEncoder"] = {}


class FaceIdentityEncoder:
    def __init__(self, model_directory: Path) -> None:
        model_path = model_directory / FACE_IDENTITY_MODEL_FILENAME
        self._prepare_model(model_path)
        try:
            self.network = cv2.FaceRecognizerSF.create(str(model_path), "")
        except cv2.error as error:
            raise RuntimeError(f"cannot load face identity model: {error}") from error
        self.lock = threading.Lock()

    def encode(self, image: np.ndarray, landmarks: list) -> np.ndarray | None:
        required_index = max(
            *FACE_IDENTITY_MOUTH_LANDMARKS,
            *(index for pair in FACE_IDENTITY_EYE_LANDMARKS for index in pair),
            FACE_IDENTITY_NOSE_LANDMARK,
        )
        if len(landmarks) <= required_index or not image.size:
            return None
        height, width = image.shape[:2]
        coordinates = np.array(
            [(point.x * width, point.y * height) for point in landmarks],
            dtype=np.float32,
        )
        eyes = [
            coordinates[list(pair)].mean(axis=0) for pair in FACE_IDENTITY_EYE_LANDMARKS
        ]
        eyes.sort(key=lambda point: point[0])
        corners = sorted(
            (coordinates[index] for index in FACE_IDENTITY_MOUTH_LANDMARKS),
            key=lambda point: point[0],
        )
        points = np.array([*eyes, coordinates[FACE_IDENTITY_NOSE_LANDMARK], *corners])
        if not np.isfinite(points).all():
            return None
        if np.linalg.norm(eyes[0] - eyes[1]) < MINIMUM_IDENTITY_EYE_DISTANCE_PIXELS:
            return None
        face = np.concatenate(
            (np.array([0, 0, width, height], dtype=np.float32), points.reshape(-1))
        )
        with self.lock:
            aligned = self.network.alignCrop(image, face)
            features = self.network.feature(aligned).reshape(-1).copy()
        if (
            features.size != FACE_IDENTITY_FEATURE_COUNT
            or not np.isfinite(features).all()
        ):
            return None
        magnitude = float(np.linalg.norm(features))
        if not np.isfinite(magnitude) or magnitude < MINIMUM_IDENTITY_FEATURE_NORM:
            return None
        return features / magnitude

    @staticmethod
    def _prepare_model(model_path: Path) -> None:
        if model_path.is_file():
            with model_path.open("rb") as source:
                checksum = hashlib.file_digest(source, "sha256").hexdigest()
            if checksum != FACE_IDENTITY_MODEL_SHA256:
                raise RuntimeError("face identity model checksum does not match")
            return
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with (
                urllib.request.urlopen(
                    FACE_IDENTITY_MODEL_URL, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                tempfile.NamedTemporaryFile(
                    dir=model_path.parent, delete=False
                ) as temporary,
            ):
                temporary_path = Path(temporary.name)
                checksum = hashlib.sha256()
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    checksum.update(chunk)
                    temporary.write(chunk)
            if checksum.hexdigest() != FACE_IDENTITY_MODEL_SHA256:
                raise RuntimeError(
                    "downloaded face identity model checksum does not match"
                )
            temporary_path.replace(model_path)
        except OSError as error:
            raise RuntimeError(
                f"cannot download face identity model: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def face_identity_encoder(model_directory: Path) -> FaceIdentityEncoder:
    model_directory = model_directory.resolve()
    with MODEL_LOADING_LOCK:
        if model_directory not in FACE_IDENTITY_ENCODERS:
            FACE_IDENTITY_ENCODERS[model_directory] = FaceIdentityEncoder(
                model_directory
            )
        return FACE_IDENTITY_ENCODERS[model_directory]
