"""Decode separate person masks with the installed OpenCV ONNX runtime."""

import hashlib
import tempfile
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_DIRECTORY

PERSON_MODEL_FILENAME = "yolo11n-seg.onnx"
PERSON_MODEL_URL = (
    "https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/"
    "0d17b24163fbc33dec51d811d7be8db15b8df274/yolo11n-seg.onnx"
)
PERSON_MODEL_SHA256 = "6481a5fea55b9973c25752870587ec9163c1e6f778884e6c8befd4eff586080f"
MODEL_IMAGE_SIZE = 640
MAXIMUM_PIXEL_VALUE = 255
LETTERBOX_COLOR = (114, 114, 114)
CLASS_COUNT = 80
BOX_COORDINATE_COUNT = 4
PERSON_CLASS = 0
PERSON_CONFIDENCE_THRESHOLD = 0.35
PERSON_NMS_THRESHOLD = 0.5
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 120
MODEL_LOADING_LOCK = threading.Lock()
PERSON_MODELS = {}


class PersonSegmenter:
    """Reuse one native network while serializing its mutable inference inputs."""

    def __init__(self, model_directory: Path) -> None:
        model_path = model_directory / PERSON_MODEL_FILENAME
        _download_model_if_missing(model_path)
        with model_path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != PERSON_MODEL_SHA256:
            raise RuntimeError("person segmentation model checksum does not match")
        try:
            self.network = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error as error:
            raise RuntimeError(
                f"cannot load person segmentation model: {error}"
            ) from error
        self.lock = threading.Lock()

    def process(self, image: np.ndarray) -> list[np.ndarray]:
        padded, bounds = letterbox(image)
        blob = cv2.dnn.blobFromImage(
            padded, scalefactor=1 / MAXIMUM_PIXEL_VALUE, swapRB=True
        )
        with self.lock:
            self.network.setInput(blob)
            outputs = self.network.forward(self.network.getUnconnectedOutLayersNames())
        return decode_person_masks(outputs, image.shape[:2], bounds)


def person_segmenter(model_directory: Path = MODEL_DIRECTORY) -> PersonSegmenter:
    model_directory = model_directory.resolve()
    with MODEL_LOADING_LOCK:
        if model_directory not in PERSON_MODELS:
            PERSON_MODELS[model_directory] = PersonSegmenter(model_directory)
        return PERSON_MODELS[model_directory]


def warm_up_person_model(model_directory: Path = MODEL_DIRECTORY) -> None:
    """Load weights before the first current-frame request."""
    person_segmenter(model_directory)


def letterbox(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = MODEL_IMAGE_SIZE / max(width, height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left = (MODEL_IMAGE_SIZE - resized_width) // 2
    top = (MODEL_IMAGE_SIZE - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    padded = np.full(
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3), LETTERBOX_COLOR, dtype=np.uint8
    )
    padded[top : top + resized_height, left : left + resized_width] = resized
    return padded, (left, top, resized_width, resized_height)


def decode_person_masks(
    outputs: list[np.ndarray],
    image_shape: tuple[int, int],
    image_bounds: tuple[int, int, int, int],
) -> list[np.ndarray]:
    detections = next((output for output in outputs if output.ndim == 3), None)
    prototypes = next((output for output in outputs if output.ndim == 4), None)
    if detections is None or prototypes is None:
        raise RuntimeError("person model did not return boxes and mask prototypes")
    predictions = detections[0].T
    prototypes = prototypes[0]
    coefficient_offset = BOX_COORDINATE_COUNT + CLASS_COUNT
    if predictions.shape[1] != coefficient_offset + prototypes.shape[0]:
        raise RuntimeError("person model output dimensions are unsupported")
    class_scores = predictions[:, BOX_COORDINATE_COUNT:coefficient_offset]
    selected = (class_scores.argmax(axis=1) == PERSON_CLASS) & (
        class_scores[:, PERSON_CLASS] >= PERSON_CONFIDENCE_THRESHOLD
    )
    predictions = predictions[selected]
    if not len(predictions):
        return []
    rectangles = predictions[:, :BOX_COORDINATE_COUNT].copy()
    rectangles[:, :2] -= rectangles[:, 2:] / 2
    keep = cv2.dnn.NMSBoxes(
        rectangles.tolist(),
        predictions[:, BOX_COORDINATE_COUNT + PERSON_CLASS].tolist(),
        PERSON_CONFIDENCE_THRESHOLD,
        PERSON_NMS_THRESHOLD,
    )
    selected_indices = np.asarray(keep).reshape(-1)
    if not len(selected_indices):
        return []
    mask_logits = predictions[
        selected_indices, coefficient_offset:
    ] @ prototypes.reshape(prototypes.shape[0], -1)
    mask_logits = mask_logits.reshape(-1, *prototypes.shape[1:])
    height, width = image_shape
    left, top, resized_width, resized_height = image_bounds
    masks = []
    for index, logits in zip(selected_indices, mask_logits):
        logits = cv2.resize(
            logits, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )
        x, y, box_width, box_height = rectangles[index]
        box_left, box_top = max(0, round(x)), max(0, round(y))
        box_right = min(MODEL_IMAGE_SIZE, round(x + box_width))
        box_bottom = min(MODEL_IMAGE_SIZE, round(y + box_height))
        foreground = logits > 0
        foreground[:box_top] = False
        foreground[box_bottom:] = False
        foreground[:, :box_left] = False
        foreground[:, box_right:] = False
        foreground = foreground[top : top + resized_height, left : left + resized_width]
        mask = cv2.resize(
            foreground.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        if mask.any():
            masks.append(mask)
    return masks


def _download_model_if_missing(model_path: Path) -> None:
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urllib.request.urlopen(
            PERSON_MODEL_URL, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            with tempfile.NamedTemporaryFile(
                dir=model_path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                checksum = hashlib.sha256()
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    checksum.update(chunk)
                    temporary.write(chunk)
        if checksum.hexdigest() != PERSON_MODEL_SHA256:
            raise RuntimeError(
                "downloaded person segmentation model checksum does not match"
            )
        temporary_path.replace(model_path)
    except OSError as error:
        raise RuntimeError(
            f"cannot download person segmentation model: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
