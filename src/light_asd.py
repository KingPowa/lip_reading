"""CPU inference wrapper for the published Light-ASD architecture."""

import logging
import urllib.request
import wave
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from python_speech_features import mfcc
from torch import nn

from .contracts import FaceActivity, FaceTrack
from .config import MAXIMUM_ACTIVITY_TRACKS, LIGHT_ASD_URL
from .media import VideoFrame
from .vision import box_at


LOGGER = logging.getLogger(__name__)
TARGET_FRAME_RATE = 25
FRAME_DURATION = 1 / TARGET_FRAME_RATE
FACE_PADDING_RATIO = 0.40
FACE_CROP_SIZE = 224
MODEL_FACE_SIZE = 112
MFCCS_PER_FRAME = 4
MAXIMUM_FRAMES_PER_BATCH = 500


class _AudioBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(input_channels, output_channels, (3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(output_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(output_channels, output_channels, (1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(output_channels, momentum=0.01, eps=0.001)
        self.m_5 = nn.Conv2d(input_channels, output_channels, (5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(output_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(output_channels, output_channels, (1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(output_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv2d(output_channels, output_channels, 1, bias=False)
        self.bn_last = nn.BatchNorm2d(output_channels, momentum=0.01, eps=0.001)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(tensor))))))
        output_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(tensor))))))
        return self.relu(self.bn_last(self.last(output_3 + output_5)))


class _VisualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, downsample: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        stride = (1, 2, 2) if downsample else (1, 1, 1)
        self.s_3 = nn.Conv3d(input_channels, output_channels, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn_s_3 = nn.BatchNorm3d(output_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv3d(output_channels, output_channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = nn.BatchNorm3d(output_channels, momentum=0.01, eps=0.001)
        self.s_5 = nn.Conv3d(input_channels, output_channels, (1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
        self.bn_s_5 = nn.BatchNorm3d(output_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv3d(output_channels, output_channels, (5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = nn.BatchNorm3d(output_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv3d(output_channels, output_channels, 1, bias=False)
        self.bn_last = nn.BatchNorm3d(output_channels, momentum=0.01, eps=0.001)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output_3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(tensor))))))
        output_5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(tensor))))))
        return self.relu(self.bn_last(self.last(output_3 + output_5)))


class _VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _VisualBlock(1, 32, downsample=True)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.block3(self.pool2(self.block2(self.pool1(self.block1(tensor)))))
        batches, frames, channels, width, height = tensor.transpose(1, 2).shape
        tensor = tensor.transpose(1, 2).reshape(batches * frames, channels, width, height)
        return self.maxpool(tensor).view(batches, frames, channels)


class _AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 2), padding=(0, 1))
        self.block3 = _AudioBlock(64, 128)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.block3(self.pool2(self.block2(self.pool1(self.block1(tensor)))))
        return torch.mean(tensor, dim=2, keepdim=True).squeeze(2).transpose(1, 2)


class _BGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru_forward = nn.GRU(128, 128, batch_first=True)
        self.gru_backward = nn.GRU(128, 128, batch_first=True)
        self.gelu = nn.GELU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.gelu(self.gru_forward(tensor)[0])
        return self.gelu(torch.flip(self.gru_backward(torch.flip(tensor, dims=[1]))[0], dims=[1]))


class _LightASD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.GRU = _BGRU()
        self.classifier = nn.Linear(128, 2)

    def forward(self, audio_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        normalized_visual = (
            visual_features.view(visual_features.shape[0], 1, *visual_features.shape[1:]) / 255 - 0.4161
        ) / 0.1688
        visual = self.visualEncoder(normalized_visual)
        audio = self.audioEncoder(audio_features.unsqueeze(1).transpose(2, 3))
        return self.classifier(self.GRU(audio + visual).reshape(-1, 128))


def score_face_activity(
    audio_path: Path,
    frames: list[VideoFrame],
    tracks: list[FaceTrack],
    model_path: Path,
    maximum_tracks: int = MAXIMUM_ACTIVITY_TRACKS,
) -> list[FaceActivity]:
    """Score every visible face with the pretrained audio-visual Light-ASD model.

    Args:
        audio_path: Normalized 16 kHz mono WAV from the source video.
        frames: Decoded video frames on their presentation timestamps.
        tracks: MediaPipe face tracks to evaluate.
        model_path: Local Light-ASD TalkSet weight file.

    Returns:
        Timestamped speaking probabilities for each face track.

    Raises:
        RuntimeError: If the supplied Light-ASD weights are unavailable or invalid.
    """
    if not tracks:
        return []
    model = _load_model(model_path)
    audio_features = _audio_features(audio_path)
    candidates = sorted(tracks, key=_track_priority, reverse=True)[:maximum_tracks]
    activities = [activity for track in candidates for activity in _score_track(model, audio_features, frames, track)]
    LOGGER.info("Light-ASD produced %d face activity samples", len(activities))
    return activities


def _track_priority(track: FaceTrack) -> int:
    return sum(box.width * box.height for box in track.boxes)


@lru_cache
def _load_model(model_path: Path) -> _LightASD:
    _download_weights_if_missing(model_path)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    converted = {
        key.removeprefix("model.").replace("lossAV.FC", "classifier"): value
        for key, value in state.items()
        if key.startswith("model.") or key.startswith("lossAV.FC")
    }
    model = _LightASD()
    model.load_state_dict(converted, strict=True)
    return model.eval()


def _download_weights_if_missing(model_path: Path) -> None:
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading Light-ASD TalkSet weights")
    try:
        urllib.request.urlretrieve(LIGHT_ASD_URL, model_path)
    except OSError as error:
        raise RuntimeError(f"cannot download Light-ASD weights: {error}") from error


def _audio_features(audio_path: Path) -> np.ndarray:
    with wave.open(str(audio_path)) as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
        if audio.getframerate() != 16000 or audio.getnchannels() != 1:
            raise RuntimeError("Light-ASD requires normalized 16 kHz mono audio")
    return mfcc(samples, 16000, numcep=13, winlen=0.025, winstep=0.010)


def _score_track(
    model: _LightASD,
    audio_features: np.ndarray,
    frames: list[VideoFrame],
    track: FaceTrack,
) -> list[FaceActivity]:
    timestamps = np.arange(track.start, track.end + FRAME_DURATION / 2, FRAME_DURATION)
    face_frames = np.stack([_face_at_timestamp(frames, track, timestamp) for timestamp in timestamps])
    scores: list[float] = []
    for start in range(0, len(face_frames), MAXIMUM_FRAMES_PER_BATCH):
        end = min(len(face_frames), start + MAXIMUM_FRAMES_PER_BATCH)
        visual = torch.from_numpy(face_frames[start:end]).unsqueeze(0).float()
        window = _mfcc_window(audio_features, track.start + start * FRAME_DURATION, end - start)
        audio = torch.from_numpy(window).unsqueeze(0).float()
        with torch.inference_mode():
            scores.extend(torch.softmax(model(audio, visual), dim=1)[:, 1].tolist())
    return [
        FaceActivity(track.track_id, float(timestamp), float(timestamp + FRAME_DURATION), score)
        for timestamp, score in zip(timestamps, scores, strict=True)
    ]


def _frame_at(frames: list[VideoFrame], timestamp: float) -> VideoFrame:
    index = bisect_left(frames, timestamp, key=lambda frame: frame.timestamp)
    if index == 0:
        return frames[0]
    if index == len(frames):
        return frames[-1]
    before, after = frames[index - 1], frames[index]
    return before if timestamp - before.timestamp <= after.timestamp - timestamp else after


def _face_at_timestamp(frames: list[VideoFrame], track: FaceTrack, timestamp: float) -> np.ndarray:
    box = box_at(track, timestamp)
    if box is None:
        raise RuntimeError(f"face track {track.track_id} is missing its own timestamped box")
    return _face_frame(_frame_at(frames, timestamp).image, box)


def _face_frame(image: object, box: object) -> np.ndarray:
    height, width = image.shape[:2]
    size = round(max(box.width, box.height) * (1 + 2 * FACE_PADDING_RATIO))
    center_x = box.x + box.width // 2
    center_y = box.y + box.height // 2
    left, top = center_x - size // 2, center_y - size // 2
    right, bottom = left + size, top + size
    padded = cv2.copyMakeBorder(image, size, size, size, size, cv2.BORDER_CONSTANT, value=(110, 110, 110))
    crop = padded[top + size : bottom + size, left + size : right + size]
    face = cv2.cvtColor(cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE)), cv2.COLOR_BGR2GRAY)
    offset = (FACE_CROP_SIZE - MODEL_FACE_SIZE) // 2
    return face[offset : offset + MODEL_FACE_SIZE, offset : offset + MODEL_FACE_SIZE]


def _mfcc_window(features: np.ndarray, start_seconds: float, frame_count: int) -> np.ndarray:
    start = round(start_seconds * 100)
    length = frame_count * MFCCS_PER_FRAME
    window = features[start : start + length]
    return np.pad(window, ((0, max(0, length - len(window))), (0, 0)))
