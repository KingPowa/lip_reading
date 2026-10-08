"""Stream a video through the live detector and tracker and print coherence metrics.

Usage: python -m tests.replay_live samples/giuseppeconte.mp4 --duration 120 --rotate 25
Not a unittest module; it needs real MediaPipe models and ffmpeg.
"""

import argparse
import json
import subprocess
from collections import Counter

import cv2
import numpy as np

from src.config.live import (
    AUDIO_PACKET_SECONDS,
    AUDIO_SAMPLE_RATE,
    JPEG_QUALITY,
    MAXIMUM_FRAME_HEIGHT,
    MAXIMUM_FRAME_WIDTH,
    TRACKING_INTERVAL_MILLISECONDS,
)
from src.core.contracts import Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.realtime import RealtimeTracker
from src.core.vision.live import LiveFaceDetector

WORD_SPACING_SECONDS = 0.3
SECONDS_PER_MINUTE = 60.0
WORD_STEPS = round(WORD_SPACING_SECONDS / AUDIO_PACKET_SECONDS)


def read_audio(path: str, offset: float, duration: float) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-ss", str(offset), "-t", str(duration), "-i", path,
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-",
    ]  # fmt: skip
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def prepare_frame(image: np.ndarray, degrees: float) -> bytes:
    height, width = image.shape[:2]
    scale = min(MAXIMUM_FRAME_WIDTH / width, MAXIMUM_FRAME_HEIGHT / height)
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    if degrees:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
    quality = round(JPEG_QUALITY * 100)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def frames(path: str, offset: float, steps: int, degrees: float):
    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
    interval = TRACKING_INTERVAL_MILLISECONDS / 1000
    image = None
    try:
        for step in range(steps):
            target = offset + (step + 1) * interval
            while video.get(cv2.CAP_PROP_POS_MSEC) / 1000 < target:
                if not video.grab():
                    return
                success, decoded = video.retrieve()
                image = decoded if success else image
            yield step, None if image is None else prepare_frame(image, degrees)
    finally:
        video.release()


def replay(path: str, offset: float, duration: float, degrees: float) -> dict:
    audio = read_audio(path, offset, duration)
    packet = round(AUDIO_PACKET_SECONDS * AUDIO_SAMPLE_RATE)
    steps = len(audio) // packet
    detector = LiveFaceDetector()
    tracker = RealtimeTracker()
    records, words, events = [], [], []
    detected = missing_mesh = no_face = total = 0
    try:
        for step, content in frames(path, offset, steps, degrees):
            timestamp = round((step + 1) * AUDIO_PACKET_SECONDS, 6)
            tracker.update_audio(audio[step * packet : (step + 1) * packet], timestamp)
            _, _, observations = detector.detect(content, timestamp)
            total += 1
            no_face += not observations
            detected += len(observations)
            missing_mesh += sum(item.aperture is None for item in observations)
            estimate = tracker.update_faces(observations, timestamp)
            records.append((timestamp, estimate, tracker.identities.next_identity - 1))
            if estimate.state != "silent" and step % WORD_STEPS == 0:
                words.append(Word("w", timestamp - 0.2, timestamp, 1.0))
                events.extend(tracker.caption_events(words[-1:]))
    finally:
        detector.close()
    return metrics(records, events, total, detected, missing_mesh, no_face)


def metrics(records, events, total, detected, missing_mesh, no_face) -> dict:
    speech = [(time, item) for time, item, _ in records if item.state != "silent"]
    minutes = max(len(records) * AUDIO_PACKET_SECONDS / SECONDS_PER_MINUTE, 1e-9)
    states = Counter(item.state for _, item in speech)
    flips = sum(a.state != b.state for (_, a), (_, b) in zip(speech, speech[1:]))
    labels = sum(a.speaker != b.speaker for (_, a), (_, b) in zip(speech, speech[1:]))
    segments = merge_word_events(events)
    return {
        "steps": total,
        "speech_fraction": round(len(speech) / max(total, 1), 3),
        "distinct_face_ids": records[-1][2] if records else 0,
        "face_ids_per_minute": round((records[-1][2] if records else 0) / minutes, 2),
        "speech_state_transitions": flips,
        "speech_state_transitions_per_minute": round(flips / minutes, 2),
        "speech_state_fractions": {
            state: round(count / max(len(speech), 1), 3)
            for state, count in sorted(states.items())
        },
        "person_label_changes": labels,
        "frames_without_any_face": round(no_face / max(total, 1), 3),
        "faces_detected_but_mesh_none": round(missing_mesh / max(detected, 1), 3),
        "caption_segments": len(segments),
        "caption_words": len(events),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--rotate", type=float, default=0.0)
    arguments = parser.parse_args()
    cv2.setNumThreads(0)
    result = replay(arguments.video, arguments.offset, arguments.duration, arguments.rotate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
