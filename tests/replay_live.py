"""Stream a video through the live detector and tracker and print coherence metrics.

Usage: python -m tests.replay_live samples/giuseppeconte.mp4 --duration 120 --rotate 25
Not a unittest module; it needs real MediaPipe models and ffmpeg.
"""

import argparse
import json
import subprocess
import time
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
from src.core.audio.speech import embed_samples, warm_up_speaker_model
from src.core.contracts import Word
from src.core.tracking.association import merge_word_events
from src.core.tracking.realtime import RealtimeTracker
from src.core.tracking.voice import SpeechBuffer
from src.core.vision.live import LiveFaceDetector
from src.config.models import MODEL_DIRECTORY

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


def prepare_frame(image: np.ndarray, degrees: float, shrink: float = 1.0) -> bytes:
    height, width = image.shape[:2]
    scale = min(MAXIMUM_FRAME_WIDTH / width, MAXIMUM_FRAME_HEIGHT / height)
    image = cv2.resize(image, (round(width * scale), round(height * scale)))
    if shrink < 1.0:
        # Smaller faces make mouth motion subtler and landmarks noisier.
        height, width = image.shape[:2]
        small = cv2.resize(image, (round(width * shrink), round(height * shrink)))
        canvas = np.zeros_like(image)
        top, left = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
        canvas[top : top + small.shape[0], left : left + small.shape[1]] = small
        image = canvas
    if degrees:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
    quality = round(JPEG_QUALITY * 100)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def frames(
    path: str, offset: float, steps: int, degrees: float, blanks=(), shrink=1.0
):
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
            hidden = any(start <= target < end for start, end in blanks)
            if hidden and image is not None:
                yield step, prepare_frame(np.zeros_like(image), degrees, shrink)
                continue
            yield step, None if image is None else prepare_frame(image, degrees, shrink)
    finally:
        video.release()


def replay(
    path: str,
    offset: float,
    duration: float,
    degrees: float,
    blanks=(),
    voice=True,
    gain=1.0,
    shrink=1.0,
    reference=None,
    dump=None,
) -> dict:
    audio = read_audio(path, offset, duration) * gain
    packet = round(AUDIO_PACKET_SECONDS * AUDIO_SAMPLE_RATE)
    steps = len(audio) // packet
    detector = LiveFaceDetector()
    tracker = RealtimeTracker()
    classifier, buffer, embed_ms = None, SpeechBuffer(), []
    if voice:
        classifier = warm_up_speaker_model(MODEL_DIRECTORY)
    records, words, events = [], [], []
    detected = missing_mesh = no_face = total = suppressed = 0
    try:
        for step, content in frames(path, offset, steps, degrees, blanks, shrink):
            timestamp = round((step + 1) * AUDIO_PACKET_SECONDS, 6)
            chunk = audio[step * packet : (step + 1) * packet]
            tracker.update_audio(chunk, timestamp)
            if classifier is not None:
                buffer.add(chunk, timestamp, tracker.speaking)
                if buffer.ready():
                    samples, start, end = buffer.take()
                    began = time.perf_counter()
                    embedding = embed_samples(classifier, samples)
                    embed_ms.append((time.perf_counter() - began) * 1000)
                    if embedding is not None:
                        tracker.update_voice(embedding, start, end)
            _, _, observations = detector.detect(content, timestamp)
            total += 1
            no_face += not observations
            detected += len(observations)
            missing_mesh += sum(item.aperture is None for item in observations)
            estimate = tracker.update_faces(observations, timestamp)
            suppressed += len(observations) - len(tracker.current_faces)
            records.append((timestamp, estimate, tracker.label_count))
            if dump is not None:
                dump.append(
                    {
                        "t": timestamp,
                        "speaker": estimate.speaker,
                        "state": estimate.state,
                        "confidence": round(estimate.confidence, 3),
                        "energy": round(float(np.sqrt(np.mean(chunk * chunk))), 5),
                        "faces": [
                            {
                                "id": tracker.labels.get(face.track_id),
                                "motion": round(face.mouth_motion, 4),
                                "area": face.box.width * face.box.height,
                                "box": [face.box.x, face.box.y, face.box.width],
                                "aperture": (
                                    tracker.mouths[face.track_id][-1][1]
                                    if tracker.mouths.get(face.track_id)
                                    else None
                                ),
                            }
                            for face in tracker.current_faces
                        ],
                    }
                )
            if estimate.state != "silent" and step % WORD_STEPS == 0:
                words.append(Word("w", timestamp - 0.2, timestamp, 1.0))
                events.extend(tracker.caption_events(words[-1:]))
    finally:
        detector.close()
    result = metrics(records, events, total, detected, missing_mesh, no_face, suppressed)
    relative = [(start - offset, end - offset) for start, end in blanks]
    result |= attribution_metrics(records, relative)
    if reference is not None:
        result |= detection_metrics(records, reference_speech(reference, offset))
    return result | voice_metrics(tracker, records, embed_ms, relative)


def reference_speech(path: str, offset: float) -> list[tuple[float, float]]:
    """Speech intervals from an offline speech.json, shifted to the replay clock."""
    words = json.loads(open(path, encoding="utf-8").read())["words"]
    return [(word["start"] - offset, word["end"] - offset) for word in words]


def detection_metrics(records, intervals) -> dict:
    def speaking(time_):
        return any(start - 0.15 <= time_ <= end + 0.15 for start, end in intervals)

    inside = [item.state != "silent" for time_, item, _ in records if speaking(time_)]
    outside = [
        item.state != "silent" for time_, item, _ in records if not speaking(time_)
    ]
    return {
        "speech_recall": round(sum(inside) / max(len(inside), 1), 3),
        "speech_false_alarm": round(sum(outside) / max(len(outside), 1), 3),
    }


def attribution_metrics(records, blanks) -> dict:
    """Single-speaker clips: how often speech goes to the dominant visible person."""
    speech = [
        item
        for time_, item, _ in records
        if item.state != "silent"
        and not any(start <= time_ < end for start, end in blanks)
    ]
    visible = Counter(item.speaker for item in speech if item.state == "visible")
    dominant = visible.most_common(1)[0][0] if visible else None
    counts = Counter(
        "dominant_visible"
        if item.state == "visible" and item.speaker == dominant
        else "other_person"
        if item.speaker.startswith("Person") and item.speaker != dominant
        else "dominant_off_screen"
        if item.speaker == dominant
        else item.state
        for item in speech
    )
    total = max(len(speech), 1)
    return {
        "dominant_person": dominant,
        "attribution": {
            key: round(value / total, 3) for key, value in sorted(counts.items())
        },
    }


def voice_metrics(tracker, records, embed_ms, blanks) -> dict:
    result = {
        "voice_ids": len(tracker.voices.centroids),
        "voices_bound": len(tracker.voices.bound_faces()),
        "identity_merges": tracker.merge_count,
    }
    if embed_ms:
        ordered = sorted(embed_ms)
        result["embedding_ms_mean"] = round(sum(ordered) / len(ordered), 1)
        result["embedding_ms_p95"] = round(ordered[int(0.95 * (len(ordered) - 1))], 1)
        result["embedding_jobs"] = len(ordered)
    if blanks:
        result["blank_window_speech"] = blank_fractions(records, blanks)
    return result


def blank_fractions(records, blanks) -> dict:
    visible = Counter(item.speaker for _, item, _ in records if item.state == "visible")
    dominant = visible.most_common(1)[0][0] if visible else None
    counts, total = Counter(), 0
    for start, end in blanks:
        # Single-speaker clips: the person most often attributed overall, not a
        # short lookback that a listener can dominate.
        original = dominant
        for time_, item, _ in records:
            if not start <= time_ < end or item.state == "silent":
                continue
            total += 1
            if item.speaker == original:
                counts["original_person"] += 1
            elif item.state in ("off_screen", "uncertain") and not item.speaker.startswith("Person"):
                counts[item.state] += 1
            else:
                counts["other_person"] += 1
    return {key: round(value / max(total, 1), 3) for key, value in sorted(counts.items())} | {"speech_steps": total}


def metrics(records, events, total, detected, missing_mesh, no_face, suppressed) -> dict:
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
        "tentative_detections_suppressed": round(suppressed / max(detected, 1), 3),
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
    parser.add_argument("--blank", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--shrink", type=float, default=1.0)
    parser.add_argument("--reference", help="offline speech.json for speech recall")
    parser.add_argument("--dump", help="write per-step speaker decisions to this JSON")
    arguments = parser.parse_args()
    cv2.setNumThreads(0)
    result = replay(
        arguments.video,
        arguments.offset,
        arguments.duration,
        arguments.rotate,
        [tuple(item) for item in arguments.blank],
        not arguments.no_voice,
        arguments.gain,
        arguments.shrink,
        arguments.reference,
        steps := [] if arguments.dump else None,
    )
    if arguments.dump:
        with open(arguments.dump, "w", encoding="utf-8") as handle:
            json.dump(steps, handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
