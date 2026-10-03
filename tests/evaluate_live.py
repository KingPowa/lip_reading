"""Compare a saved browser replay with the supplied timestamped transcription."""

import argparse
import json
import re
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMECODE = re.compile(r"^(\d+):(\d+)$")
TIME_DESCRIPTION = re.compile(r"^\d+ (?:second[oi]|minut[oi](?: e \d+ second[oi])?)$")
NUMBER_WORDS = {
    "zero": "0",
    "uno": "1",
    "due": "2",
    "tre": "3",
    "quattro": "4",
    "cinque": "5",
    "sei": "6",
    "sette": "7",
    "otto": "8",
    "nove": "9",
    "dieci": "10",
}


def reference_sections(text: str) -> list[tuple[float, str]]:
    sections = []
    words = []
    for line in text.splitlines():
        match = TIMECODE.fullmatch(line.strip())
        if match:
            end = int(match[1]) * 60 + int(match[2])
            sections.append((end, " ".join(words)))
            words = []
            continue
        if line.strip() and not TIME_DESCRIPTION.fullmatch(line.strip()):
            words.append(line.strip())
    return sections


def tokens(text: str) -> list[str]:
    return [
        NUMBER_WORDS.get(word, word) for word in re.findall(r"\w+", text.casefold())
    ]


def edit_distance(reference: list[str], recognized: list[str]) -> int:
    previous = list(range(len(recognized) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(recognized, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def evaluate(report: dict, reference: str) -> dict:
    sections = [
        section
        for section in reference_sections(reference)
        if section[0] <= report["sourceTime"]
    ]
    end = sections[-1][0]
    expected = tokens(" ".join(text for _, text in sections))
    responses = [
        response for response in report["responses"] if response["status"] == 200
    ]
    events = [
        event
        for response in responses
        for event in response["events"]
        if event["start"] < end
    ]
    recognized = tokens(" ".join(event["text"] for event in events))
    durations = [response["timings"]["total"] for response in responses]
    delays = [
        response["receivedAt"] - event["end"]
        for response in responses
        for event in response["events"]
    ]
    display_delays = []
    for response in responses:
        for event in response["events"]:
            displayed_at = next(
                (
                    snapshot["at"]
                    for snapshot in report.get("snapshots", [])
                    if snapshot["at"] >= response["receivedAt"]
                    and f"{event['speaker']}: {event['text']}" in snapshot["caption"]
                ),
                None,
            )
            if displayed_at is not None:
                display_delays.append(displayed_at - event["end"])
    return {
        "referenceEndSeconds": end,
        "referenceWords": len(expected),
        "recognizedWords": len(recognized),
        "wordErrorRate": edit_distance(expected, recognized) / len(expected),
        "medianProcessingSeconds": statistics.median(durations),
        "maximumProcessingSeconds": max(durations),
        "medianCaptionArrivalDelaySeconds": statistics.median(delays),
        "maximumCaptionArrivalDelaySeconds": max(delays),
        "observedDisplayedEvents": len(display_delays),
        "medianCaptionDisplayDelaySeconds": statistics.median(display_delays)
        if display_delays
        else None,
        "maximumCaptionDisplayDelaySeconds": max(display_delays)
        if display_delays
        else None,
        "speakerLabels": sorted(
            {event["speaker"] for event in events if event["faceTrack"]}
        ),
        "maximumConcurrentUploads": report["maximumConcurrent"],
        "bufferDrained": report["drained"],
        "restartEnabled": report["startEnabled"],
        "visual": report["visual"],
        "transcription": " ".join(event["text"] for event in events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=PROJECT_ROOT / "samples/transcript.txt"
    )
    arguments = parser.parse_args()
    metrics = evaluate(
        json.loads(arguments.report.read_text()), arguments.reference.read_text()
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
