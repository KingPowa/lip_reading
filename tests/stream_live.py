"""Live-modality evaluation: stream a video to the live server over HTTP in real time.

Mimics the browser client: 100 ms PCM to /activity, serialized /audio jobs with
the unacknowledged backlog (<= 1 s), ~10 fps JPEG frames to /track (skipping while
an upload is in flight). Records every response and server RSS / CPU.

Usage: .venv/bin/python -m tests.stream_live . --name NAME [--duration 180] [--mix]
--mix splices English text-to-speech (macOS `say`) into the Italian sample to measure
language switching. Not a unittest module; needs real models, ffmpeg and node.
"""

import argparse
import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MAIN = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
VIDEO = MAIN / "samples/giuseppeconte.mp4"
REFERENCE = MAIN / "samples/transcript.txt"
HERE = MAIN / "output" / "stream-live"
TTS = (
    "Thank you, Mister President. I would like to explain why this government has failed to deliver on its promises to the families of this country.",
    "We need a serious industrial plan, more investment in schools and hospitals, and lower taxes for workers and small businesses.",
    "Inflation is still too high, wages are falling, and people are asking us for real answers instead of slogans.",
)
RATE = 16000
PACKET = 0.1
MAX_JOB_SECONDS = 1.0


def read_audio(offset: float, duration: float) -> bytes:
    command = ["ffmpeg", "-v", "error", "-ss", str(offset), "-t", str(duration),
               "-i", str(VIDEO), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    return subprocess.run(command, check=True, capture_output=True).stdout


class Frames:
    def __init__(self, offset: float) -> None:
        self.video = cv2.VideoCapture(str(VIDEO))
        self.video.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
        self.offset = offset
        self.image = None

    def at(self, seconds: float) -> bytes | None:
        target = self.offset + seconds
        while self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000 < target:
            if not self.video.grab():
                break
            ok, image = self.video.retrieve()
            if ok:
                self.image = image
        if self.image is None:
            return None
        return cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, 65])[1].tobytes()


class Client:
    def __init__(self, port: int, session: str) -> None:
        self.port = port
        self.session = session
        self.local = threading.local()

    def post(self, path: str, body: bytes, headers: dict) -> tuple[int, dict]:
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
            self.local.connection = connection
        headers = {"X-Capture-Session": self.session, "Content-Length": str(len(body))} | headers
        try:
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException):
            self.local.connection = None
            raise
        try:
            payload = json.loads(data)
        except ValueError:
            payload = {"raw": data[:200].decode(errors="replace")}
        return response.status, payload


def start_server(root: Path, port: int, language: str, log: Path) -> tuple[subprocess.Popen, float]:
    began = time.monotonic()
    env = os.environ | {"PYTHONPATH": str(root), "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        [str(PYTHON), "-m", "src.app.live", "--language", language, "--port", str(port),
         "--model-directory", str(MAIN / "models")],
        cwd=root, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
    )
    while time.monotonic() - began < 300:
        if process.poll() is not None:
            raise RuntimeError(f"server exited; see {log}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/")
            if connection.getresponse().status == 200:
                return process, time.monotonic() - began
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def process_stats(pid: int) -> tuple[float, float] | None:
    """RSS MB and CPU seconds for the process tree rooted at pid."""
    output = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,time="], capture_output=True, text=True).stdout
    rows = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 4:
            rows[int(parts[0])] = (int(parts[1]), int(parts[2]), parts[3])
    if pid not in rows:
        return None
    tree, frontier = {pid}, [pid]
    while frontier:
        parent = frontier.pop()
        for child, (ppid, _, _) in rows.items():
            if ppid == parent and child not in tree:
                tree.add(child)
                frontier.append(child)
    rss = sum(rows[item][1] for item in tree) / 1024
    # macOS footprint counts dirty+compressed memory, not mmapped model pages, so it is stable.
    footprint = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    found = re.search(r"Footprint: ([0-9.]+) (KB|MB|GB)", footprint)
    if found:
        rss = float(found[1]) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[found[2]]
    cpu = 0.0
    for item in tree:
        text = rows[item][2]
        fields = [float(value) for value in re.split("[:-]", text)]
        seconds = 0.0
        for value in fields:
            seconds = seconds * 60 + value
        cpu += seconds
    return rss, cpu


MIX_AT = (40.0, 100.0, 150.0)


def english_clip(index: int) -> bytes:
    path = HERE / "tts" / f"en{index}.pcm"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        speech = path.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(speech), TTS[index - 1]], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(speech), "-ac", "1", "-ar", str(RATE),
                        "-f", "s16le", str(path)], check=True)
    return path.read_bytes()


def splice_english(pcm: bytes) -> tuple[bytes, list[tuple[float, float]]]:
    """Replace stretches of the stream with English TTS; return their intervals."""
    data, intervals = bytearray(pcm), []
    for index, at in enumerate(MIX_AT, start=1):
        clip = english_clip(index)
        start = round(at * RATE) * 2
        if start + len(clip) > len(data):
            break
        data[start : start + len(clip)] = clip
        intervals.append((at, at + len(clip) / 2 / RATE))
    return bytes(data), intervals


def run(root: Path, name: str, language: str, offset: float, duration: float, port: int, speed: float,
        mix: bool = False) -> dict:
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    pcm = read_audio(offset, duration)
    english = []
    if mix:
        pcm, english = splice_english(pcm)
    packet_bytes = round(PACKET * RATE) * 2
    packets = len(pcm) // packet_bytes
    total_seconds = packets * PACKET
    process, startup = start_server(root, port, language, out / "server.log")
    base = process_stats(process.pid)
    session = f"eval{int(time.time())}"
    client = Client(port, session)
    status, payload = client.post("/start", b"", {"X-Whisper-Language": language})
    assert status == 200, payload
    t0 = time.monotonic()
    import faulthandler
    stacks = (out / "stacks.txt").open("w")
    # A run that overruns its stream by 5 minutes dumps every thread's stack once.
    faulthandler.dump_traceback_later(total_seconds / speed + 300, exit=False, file=stacks)

    def now() -> float:
        return (time.monotonic() - t0) * speed

    record = {"audio": [], "activity": [], "track": [], "errors": [], "stats": [], "lag": []}
    lock = threading.Lock()
    done = threading.Event()

    def available() -> int:
        return min(packets, int(now() / PACKET + 1e-9))

    def audio_loop() -> None:
        submitted, job = 0, 0
        while True:
            ready = available()
            final = ready >= packets and submitted >= packets
            if ready - submitted < 1 and not final:
                time.sleep(0.01)
                continue
            end = min(ready, submitted + round(MAX_JOB_SECONDS / PACKET))
            body = pcm[submitted * packet_bytes : end * packet_bytes]
            is_final = end >= packets and not body
            if end >= packets and body:
                is_final = False
            job += 1
            headers = {"X-Capture-Offset": f"{submitted * PACKET:.6f}", "X-Whisper-Language": language,
                       "X-Capture-Job": str(job), "X-Capture-Final": "true" if is_final else "false"}
            while True:
                sent = now()
                try:
                    status, payload = client.post("/audio", body, headers)
                except (OSError, http.client.HTTPException) as error:
                    with lock:
                        record["errors"].append({"path": "/audio", "error": str(error), "at": sent})
                    time.sleep(0.2)
                    continue
                received = now()
                if status == 429:
                    with lock:
                        record["errors"].append({"path": "/audio", "status": 429, "at": sent})
                    time.sleep(0.2)
                    continue
                break
            with lock:
                record["lag"].append(received - end * PACKET)
                if status != 200:
                    record["errors"].append({"path": "/audio", "status": status, "payload": payload, "at": sent})
                else:
                    record["audio"].append(payload | {"sentAt": sent, "receivedAt": received})
            submitted = end
            if is_final or status != 200 and end >= packets:
                return

    def activity_loop() -> None:
        index = 0
        while index < packets:
            ready = available()
            if ready <= index:
                time.sleep(0.005)
                continue
            index = ready  # like the client, only the latest pending packet is sent
            body = pcm[(index - 1) * packet_bytes : index * packet_bytes]
            sent = now()
            try:
                status, payload = client.post("/activity", body, {"X-Capture-Offset": f"{index * PACKET:.6f}"})
            except (OSError, http.client.HTTPException) as error:
                record["errors"].append({"path": "/activity", "error": str(error), "at": sent})
                continue
            with lock:
                if status != 200:
                    record["errors"].append({"path": "/activity", "status": status, "at": sent})
                else:
                    record["activity"].append({"end": index * PACKET, "receivedAt": now(), "speaker": payload.get("speaker")})

    def track_loop() -> None:
        frames = Frames(offset)
        next_at = 0.1
        while now() < total_seconds:
            if now() < next_at:
                time.sleep(0.005)
                continue
            stamp = now()
            next_at = stamp + 0.1
            content = frames.at(stamp)
            if content is None:
                continue
            try:
                status, payload = client.post("/track", content, {"X-Capture-Offset": f"{stamp:.6f}"})
            except (OSError, http.client.HTTPException) as error:
                record["errors"].append({"path": "/track", "error": str(error), "at": stamp})
                continue
            with lock:
                if status != 200:
                    record["errors"].append({"path": "/track", "status": status, "at": stamp})
                else:
                    record["track"].append({"timestamp": stamp, "receivedAt": now(),
                                            "faces": len(payload.get("faces", [])), "speaker": payload.get("speaker")})

    def stats_loop() -> None:
        while not done.is_set():
            stats = process_stats(process.pid)
            if stats:
                record["stats"].append({"at": now(), "rssMb": stats[0], "cpuSeconds": stats[1]})
            done.wait(1.0)

    threads = [threading.Thread(target=target, daemon=True) for target in (audio_loop, activity_loop, track_loop)]
    monitor = threading.Thread(target=stats_loop, daemon=True)
    monitor.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    finished = now()
    faulthandler.cancel_dump_traceback_later()
    end_status = client.post("/end", b"", {})[0]
    done.set()
    monitor.join()
    final_stats = process_stats(process.pid)
    process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
    record |= {"name": name, "root": str(root), "language": language, "offset": offset, "duration": total_seconds,
               "speed": speed, "startupSeconds": startup, "baseline": base, "final": final_stats,
               "finishedAt": finished, "endStatus": end_status, "english": english}
    (out / "record.json").write_text(json.dumps(record, ensure_ascii=False))
    return record


# ---------------------------------------------------------------- metrics

from tests.evaluate_live import edit_distance, reference_sections, tokens  # noqa: E402


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 3)


def render_cards(root: Path, record: dict) -> dict:
    responses = HERE / "runs" / record["name"] / "responses.json"
    responses.write_text(json.dumps(record["audio"], ensure_ascii=False))
    output = subprocess.run(["node", str(MAIN / "tests" / "render_cards.cjs"), str(root), str(responses)],
                            capture_output=True, text=True)
    if output.returncode:
        return {"renderError": output.stderr[-500:]}
    return json.loads(output.stdout)


def evaluate(record: dict) -> dict:
    root = Path(record["root"])
    events = [event for response in record["audio"] for event in response["events"]]
    offset, duration = record["offset"], record["duration"]
    sections = [s for s in reference_sections(REFERENCE.read_text()) if offset <= s[0] - 0 and s[0] <= offset + duration]
    reference_end = sections[-1][0] - offset if sections else duration
    scored = [event for event in events if event["start"] < reference_end]
    expected = tokens(" ".join(text for _, text in sections))
    recognized = tokens(" ".join(event["text"] for event in scored))
    wer = edit_distance(expected, recognized) / max(len(expected), 1)
    words = [(event, len(event["text"].split())) for event in events]
    word_total = sum(count for _, count in words) or 1
    languages = {}
    for event, count in words:
        languages[event.get("language")] = languages.get(event.get("language"), 0) + count
    language_switches = sum(
        1 for a, b in zip(events, events[1:]) if a.get("language") and b.get("language") and a["language"] != b["language"]
    )
    speaker_switches = sum(1 for a, b in zip(events, events[1:]) if a["speaker"] != b["speaker"])
    speakers = {}
    for event, count in words:
        speakers[event["speaker"]] = speakers.get(event["speaker"], 0) + count
    # Per-word lag: words spread evenly over their event, so merging events does not bias it.
    delays = [
        response["receivedAt"] - (event["start"] + (event["end"] - event["start"]) * (i + 1) / n)
        for response in record["audio"]
        for event in response["events"]
        for n in [max(len(event["text"].split()), 1)]
        for i in range(n)
    ]
    rtt = [r["receivedAt"] - r["sentAt"] for r in record["audio"]]
    track_age = [t["receivedAt"] - t["timestamp"] for t in record["track"]]
    stats = record["stats"]
    minutes = duration / 60
    cpu = None
    if record["baseline"] and record["final"]:
        cpu = (record["final"][1] - record["baseline"][1]) / duration
    first_partial = next((r["receivedAt"] for r in record["audio"] if r.get("provisionalText")), None)
    statuses = {}
    for error in record["errors"]:
        key = f"{error['path']}:{error.get('status', 'exception')}"
        statuses[key] = statuses.get(key, 0) + 1
    result = {
        "name": record["name"],
        "seconds": round(duration, 1),
        # precision
        "wer": round(wer, 3),
        "referenceWords": len(expected),
        "recognizedWords": len(recognized),
        # language mixing
        "languageWordShare": {str(k): round(v / word_total, 3) for k, v in sorted(languages.items(), key=lambda x: -x[1])},
        "languageSwitches": language_switches,
        # fragmentation
        "events": len(events),
        "speakerSwitches": speaker_switches,
        "speakerWordShare": {k: round(v / word_total, 3) for k, v in sorted(speakers.items(), key=lambda x: -x[1])},
        # speed
        "captionDelayP50": quantile(delays, 0.5),
        "captionDelayP95": quantile(delays, 0.95),
        "audioRttP50": quantile(rtt, 0.5),
        "audioRttP95": quantile(rtt, 0.95),
        "audioBacklogP95": quantile(record["lag"], 0.95),
        "trackAgeP50": quantile(track_age, 0.5),
        "trackAgeP95": quantile(track_age, 0.95),
        "trackFps": round(len(record["track"]) / duration, 2),
        "firstPartialAt": first_partial,
        # lightweight
        "startupSeconds": round(record["startupSeconds"], 1),
        "rssPeakMb": round(max((s["rssMb"] for s in stats), default=0), 0),
        "rssMeanMb": round(statistics.mean(s["rssMb"] for s in stats), 0) if stats else None,
        "cpuCoresMean": round(cpu, 3) if cpu is not None else None,
        # reliability
        "errors": statuses,
        "endStatus": record["endStatus"],
    }
    if record.get("english"):
        def truth(event):
            middle = (event["start"] + event["end"]) / 2
            return "en" if any(a - 0.3 <= middle <= b + 0.3 for a, b in record["english"]) else "it"
        right = sum(count for event, count in words if event.get("language") == truth(event))
        en_words = [(e, c) for e, c in words if truth(e) == "en"]
        it_words = [(e, c) for e, c in words if truth(e) == "it"]
        result |= {
            "mixLanguageAccuracy": round(right / word_total, 3),
            "mixEnglishRecall": round(sum(c for e, c in en_words if e.get("language") == "en") / max(sum(c for _, c in en_words), 1), 3),
            "mixItalianPurity": round(sum(c for e, c in it_words if e.get("language") == "it") / max(sum(c for _, c in it_words), 1), 3),
            "mixIdealSwitches": 2 * len(record["english"]),
        }
        result.pop("wer", None)
    # What the user sees live: the status pill and the outlined (highlighted) person.
    live = sorted(
        [(item["receivedAt"], item["speaker"]) for item in record["activity"] + record["track"] if item.get("speaker")],
        key=lambda pair: pair[0],
    )
    pill = [s["speaker"] if s["state"] == "visible" else s["state"] for _, s in live]
    outlined = [s["speaker"] for _, s in live if s["state"] == "visible"]
    speaking = [s for _, s in live if s["state"] != "silent"]
    result |= {
        "pillChangesPerMinute": round(sum(a != b for a, b in zip(pill, pill[1:])) / minutes, 1),
        "highlightSwitchesPerMinute": round(sum(a != b for a, b in zip(outlined, outlined[1:])) / minutes, 1),
        "unsureShare": round(sum(s["state"] == "uncertain" for s in speaking) / max(len(speaking), 1), 3),
    }
    cards = render_cards(root, record)
    result |= {"cards": cards.get("cards"), "cardsPerMinute": round(cards["cards"] / minutes, 1) if "cards" in cards else None,
               "wordsPerCard": cards.get("wordsPerCard"), "cardSpeakers": cards.get("speakers"),
               **({"renderError": cards["renderError"]} if "renderError" in cards else {})}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--language", default="it,en")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--mix", action="store_true", help="splice English TTS clips in")
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    if arguments.metrics_only:
        record = json.loads((HERE / "runs" / arguments.name / "record.json").read_text())
    else:
        record = run(arguments.root.resolve(), arguments.name, arguments.language, arguments.offset,
                     arguments.duration, arguments.port, arguments.speed, arguments.mix)
    result = evaluate(record)
    (HERE / "runs" / arguments.name / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
