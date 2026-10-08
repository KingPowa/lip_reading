# Active-Speaker Captioning

Turn a prerecorded video or camera recording into an annotated copy with:

- readable, speaker-labelled captions;
- a coloured upper-body highlight around each visible person who is speaking;
- a `.vtt` subtitle file for a video player; and
- JSON files that make it easy to inspect what the pipeline decided.

It names people `Person 1`, `Person 2`, and so on **within one recording**. It does not identify real people or remember anyone between recordings.

## Use your camera

```bash
uv sync
uv run --python .venv/bin/python -m src.app.live --language it
```

Open the printed address and choose **Start camera**. Select the language before capture. Italian, English, Spanish, French, and German have configured small [Vosk models](https://alphacephei.com/vosk/models). The selected model downloads once into `models/` and loads before capture starts. Camera and microphone access require localhost or HTTPS.

Live mode runs three independent channels:

- **Audio:** an AudioWorklet produces mono 16 kHz PCM every 100 ms. A persistent Vosk decoder consumes each packet once and returns partial text while you speak. Final text and word times enter the transcript at an utterance boundary or on Stop. A provisional transcript row updates throughout the utterance.
- **Vision:** small current JPEGs run through face detection, mouth landmarks, and local SFace identity embeddings about ten times per second. Session-local features restore a returning person's label after leaving the frame, including returns at a different position. Distinct feature matches take priority over box overlap. There is no person segmentation, ECAPA clustering, or Light-ASD inference in the live path.
- **Attribution:** recent mouth movement and microphone energy select a current face. Competing evidence is `uncertain`; speech with no moving visible mouth is estimated `off_screen`. Highlights update from current evidence and continue independently of caption timing. Silence releases a highlight, and stale camera geometry expires.

The one second target applies to current visual evidence and provisional recognition. Capture cadence is 100 ms; processing and freshness are measured separately. A decoder can need more acoustic context to resolve an ambiguous word, so an absolute one second guarantee for every word on every device is not established. Final utterance commitment can take longer; partial text remains visible. See [the current replay measurements](progress/06-independent-realtime.md).

Only one audio upload per capture runs at a time. Unacknowledged PCM remains in a bounded buffer, retries reuse the same job, and Stop flushes the worklet and decoder before releasing the session. If network or decoding falls behind by twelve seconds, capture pauses and drains retained speech. Camera and audio activity requests continue independently of a busy decoder. **Retry recognition** retains failed audio. Stale responses cannot replace evidence from another capture session.

`--chunk-seconds` now controls PCM upload cadence (0.05–1 second, default 0.1). Use the default for the latency target. Live mode requires a selected language; it does not perform automatic language detection. `--speech-model /path/to/vosk-model` accepts a compatible local model. Whisper model selection belongs to the offline command. Existing `python -m src.live` and `live-speaker` entry points also work.

The first live startup also downloads the 37 MB [OpenCV SFace model](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) into `models/` (or `--model-directory`) and verifies its checksum. Identity features remain in memory for the capture session, with up to 128 identities and five recent feature examples per identity; the oldest absent identities are evicted at capacity. Features are discarded with the session. Missing landmarks use short box continuity; weak or ambiguous feature matches can receive a new label. See [the identity regression checks](progress/07-live-identity.tdd.md).

For access from another device, supply a trusted certificate:

```bash
uv run --python .venv/bin/python -m src.app.live --language it --host 0.0.0.0 --cert certificate.pem --key private-key.pem
```

Press `Ctrl-C` to stop the server.

## Start here

1. Put an MP4 with both video and audio in `samples/`.
2. Run the command below.
3. Open `output/output.annotated.mp4` to watch the result.
4. If a result looks wrong, open `output/viewer.html`, then inspect `events.json`, `speech.json`, and `faces.json` in that order.

```bash
uv run --python .venv/bin/python -m src.main samples/giuseppeconte.mp4 --output output
```

The first run downloads missing model weights into `models/`. Later runs reuse them.

## What you need

- Python 3.12–3.14
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe available on your `PATH`
- Enough disk space for the models and generated video

Install the Python environment once:

```bash
uv sync
```

On macOS, FFmpeg can be installed with `brew install ffmpeg`.

## Run it

```bash
uv run --python .venv/bin/python -m src.main INPUT.mp4 --output OUTPUT_DIRECTORY
```

Useful options:

```bash
# Tell Whisper the language when you already know it.
uv run --python .venv/bin/python -m src.main INPUT.mp4 --language it

# Use a faster, less accurate model for a quick check.
uv run --python .venv/bin/python -m src.main INPUT.mp4 --whisper-model small

# Use the middle ground for a faster, still more accurate run.
uv run --python .venv/bin/python -m src.main INPUT.mp4 --whisper-model medium

# See detailed progress messages.
uv run --python .venv/bin/python -m src.main INPUT.mp4 --verbose
```

`large-v3` is the default Whisper model because the supplied Italian speech still contains word-level errors with `small` and `medium`. For Italian, pass `--language it` to avoid automatic language detection. Use `small` or `medium` for faster review runs.

## Outputs

| File                   | Open this when you want…                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `output.annotated.mp4` | The finished video, with coloured upper-body highlights and matching captions burned in.                                       |
| `captions.vtt`         | Captions to load in a compatible video player.                                                                                 |
| `viewer.html`          | A local review page: seek from the colour-matched transcript and inspect active upper-body overlays. Keep it beside the video. |
| `events.json`          | Final speaker, text, face-track, confidence, state, and display colour for each speech event.                                  |
| `speech.json`          | Raw Whisper words plus anonymous voice clusters. Use it to diagnose transcription or diarization.                              |
| `faces.json`           | Detected face tracks and their timestamped boxes. Use it to diagnose visual tracking.                                          |
| `audio.wav`            | The 16 kHz mono audio extracted for the speech and active-speaker models.                                                      |

## How it decides who is speaking

Every stage uses the video timestamps, so audio, face boxes, and captions stay on the same clock.

```text
video
  ├─ FFmpeg → 16 kHz mono audio
  ├─ Whisper → words and timestamps
  ├─ ECAPA → anonymous voice clusters
  ├─ MediaPipe → visible face tracks
  └─ Light-ASD → speaking probability for each candidate face at each time
                         ↓
               association rules → captions, JSON, annotated video
```

For each voice turn, the association step picks a face only when its time-weighted Light-ASD score clears both a minimum confidence threshold and the runner-up margin. If the evidence is weak, the result is deliberately labelled `uncertain`; if no face is visible, it is `off_screen`. This is safer than drawing a confident box around the wrong person.

A face track cannot be assigned two person labels. If overlapping input would assign two boxes to the same person, the lower-confidence event is marked `uncertain`, so the rendered video has one highlight per person.

Only attributed speakers are highlighted. The offline renderer estimates an upper-body rectangle from each face. The live preview outlines the current speaking face.

## Improving a result

Use this quick triage order:

1. **Words are wrong:** use `--language it` for Italian. `large-v3` is already the default; use `small` or `medium` only when faster processing matters more than transcription accuracy. Check that the source audio is clear and not dominated by music or applause.
2. **Wrong person is highlighted:** inspect `faces.json` and `events.json` at the same timestamp. Every stable face track is evaluated. Attribution thresholds still need calibration against labelled multi-speaker recordings.
3. **No person is highlighted:** this is usually intentional `uncertain` or `off_screen` output. Check whether the speaker’s face is actually visible.
4. **Captions look cut off:** rerun with the current version. Captions are wrapped using their actual rendered width, not an approximate character count.

## Limits to expect

- Live captions use provisional streaming text. Recognition accuracy and word latency depend on the speech model, audio clarity, and device load.
- It does not perform real-name recognition or cross-video identity matching.
- Overlapping speech is marked when detected, but exact transcription of two people talking at once remains difficult.
- Accurate local transcription and active-speaker scoring are computationally expensive. A long video can take significantly longer than its runtime on CPU.
- Offline active-speaker inference receives consecutive images stored in temporary disk-backed memory maps; the full 640×360 sample needs about 9.8 GB of temporary frame storage. Live mode sends PCM and separate JPEGs up to 640×480; it does not decode video recordings.
- Live speaker attribution remains a mouth-motion heuristic rather than calibrated active-speaker inference. Face identity uses local embeddings within each capture. Background noise, yawning, small faces, occlusion, camera cuts, and simultaneous speech can cause uncertain or incorrect attribution. Poor face images, lookalikes, or missing landmarks can still change identities. Offline voice clustering remains probabilistic.
- Words are decoded from audio; this is not a visual-to-text model for silent lip reading.
- Burned-in captions use a Unicode-capable font, so Italian accents and apostrophes render as text instead of replacement characters.
- The result is evidence-based, not a ground-truth transcript. Review important captions before publishing them.

## Development checks

```bash
.venv/bin/python -m unittest -v
.venv/bin/python -m compileall -q src tests
node tests/live_client_test.cjs
```

The tests cover concurrent audio/vision, incremental decoding, PCM continuity and retries, session isolation, Stop flush, resampling, mouth evidence, current geometry, offline recognition, rendering, and media validation. The real browser replay and its limits are documented in [the independent pipeline report](progress/06-independent-realtime.md). Earlier reports under `progress/` describe the previous architecture.

## Project map

```text
src/
  core/
    audio/            incremental Vosk, model storage, offline Whisper and voices
    vision/           current face/mouth observations, offline faces and Light-ASD
    tracking/         current speaker estimates, offline association rules
    contracts.py      domain data
    media.py          offline FFmpeg and frame access
  config/             model catalog, tuning, transport limits, display settings
  app/
    live.py           live command-line startup
    offline.py        offline pipeline and command line
    service.py        independent channel coordination
    server.py         HTTP transport
    sessions.py       capture lifecycle and concurrency
    rendering/        JSON display payloads, pages, offline video rendering
    templates/        live HTML
    static/           PCM buffer, worklet, transport controller, live renderer
  live.py             compatibility command entry point
  main.py             compatibility command entry point
progress/             implementation and measurement reports
```
