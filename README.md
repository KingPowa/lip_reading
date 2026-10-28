# Active-Speaker Captioning

**Disclaimer**: this code was created by Claude with personal supervision on the design.

Find out who is speaking, and what they say, in a video or live from your camera. Everything runs locally. Two modes share the same core code:

- **Live mode** streams your camera and microphone to a local server and shows speaker-labelled captions and a highlight around the speaking face as you talk.
- **Offline mode** turns a prerecorded MP4 into an annotated video, a `.vtt` subtitle file, and JSON files that show what the pipeline decided.

People are named `Person 1`, `Person 2`, and so on **within one recording or capture**. The project does not identify real people and does not remember anyone between recordings.

## What you need

- Python 3.12–3.14
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe on your `PATH` (offline mode; on macOS, `brew install ffmpeg`)
- Disk space for the models and generated video

```bash
uv sync
```

Models download on first use into `models/` (or `--model-directory`) and are reused afterwards:

| Model                                   | Used by                                        |
| --------------------------------------- | ---------------------------------------------- |
| Whisper `large-v3`                      | offline transcription                          |
| ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) | voice clustering (offline) and voice identity (live; torch-free via `models/speaker-ecapa/ecapa.onnx`, created once by `PYTHONPATH=. python scripts/export_ecapa_onnx.py` with `onnx` and `onnxscript` installed, otherwise SpeechBrain is used) |
| Light-ASD weights                       | offline speaking scores                        |
| MediaPipe face landmarker               | offline face tracks (live uses MediaPipe's bundled face detection and FaceMesh) |
| OpenCV SFace (about 37 MB, checksum-verified) | live face identity                       |
| Vosk small model per language           | live transcription                             |
| faster-whisper `tiny`                   | live language ID, only when several languages are selected |

## Live captioning app

Speaker Captions is a Mac app that captions your camera and microphone live: who is speaking, what they say, and in which language.

```bash
uv sync                          # once: Python environment and models' dependencies
scripts/install-desktop-app.sh   # builds "Speaker Captions.app" and copies it to ~/Applications
```

Open **Speaker Captions** from Applications or Spotlight. The first launch shows a loading screen while the models warm up (the first time, they also download), then asks for camera and microphone access. Pick one to three languages with the chips under the video and press **Start**.

The window shows:

- the camera with a glowing outline and name chip on whoever is speaking, and a live caption bar;
- a status pill (`Listening`, `Person 2 speaking`, `Person 1 · off camera`, `Not sure who is speaking`);
- a transcript of colour-coded cards per person, with language and "off camera" badges.

The app is an Electron shell (`desktop/`) around the local Python server below: it starts the server on a free local port, grants camera and microphone access only to it, and stops it when you quit. It is unsigned and uses this repository's `.venv`, so keep the repository in place. For development, `npm --prefix desktop install && npm --prefix desktop start`. See `desktop/README.md`.

### From a browser

```bash
uv run --python .venv/bin/python -m src.app.live --language it
```

Without the app, start the server and open the printed address, then press **Start**. Camera and microphone access need localhost or HTTPS. `--language` takes `en`, `it`, `es`, `fr`, or `de`, or a comma list of up to three such as `--language it,en`. The page also has language chips. The `live-speaker` script and `python -m src.live` do the same thing.

How it works:

- **Audio.** The browser sends mono 16 kHz PCM every 100 ms (with auto gain requested) over plain HTTP POSTs (`/start`, `/audio`, `/activity`, `/track`, `/end`); there is no WebSocket. A persistent Vosk decoder (catalog models use a wider search than their defaults: beam 13, 7000 active states) returns partial text while you speak and final text with word times at the end of an utterance or on Stop. Fluent speakers rarely pause long enough for Vosk's own endpointer, so an utterance is also closed at a 0.25 s pause once it is 1.2 s long, at a dip after 7 s, or at 10 s. An adaptive speech gate tracks the room's noise floor, with an absolute floor of 0.0025 RMS, so quiet speech is detected.
- **Vision.** About ten times per second, face detection and mouth landmarks run on small JPEGs; SFace identity embeddings are recomputed only for new or moved faces, or once a second. Session-local features restore a returning person's label after they leave and re-enter the frame, even at a different position. Features are kept only in memory for the session.
- **Attribution.** Mouth motion is the max-minus-min mouth opening over 0.4 s, minus a per-face jitter baseline learned while the room is silent. Because a single 0.4 s window is noisy (listeners chew, nod, murmur), each face accumulates *speaking evidence*: motion integrated over about 2 s while speech is present, decaying during silence. Faces with at least 0.01 evidence are candidates; the winner needs a lead of min(0.008, half its score), otherwise the result is `uncertain`. Small faces keep at least half weight, and a recognised voice adds a bonus to its face. States are `visible`, `uncertain`, `off_screen`, and `silent`. Captions are attributed per utterance, not per word: a duration-weighted vote over the utterance, where a named person with at least 30 % beats generic labels, and a second person only takes words after an uninterrupted 1.5 s stretch. Once a voice has 8 s of face evidence and its leading face has twice the runner-up's share, its utterances and the live highlight go to that face, so listeners' mouth jitter cannot steal a monologue. The transcript groups consecutive captions of one speaker into paragraphs of up to 30 s.
- **Voice identity.** ECAPA embeddings of roughly one second of speech are bound to a face after repeated clear evidence (at least 5 samples in a one-second window where that face's evidence is at least twice any other face's, 70 % dominance, 3 confirmations). That needs a few seconds of visible speaking first. Once a voice is bound, the person stays `Person N` when they leave the frame and keep talking: the status reads "Person N · off camera", the transcript card gets an "off camera" badge, and the colour stays the same. Generic "Off-screen speaker" moments inside a recognised voice window are relabelled, so captions finalised afterwards carry the name (captions already shown are not rewritten). Unknown voices stay "Off-screen speaker".
- **Several languages.** One Vosk decoder (the active language) runs on the audio. When an utterance ends, spoken language ID (faster-whisper `tiny`) may switch the language: the other language needs >= 0.75 of the probability among the selected languages on an utterance of at least 1 s, and the utterance is held back until the next one decides: if that also points to the other language, both are decoded again in it and the language switches; otherwise the held words are shown in the current language, unless the held utterance alone was at least 0.9 for the other language over at least 2 s, in which case it alone is decoded in that language (a one-off foreign sentence) without switching. Without the ID model the first language is kept. Captions show a tag such as `[IT]` when several languages are active. With one language, behaviour is unchanged.

Measured by streaming the 9.5 minute sample to the live server in real time (`tests/stream_live.py`, languages `it,en`, Apple M3), against the previous design: final captions arrive a median 1.9 s after a word (was 8.2 s; p95 5.7 s, was 18.5 s); the server uses 0.55 CPU cores (was 0.88) and 863 MB mean memory (was 1166 MB); word error rate is 0.167 (was 0.175); 99.9 % of words go to the one person speaking (was 60 %, spread over 16 labels), with 2 speaker changes in the transcript (was 349); the on-video highlight changes person 9 times a minute (was 36) and the status shows "not sure" during 1.9 % of speech (was 5 %). With three English sentences spliced into the Italian speech, 99.5 % of words carry the right language tag (was 90.7 %). Streamed three times faster than real time (a slower machine), the camera channel still processes 8.4 frames per second (was 4.8). Single-machine, single-recording measurements; the speaker numbers come from a recording with one speaker in a crowded frame. See [the evaluation on the sample video](progress/08-evaluation-and-tuning.md), [the replay measurements](progress/06-independent-realtime.md) and [the identity checks](progress/07-live-identity.tdd.md).

Audio uploads run one at a time, unacknowledged PCM stays in a bounded buffer, and Stop flushes the decoder before the session ends. If the network or decoder falls twelve seconds behind, capture pauses and drains. **Retry recognition** keeps failed audio.

Options:

| Option                     | Meaning                                                                         |
| -------------------------- | ------------------------------------------------------------------------------- |
| `--language`               | One language or a comma list of up to three                                     |
| `--host`, `--port`         | Address to bind (default port 8765)                                             |
| `--chunk-seconds`          | PCM upload interval, 0.05–1 (default 0.1)                                       |
| `--speech-model DIR`       | Custom local Vosk model; single language only                                   |
| `--model-directory`        | Where models are stored                                                         |
| `--cert`, `--key`          | TLS certificate and key                                                         |

For another device, supply a trusted certificate:

```bash
uv run --python .venv/bin/python -m src.app.live --language it --host 0.0.0.0 --cert certificate.pem --key private-key.pem
```

Press `Ctrl-C` to stop the server.

## Offline mode

Start here:

1. Put an MP4 with video and audio in `samples/`.
2. Run the command below.
3. Open `output/output.annotated.mp4`.
4. If a result looks wrong, open `output/viewer.html`, then inspect `events.json`, `speech.json`, and `faces.json` in that order.

```bash
uv run --python .venv/bin/python -m src.main samples/giuseppeconte.mp4 --output output
```

Options:

```bash
# Tell Whisper the language (omit it and Whisper detects it per file).
uv run --python .venv/bin/python -m src.main INPUT.mp4 --language it

# Faster, less accurate models (default is large-v3).
uv run --python .venv/bin/python -m src.main INPUT.mp4 --whisper-model small

# Detailed progress messages; --model-directory changes where models live.
uv run --python .venv/bin/python -m src.main INPUT.mp4 --verbose
```

`large-v3` is the default because the supplied Italian speech still has word-level errors with `small` and `medium`.

### Outputs

| File                   | Open this when you want…                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| `output.annotated.mp4` | The finished video, with coloured upper-body highlights and matching captions burned in.                       |
| `captions.vtt`         | Captions to load in a compatible video player.                                                                 |
| `viewer.html`          | A local review page: seek from the colour-matched transcript and inspect overlays. Keep it beside the video.   |
| `events.json`          | Final speaker, text, face-track, confidence, state, and colour for each speech event.                          |
| `speech.json`          | Raw Whisper words plus anonymous voice clusters. Use it to diagnose transcription or diarization.              |
| `faces.json`           | Detected face tracks and their timestamped boxes. Use it to diagnose visual tracking.                          |
| `audio.wav`            | The 16 kHz mono audio extracted for the models.                                                                |

### How it decides who is speaking

Every stage uses the video timestamps, so audio, face boxes, and captions share one clock.

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

For each voice turn, a face is picked only when its time-weighted Light-ASD score is at least 0.20 and leads the runner-up by at least 0.12. Weak evidence is labelled `uncertain`; no visible face is `off_screen`. This is safer than boxing the wrong person. A face track cannot carry two person labels: if overlapping input would do that, the lower-confidence event becomes `uncertain`. Only attributed speakers are highlighted, using an upper-body rectangle estimated from the face.

### Improving a result

1. **Words are wrong:** pass `--language`; check the audio is clear and not dominated by music or applause. Use `small` or `medium` only when speed matters more.
2. **Wrong person highlighted:** compare `faces.json` and `events.json` at the same timestamp.
3. **No one highlighted:** usually intentional `uncertain` or `off_screen`. Check whether the speaker's face is visible.

## Current status and limits

- Live thresholds were tuned by replaying the single-speaker sample video (one standing speaker among small seated listeners); see [the evaluation](progress/08-evaluation-and-tuning.md). On that video the speaker is picked correctly about 77 % of speech time and a listener about 14 %, at normal and at −22 dB volume. They have not been calibrated on labelled multi-speaker conversations, and offline thresholds are untouched.
- Very small faces (about 20 px wide) defeat mouth landmarks; attribution then falls back to `uncertain`/`off_screen` and voices rarely bind.
- Multilanguage is decided per utterance, not per word, so switching languages inside one sentence is not handled. Partial text comes from the currently active language until the utterance closes.
- Live attribution is a mouth-motion heuristic, not calibrated active-speaker inference. Background noise, yawning, small faces, occlusion, camera cuts, lookalikes, and simultaneous speech can cause uncertain or wrong results. Voice binding needs a few seconds of clear visible speech first.
- Live captions are provisional streaming text from small models; accuracy and latency depend on the model, audio, and device load. An absolute one-second guarantee for every word is not established.
- Words come from audio; this is not silent lip reading.
- Offline processing is expensive: long videos can take much longer than their runtime on CPU, and the 640×360 sample needs about 9.8 GB of temporary frame storage.
- Burned-in captions use a hard-coded macOS font path (`CAPTION_FONT_PATH` in `src/config/rendering.py`), so offline rendering on other systems needs that path changed.
- Review important captions before publishing them.

## Development checks

```bash
.venv/bin/python -m unittest
.venv/bin/python -m compileall -q src tests
node tests/live_client_test.cjs
npm --prefix desktop test
```

`tests/replay_live.py` replays a video through the live tracker (options `--gain` for quieter audio, `--shrink` for smaller faces, `--blank START END` to hide the camera, `--reference speech.json` for speech recall, `--dump` for per-step decisions); `tests/evaluate_live.py` scores a saved browser replay against `samples/transcript.txt`; `tests/stream_live.py` streams a video to the live server in real time and writes latency, CPU, memory, accuracy, language and transcript-card metrics to `output/stream-live/`. None of them is part of the unit tests. Long runs need the machine awake (`caffeinate -i` on macOS). pytest is configured in `pyproject.toml` but is not a dependency. `src/core/vision/person_segmentation.py` (YOLO) is only used by its tests, not by either pipeline.

## Project map

```text
src/
  core/
    audio/            realtime Vosk and multilanguage selection (realtime.py), speech gate
                      (activity.py), language ID (language.py), model storage, offline Whisper
    vision/           live face/mouth/identity, offline faces and Light-ASD
    tracking/         live speaker estimates, identity, voice gallery, offline association rules
    contracts.py      domain data
    media.py          offline FFmpeg and frame access
  config/             model catalog, tuning, transport limits, display settings
  app/
    live.py           live command-line startup
    offline.py        offline pipeline and command line
    service.py        channel coordination
    server.py         HTTP transport
    sessions.py       per-capture state
    rendering/        display payloads, pages, offline video rendering
    templates/        live HTML
    static/           PCM buffer, worklet, transport controller, live renderer
  live.py, main.py    compatibility entry points
desktop/              Electron app shell (window, permissions, backend lifecycle, packaging)
scripts/              install-desktop-app.sh
tests/                unit tests, browser-client test, replay tools
progress/             reports; 06–08 describe the current live design and its evaluation, earlier ones older designs
```
