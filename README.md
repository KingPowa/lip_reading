# Active-Speaker Captioning

Turn a prerecorded video or camera recording into an annotated copy with:

- readable, speaker-labelled captions;
- a coloured upper-body highlight around each visible person who is speaking;
- a `.vtt` subtitle file for a video player; and
- JSON files that make it easy to inspect what the pipeline decided.

It names people `Person 1`, `Person 2`, and so on **within one recording**. It does not identify real people or remember anyone between recordings.

## Use your camera

Launch the local camera app, then open the printed address in a browser:

```bash
uv run --python .venv/bin/python -m src.live
```

The server loads Whisper, Light-ASD, ECAPA, and the person instance model before it prints the address. The first launch downloads missing weights, including the 11.7 MB silhouette model. Grant camera and microphone access, confirm the spoken language (it defaults from your browser locale), and click **Start camera**. The preview remains live. Capture uses overlapping continuous recordings, with cumulative updates every four seconds by default. One upload runs at a time; pending updates combine without dropping their captured audio, and busy responses are retried. Stop finishes buffered speech before enabling another capture session.

Local faster-whisper is the single text source for live captions, using CPU INT8 inference. Its uncommitted tail is labelled **Provisional**; word timestamps, three seconds of preceding audio, and text-overlap reconciliation connect successive updates. Live mode defaults to `base`; choose `small` or larger for more recognition detail. Light-ASD attributes individual words to visible faces and forms speaking runs inside transcription segments. ECAPA keeps anonymous voice labels for those runs within the capture session. `--chunk-seconds` (1–10) controls upload cadence. Select a language in the page or use `--language it` for Italian.

Current-frame face detection and person instance segmentation run independently of speech recognition. A coloured silhouette follows the current position of the last attributed speaker, including when people touch. If occlusion or competing masks make the match ambiguous, the app shows the current face outline. Geometry expires rather than freezing at an old position. Hover or focus a highlight to read the attributed speech. Preview captions keep spoken-time order within an update and yield to newer results; long batches fit within the upload cadence. The transcript retains every caption with its session timestamp.

Light-ASD automatically uses an available CUDA or Apple MPS GPU, warms its inference before capture, and falls back to CPU if warm-up fails. Use `--activity-device cpu` to force CPU inference. OpenCV runs without its competing parallel pool. Startup logs show the selected backend and thread counts; each recognition update logs its capture interval and stage timings. Whisper consumes the already extracted PCM samples directly. FFmpeg and FFprobe use process spawning to avoid the observed OpenBLAS/fork deadlock in the threaded server.

If recognition falls more than 60 seconds behind, capture pauses and drains its retained recordings. Errors retain the failed upload and expose **Retry recognition**. On the tested Apple M3, four Torch CPU threads were used:

```bash
OMP_NUM_THREADS=4 uv run --python .venv/bin/python -m src.live --language it
```

The default `http://127.0.0.1:8765` works on the computer running the app. To use another device's camera, bind the server to your network and supply a trusted TLS certificate (browsers require HTTPS before they grant camera access to a non-localhost page):

```bash
uv run --python .venv/bin/python -m src.live --host 0.0.0.0 --cert certificate.pem --key private-key.pem
```

Open the resulting `https://YOUR-COMPUTER-IP:8765` address on that device. Press `Ctrl-C` in the terminal to stop the app. For a shell executable after `uv sync`, run `live-speaker` instead.

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

Only attributed speakers are highlighted. The offline renderer estimates an upper-body rectangle from each face. The live preview uses current person contours with a face-outline fallback when segmentation is ambiguous.

## Improving a result

Use this quick triage order:

1. **Words are wrong:** use `--language it` for Italian. `large-v3` is already the default; use `small` or `medium` only when faster processing matters more than transcription accuracy. Check that the source audio is clear and not dominated by music or applause.
2. **Wrong person is highlighted:** inspect `faces.json` and `events.json` at the same timestamp. Every stable face track is evaluated. Attribution thresholds still need calibration against labelled multi-speaker recordings.
3. **No person is highlighted:** this is usually intentional `uncertain` or `off_screen` output. Check whether the speaker’s face is actually visible.
4. **Captions look cut off:** rerun with the current version. Captions are wrapped using their actual rendered width, not an approximate character count.

## Limits to expect

- Camera captions arrive after local analysis; processing time and backlog appear in the status. Current visual tracking does not make transcription instantaneous.
- It does not perform real-name recognition or cross-video identity matching.
- Overlapping speech is marked when detected, but exact transcription of two people talking at once remains difficult.
- Accurate local transcription and active-speaker scoring are computationally expensive. A long video can take significantly longer than its runtime on CPU.
- Active-speaker inference receives consecutive images; face detection samples separately and includes the last frame. Images use temporary disk-backed memory maps. The full 640×360 sample needs about 9.8 GB of temporary frame storage. Live recordings are limited to 32 seconds, 1280×720 pixels, and 25 fps.
- Voice clustering and face matching remain probabilistic. Occlusion, camera cuts, overlapping voices, and imperfect person masks can produce uncertain attribution or identity changes. No instance-mask quality benchmark or multi-speaker error-rate benchmark is included.
- Words are decoded from audio; this is not a visual-to-text model for silent lip reading.
- Burned-in captions use a Unicode-capable font, so Italian accents and apostrophes render as text instead of replacement characters.
- The result is evidence-based, not a ground-truth transcript. Review important captions before publishing them.

## Development checks

```bash
.venv/bin/python -m unittest -v
.venv/bin/python -m compileall -q src tests
node tests/live_client_test.cjs
```

The tests cover asynchronous capture and retry, session isolation and cleanup, word commitment, voice continuity, current geometry and mask association, consecutive-frame evidence, rendering, and media limits. A saved browser replay can be compared with the supplied transcription using `python tests/evaluate_live.py output/streaming-check/review-final.json`. See [the faster-whisper replay](progress/05-live-asr-performance.md) for current live latency and its accuracy tradeoff, [the latency investigation](progress/04-streaming-latency-investigation.md) for prior measurements and deadlock evidence, and [the earlier repair report](progress/03-streaming-repair.md) for the preceding implementation.

## Project map

```text
src/main.py          command-line pipeline
src/live.py          local camera server and recognizer endpoint
src/live_view.py     browser page
src/live_client.js   capture queue, current highlights, and transcript interface
src/live_tracking.py current face identities and person-mask association
src/person_segmentation.py cached OpenCV person instance masks
src/colors.py        stable speaker colour palette
src/media.py         FFmpeg/FFprobe validation, timestamps, audio extraction
src/speech.py        Whisper transcription and ECAPA voice clustering
src/vision.py        MediaPipe face detection and tracking
src/light_asd.py     audio-visual active-speaker probabilities
src/association.py   speaker-to-face decision rules
src/render.py        video, VTT, JSON, and review-page output
progress/            plain-language notes for each improvement decision
```
