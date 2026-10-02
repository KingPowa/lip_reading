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

The server loads Whisper before it prints the address. Grant camera and microphone access, confirm the spoken language (it defaults from your browser locale), and click **Start camera**. The camera never replays recordings: it remains the only video shown. The app continuously sends two-second windows for local transcription and speaker attribution, then places the resulting text and coloured speaker highlight on that live preview. Each detected speaker keeps the same colour for the camera overlay, caption, and transcript entry. Hover or focus a highlight to read that person's speech.

Chromium-based browsers show their native live transcript immediately; local Whisper then supplies the speaker-coloured attribution. Other browsers still receive local caption results after each analysis window. To stay responsive, live attribution keeps face identities between windows instead of running the slower offline voice-clustering stage. Captions arrive after the local analysis time for each short window, rather than after a recording is played back. Use `--chunk-seconds` (1–10) to trade latency for more speech context. Select a language in the page, or lock one with `--language it`; this prevents short clips from being misdetected and Whisper is explicitly run in transcription mode, never translation mode.

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

| File | Open this when you want… |
| --- | --- |
| `output.annotated.mp4` | The finished video, with coloured upper-body highlights and matching captions burned in. |
| `captions.vtt` | Captions to load in a compatible video player. |
| `viewer.html` | A local review page: seek from the colour-matched transcript and inspect active upper-body overlays. Keep it beside the video. |
| `events.json` | Final speaker, text, face-track, confidence, state, and display colour for each speech event. |
| `speech.json` | Raw Whisper words plus anonymous voice clusters. Use it to diagnose transcription or diarization. |
| `faces.json` | Detected face tracks and their timestamped boxes. Use it to diagnose visual tracking. |
| `audio.wav` | The 16 kHz mono audio extracted for the speech and active-speaker models. |

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

Only the visible speaker is drawn. The existing face track estimates an upper-body rectangle, so the overlay remains lightweight without loading a pose or segmentation model. Faces that are merely visible do not receive a neutral box.

## Improving a result

Use this quick triage order:

1. **Words are wrong:** use `--language it` for Italian. `large-v3` is already the default; use `small` or `medium` only when faster processing matters more than transcription accuracy. Check that the source audio is clear and not dominated by music or applause.
2. **Wrong person is highlighted:** inspect `faces.json` and `events.json` at the same timestamp. The pipeline only evaluates the eight largest face tracks, which keeps local processing practical but can miss a small or distant speaker.
3. **No person is highlighted:** this is usually intentional `uncertain` or `off_screen` output. Check whether the speaker’s face is actually visible.
4. **Captions look cut off:** rerun with the current version. Captions are wrapped using their actual rendered width, not an approximate character count.

## Limits to expect

- Camera mode is continuous short-segment recognition. Captions and highlights arrive after local analysis of the current window; the preview is always live and no recorded audio or video is played back.
- It does not perform real-name recognition or cross-video identity matching.
- Overlapping speech is marked when detected, but exact transcription of two people talking at once remains difficult.
- Accurate local transcription and active-speaker scoring are computationally expensive. A long video can take significantly longer than its runtime on CPU.
- To keep memory bounded, visual analysis samples every fifth source frame while retaining the source timestamps. Rendering separately streams every source frame, so the final annotated video keeps the original duration, frame rate, and audio.
- Burned-in captions use a Unicode-capable font, so Italian accents and apostrophes render as text instead of replacement characters.
- The result is evidence-based, not a ground-truth transcript. Review important captions before publishing them.

## Development checks

```bash
.venv/bin/python -m unittest -v
.venv/bin/python -m compileall -q src tests
```

The tests cover association confidence, upper-body overlay payloads, colour handoff, and caption width.

## Project map

```text
src/main.py          command-line pipeline
src/live.py          local camera server and recognizer endpoint
src/live_view.py     browser camera, highlights, and transcript interface
src/colors.py        stable speaker colour palette
src/media.py         FFmpeg/FFprobe validation, timestamps, audio extraction
src/speech.py        Whisper transcription and ECAPA voice clustering
src/vision.py        MediaPipe face detection and tracking
src/light_asd.py     audio-visual active-speaker probabilities
src/association.py   speaker-to-face decision rules
src/render.py        video, VTT, JSON, and review-page output
progress/            plain-language notes for each improvement decision
```
