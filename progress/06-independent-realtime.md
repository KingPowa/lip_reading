# Independent real-time camera and microphone pipelines — 2026-10-03

The previous live app collected video on a four second cadence and ran video normalization, Whisper, face analysis, Light-ASD, and voice clustering before returning captions. Speaking highlights depended on the lifetime of those delayed caption events. Reducing one model's inference time did not remove that dependency.

Applied ECC's `latency-critical-systems` and `python-patterns` skills. The refactor separates current audio, current vision, speaker estimation, HTTP/session coordination, and presentation. No commits or pushes were made.

## Current live path

```text
microphone → AudioWorklet → 100 ms mono PCM packets
                         ├─ /audio → persistent Vosk decoder → provisional/final text
                         └─ /activity → current microphone energy
camera → small JPEG every ~100 ms → /track → faces and mouth aperture
                                               ↓
                              rough box continuity + recent mouth movement
                                               ↓
                              current speaker / uncertain / off-screen
```

Audio and camera inference have separate session locks. The short shared state lock only updates measurements and estimates; no native inference runs while holding it. Regression tests block each native path in turn and prove the other path completes. Transcription never invokes face detection, video decoding, segmentation, Light-ASD, or voice embeddings.

Vosk consumes every PCM packet once, retaining its decoder state. The browser retains unacknowledged audio and retries the same job identifier. The server returns its cached response for a retried job, preventing duplicate decoder input. `acknowledgedUntil` means PCM accepted by the decoder, not a finalized word boundary. Speech cannot disappear because the camera finds no face. Final utterance results have word times; provisional text is visible in both the caption and transcript before finalization.

The browser sends little-endian 16-bit PCM directly; FFmpeg is absent from the live path. Worklet resampling preserves its phase between input blocks at 44.1 and 48 kHz. Stop flushes the final partial worklet packet, submits remaining PCM, flushes the decoder, waits for camera/activity requests, and releases the session. A twelve second buffer limit pauses capture and drains retained speech rather than silently discarding it.

Live detection uses the bundled full-range face detector and mouth landmarks from small face crops. It considers up to four large faces, matches current boxes by overlap, ignores missing mouth landmarks, and scores recent normalized mouth variation. Relative face size and a small continuity bonus reduce jitter from distant faces. Microphone energy gates speaking estimates. Ambiguous competing mouths yield `uncertain`; voice without a moving visible mouth is estimated `off_screen`. Stale camera evidence yields `uncertain`. Highlights read these current estimates, independently of caption expiry.

## Structure

- `src/core/audio`: streaming PCM/decoder behavior, model loading, existing offline speech.
- `src/core/vision`: current face/mouth observations and existing offline visual inference.
- `src/core/tracking`: current estimates and offline association rules.
- `src/config`: model catalog, thresholds, resource limits, transport, rendering settings.
- `src/app`: command startup, HTTP, session lifecycle, channel coordination.
- `src/app/rendering`, `templates`, `static`: public display payloads, pages, renderer, and browser capture/transport.

`src/live.py` and `src/main.py` are small compatibility command entry points. Existing offline Whisper/ECAPA/Light-ASD behavior remains available. The legacy segmentation helpers, tests, and faster-whisper adapter were preserved: automatic approval review rejected their proposed removal because it could affect offline functionality. They are absent from the current live inference path.

## Measurements

Replayed the first 45 seconds of the supplied video through the real browser AudioWorklet, HTTP server, Vosk Italian small model, face/mouth detection, rendering, and Stop. This is one Apple M3 run, not an accuracy benchmark or a hardware-independent latency guarantee.

| Measurement | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Audio response age since packet end | 6 ms | 20 ms | 42 ms | 91 ms |
| Current camera response age since capture | 39 ms | 47 ms | 55 ms | 66 ms |
| Incremental speech processing | 5 ms | 17 ms | 36 ms | 90 ms |
| Audio activity response age since packet end | 2 ms | 6 ms | 7 ms | 8 ms |

There were 315 successful audio updates, 446 camera updates, and 451 activity updates, with no failed requests. Maximum unacknowledged audio was 200 ms. Stop drained the stream and enabled restart. Current highlights appeared before the first provisional caption. The first provisional phrase appeared about 0.8 seconds after the decoder's estimated start of the first word; the first outline appeared about 0.4 seconds after that start.

The later 20 second smoke replay included the final face-size weighting and domain/presentation separation. It completed 127 audio and 199 camera updates without errors; maximum observed audio response age was 74 ms, camera response age 63 ms, and unacknowledged audio 100 ms. A final eight second browser replay after the stream acknowledgement naming and startup changes transcribed speech, drained the queue to zero on Stop, and enabled restart. The only browser console error was a missing favicon.

Raw replay artifacts live under the ignored `output/realtime-check/` directory: `streaming-replay-final.json` and `streaming-replay-smoke.json`. The fixture was the existing `/private/tmp/lip-reading-browser.mp4`, previously extracted from the supplied video.

These timings describe fresh packet processing, not word correctness or the time needed to disambiguate every word. Provisional text can revise. A long utterance can finalize much later at silence or Stop; current text is already visible. The simple mouth/energy heuristic can confuse background noise, listeners moving their mouths, small faces, camera cuts, yawns, and overlapping voices. No claim of calibrated multi-speaker accuracy is made. The one second target is supported by the measured pipeline and initial response on this fixture; it is not proven for every word or device.

## Run and checks

```bash
uv sync
uv run --python .venv/bin/python -m src.app.live --language it
```

Small language models download once from the [official Vosk catalog](https://alphacephei.com/vosk/models), before microphone capture begins. Italian is about 48 MB compressed. macOS uses the published universal Vosk 0.3.42 wheel; other platforms use 0.3.45. Existing model files remain in `models/`. `--speech-model` selects a custom compatible local Vosk model. Live `--chunk-seconds` is now a floating-point PCM update interval, default 0.1 seconds; offline `--whisper-model` remains available.

Validation: 75 Python tests; browser controller/buffer/worklet checks; source compilation; Ruff import/lint and formatting; Prettier for browser assets; dependency lock check; wheel/sdist build and packaged-asset inspection; Git whitespace check. Camera warm-up failure does not prevent microphone startup. The preserved offline tests pass after their imports move to the new packages.
