# Streaming latency and missing highlights — 2026-10-03

The replay reproduced both symptoms. There are two different failure modes: expensive serial inference builds a backlog, and a native process-creation deadlock can stop both recognition and tracking completely. Person segmentation itself returned contours successfully; that did not guarantee a speaking highlight would reach the screen.

Used ECC's latency-critical-systems skill with Ponytail. Reused the installed models and standard-library media/server code. No dependencies, commits, or pushes were added. The requested assessment is absent from the working tree, so its tracked version was read; its deletion and `progress/.md` were left untouched.

## Confirmed causes and repairs

1. **The server can freeze while starting FFmpeg/FFprobe.** During the `base` comparison, responses stopped at about 31 seconds of capture. The [native stack sample](../output/streaming-check/review-base-stacks.txt) shows a request inside `subprocess_fork_exec → fork → OpenBLAS blas_thread_shutdown → pthread_join`, while another request is performing mask matrix multiplication. Other Python threads wait for the GIL. This explains simultaneous loss of face tracking and captions without a model error. The shared media runner now resolves the executable and uses the standard-library `posix_spawn` path. Python-created sockets/files remain non-inheritable. Whisper now receives the already extracted PCM array, eliminating its additional FFmpeg process. A regression forbids the fork path while running a real FFprobe command. The fixed comparison crossed the recorder rotation, drained, and released its session.

   The installed CPython 3.12 source confirms the spawn prerequisites. [Python's subprocess documentation](https://docs.python.org/3/library/subprocess.html) also explains why a subprocess timeout cannot reliably interrupt initial process creation; increasing the previous timeout would not fix this freeze.

2. **Visual inference delays text, and repeats already committed context.** The original replay processed an update in 31.25 seconds; Light-ASD used 18.86 seconds of that time. Speech, face analysis, activity scoring, and voice clustering execute before one combined response. Light-ASD now automatically uses an available CUDA or Apple MPS GPU, with a real inference warm-up and CPU fallback if warm-up fails. It scores all candidate faces only around newly committed words, retaining one second of temporal context. Whisper keeps its three seconds of audio context. Old words are no longer voice-clustered again. Consecutive images and the models' input resolution are preserved.

   Local CPU/MPS checks on 100- and 250-frame inputs differed by at most approximately `1.7e-6` in logits. This checks numerical agreement, not model accuracy. The architecture was checked against the [published Light-ASD implementation](https://github.com/Junhua-Liao/Light-ASD/blob/main/model/Model.py).

3. **Native CPU pools compete.** OpenCV reported eight threads alongside four Torch threads. On this build's GCD backend, `setNumThreads(1)` still reports eight; `setNumThreads(0)` reports one and disables parallel execution. Live startup applies the latter and logs the effective backend/thread counts. Each completed recognition also logs capture bounds, commitment, track/event counts, and stage timings.

4. **The browser adds another playback backlog.** Previously, every response waited behind the full spoken duration of earlier responses. In the reproduced failure, median caption arrival delay was 44.09 seconds and median display delay was 56.02 seconds. Preview captions now yield to a newer update and compress long batches into one upload cadence, preserving order within the update. Every event remains in the transcript. This bounds the preview queue; it does not make inference instantaneous or guarantee every brief event remains on screen for its original duration.

5. **A detector miss breaks the highlight's identity.** Conte's face disappeared from detections around 40 seconds, then returned with a new ID. The old two-second retention discarded his identity while delayed captions still referenced it. Recognition also consumed a historical ID once per recording, leaving later non-overlapping fragments unmatched. Brief misses now retain identity for five seconds, without returning stale geometry. Sequential fragments can reuse their historical identity; overlapping competing faces cannot. Position-based continuity can still confuse replacements/crossings.

6. **Weight loading is not inference warm-up.** The initial Whisper job was slower than subsequent jobs. Startup now performs inference on a short silent PCM array before opening the server. Whisper remains on its supported default backend. A local attempt to use MPS with word timestamps failed with an unsupported float64 conversion, so no Whisper GPU workaround was shipped.

## Browser replay and transcription comparison

The fixture is the first 60 seconds of `samples/giuseppeconte.mp4`, copied with FFmpeg. The browser plays it and supplies its actual audio/video capture stream to `getUserMedia`. The app's MediaRecorder, uploads, tracking, real models, Stop, and session cleanup run normally. Capture stops just after source time 52 seconds, crossing the 30-second recording rotation. The comparison uses the first 99 normalized words in `samples/transcript.txt`; this supplied text has errors and is not a manually verified gold transcript.

`output/streaming-check/replay-browser.js` records HTTP responses and 250 ms DOM snapshots. It can be run with the ECC Playwright `browser_run_code_unsafe` tool against the local page. The saved reports include tracking response latency, current and attributed IDs, contour points, captions, and upload concurrency. [The screenshot](../output/streaming-check/review-highlight.png) shows Conte highlighted during a real replay. Its mask also includes part of a nearby occluded person; pixel-perfect segmentation is not established.

| Measurement                              | Reproduced original (`small`) | Final repairs (`small`) | Fixed faster model (`base`) |
| ---------------------------------------- | ----------------------------: | ----------------------: | --------------------------: |
| Recognition processing, median / maximum |               20.36 / 31.25 s |           5.78 / 8.02 s |               3.52 / 4.54 s |
| Caption arrival delay, median / maximum  |               44.09 / 58.85 s |         11.43 / 15.56 s |               5.73 / 8.54 s |
| Caption display delay, median / maximum  |               56.02 / 62.79 s |         12.43 / 15.69 s |               6.99 / 8.98 s |
| Reference word error rate                |                         9.09% |                  16.16% |                      37.37% |
| Recognition updates                      |                             5 |                      10 |                          15 |
| Highlight / silhouette snapshots         |                       11 / 11 |                 68 / 57 |                   127 / 116 |
| Stop drained / restart enabled           |                     Yes / yes |               Yes / yes |                   Yes / yes |

Final empirical percentiles (seconds):

| Path / stage     |   p50 |   p95 |   p99 |
| ---------------- | ----: | ----: | ----: |
| total            |  5.78 |  7.75 |  7.97 |
| transcription    |  2.77 |  3.51 |  3.64 |
| activity         |  1.78 |  2.61 |  2.68 |
| captionArrival   | 11.43 | 15.03 | 15.45 |
| trackingResponse |  0.17 |  0.27 |  0.36 |

Final run: 117 successful tracking responses, 117 containing contours; maximum concurrent recognition uploads one; no recorded HTTP or fetch failures. The contour count includes all detected people, while speaking-highlight snapshots count visible attributed overlays. A snapshot count is neither mask accuracy nor a proportion of correctly attributed frames.

The initial baseline briefly overlapped the separate model-device microbenchmarks. Its absolute throughput is therefore not an isolated hardware benchmark. The previous repair report independently measured approximately 18 seconds of median display delay. Recorder boundaries and changing backlog also produce different inference windows across runs; compare end-to-end outcomes, not these stage measurements as identical-input microbenchmarks. Sample counts are too small to establish a latency SLA.

The `base` run completed after the spawn repair and reduced median display delay to 6.99 seconds, but its reference error rate was 37.37%. The default remains `small`; the faster model substantially worsened this Italian passage. The final `small` run still attributed “Lei continua” around 37 seconds to the wrong face (`Person 2`), and some later speech remained uncertain. No thresholds were tuned to hide those errors on this one recording. Both remain batch recognition behind live video. Highlights show the current position of a previously attributed speaker, not a fresh audio-visual decision at every preview frame. Faces can still disappear when occluded, masks can bleed, and attribution remains probabilistic.

## Run and inspect

Restart the app to load the repairs:

```bash
OMP_NUM_THREADS=4 uv run --python .venv/bin/python -m src.live --language it
```

Startup should report `Light-ASD=mps` and `OpenCV threads=1` on this Mac. `--activity-device cpu` selects the diagnostic CPU path. `--whisper-model base` selects the measured faster, less accurate alternative.

Saved logs and results are in `output/streaming-check/review-final-server.log`, `review-final.json`, and `review-final-metrics.json`. The reproduced failure is in `review-before.json`; the deadlock evidence is in `review-base-stalled.json` and `review-base-stacks.txt`; the fixed lighter-model comparison is in `review-base-fixed.json`. Recompute accuracy/delay with:

```bash
.venv/bin/python tests/evaluate_live.py output/streaming-check/review-final.json
```

Validation: 71 Python tests pass, including actual spawn verification and PCM input, GPU fallback, bounded visual scoring, fragment identity, and detector-miss regressions. Node checks verify caption queue bounds/order, retention of every transcript event, serialized uploads, retries, and current geometry. Compilation, formatting, production-code lint checks, and `git diff --check` pass. This checks the first 52 seconds of one recording, not the entire 568-second video or a multi-speaker benchmark. Remaining latency and accuracy limits are measured above rather than treated as solved by unit tests.
