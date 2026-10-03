# Live ASR performance — 2026-10-03

The prior repair still had a median **12.43 s** delay from spoken word to visible caption. The largest stages were Whisper transcription (2.77 s median) and Light-ASD (1.78 s median) on each short update. Each upload was only one part of the delay; media normalization and queueing added more.

## RealtimeSTT fit

The adjacent `realtime_stt` project uses RealtimeSTT 1.1.2 around faster-whisper base, CPU INT8, and Silero VAD. Its recorder consumes microphone PCM and returns preview/final text. This app receives timestamped browser video/audio uploads and needs word timestamps to align words with faces. Using `AudioToTextRecorder` here would require a second persistent audio-feed/VAD lifecycle and would still need a word-alignment pass. The smaller fit is faster-whisper directly, retaining this app's word-to-face pipeline and browser capture.

## Change

Live recognition now uses faster-whisper 1.2.1 with the same Systran base model already cached for RealtimeSTT. CPU INT8 is used because CTranslate2 has no Apple MPS device. Earlier transcript text remains available to overlap reconciliation, but it is not supplied as an ASR prompt: the first replay showed that prompting partial windows caused repeated opening words later in the recording. Whisper remains in use for offline rendering. No RealtimeSTT dependency or microphone recorder was added.

## Browser replay

Replayed the first 52 seconds of `samples/giuseppeconte.mp4` through the browser's real MediaRecorder, live HTTP server, models, face tracker, Stop, and session cleanup. The supplied transcript is a rough reference, not verified ground truth.

| Measurement | Prior live small Whisper | faster-whisper base |
| --- | ---: | ---: |
| Recognition work, p50 / p95 | 5.78 / 7.75 s | **3.33 / 5.44 s** |
| Transcription stage, p50 | 2.77 s | **1.24 s** |
| Caption arrival delay, p50 | 11.43 s | **6.89 s** |
| Caption display delay, p50 | 12.43 s | **7.62 s** |
| Reference WER | 16.16% | 23.23% |

The updated replay completed 15 successful recognition updates, serialized uploads (maximum concurrency one), 118/118 tracking responses with contours, and drained speech on Stop with restart enabled. The maximum processing interval was 6.82 s; p95 caption display delay was 10.83 s. Recognition and speaker attribution remain probabilistic. This single passage shows a speed/accuracy tradeoff, not a general quality result.

Replay data: [faster-whisper run](../output/streaming-check/faster-whisper-replay-noprompt.json). The first test with ASR context prompts is retained as [the rejected run](../output/streaming-check/faster-whisper-replay.json) (repeated phrase, 50.5% WER).

## Run

After syncing the new dependency, start with:

```bash
OMP_NUM_THREADS=4 uv run --python .venv/bin/python -m src.live --language it
```

Use `--whisper-model small` to request the faster-whisper small model if its extra accuracy is worth the added model size and processing time. The default is base for latency. Offline `src.main` continues to use OpenAI Whisper.

Validation: 72 Python tests pass; live-client JavaScript checks, compilation, `uv lock --check`, and `git diff --check` pass. The replay used the cached faster-whisper base weights; `uv sync` then installed the pinned dependency into this workspace venv.
