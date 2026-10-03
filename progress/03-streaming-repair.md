# Streaming repairs and sample check — 2026-10-03

The confirmed data-loss, timing, session, motion-evidence, and speaker-boundary defects from the assessment are repaired. The local browser app was exercised with real audio and video from `samples/giuseppeconte.mp4`. This is still a probabilistic captioning system: the supplied passage contains recognition errors, delayed captions, and ambiguous person masks. The changes add separate silhouettes in the crowded scene but do not establish production accuracy or mask quality on a labelled dataset.

Used ECC's defect workflow and Python verification guidance with Ponytail: reused the standard-library server, browser MediaRecorder, and installed MediaPipe, Whisper, SpeechBrain, and Light-ASD. No dependencies were added. No Git commit or push was made.

## Changes corresponding to the assessment

| Finding                                                                | Repair                                                                                                                                                                                                                                                                                                                                                                      | Evidence and remaining limit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Successful responses discarded after newer rejected uploads            | One browser upload at a time; cumulative pending recordings coalesce while retaining their audio. Retryable responses retain the latest complete recording. Valid results belong to a session, rather than the newest submitted number. Explicit retry covers other failures.                                                                                               | Executed JavaScript regressions hold a successful response while newer recordings arrive and simulate 429. Real replay had maximum upload concurrency one, drained after Stop, and enabled restart. This verifies queue behavior, not bit-perfect recording coverage on every browser.                                                                                                                                                                                                                                                                                |
| Independent clips and competing text engines                           | Removed browser SpeechRecognition. Continuous recorder windows overlap for four seconds; cumulative updates default to four seconds. Whisper receives three seconds of preceding audio, recent transcript context, and word timestamps. One-second provisional tails defer boundary words; overlap text anchors handle revised timestamps. Stop commits the remaining tail. | Word-boundary, repeated-text, revised-timestamp, and final-tail regressions pass. Real transcription was compared with the supplied timestamped text. Whole-phrase ASR rewrites can still defeat the simple anchor. This is contextual batch inference, not a streaming ASR decoder with voice-activity endpointing.                                                                                                                                                                                                                                                  |
| Body rectangles presented as silhouettes                               | Current-frame YOLO11n instance segmentation produces separate person masks, decoded with installed OpenCV. Face coverage, center inclusion, assignment margin, and one-face-per-mask checks select conservative contours. Ambiguous matches use the current face outline.                                                                                                   | Touching-person, duplicate-detection, nonperson, letterbox, contour-shape, and weak-match regressions pass. The first semantic segmentation approach failed the crowded sample and was replaced. Real frames at 0/15/25/40/50 seconds produced separate people; Conte had a contour at four of five sampled frames (face detection missed him at 40 seconds). A [browser screenshot](../output/streaming-check/instance-browser.png) shows the speaking silhouette. Mask bleed and occlusion remain probabilistic; no labelled pixel-quality benchmark was performed. |
| Frozen geometry, simultaneous sequential captions, clip-relative times | JPEG tracking runs independently of delayed speech inference. Capture-time history maps recognition tracks to current IDs. Current geometry expires after 1.5 seconds. Captions retain relative spoken-time order, overlays deduplicate by person, and transcript times use the capture clock. Sessions isolate state; `/end` releases native tracking resources.           | Browser checks confirm a moving outline updates the same DOM node, retains keyboard focus, and disappears when geometry expires. Replay crossed a recorder rotation and stopped cleanly. Current outlines are spatially current, while their text is delayed; they do not imply that the person is speaking at that exact preview instant.                                                                                                                                                                                                                            |
| Repeated images and global activity-track cap                          | All decoded frames are retained for Light-ASD; detector sampling is separate and includes the last frame. Every stable track is scored. Full frames use temporary disk-backed arrays instead of occupying RAM.                                                                                                                                                              | Consecutive-frame, final-frame, and activity-candidate regressions pass. The full 568.32-second sample would use about 9.8 GB of temporary frame storage; the full video was not inferred in this check.                                                                                                                                                                                                                                                                                                                                                              |
| Position-only identity and switches inside Whisper segments            | Session ECAPA centroids retain voice identities. Word-level visual attribution forms speaking runs inside a Whisper segment before voice embeddings are computed. Brief replies retain their distinct face identity when too short for an embedding. Weak or absent visual evidence remains uncertain/off-screen.                                                           | Regressions distinguish two voices inside one segment, retain one voice after a face ID changes, preserve short replies, and retain uncertain/off-screen speech. The final replay also exposed two wrong Person 2 events during Giuseppe's speech; persistent embeddings and word-level attribution do not eliminate model errors. Crossings, occlusion, and simultaneous voices remain probabilistic; no real multi-speaker benchmark was performed.                                                                                                                 |
| Avoidable setup and unmeasured latency                                 | Whisper, Light-ASD, and ECAPA warm before serving. Native live face detectors and the instance network are cached; the network serializes mutable inputs across sessions. Normalization seeks to needed context rather than repeatedly processing the entire cumulative history. Every response reports stage durations.                                                    | Real measurements below identify Light-ASD as the largest stage. FFmpeg normalization and audio extraction remain separate bounded operations. No latency target was given, and interactive low latency has not been achieved on this CPU.                                                                                                                                                                                                                                                                                                                            |

## Real browser replay

Ran the local server with cached weights, Italian, Whisper `small`, and four CPU threads on the available Apple M3:

```bash
MPLCONFIGDIR=/private/tmp/lip-reading-matplotlib HF_HUB_OFFLINE=1 \
  OMP_NUM_THREADS=4 .venv/bin/python -m src.live --language it --port 8765
```

Used FFmpeg to copy the first 60 seconds into a browser fixture:

```bash
ffmpeg -y -i samples/giuseppeconte.mp4 -t 60 -c copy /private/tmp/lip-reading-browser.mp4
```

Playwright supplied that file through a file input, played it in an HTML video, and returned `video.captureStream()` from the fixture's `getUserMedia`. Both audio and video tracks were present. The real Start/Stop controls, MediaRecorder, HTTP endpoints, and model pipeline ran unchanged. Capture stopped at source time 52.02 seconds, crossing the 30-second recorder window with its four-second overlap. Fetch interception saved responses and timestamps; DOM snapshots recorded captions and highlights every 500 milliseconds. The first immediate state read ran before asynchronous camera startup; a subsequent check confirmed both tracks and active capture.

Saved evidence is in ignored `output/streaming-check/final-instance.json`. Recompute its comparison with the supplied `samples/transcript.txt`:

```bash
.venv/bin/python tests/evaluate_live.py output/streaming-check/final-instance.json
```

| Measurement                                            |                                Final run |
| ------------------------------------------------------ | ---------------------------------------: |
| Completed reference interval                           |                             0–52 seconds |
| Reference / recognized normalized words                |                                  99 / 96 |
| Word error rate                                        |                                   12.12% |
| Median / maximum recognition processing                |                     6.57 / 10.15 seconds |
| Median / maximum caption response arrival delay        |                    14.04 / 19.52 seconds |
| Median / maximum observed caption display delay        |                    18.19 / 20.97 seconds |
| Caption events observed on screen                      |                                 18 of 19 |
| Successful recognition updates                         |                                        9 |
| Maximum simultaneous recognition uploads               |                                        1 |
| Current-frame tracking responses                       |                                      141 |
| Tracking responses containing any contour              |                                      141 |
| Speaking-highlight snapshots / those using silhouettes |                                  48 / 43 |
| Attributed person labels                               | Person 1 and Person 2 (two false events) |
| Stop buffer drained / restart enabled                  |                                Yes / yes |

Arrival delay means response-received time minus the event's spoken end. Display delay uses the first 500 ms DOM sample showing that complete event after its response arrived; very short events can fall between samples. Captions are queued in spoken order, so display delay exceeds arrival delay. This distinction prevents a faster HTTP response from being misreported as faster visible captions.

The comparison lowercases words, removes punctuation, and normalizes Italian number words zero through ten. Reference sections are included only when their supplied end time is complete. The reference has coarse timings and was not independently corrected or labelled for identity/masks. Errors include missing “anno” and “sveglia” and “braga di tela” for “braghe di tela.” Earlier development runs varied between approximately 12% and 17%; they are not a controlled original-version accuracy baseline. No improved word-error rate is claimed. Two of 19 caption events incorrectly attribute speech to the far-right face as Person 2 (around 37–39.8 seconds). This manually observed error is not a calibrated multi-speaker score. The default activity thresholds and greedy voice matching still require labelled calibration; raising thresholds from this single passage would not establish general accuracy.

| Stage            |  Median | Maximum |
| ---------------- | ------: | ------: |
| Normalization    | 0.461 s | 0.621 s |
| Audio extraction | 0.039 s | 0.048 s |
| Decode           | 0.257 s | 0.303 s |
| Whisper          | 2.447 s | 2.959 s |
| Voice clustering | 0.083 s | 0.132 s |
| Face analysis    | 0.173 s | 0.277 s |
| Light-ASD        | 3.301 s | 6.317 s |

Stage medians do not add to the median total. Temporary I/O, probes, and association contribute to the total. Removing the activity cap preserves candidate coverage but increases work. A recording stops and drains if recognition falls more than 60 seconds behind; it does not silently drop old audio to claim real-time behavior.

## Verification and limits

```bash
MPLCONFIGDIR=/private/tmp/lip-reading-matplotlib .venv/bin/python -m unittest -q
.venv/bin/python -m compileall -q src tests
node tests/live_client_test.cjs
git diff --check
```

The Python suite has 64 passing tests, including rejection of corrupt model weights before loading native code. The Node checks execute the actual browser source with controlled asynchronous uploads and DOM state. A native-browser check separately verified current geometry, expiry, and focus retention. Live normalization was also exercised with real cumulative WebM: a 30-second stream succeeds after seeking to required context, while an oversized 34-second stream is rejected. Live limits are 32 seconds of recording, 1280×720 pixels, 25 fps normalized video, and bounded subprocess timeouts. The package build includes the browser JavaScript resource.

Remaining work depends on concrete requirements and labelled recordings: choose an acceptable latency budget, evaluate mask quality on labelled crowded footage, and evaluate multi-speaker identity/attribution and overlapping speech. This implementation decodes words from audio; it does not add silent visual lip reading. The offline renderer still uses estimated upper-body rectangles.

The original `02-current-app-assessment.md` was removed and `progress/.md` appeared during concurrent workspace changes. Neither that removal nor `.md` was edited or restored as part of these repairs.

## Silhouette model provenance

The 11.7 MB [YOLO11n-seg ONNX export](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/blob/0d17b24163fbc33dec51d811d7be8db15b8df274/yolo11n-seg.onnx) is pinned to repository revision `0d17b24163fbc33dec51d811d7be8db15b8df274` and SHA256 `6481a5fea55b9973c25752870587ec9163c1e6f778884e6c8befd4eff586080f`. Downloads are checked and atomically installed; cached weights are checked at initial load. The model uses separate boxes and mask prototypes described by the [Ultralytics segmentation reference](https://docs.ultralytics.com/reference/models/yolo/segment/predict). OpenCV runs inference; NumPy and OpenCV decode masks without installing Ultralytics or an additional ONNX runtime.
