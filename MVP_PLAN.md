# Active-Speaker Captioning MVP Plan

## Goal

Process one prerecorded video with embedded audio and produce:

```text
input.mp4
  -> output.annotated.mp4
  -> captions.vtt
  -> events.json
```

For the MVP, recognising people means distinguishing participants within the
video as `Person 1`, `Person 2`, and so on. Real-name identification and
cross-video recognition are outside the MVP.

## Scope

### Included

- Detect and track visible faces.
- Detect which visible face is speaking.
- Distinguish recurring speakers with stable labels within the video.
- Transcribe speech and attribute it to those labels.
- Highlight every active speaker.
- Prefix captions with the corresponding speaker label.
- Represent off-screen or uncertain speakers without guessing.
- Highlight multiple people during overlapping speech when detected.

### Excluded

- Webcam and live streaming.
- Real-world identity recognition.
- Cross-video identity memory.
- Mobile applications and user accounts.
- Database or cloud deployment.
- Model training.
- Automatic translation.
- A full caption-editing interface.

## Architecture

```text
                     +-- Whisper ------- words + timestamps
Video -> normalization+-- pyannote ------ voice clusters + turns
                     +-- face tracking -- visible face tracks
                                            |
                                      Light-ASD scores
                                            |
                                voice-to-face association
                                            |
                          annotated video + captions + JSON
```

All components must use the video's presentation timestamps as their common
timeline. Results must not be synchronized using processing time or frame
numbers alone.

## Main components

| Component | Responsibility | Software |
|---|---|---|
| Media preparation | Validate video, extract 16 kHz mono audio, normalize frames, and preserve timestamps | FFmpeg |
| Face tracking | Detect faces and maintain temporary IDs across consecutive frames | MediaPipe and simple intersection-over-union tracking |
| Transcription | Produce words with start and end timestamps | Whisper |
| Voice diarization | Estimate distinct voices and their speaking intervals | pyannote `speaker-diarization-community-1` |
| Active-speaker detection | Score whether each visible face is speaking | Light-ASD |
| Association | Connect voice clusters, face tracks, and words | Project code |
| Rendering | Draw labels and highlights and write named captions | OpenCV and FFmpeg |
| Verification | Test association logic and compare output with a manually labelled video | Python assertions or one small test file |

## Data contracts

Model-specific results should be normalized before they enter the association
logic:

```python
FaceTrack(
    track_id,
    start,
    end,
    boxes,  # timestamp -> bounding box
)

SpeakerTurn(
    speaker_id,
    start,
    end,
    confidence,
)

Word(
    text,
    start,
    end,
    confidence,
)

FaceActivity(
    track_id,
    start,
    end,
    speaking_probability,
)
```

The final event format should be:

```json
{
  "start": 12.1,
  "end": 14.7,
  "speaker": "Person 2",
  "faceTrack": "face-7",
  "text": "We can start the presentation.",
  "confidence": 0.91,
  "state": "visible"
}
```

Allowed states:

- `visible`: confidently associated with a face.
- `off_screen`: a voice is present but no candidate face is visible.
- `uncertain`: candidate faces exist, but the evidence is insufficient.
- `overlapping`: more than one speaker is active.

## Association algorithm

For each diarized speaker turn:

1. Find face tracks visible during the interval.
2. Obtain Light-ASD speaking probabilities for those tracks.
3. Calculate each face's time-weighted average probability.
4. Select a face only when its score exceeds `ACTIVE_SPEAKER_THRESHOLD` and
   exceeds the runner-up by `ASSIGNMENT_MARGIN`.
5. Otherwise emit `uncertain` or `off_screen`.
6. Accumulate successful assignments from the same diarized voice, for example
   `SPEAKER_00 -> Person 1`.
7. Assign each Whisper word according to the speaker turn containing its
   midpoint.
8. Combine consecutive words belonging to the same person into readable
   caption lines.

The two thresholds belong in one configuration file because they will need
tuning against real videos.

For simultaneous speech, use pyannote's overlap-aware output instead of its
exclusive diarization output. Multiple faces may then be highlighted. Exact
transcription of overlapping speech remains a known MVP limitation.

## Rendering rules

For each frame:

- Interpolate the active face's bounding box from its track.
- Draw a thick, high-contrast border.
- Add a speaking icon and `Person N` label.
- Do not communicate speaker state through color alone.
- Use a neutral or absent border for non-speaking faces.
- Show `Off-screen speaker` when speech cannot be mapped to a visible face.
- Preserve the original audio, resolution, and frame rate.

Generate:

- `output.annotated.mp4` for visual inspection.
- `captions.vtt` for accessible playback.
- `events.json` for debugging and future interfaces.

## Repository structure

```text
lip-reading/
|-- pyproject.toml
|-- src/
|   |-- main.py          # command-line entry point and orchestration
|   |-- media.py         # FFmpeg input/output and timestamps
|   |-- vision.py        # face detection and tracking
|   |-- speech.py        # Whisper and pyannote execution
|   |-- association.py   # active-speaker and identity assignment
|   `-- render.py        # video, VTT, and JSON output
|-- tests/
|   `-- test_association.py
`-- samples/
    `-- README.md
```

Do not add a web server, database, Docker setup, plugin system, or abstract
model interfaces to the MVP.

## Implementation milestones

### 1. Environment and vertical spike

- Install FFmpeg, PyTorch, MediaPipe, Whisper, pyannote, and Light-ASD.
- Confirm that each pretrained model runs independently on a short video.
- Verify GPU use when available and CPU fallback where supported.
- Resolve dependency compatibility before adding project abstractions.

Done when one short video can be transcribed, diarized, and passed through
Light-ASD independently.

### 2. Media preparation

Implement:

```bash
python -m src.main input.mp4 --output output/
```

The command should:

- Reject missing or unsupported files.
- Confirm the presence of video and audio streams.
- Extract normalized audio.
- Retain a mapping to the original video timestamps.

Done when extracted audio and frames remain synchronized throughout the video.

### 3. Face detection and tracking

- Run MediaPipe face detection.
- Connect nearby detections between frames using bounding-box overlap.
- End a track after a configurable number of missing frames.
- Store timestamped boxes for each track.

Done when every visible participant has a stable track during a continuous
shot.

### 4. Speech analysis

- Run Whisper with word timestamps.
- Run pyannote diarization.
- Save both normalized outputs as intermediate JSON files.

Done when the system produces timed words and stable anonymous speaker
clusters.

### 5. Active-speaker association

- Extract the face clips required by Light-ASD.
- Calculate speaking probabilities.
- Implement the threshold-and-margin association algorithm.
- Assign `Person N` labels.
- Preserve uncertainty instead of forcing matches.

Done when a two-person conversation correctly switches the active label as
speakers take turns.

### 6. Output generation

- Draw active-speaker highlights.
- Add speaker-labelled captions.
- Generate VTT and event JSON.
- Remux the untouched original audio into the rendered video.

Done when the output plays with correct audio/video synchronization and the
highlight follows the current speaker.

### 7. Verification and tuning

Prepare one manually annotated evaluation video containing:

- Two or more visible speakers.
- Normal turn-taking.
- A short overlap.
- One off-screen utterance.
- A shot change or temporary occlusion.

Measure:

- Wrong-person assignment rate.
- Missed active-speaker rate.
- False-highlight rate.
- Transcription word error rate.
- Diarization error rate.
- Audio/video synchronization drift.

Tune only `ACTIVE_SPEAKER_THRESHOLD`, `ASSIGNMENT_MARGIN`, and tracking
tolerances against this sample.

## Acceptance criteria

The MVP is complete when:

- One MP4 command produces all three output files.
- The same voice retains the same `Person N` label throughout the video.
- The currently speaking visible person is highlighted.
- Turn changes update the highlight correctly.
- Two simultaneous visible speakers can both be highlighted.
- Off-screen speech does not highlight an unrelated face.
- Low-confidence attribution is visibly marked as uncertain.
- Captions contain speaker labels and remain synchronized.
- Processing failure produces a clear error rather than a partial,
  apparently successful result.
- One automated association test and one manually verified end-to-end sample
  pass.

## Deferred work

Add live streaming only after the offline association is demonstrably accurate.
Add biometric identity only if stable real-world names are required and consent,
retention, encryption, and deletion requirements have been defined.
