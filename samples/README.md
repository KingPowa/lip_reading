# Samples

Put a short `.mp4` with both video and audio here, then run:

```bash
uv run --python .venv/bin/python -m src.main samples/example.mp4 --output output
```

The pipeline uses local Whisper transcription, ECAPA anonymous voice clustering,
MediaPipe face tracking, and Light-ASD audio-visual speaker scoring. Missing
model files are downloaded to `models/` on the first run.
