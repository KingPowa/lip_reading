"""Incremental speech model catalog and loading limits."""

SPEECH_MODEL_URL = "https://alphacephei.com/vosk/models/{name}.zip"
SPEECH_MODELS = {
    "en": "vosk-model-small-en-us-0.15",
    "it": "vosk-model-small-it-0.22",
    "es": "vosk-model-small-es-0.42",
    "fr": "vosk-model-small-fr-0.22",
    "de": "vosk-model-small-de-0.15",
}
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 60
MODEL_DOWNLOAD_BLOCK_BYTES = 1024 * 1024
MAXIMUM_MODEL_ARCHIVE_BYTES = 100 * 1024 * 1024
MAXIMUM_MODEL_EXTRACTED_BYTES = 400 * 1024 * 1024
SAMPLE_TIMESTAMP_TOLERANCE = 1 / 16000
# Wider search than the small models' defaults (beam 10, 3000 active, lattice 2): on the
# 9.5 min Italian sample WER 0.172 -> 0.164 for about +0.02 CPU cores.
SPEECH_DECODER_OPTIONS = {"--beam": "13.0", "--max-active": "7000", "--lattice-beam": "4.0"}
LANGUAGE_ID_MODEL = "tiny"
LANGUAGE_ID_DIRECTORY = "faster-whisper"
