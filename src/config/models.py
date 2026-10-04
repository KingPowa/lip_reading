"""Models configuration."""

from pathlib import Path

MODEL_DIRECTORY = Path("models")
FACE_LANDMARKER_FILENAME = "face_landmarker.task"
LIGHT_ASD_FILENAME = "light-asd-talkset.model"
LIGHT_ASD_URL = "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/finetuning_TalkSet.model"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
WHISPER_MODEL_NAME = "large-v3"
SPEAKER_MODEL_DIRECTORY = "speaker-ecapa"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
