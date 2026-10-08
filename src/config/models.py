"""Models configuration."""

from pathlib import Path

MODEL_DIRECTORY = Path("models")
FACE_LANDMARKER_FILENAME = "face_landmarker.task"
FACE_IDENTITY_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"
FACE_IDENTITY_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
FACE_IDENTITY_MODEL_SHA256 = (
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)
LIGHT_ASD_FILENAME = "light-asd-talkset.model"
LIGHT_ASD_URL = "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/finetuning_TalkSet.model"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
WHISPER_MODEL_NAME = "large-v3"
SPEAKER_MODEL_DIRECTORY = "speaker-ecapa"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
