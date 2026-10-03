"""Central tuning values for the offline MVP."""

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

ACTIVE_SPEAKER_THRESHOLD = 0.20
ASSIGNMENT_MARGIN = 0.12
VOICE_CLUSTER_THRESHOLD = 0.72
MINIMUM_VOICE_SEGMENT_SECONDS = 0.75
FACE_TRACK_IOU_THRESHOLD = 0.20
MISSING_FACE_LIMIT = 25
ANALYSIS_FRAME_INTERVAL = 5
MINIMUM_TRACK_DETECTIONS = 15
MINIMUM_TRACK_DURATION_SECONDS = 3.0
MINIMUM_LIVE_FACE_DETECTIONS = 2
MINIMUM_LIVE_TRACK_DURATION_SECONDS = 0.0
CAPTION_HORIZONTAL_MARGIN = 16
CAPTION_LINE_HEIGHT = 28
CAPTION_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Verdana.ttf")
CAPTION_FONT_SIZE = 22
