"""
One-off: export the SpeechBrain ECAPA embedding model (features -> 192-d) to ONNX.

Needs `onnx` and `onnxscript` installed (not project dependencies).
Usage: PYTHONPATH=. python scripts/export_ecapa_onnx.py [models-directory]
"""

import sys
from pathlib import Path

import torch

from src.config.models import MODEL_DIRECTORY, SPEAKER_MODEL_DIRECTORY
from src.core.audio.speech import _speaker_classifier
from src.core.audio.voice_onnx import ONNX_FILENAME


class _Embedding(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features).squeeze(1)


directory = Path(sys.argv[1] if len(sys.argv) > 1 else MODEL_DIRECTORY) / SPEAKER_MODEL_DIRECTORY
model = _Embedding(_speaker_classifier(directory).mods.embedding_model).eval()
torch.onnx.export(
    model,
    (torch.randn(1, 200, 80),),
    str(directory / ONNX_FILENAME),
    input_names=["features"],
    output_names=["embedding"],
    dynamic_axes={"features": {1: "frames"}},
    dynamo=False,
)
print("wrote", directory / ONNX_FILENAME)
