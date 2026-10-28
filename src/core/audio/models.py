"""Cache and load local streaming speech models before microphone capture."""

import shutil
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

from src.config.streaming import (
    MAXIMUM_MODEL_ARCHIVE_BYTES,
    MAXIMUM_MODEL_EXTRACTED_BYTES,
    MODEL_DOWNLOAD_BLOCK_BYTES,
    MODEL_DOWNLOAD_TIMEOUT_SECONDS,
    SPEECH_DECODER_OPTIONS,
    SPEECH_MODEL_URL,
    SPEECH_MODELS,
)


class SpeechModelStore:
    def __init__(self, directory: Path, custom_model: Path | None = None) -> None:
        self.directory = directory
        self.custom_model = custom_model
        self.lock = threading.Lock()
        self.models: dict[str, object] = {}

    def load(self, language: str) -> object:
        from vosk import Model, SetLogLevel

        if language not in SPEECH_MODELS and self.custom_model is None:
            raise ValueError(
                f"choose a supported speech language: {', '.join(SPEECH_MODELS)}"
            )
        with self.lock:
            if language in self.models:
                return self.models[language]
            model_path = self.custom_model or self.directory / SPEECH_MODELS[language]
            if self.custom_model is not None and not model_path.is_dir():
                raise ValueError(
                    "--speech-model must be an existing Vosk model directory"
                )
            if not model_path.is_dir():
                download_model(self.directory, SPEECH_MODELS[language])
            if self.custom_model is None:
                widen_search(model_path / "conf" / "model.conf")
            SetLogLevel(-1)
            try:
                model = Model(str(model_path))
            except Exception as error:
                raise RuntimeError(
                    f"cannot load speech model at {model_path}: {error}"
                ) from error
            self.models[language] = model
            return model


def widen_search(config: Path) -> None:
    """Set the decoder search options in a catalog model's config (idempotent)."""
    lines = [
        line
        for line in config.read_text().splitlines()
        if line.split("=", 1)[0] not in SPEECH_DECODER_OPTIONS
    ]
    lines += [f"{name}={value}" for name, value in SPEECH_DECODER_OPTIONS.items()]
    config.write_text("\n".join(lines) + "\n")


def download_model(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=directory, prefix="speech-download-"
    ) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "model.zip"
        try:
            with (
                urllib.request.urlopen(
                    SPEECH_MODEL_URL.format(name=name),
                    timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS,
                ) as response,
                archive.open("wb") as output,
            ):
                total = 0
                while block := response.read(MODEL_DOWNLOAD_BLOCK_BYTES):
                    total += len(block)
                    if total > MAXIMUM_MODEL_ARCHIVE_BYTES:
                        raise RuntimeError(
                            "speech model archive exceeds the download limit"
                        )
                    output.write(block)
            extract_model(archive, temporary_path, name)
            shutil.move(str(temporary_path / name), directory / name)
        except (OSError, zipfile.BadZipFile) as error:
            raise RuntimeError(
                f"cannot download streaming speech model: {error}"
            ) from error


def extract_model(archive: Path, destination: Path, name: str) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if sum(member.file_size for member in members) > MAXIMUM_MODEL_EXTRACTED_BYTES:
            raise ValueError("speech model exceeds the extraction limit")
        root = (destination / name).resolve()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("speech model archive contains an invalid path")
        source.extractall(destination)
