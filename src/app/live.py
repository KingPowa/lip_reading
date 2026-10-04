"""Command-line startup for the live camera and microphone application."""

import argparse
import logging
import re
import ssl
from pathlib import Path

from src.app.server import LiveServer
from src.app.service import RecognitionService
from src.config.live import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TRANSCRIPTION_INTERVAL_SECONDS,
    LANGUAGE_PATTERN,
    MAXIMUM_TRANSCRIPTION_INTERVAL_SECONDS,
    MINIMUM_TRANSCRIPTION_INTERVAL_SECONDS,
)
from src.config.models import MODEL_DIRECTORY

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent live microphone captions and camera speaker location"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_TRANSCRIPTION_INTERVAL_SECONDS,
        help="PCM update interval (0.05–1 second)",
    )
    parser.add_argument(
        "--speech-model", type=Path, help="Custom local Vosk model directory"
    )
    parser.add_argument(
        "--language",
        help="Speech model language (en, it, es, fr, de)",
    )
    parser.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    parser.add_argument(
        "--cert", type=Path, help="TLS certificate for access from another device"
    )
    parser.add_argument("--key", type=Path, help="TLS private key")
    return parser


def serve(arguments: argparse.Namespace) -> None:
    if (
        not MINIMUM_TRANSCRIPTION_INTERVAL_SECONDS
        <= arguments.chunk_seconds
        <= MAXIMUM_TRANSCRIPTION_INTERVAL_SECONDS
    ):
        raise ValueError("--chunk-seconds must be between 0.05 and 1.0")
    if bool(arguments.cert) != bool(arguments.key):
        raise ValueError("--cert and --key must be supplied together")
    if arguments.language and not re.fullmatch(LANGUAGE_PATTERN, arguments.language):
        raise ValueError("invalid transcription language")
    service = RecognitionService(
        arguments.language, arguments.model_directory, arguments.speech_model
    )
    try:
        service.warm_up()
        server = LiveServer(
            (arguments.host, arguments.port), service, arguments.chunk_seconds
        )
        try:
            scheme = "https" if arguments.cert else "http"
            if arguments.cert:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(arguments.cert, arguments.key)
                server.socket = context.wrap_socket(server.socket, server_side=True)
            LOGGER.info(
                "open %s://%s:%d; independent streaming ASR=%s and face/mouth tracking",
                scheme,
                arguments.host,
                server.server_port,
                arguments.language or "en",
            )
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        service.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    try:
        serve(parser.parse_args())
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
