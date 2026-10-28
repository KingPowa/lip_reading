"""Torch-free ECAPA voice embeddings: numpy Fbank + ONNX Runtime."""

from pathlib import Path

import numpy as np

ONNX_FILENAME = "ecapa.onnx"
_SAMPLE_RATE, _N_FFT, _HOP, _N_MELS, _TOP_DB = 16000, 400, 160, 80, 80.0


def _mel_filters() -> np.ndarray:
    """SpeechBrain Filterbank (triangular, f_min 0, f_max 8000), shape (201, 80)."""
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)  # noqa: E731
    hz = 700 * (10 ** (np.linspace(to_mel(0), to_mel(8000), _N_MELS + 2) / 2595) - 1)
    centre, band = hz[1:-1], (hz[1:] - hz[:-1])[:-1]
    freqs = np.linspace(0, _SAMPLE_RATE // 2, _N_FFT // 2 + 1)[:, None]
    slope = (freqs - centre) / band
    return np.maximum(0, np.minimum(slope + 1, 1 - slope)).astype(np.float32)


_FILTERS = _mel_filters()
_WINDOW = np.hamming(_N_FFT + 1)[:-1].astype(np.float32)  # periodic, like torch


def fbank(samples: np.ndarray) -> np.ndarray:
    """Mean-normalised log-mel features (1, frames, 80), as SpeechBrain computes them."""
    padded = np.pad(samples.astype(np.float32), _N_FFT // 2)  # center=True, zeros
    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    # einsum avoids BLAS, whose worker threads spin after each call and burn CPU.
    db = 10 * np.log10(np.clip(np.einsum("tf,fm->tm", power, _FILTERS), 1e-10, None))
    db = np.maximum(db, db.max() - _TOP_DB)
    return (db - db.mean(axis=0))[None].astype(np.float32)


class OnnxSpeakerEmbedder:
    def __init__(self, path: Path) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # Idle worker threads spin by default, which multiplied CPU use about 3x live.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"features": fbank(samples)})[0].reshape(-1)
