"""Audio load / normalize helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SR = 22_050
TARGET_LUFS = -14.0
MIN_DURATION_S = 5.0
MAX_DURATION_S = 8 * 60.0


def load_audio(path: str | Path, *, sr: int = SR) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at `sr`. Raises ValueError on duration bounds."""
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / sr
    if dur < MIN_DURATION_S:
        raise ValueError(f"Audio too short ({dur:.1f}s < {MIN_DURATION_S}s): {path}")
    if dur > MAX_DURATION_S:
        raise ValueError(f"Audio too long ({dur:.1f}s > {MAX_DURATION_S}s): {path}")
    return y.astype(np.float32, copy=False), sr


def loudness_normalize(y: np.ndarray, sr: int, *, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Normalize to target LUFS; fall back to peak norm if metering fails (near-silence)."""
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y.astype(np.float64))
        if not np.isfinite(loudness):
            raise ValueError("non-finite loudness")
        out = pyln.normalize.loudness(y.astype(np.float64), loudness, target_lufs)
    except Exception:
        peak = float(np.max(np.abs(y))) or 1.0
        out = y.astype(np.float64) / peak * 0.9
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def load_and_normalize(path: str | Path, *, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr = load_audio(path, sr=sr)
    return loudness_normalize(y, sr), sr


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(y))) or 1.0
    y_out = (y / peak * 0.89).astype(np.float32)  # ~-1 dBFS
    sf.write(str(path), y_out, sr, subtype="PCM_16")
