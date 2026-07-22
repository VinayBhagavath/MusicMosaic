"""Audio load / normalize helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22_050
MIN_DURATION_S = 5.0
MAX_DURATION_S = 8 * 60.0


def load_audio(path: str | Path, *, sr: int = SR) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at `sr`. Raises ValueError on duration bounds."""
    y, _file_sr = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / sr
    name = Path(path).name
    if dur < MIN_DURATION_S:
        raise ValueError(f"Audio too short ({dur:.1f}s < {MIN_DURATION_S}s): {name}")
    if dur > MAX_DURATION_S:
        raise ValueError(f"Audio too long ({dur:.1f}s > {MAX_DURATION_S}s): {name}")
    return y.astype(np.float32, copy=False), sr


def peak_normalize(y: np.ndarray, *, peak: float = 0.9) -> np.ndarray:
    """Fast peak normalize — preferred for matching (LUFS is slow and not needed mid-pipeline)."""
    p = float(np.max(np.abs(y))) or 1.0
    return np.clip(y / p * peak, -1.0, 1.0).astype(np.float32)


def load_and_normalize(path: str | Path, *, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr = load_audio(path, sr=sr)
    return peak_normalize(y), sr


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_out = peak_normalize(y, peak=0.89)
    sf.write(str(path), y_out, sr, subtype="PCM_16")
