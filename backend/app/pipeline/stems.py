"""Optional stem separation (Demucs) for role-true mosaic reconstruction.

Prefers demucs-mlx (Metal / MLX on Apple Silicon). Falls back to PyTorch Demucs
on CPU. Separators are reused across songs so model weights load once per job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import librosa
import numpy as np

ProgressCb = Callable[[float, str], None]

STEM_NAMES = ("drums", "bass", "other", "vocals")

_mlx_separator: Any | None = None
_torch_separator: Any | None = None
_torch_device: str | None = None


@dataclass(slots=True)
class StemBundle:
    """Per-song stem waveforms at pipeline SR."""

    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    vocals: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "drums": self.drums,
            "bass": self.bass,
            "other": self.other,
            "vocals": self.vocals,
        }


def demucs_mlx_available() -> bool:
    try:
        import demucs_mlx  # noqa: F401

        return True
    except Exception:
        return False


def demucs_torch_available() -> bool:
    try:
        import demucs.api  # noqa: F401

        return True
    except Exception:
        return False


def demucs_available() -> bool:
    return demucs_mlx_available() or demucs_torch_available()


def _fit_len(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) > n:
        return arr[:n]
    if len(arr) < n:
        return np.pad(arr, (0, n - len(arr)))
    return arr


def _bundle_from_dict(out: dict[str, np.ndarray], n: int) -> StemBundle | None:
    if len(out) < 3:
        return None
    zeros = np.zeros(n, dtype=np.float32)
    return StemBundle(
        drums=out.get("drums", zeros),
        bass=out.get("bass", zeros),
        other=out.get("other", zeros),
        vocals=out.get("vocals", zeros),
    )


def _get_mlx_separator(model_name: str = "htdemucs"):
    global _mlx_separator
    if _mlx_separator is None:
        from demucs_mlx import Separator

        # shifts=0 avoids multi-pass; batch_size uses GPU better on M-series
        _mlx_separator = Separator(model=model_name, shifts=0, batch_size=8)
    return _mlx_separator


def _get_torch_separator(model_name: str = "htdemucs"):
    global _torch_separator, _torch_device
    if _torch_separator is None:
        from demucs.api import Separator

        # PyTorch Demucs on MPS is unreliable (complex STFT ops); keep CPU.
        _torch_device = "cpu"
        _torch_separator = Separator(model=model_name, device=_torch_device)
    return _torch_separator


def _separate_mlx(
    y: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
) -> StemBundle | None:
    separator = _get_mlx_separator(model_name)
    model_sr = int(separator.samplerate)
    # [C, T] stereo at model rate
    if sr != model_sr:
        y_m = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=model_sr)
    else:
        y_m = y.astype(np.float32, copy=False)
    wav = np.stack([y_m, y_m], axis=0)
    _origin, stems = separator.separate_tensor(wav)
    out: dict[str, np.ndarray] = {}
    for name in STEM_NAMES:
        if name not in stems:
            continue
        s = np.asarray(stems[name], dtype=np.float32)
        if s.ndim == 2:
            s = s.mean(axis=0)
        else:
            s = np.squeeze(s)
        if model_sr != sr:
            s = librosa.resample(s, orig_sr=model_sr, target_sr=sr).astype(np.float32)
        out[name] = _fit_len(s.astype(np.float32, copy=False), len(y))
    return _bundle_from_dict(out, len(y))


def _separate_torch(
    y: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
) -> StemBundle | None:
    import torch
    import torchaudio

    separator = _get_torch_separator(model_name)
    wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # [1, T]
    wav = wav.repeat(2, 1)  # stereo
    if sr != separator.samplerate:
        wav = torchaudio.functional.resample(wav, sr, separator.samplerate)

    _origin, stems = separator.separate_tensor(wav)
    out: dict[str, np.ndarray] = {}
    for name in STEM_NAMES:
        if name not in stems:
            continue
        s = stems[name]
        if s.dim() == 2:
            s = s.mean(0)
        else:
            s = s.squeeze()
        if separator.samplerate != sr:
            s = torchaudio.functional.resample(
                s.unsqueeze(0), separator.samplerate, sr
            ).squeeze(0)
        arr = s.detach().cpu().numpy().astype(np.float32)
        out[name] = _fit_len(arr, len(y))
    return _bundle_from_dict(out, len(y))


def separate_stems(
    y: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
    on_progress: ProgressCb | None = None,
) -> StemBundle | None:
    """Run Demucs; returns None if unavailable or on failure.

    Output stems are resampled to `sr` and length-matched to `y`.
    """
    if not demucs_available() or len(y) < sr:
        return None
    try:
        if on_progress:
            on_progress(0.05, "Loading Demucs")
        if demucs_mlx_available():
            if on_progress:
                on_progress(0.2, "Separating stems (MLX)")
            bundle = _separate_mlx(y, sr, model_name=model_name)
        else:
            if on_progress:
                on_progress(0.2, "Separating stems (CPU)")
            bundle = _separate_torch(y, sr, model_name=model_name)
        if bundle is not None and on_progress:
            on_progress(1.0, "Stems ready")
        return bundle
    except Exception as e:
        print(f"[musicmosaic] Demucs stem sep failed: {e}", flush=True)
        # If MLX fails mid-job, try torch once
        if demucs_mlx_available() and demucs_torch_available():
            try:
                print("[musicmosaic] Falling back to PyTorch Demucs", flush=True)
                return _separate_torch(y, sr, model_name=model_name)
            except Exception as e2:
                print(f"[musicmosaic] Torch Demucs also failed: {e2}", flush=True)
        return None


def separate_many(
    songs: dict[str, np.ndarray],
    sr: int,
    *,
    on_progress: ProgressCb | None = None,
) -> dict[str, StemBundle]:
    """Separate each song; skip failures silently. Reuses one loaded model."""
    out: dict[str, StemBundle] = {}
    items = list(songs.items())
    backend = "MLX" if demucs_mlx_available() else "CPU"
    # Warm separator once before the loop
    if items and demucs_available():
        try:
            if demucs_mlx_available():
                _get_mlx_separator()
            else:
                _get_torch_separator()
        except Exception as e:
            print(f"[musicmosaic] Demucs load failed: {e}", flush=True)
            return out

    for i, (sid, y) in enumerate(items):
        if on_progress:
            on_progress(i / max(1, len(items)), f"Stems {sid} ({backend})")
        bundle = separate_stems(y, sr)
        if bundle is not None:
            out[sid] = bundle
    return out
