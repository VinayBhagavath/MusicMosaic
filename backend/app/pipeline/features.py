"""Feature extraction: handcrafted acoustic descriptors."""

from __future__ import annotations

from typing import Callable, Protocol

import librosa
import numpy as np

from app.pipeline.segment import Segment

ProgressCb = Callable[[float, str], None]


class FeatureExtractor(Protocol):
    def embed(self, y: np.ndarray, sr: int) -> np.ndarray: ...

    def embed_batch(
        self,
        segments: list[Segment],
        sr: int,
        *,
        on_progress: ProgressCb | None = None,
    ) -> np.ndarray: ...


def _mean_std(x: np.ndarray, axis: int = 1) -> np.ndarray:
    mu = np.mean(x, axis=axis)
    sd = np.std(x, axis=axis)
    return np.concatenate([mu, sd], axis=0)


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / max(n, eps)).astype(np.float32)


class HandcraftedExtractor:
    """MFCC + chroma + centroid + RMS → ~54-D L2-normalized vector.

    Tuned for speed: hop=512 at 22.05 kHz (~23 ms frames) is enough for 0.5 s windows.
    """

    def __init__(self, *, n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, **kw)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, **kw)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)

        parts = [
            _mean_std(mfcc),
            _mean_std(chroma),
            _mean_std(centroid),
            _mean_std(rms),
        ]
        return _l2_normalize(np.concatenate(parts).astype(np.float32))

    def embed_batch(
        self,
        segments: list[Segment],
        sr: int,
        *,
        on_progress: ProgressCb | None = None,
    ) -> np.ndarray:
        n = len(segments)
        if n == 0:
            return np.zeros((0, 54), dtype=np.float32)
        out = np.empty((n, self.embed(segments[0].waveform, sr).shape[0]), dtype=np.float32)
        report_every = max(1, n // 20)
        for i, seg in enumerate(segments):
            out[i] = self.embed(seg.waveform, sr)
            if on_progress and (i % report_every == 0 or i == n - 1):
                on_progress((i + 1) / n, f"features {i + 1}/{n}")
        return out
