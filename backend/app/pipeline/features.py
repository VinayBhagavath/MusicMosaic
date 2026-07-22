"""Feature extraction: handcrafted acoustic descriptors.

Performance choice: compute frame-level features once per track, then
aggregate mean/std inside each window. That is ~O(song) STFTs instead of
~O(segments) — typically 5–15× faster for overlapping windows.
"""

from __future__ import annotations

from typing import Callable, Protocol

import librosa
import numpy as np

from app.pipeline.segment import Segment

ProgressCb = Callable[[float, str], None]


class FeatureExtractor(Protocol):
    def embed(self, y: np.ndarray, sr: int) -> np.ndarray: ...

    def embed_segments(
        self,
        y: np.ndarray,
        sr: int,
        segments: list[Segment],
        *,
        on_progress: ProgressCb | None = None,
    ) -> np.ndarray: ...


def _mean_std_cols(x: np.ndarray) -> np.ndarray:
    """x shape (n_features, n_frames) → [mean..., std...] length 2*n_features."""
    if x.size == 0 or x.shape[1] == 0:
        n = x.shape[0] if x.ndim == 2 else 0
        return np.zeros(2 * n, dtype=np.float32)
    mu = np.mean(x, axis=1)
    sd = np.std(x, axis=1)
    return np.concatenate([mu, sd]).astype(np.float32)


def _l2_normalize_rows(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, eps)).astype(np.float32)


class HandcraftedExtractor:
    """MFCC + chroma + centroid + RMS → ~54-D L2-normalized vectors."""

    def __init__(self, *, n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _frame_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Return stacked frame features shape (54/2 wait - raw frames before mean/std).

        Actually returns list of feature matrices; we stack channels as rows later.
        """
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, **kw)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, **kw)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        # Align frame counts (rms can differ by 1)
        n = min(mfcc.shape[1], chroma.shape[1], centroid.shape[1], rms.shape[1])
        return np.vstack(
            [mfcc[:, :n], chroma[:, :n], centroid[:, :n], rms[:, :n]]
        ).astype(np.float32)

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        frames = self._frame_features(y, sr)
        return _l2_normalize_rows(_mean_std_cols(frames)[None, :])[0]

    def embed_segments(
        self,
        y: np.ndarray,
        sr: int,
        segments: list[Segment],
        *,
        on_progress: ProgressCb | None = None,
    ) -> np.ndarray:
        """One feature pass over `y`, then windowed mean/std for each segment."""
        n = len(segments)
        if n == 0:
            return np.zeros((0, 54), dtype=np.float32)

        if on_progress:
            on_progress(0.05, "computing frame features")
        frames = self._frame_features(y, sr)  # (F, T)
        n_frames = frames.shape[1]
        out = np.empty((n, frames.shape[0] * 2), dtype=np.float32)

        report_every = max(1, n // 5)
        for i, seg in enumerate(segments):
            f0 = int(seg.start_s * sr) // self.hop_length
            f1 = int(seg.end_s * sr) // self.hop_length
            f0 = max(0, min(f0, n_frames - 1))
            f1 = max(f0 + 1, min(f1, n_frames))
            out[i] = _mean_std_cols(frames[:, f0:f1])
            if on_progress and (i % report_every == 0 or i == n - 1):
                on_progress(0.1 + 0.9 * (i + 1) / n, f"windows {i + 1}/{n}")

        return _l2_normalize_rows(out)

    # Back-compat for older call sites / tests
    def embed_batch(
        self,
        segments: list[Segment],
        sr: int,
        *,
        on_progress: ProgressCb | None = None,
    ) -> np.ndarray:
        """Fallback: embed each segment waveform independently (slower)."""
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
