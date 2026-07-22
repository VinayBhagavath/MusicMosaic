"""Musical feature extraction tuned for instrumental collage.

Design (human-listening first):
- Chroma dominates (harmony / key feel) with key-invariant matching downstream
- MFCC carries timbre (piano vs strings vs synth)
- Energy keeps loudness contour aligned
- Frame features once per track, then window aggregate (fast)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np

from app.pipeline.segment import Segment

ProgressCb = Callable[[float, str], None]


def _l2_rows(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, eps)).astype(np.float32)


def _mean_std(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or x.shape[1] == 0:
        return np.zeros(x.shape[0] * 2, dtype=np.float32)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


@dataclass(slots=True)
class EmbPack:
    """Per-segment musical descriptors (rows L2-normalized where noted)."""

    chroma: np.ndarray  # [n, 12]
    timbre: np.ndarray  # [n, n_mfcc*2]
    energy: np.ndarray  # [n, 4] rms+centroid mean/std


class MusicalExtractor:
    """Instrumental-oriented features. Prefer chroma_cqt quality vs speed? 

    Decision: chroma_stft (faster). Quality comes from key-invariant scoring +
    longer windows + continuity, not from a heavier spectrogram.
    """

    def __init__(self, *, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _frames(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        y_h = y  # skip HPSS — too slow for little gain vs key-invariant chroma
        chroma = librosa.feature.chroma_stft(y=y_h, sr=sr, **kw)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, **kw)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = min(chroma.shape[1], mfcc.shape[1], cent.shape[1], rms.shape[1])
        return chroma[:, :n], mfcc[:, :n], cent[:, :n], rms[:, :n]

    def embed_segments(
        self,
        y: np.ndarray,
        sr: int,
        segments: list[Segment],
        *,
        on_progress: ProgressCb | None = None,
    ) -> EmbPack:
        n = len(segments)
        if n == 0:
            return EmbPack(
                np.zeros((0, 12), np.float32),
                np.zeros((0, self.n_mfcc * 2), np.float32),
                np.zeros((0, 4), np.float32),
            )

        if on_progress:
            on_progress(0.05, "frame features")
        chroma_f, mfcc_f, cent_f, rms_f = self._frames(y, sr)
        n_frames = chroma_f.shape[1]

        chroma = np.empty((n, 12), np.float32)
        timbre = np.empty((n, self.n_mfcc * 2), np.float32)
        energy = np.empty((n, 4), np.float32)

        report_every = max(1, n // 5)
        for i, seg in enumerate(segments):
            f0 = int(seg.start_s * sr) // self.hop_length
            f1 = int(seg.end_s * sr) // self.hop_length
            f0 = max(0, min(f0, n_frames - 1))
            f1 = max(f0 + 1, min(f1, n_frames))
            # Chroma: mean only (std of chroma is noisy); L2 later
            c = chroma_f[:, f0:f1].mean(axis=1)
            chroma[i] = c
            timbre[i] = _mean_std(mfcc_f[:, f0:f1])
            energy[i] = _mean_std(np.vstack([rms_f[:, f0:f1], cent_f[:, f0:f1]]))
            if on_progress and (i % report_every == 0 or i == n - 1):
                on_progress(0.1 + 0.9 * (i + 1) / n, f"windows {i + 1}/{n}")

        return EmbPack(
            chroma=_l2_rows(chroma),
            timbre=_l2_rows(timbre),
            energy=_l2_rows(energy),
        )


# Back-compat alias used by older imports/tests
HandcraftedExtractor = MusicalExtractor
