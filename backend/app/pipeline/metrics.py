"""Listening-oriented objective metrics for mosaic reconstruction.

These are diagnostics, not claims of perceptual equivalence.  They provide
stable signals for comparing pipeline revisions: harmony, spectral envelope,
rhythm, and splice smoothness.
"""

from __future__ import annotations

import librosa
import numpy as np


def _same_length(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(reference), len(estimate))
    if n <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return (
        np.asarray(reference[:n], dtype=np.float32),
        np.asarray(estimate[:n], dtype=np.float32),
    )


def multi_resolution_log_mel_distance(
    reference: np.ndarray, estimate: np.ndarray, sr: int
) -> float:
    """Mean normalized log-mel distance; lower is better."""
    ref, est = _same_length(reference, estimate)
    if len(ref) < 64:
        return 0.0
    distances: list[float] = []
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        kwargs = dict(sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64, fmax=sr / 2)
        r = np.log1p(librosa.feature.melspectrogram(y=ref, **kwargs))
        e = np.log1p(librosa.feature.melspectrogram(y=est, **kwargs))
        n = min(r.shape[1], e.shape[1])
        scale = float(np.mean(np.abs(r[:, :n]))) + 1e-6
        distances.append(float(np.mean(np.abs(r[:, :n] - e[:, :n])) / scale))
    return float(np.mean(distances))


def frame_chroma_similarity(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:
    """Frame-wise chroma cosine similarity in [0, 1]; higher is better."""
    ref, est = _same_length(reference, estimate)
    if len(ref) < 64:
        return 0.0
    r = librosa.feature.chroma_stft(y=ref, sr=sr, n_fft=2048, hop_length=512)
    e = librosa.feature.chroma_stft(y=est, sr=sr, n_fft=2048, hop_length=512)
    n = min(r.shape[1], e.shape[1])
    r, e = r[:, :n], e[:, :n]
    denom = np.linalg.norm(r, axis=0) * np.linalg.norm(e, axis=0)
    valid = denom > 1e-8
    if not np.any(valid):
        return 0.0
    sims = np.sum(r[:, valid] * e[:, valid], axis=0) / denom[valid]
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def onset_envelope_correlation(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:
    """Correlation of onset-strength envelopes in [-1, 1]; higher is better."""
    ref, est = _same_length(reference, estimate)
    if len(ref) < 64:
        return 0.0
    r = librosa.onset.onset_strength(y=ref, sr=sr, hop_length=256)
    e = librosa.onset.onset_strength(y=est, sr=sr, hop_length=256)
    n = min(len(r), len(e))
    r, e = r[:n], e[:n]
    if n < 2 or float(np.std(r)) < 1e-8 or float(np.std(e)) < 1e-8:
        return 0.0
    return float(np.clip(np.corrcoef(r, e)[0, 1], -1.0, 1.0))


def boundary_discontinuity(
    audio: np.ndarray, sr: int, boundaries_s: list[float] | np.ndarray
) -> float:
    """Median normalized sample jump at tile boundaries; lower is better."""
    y = np.asarray(audio, dtype=np.float32)
    if len(y) < 3:
        return 0.0
    scale = float(np.sqrt(np.mean(y * y))) + 1e-6
    jumps = []
    for boundary_s in boundaries_s:
        i = int(round(float(boundary_s) * sr))
        if 1 <= i < len(y):
            jumps.append(abs(float(y[i] - y[i - 1])) / scale)
    return float(np.median(jumps)) if jumps else 0.0


def quality_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    *,
    boundaries_s: list[float] | np.ndarray = (),
) -> dict[str, float]:
    """Compute compact metrics suitable for mosaic.json stats."""
    return {
        "log_mel_distance": round(
            multi_resolution_log_mel_distance(reference, estimate, sr), 4
        ),
        "chroma_similarity": round(frame_chroma_similarity(reference, estimate, sr), 4),
        "onset_correlation": round(onset_envelope_correlation(reference, estimate, sr), 4),
        "boundary_discontinuity": round(
            boundary_discontinuity(estimate, sr, boundaries_s), 4
        ),
    }


def reconstruction_quality_score(metrics: dict[str, float]) -> float:
    """Scalar used only to choose between two source-only render candidates.

    Harmony and rhythm carry most of a song's identity. Log-mel distance keeps
    the selector from choosing an unnaturally colored candidate merely because
    its dominant notes line up.
    """
    chroma = float(metrics.get("chroma_similarity", 0.0))
    onset = float(metrics.get("onset_correlation", 0.0))
    log_mel = float(metrics.get("log_mel_distance", 2.0))
    boundary = float(metrics.get("boundary_discontinuity", 0.0))
    return (
        0.48 * chroma
        + 0.29 * onset
        - 0.18 * min(2.0, log_mel) / 2.0
        - 0.05 * min(1.0, boundary)
    )


def candidate_improves(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    min_score_gain: float = 0.02,
) -> bool:
    """Conservative auto-render acceptance gate.

    The new renderer must improve the composite by a perceptually meaningful
    margin and may not trade away harmony, rhythm, or spectral fit materially.
    """
    if float(candidate.get("chroma_similarity", 0.0)) + 0.005 < float(
        baseline.get("chroma_similarity", 0.0)
    ):
        return False
    if float(candidate.get("onset_correlation", 0.0)) + 0.03 < float(
        baseline.get("onset_correlation", 0.0)
    ):
        return False
    if float(candidate.get("log_mel_distance", 2.0)) > 1.05 * float(
        baseline.get("log_mel_distance", 2.0)
    ):
        return False
    baseline_boundary = float(baseline.get("boundary_discontinuity", 0.0))
    candidate_boundary = float(candidate.get("boundary_discontinuity", 0.0))
    if candidate_boundary > max(
        baseline_boundary + 0.02,
        baseline_boundary * 1.5,
    ):
        return False
    return (
        reconstruction_quality_score(candidate)
        >= reconstruction_quality_score(baseline) + min_score_gain
    )
