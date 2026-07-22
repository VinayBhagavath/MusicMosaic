import numpy as np

from app.pipeline.metrics import (
    boundary_discontinuity,
    frame_chroma_similarity,
    multi_resolution_log_mel_distance,
    onset_envelope_correlation,
    quality_metrics,
)


def _tone(sr: int, hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_identical_audio_has_best_spectral_and_chroma_scores():
    sr = 22050
    y = _tone(sr, 440.0)
    assert multi_resolution_log_mel_distance(y, y, sr) < 1e-6
    assert frame_chroma_similarity(y, y, sr) > 0.999


def test_metrics_detect_pitch_and_rhythm_mismatch():
    sr = 22050
    ref = _tone(sr, 440.0, 2.0)
    estimate = _tone(sr, 523.25, 2.0)
    ref[:: sr // 4] += 0.8
    estimate[sr // 8 :: sr // 4] += 0.8
    metrics = quality_metrics(ref, estimate, sr, boundaries_s=[0.5, 1.0, 1.5])
    assert metrics["log_mel_distance"] > 0.05
    assert metrics["chroma_similarity"] < 0.9
    assert metrics["onset_correlation"] < 0.9


def test_boundary_discontinuity_reports_hard_splice():
    sr = 1000
    smooth = np.zeros(1000, np.float32)
    hard = smooth.copy()
    hard[500:] = 1.0
    assert boundary_discontinuity(smooth, sr, [0.5]) == 0.0
    assert boundary_discontinuity(hard, sr, [0.5]) > 1.0


def test_constant_onset_envelopes_are_handled():
    sr = 22050
    silence = np.zeros(sr, np.float32)
    assert onset_envelope_correlation(silence, silence, sr) == 0.0
