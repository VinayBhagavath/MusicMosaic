"""Tests for cohesion transforms (pitch/tempo/spectrum/roles)."""

import numpy as np
import pytest

from app.pipeline.cohesion import (
    classify_role,
    harmonic_reconstruct,
    highpass,
    match_loudness,
    match_spectrum,
)
from app.pipeline.transform import _rubberband_ok, fit_length, pitch_shift, prepare_clip, time_stretch


@pytest.mark.skipif(not _rubberband_ok(), reason="Rubber Band system binary is not installed")
def test_time_stretch_changes_length():
    sr = 22050
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.2
    out = time_stretch(y, sr, 1.25)
    # Faster → shorter
    assert len(out) < len(y) * 0.95


@pytest.mark.skipif(not _rubberband_ok(), reason="Rubber Band system binary is not installed")
def test_pitch_shift_fractional_runs():
    sr = 22050
    y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32) * 0.2
    out = pitch_shift(y, sr, 1.5)
    assert len(out) > sr // 2
    assert np.max(np.abs(out)) > 0.01


@pytest.mark.skipif(not _rubberband_ok(), reason="Rubber Band system binary is not installed")
def test_prepare_clip_fits_target_n():
    sr = 22050
    song = np.sin(2 * np.pi * 330 * np.arange(sr * 3) / sr).astype(np.float32) * 0.2
    clip = prepare_clip(song, sr, 0.5, target_n=sr, n_steps=2.0)
    assert len(clip) == sr


def test_residual_fill_adds_missing_band_energy():
    from app.pipeline.cohesion import residual_fill

    sr = 22050
    t = np.arange(sr) / sr
    # Primary: only 220 Hz. Target wants 220 + 660. Secondary has 660.
    primary = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    secondary = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    ref = (primary + secondary).astype(np.float32)
    out = residual_fill(primary, secondary, ref, sr, amount=0.8)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / sr)
    e220 = float(spec[np.argmin(np.abs(freqs - 220))])
    e660 = float(spec[np.argmin(np.abs(freqs - 660))])
    e660_primary = float(np.abs(np.fft.rfft(primary))[np.argmin(np.abs(freqs - 660))])
    assert e220 > 0.1
    assert e660 > e660_primary * 5


def test_match_loudness_moves_rms():
    quiet = np.ones(4096, np.float32) * 0.05
    loud = np.ones(4096, np.float32) * 0.4
    out = match_loudness(quiet, loud, blend=1.0)
    assert float(np.sqrt(np.mean(out**2))) > float(np.sqrt(np.mean(quiet**2))) * 1.5


def test_match_spectrum_runs():
    sr = 22050
    y = np.random.randn(sr).astype(np.float32) * 0.05
    ref = np.sin(2 * np.pi * 100 * np.arange(sr) / sr).astype(np.float32) * 0.3
    out = match_spectrum(y, ref, sr, strength=0.5)
    assert len(out) == len(y)


def _band_energy(y: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(y.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    band = (freqs >= lo) & (freqs < hi)
    return float(np.sum(spec[band] ** 2))


def test_match_spectrum_transfers_tonal_balance_toward_dark_target():
    sr = 22050
    rng = np.random.default_rng(0)
    y = rng.standard_normal(sr).astype(np.float32) * 0.1  # bright, flat noise
    # Dark target: low-pass filtered noise (little high-frequency energy)
    ref = highpass(rng.standard_normal(sr).astype(np.float32) * 0.1, sr, cutoff_hz=1.0)
    from app.pipeline.cohesion import lowpass

    ref = lowpass(ref, sr, cutoff_hz=800.0)
    out = match_spectrum(y, ref, sr, strength=1.0)

    hi_before = _band_energy(y, sr, 4000, 9000) / (_band_energy(y, sr, 100, 800) + 1e-9)
    hi_after = _band_energy(out, sr, 4000, 9000) / (_band_energy(out, sr, 100, 800) + 1e-9)
    # Output should be darker (less HF relative to LF) than the bright input.
    assert hi_after < hi_before


def test_harmonic_reconstruct_imposes_target_pitch():
    """Legibility: harmonic detail should move toward the target's notes."""
    sr = 22050
    rng = np.random.default_rng(0)
    # Source: broadband noise (no clear pitch of its own).
    y = rng.standard_normal(sr).astype(np.float32) * 0.1
    # Target: a clear 300 Hz tone (the "note" we want to become legible).
    t = np.arange(sr) / sr
    ref = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    out = harmonic_reconstruct(y, ref, sr, strength=0.9)

    spec = np.abs(np.fft.rfft(out.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)
    peak_hz = float(freqs[int(np.argmax(spec))])
    # The reconstructed note now peaks near the target's fundamental.
    assert abs(peak_hz - 300.0) < 30.0


def test_harmonic_reconstruct_zero_strength_is_noop():
    sr = 22050
    t = np.arange(sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ref = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    out = harmonic_reconstruct(y, ref, sr, strength=0.0)
    assert np.allclose(out, y)


def test_harmonic_reconstruct_preserves_length():
    sr = 22050
    t = np.arange(sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ref = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = harmonic_reconstruct(y, ref, sr, strength=0.5)
    assert len(out) == len(y)


def test_match_spectrum_preserves_source_pitch():
    sr = 22050
    t = np.arange(sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ref = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    out = match_spectrum(y, ref, sr, strength=1.0)
    # Envelope transfer must not repitch: fundamental stays at the source's 220 Hz.
    spec = np.abs(np.fft.rfft(out.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)
    peak_hz = float(freqs[int(np.argmax(spec))])
    assert abs(peak_hz - 220.0) < 20.0


def test_highpass_reduces_dc():
    sr = 22050
    y = np.ones(sr, np.float32) * 0.5
    out = highpass(y, sr, cutoff_hz=120)
    assert abs(float(out.mean())) < 0.15


def test_classify_role_returns_label():
    sr = 22050
    # Low sine → bass-ish
    y = np.sin(2 * np.pi * 60 * np.arange(sr) / sr).astype(np.float32) * 0.4
    role = classify_role(y, sr)
    assert role in {"bass", "drums", "harmonic", "full"}


def test_match_envelope_follows_ref_shape():
    from app.pipeline.cohesion import match_envelope

    sr = 22050
    n = sr
    # Flat source vs decaying reference
    y = np.ones(n, np.float32) * 0.3
    t = np.linspace(0, 1, n, dtype=np.float32)
    ref = (0.5 * np.exp(-3.0 * t)).astype(np.float32)
    out = match_envelope(y, ref, sr, blend=1.0)
    # Early louder than late
    assert float(np.mean(np.abs(out[: n // 8]))) > float(np.mean(np.abs(out[-n // 8 :]))) * 1.3


def test_onset_align_moves_attack():
    from app.pipeline.transform import align_onset_to_ref

    sr = 22050
    n = sr
    y = np.zeros(n, np.float32)
    ref = np.zeros(n, np.float32)
    y[int(0.4 * sr) : int(0.4 * sr) + 64] = 1.0
    ref[int(0.1 * sr) : int(0.1 * sr) + 64] = 1.0
    out = align_onset_to_ref(y, ref, sr)
    # Peak should move earlier toward ref
    assert int(np.argmax(np.abs(out))) < int(0.25 * sr)


def test_onset_align_zero_pads_instead_of_wrapping_tail():
    from app.pipeline.transform import align_onset_to_ref

    sr = 22050
    y = np.zeros(sr, np.float32)
    ref = np.zeros(sr, np.float32)
    y[int(0.1 * sr) : int(0.1 * sr) + 64] = 1.0
    y[int(0.9 * sr) : int(0.9 * sr) + 64] = 0.5
    ref[int(0.4 * sr) : int(0.4 * sr) + 64] = 1.0
    out = align_onset_to_ref(y, ref, sr)
    assert np.max(np.abs(out[: int(0.25 * sr)])) == 0.0


def test_f0_semitone_delta_octave():
    from app.pipeline.transform import f0_semitone_delta

    sr = 22050
    t = np.arange(sr) / sr
    src = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ref = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    d = f0_semitone_delta(src, ref, sr)
    assert d is not None
    assert abs(d - 12.0) < 1.5


def test_fit_length():
    assert len(fit_length(np.ones(10, np.float32), 5)) == 5
    assert len(fit_length(np.ones(3, np.float32), 8)) == 8

