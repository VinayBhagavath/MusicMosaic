"""Spectral / bass / loudness cohesion toward a target reference window."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _rms(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12))


def highpass(y: np.ndarray, sr: int, cutoff_hz: float = 150.0) -> np.ndarray:
    """Remove competing sub/bass from secondary layers."""
    if len(y) < 64 or cutoff_hz <= 0:
        return y.astype(np.float32, copy=False)
    try:
        sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
        out = sosfiltfilt(sos, y.astype(np.float64))
        return out.astype(np.float32)
    except Exception:
        return y.astype(np.float32, copy=False)


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float = 200.0) -> np.ndarray:
    if len(y) < 64 or cutoff_hz <= 0:
        return y.astype(np.float32, copy=False)
    try:
        sos = butter(2, cutoff_hz, btype="lowpass", fs=sr, output="sos")
        out = sosfiltfilt(sos, y.astype(np.float64))
        return out.astype(np.float32)
    except Exception:
        return y.astype(np.float32, copy=False)


def match_loudness(y: np.ndarray, ref: np.ndarray, *, blend: float = 0.85) -> np.ndarray:
    """Match RMS toward reference (blend=1 → full match)."""
    if len(y) == 0 or len(ref) == 0:
        return y
    cur = _rms(y)
    tgt = _rms(ref)
    if cur < 1e-8 or tgt < 1e-8:
        return y.astype(np.float32, copy=False)
    scale = (tgt / cur) ** blend
    scale = float(np.clip(scale, 0.35, 3.5))
    out = y * scale
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out.astype(np.float32)


def match_envelope(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    blend: float = 0.85,
    hop: int = 256,
) -> np.ndarray:
    """Morph the time-varying loudness envelope of `y` toward `ref`.

    This is the main rhythm/dynamics synth step after a note is selected —
    attacks swell and decays follow the target shape.
    """
    if blend <= 0 or len(y) < hop * 4 or len(ref) < hop * 2:
        return y.astype(np.float32, copy=False)
    try:
        n = len(y)
        ref = ref[:n] if len(ref) >= n else np.pad(ref, (0, n - len(ref)))
        # Frame RMS
        n_frames = max(1, (n - 1) // hop + 1)

        def _frame_rms(sig: np.ndarray) -> np.ndarray:
            out = np.empty(n_frames, dtype=np.float64)
            for i in range(n_frames):
                a = i * hop
                b = min(n, a + hop * 2)
                out[i] = np.sqrt(np.mean(sig[a:b].astype(np.float64) ** 2) + 1e-12)
            return out

        ey = _frame_rms(y)
        er = _frame_rms(ref)
        # Smooth
        k = 5
        ker = np.ones(k, dtype=np.float64) / k
        ey_s = np.convolve(ey, ker, mode="same")
        er_s = np.convolve(er, ker, mode="same")
        ratio = (er_s + 1e-8) / (ey_s + 1e-8)
        ratio = np.clip(ratio, 0.25, 4.0)
        ratio = 1.0 + blend * (ratio - 1.0)
        # Upsample to samples
        x_f = (np.arange(n_frames) * hop + hop).astype(np.float64)
        x_s = np.arange(n, dtype=np.float64)
        gain = np.interp(x_s, x_f, ratio, left=ratio[0], right=ratio[-1]).astype(
            np.float32
        )
        out = y.astype(np.float32) * gain
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.98:
            out *= 0.98 / peak
        return out
    except Exception:
        return y.astype(np.float32, copy=False)


def match_transient(y: np.ndarray, ref: np.ndarray, sr: int, *, amount: float = 0.35) -> np.ndarray:
    """Gently boost/cut attack to follow target onset energy."""
    if amount <= 0 or len(y) < sr // 4 or len(ref) < sr // 4:
        return y.astype(np.float32, copy=False)
    try:
        # Compare early-window energy (attack) vs whole
        n = min(len(y), len(ref), int(0.08 * sr))
        if n < 16:
            return y.astype(np.float32, copy=False)
        y_att = _rms(y[:n])
        r_att = _rms(ref[:n])
        y_all = _rms(y) + 1e-8
        r_all = _rms(ref) + 1e-8
        y_ratio = y_att / y_all
        r_ratio = r_att / r_all
        if y_ratio < 1e-6:
            return y.astype(np.float32, copy=False)
        gain = (r_ratio / y_ratio) ** amount
        gain = float(np.clip(gain, 0.7, 1.45))
        # Apply fade on attack region only
        out = y.astype(np.float32, copy=True)
        env = np.linspace(gain, 1.0, n, dtype=np.float32)
        out[:n] *= env
        return out
    except Exception:
        return y.astype(np.float32, copy=False)


def _spectral_envelope(mag: np.ndarray, lifter: int) -> np.ndarray:
    """Smooth log-magnitude envelope via cepstral liftering (per frame).

    Keeps only the low-quefrency part of the cepstrum so the result captures the
    broad spectral shape (formants / tonal balance) and ignores the fine
    harmonic comb — exactly the part the ear reads as timbre/tone.
    """
    log_mag = np.log(mag + 1e-6)
    cep = np.fft.irfft(log_mag, axis=0)
    m = cep.shape[0]
    lifter = int(np.clip(lifter, 1, m // 2))
    liftered = np.zeros_like(cep)
    liftered[:lifter] = cep[:lifter]
    if lifter > 1:
        liftered[-(lifter - 1):] = cep[-(lifter - 1):]
    env = np.fft.rfft(liftered, axis=0).real
    return env[: mag.shape[0]]


def match_spectrum(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    strength: float = 0.55,
    n_fft: int = 2048,
) -> np.ndarray:
    """Transfer the target's tonal envelope onto `y`.

    Instead of forcing the source magnitude onto the target frame-by-frame
    (which smears harmonics and sounds phasey), we match only the smooth
    spectral *envelope*. The source keeps its own harmonic fine structure, so
    the note stays natural but its tone/color moves toward the target.
    """
    if strength <= 0 or len(y) < n_fft // 2 or len(ref) < n_fft // 4:
        return y.astype(np.float32, copy=False)
    try:
        # Shorter windows for note-sized clips
        n_fft = int(min(n_fft, 1 << int(np.floor(np.log2(max(256, len(y)))))))
        hop = max(64, n_fft // 4)
        Y = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop)
        R = librosa.stft(ref.astype(np.float32), n_fft=n_fft, hop_length=hop)
        mag_y = np.abs(Y) + 1e-6
        mag_r = np.abs(R) + 1e-6
        t = min(mag_y.shape[1], mag_r.shape[1])
        mag_y = mag_y[:, :t]
        mag_r = mag_r[:, :t]
        Y = Y[:, :t]

        # ~1.5 kHz envelope resolution: enough for formant/tone, not harmonics.
        lifter = max(6, int(round(n_fft / max(1.0, sr / 1500.0))))
        env_y = _spectral_envelope(mag_y, lifter)
        env_r = _spectral_envelope(mag_r, lifter)
        # Per-frame tonal-balance difference is noisy on short clips; average the
        # envelope over time so we apply one stable "tone curve" per note.
        tone_gain = np.exp(
            strength * (env_r.mean(axis=1, keepdims=True) - env_y.mean(axis=1, keepdims=True))
        )
        tone_gain = np.clip(tone_gain, 0.3, 3.0).astype(np.float32)

        Y_eq = Y * tone_gain
        out = librosa.istft(Y_eq, hop_length=hop, length=len(y))
        return out.astype(np.float32)
    except Exception:
        return y.astype(np.float32, copy=False)


def harmonic_reconstruct(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    strength: float = 0.35,
    n_fft: int = 1024,
) -> np.ndarray:
    """Impose the target's harmonic *fine structure* (which notes sound) onto `y`.

    This is the complement of :func:`match_spectrum`. That function transfers
    only the smooth tonal *envelope* (formants / color) and deliberately leaves
    the source's own harmonic comb in place — which keeps timbre natural but
    also keeps the *wrong pitches*, so the target melody/chords are hard to
    follow. Here we do the opposite split: keep `y`'s broad spectral envelope
    (its timbre) but morph the harmonic detail toward the target's, so the
    reconstruction becomes legible as the target song.

    This is a lightweight, per-note version of NMF spectral mosaicing
    (Driedger, Prätzlich, Müller, "Let It Bee", ISMIR 2015): the target
    spectrum is expressed through the source's sound instead of copied wholesale.
    The source's original phase is retained (`Y * gain`), so no phase estimation
    is needed and the note's onset stays intact.
    """
    if strength <= 0 or len(y) < n_fft // 2 or len(ref) < n_fft // 4:
        return y.astype(np.float32, copy=False)
    try:
        n_fft = int(min(n_fft, 1 << int(np.floor(np.log2(max(256, len(y)))))))
        hop = max(64, n_fft // 4)
        Y = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop)
        R = librosa.stft(ref.astype(np.float32), n_fft=n_fft, hop_length=hop)
        mag_y = np.abs(Y) + 1e-6
        mag_r = np.abs(R) + 1e-6
        t = min(mag_y.shape[1], mag_r.shape[1])
        if t < 1:
            return y.astype(np.float32, copy=False)
        mag_y = mag_y[:, :t]
        mag_r = mag_r[:, :t]
        Y = Y[:, :t]

        # ~1.5 kHz envelope resolution: broad tone/formant shape, not harmonics.
        lifter = max(6, int(round(n_fft / max(1.0, sr / 1500.0))))
        log_y = np.log(mag_y)
        log_r = np.log(mag_r)
        env_y = _spectral_envelope(mag_y, lifter)  # log-domain source timbre
        env_r = _spectral_envelope(mag_r, lifter)  # log-domain target timbre
        detail_y = log_y - env_y  # source harmonic comb (which notes it has)
        detail_r = log_r - env_r  # target harmonic comb (which notes we want)

        b = float(np.clip(strength, 0.0, 1.0))
        # Keep the source envelope (timbre); blend harmonic detail toward target.
        new_log = env_y + (1.0 - b) * detail_y + b * detail_r
        # Bound the per-bin correction so a single note can't be over-EQ'd.
        gain = np.exp(np.clip(new_log - log_y, -1.386, 1.386)).astype(np.float32)
        Y_eq = Y * gain
        out = librosa.istft(Y_eq, hop_length=hop, length=len(y))
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.99:
            out *= 0.99 / peak
        return out.astype(np.float32)
    except Exception:
        return y.astype(np.float32, copy=False)


def residual_fill(
    primary: np.ndarray,
    secondary: np.ndarray,
    ref: np.ndarray | None,
    sr: int,
    *,
    amount: float = 0.35,
    n_fft: int = 1024,
) -> np.ndarray:
    """Add `secondary` only where the target still has energy the primary lacks.

    Plain weighted mixing of multi-song layers muddies Interstellar-style piano
    (phase fights + wrong notes stacked). This keeps the primary intact and
    lets the secondary fill residual spectral holes toward `ref`.
    """
    if amount <= 0 or len(secondary) == 0:
        return primary.astype(np.float32, copy=False)
    n = len(primary)
    sec = secondary[:n] if len(secondary) >= n else np.pad(secondary, (0, n - len(secondary)))
    if ref is None or len(ref) < n_fft // 4 or n < n_fft // 2:
        out = primary + float(amount) * sec.astype(np.float32)
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.99:
            out *= 0.99 / peak
        return out.astype(np.float32)
    try:
        n_fft = int(min(n_fft, 1 << int(np.floor(np.log2(max(256, n))))))
        hop = max(64, n_fft // 4)
        P = librosa.stft(primary.astype(np.float32), n_fft=n_fft, hop_length=hop)
        S = librosa.stft(sec.astype(np.float32), n_fft=n_fft, hop_length=hop)
        R = librosa.stft(ref[:n].astype(np.float32), n_fft=n_fft, hop_length=hop)
        t = min(P.shape[1], S.shape[1], R.shape[1])
        P, S, R = P[:, :t], S[:, :t], R[:, :t]
        mag_p = np.abs(P) + 1e-6
        mag_s = np.abs(S) + 1e-6
        mag_r = np.abs(R) + 1e-6
        need = np.maximum(0.0, mag_r - mag_p)
        supply = np.minimum(need, mag_s)
        gain = amount * (supply / mag_s)
        gain = np.clip(gain, 0.0, amount).astype(np.float32)
        Y = P + S * gain
        out = librosa.istft(Y, hop_length=hop, length=n)
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.99:
            out *= 0.99 / peak
        return out.astype(np.float32)
    except Exception:
        out = primary + float(amount) * sec.astype(np.float32)
        return out.astype(np.float32)


def soft_limit(y: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > ceiling:
        y = y * (ceiling / peak)
    return y.astype(np.float32)


def classify_role(y: np.ndarray, sr: int) -> str:
    """Heuristic role: bass | drums | harmonic | full."""
    if len(y) < sr // 4:
        return "full"
    try:
        # Band energies
        S = np.abs(librosa.stft(y.astype(np.float32), n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        total = float(S.mean()) + 1e-8
        bass = float(S[freqs < 150].mean()) / total
        high = float(S[freqs > 3000].mean()) / total
        # Percussiveness via HPSS
        _h, p = librosa.effects.hpss(y.astype(np.float32))
        perc_ratio = _rms(p) / (_rms(y) + 1e-8)
        if bass > 0.45 and perc_ratio < 0.55:
            return "bass"
        if perc_ratio > 0.55 or (high > 0.25 and perc_ratio > 0.4):
            return "drums"
        if bass < 0.25 and perc_ratio < 0.4:
            return "harmonic"
        return "full"
    except Exception:
        return "full"
