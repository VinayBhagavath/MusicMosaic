"""Pitch / tempo transforms for target cohesion.

Tries Rubber Band (pyrubberband) when available; falls back to librosa
phase-vocoder. Supports fractional semitone shifts (cents) and per-window F0.
"""

from __future__ import annotations

import shutil

import librosa
import numpy as np

_HAS_RB: bool | None = None


def _rubberband_ok() -> bool:
    global _HAS_RB
    if _HAS_RB is not None:
        return _HAS_RB
    try:
        import pyrubberband  # noqa: F401

        _HAS_RB = shutil.which("rubberband") is not None
    except Exception:
        _HAS_RB = False
    return _HAS_RB


def require_rubberband() -> None:
    """Fail early when fidelity transforms cannot use Rubber Band."""
    if not _rubberband_ok():
        raise RuntimeError(
            "Rubber Band is required for fidelity mode. Install the system binary "
            "(`brew install rubberband` on macOS) and the Python package "
            "(`uv sync --extra cohesion`)."
        )


def estimate_tuning_cents(y: np.ndarray, sr: int) -> float:
    """Fine tuning offset in cents (−50..+50). 0 = A440-ish grid."""
    if len(y) < sr // 4:
        return 0.0
    try:
        t = float(librosa.estimate_tuning(y=y.astype(np.float32), sr=sr))
        return float(np.clip(t * 100.0, -50.0, 50.0))
    except Exception:
        return 0.0


def estimate_f0_hz(y: np.ndarray, sr: int) -> float | None:
    """Median voiced F0 in Hz, or None if unpitched / too short.

    Uses YIN (not pYIN): ~10–50× faster on note-sized windows and accurate
    enough for per-tile pitch alignment. pYIN's probabilistic voicing is not
    worth the reconstruct-stage cost at mosaic scale.
    """
    if len(y) < sr // 5:
        return None
    try:
        f0 = librosa.yin(
            y.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        vals = f0[np.isfinite(f0) & (f0 > 40) & (f0 < 2000)]
        if vals.size < 3:
            return None
        med = float(np.median(vals))
        if med < 40 or med > 2000:
            return None
        return med
    except Exception:
        return None


def f0_semitone_delta(src: np.ndarray, ref: np.ndarray, sr: int) -> float | None:
    """How many semitones to shift `src` so its F0 matches `ref`."""
    f_src = estimate_f0_hz(src, sr)
    f_ref = estimate_f0_hz(ref, sr)
    if f_src is None or f_ref is None or f_src < 1e-6:
        return None
    delta = 12.0 * np.log2(f_ref / f_src)
    if abs(delta) > 14:
        while delta > 6:
            delta -= 12
        while delta < -6:
            delta += 12
        if abs(delta) > 8:
            return None
    return float(delta)


def onset_peak_index(y: np.ndarray, sr: int) -> int:
    """Sample index of the strongest early onset (attack) in `y`."""
    if len(y) < 64:
        return 0
    try:
        hop = 256
        env = librosa.onset.onset_strength(y=y.astype(np.float32), sr=sr, hop_length=hop)
        if env.size == 0:
            return int(np.argmax(np.abs(y)))
        limit = max(1, int(0.7 * env.size))
        frame = int(np.argmax(env[:limit]))
        return int(np.clip(frame * hop, 0, len(y) - 1))
    except Exception:
        return int(np.argmax(np.abs(y)))


def align_onset_to_ref(
    y: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    max_shift_s: float = 0.04,
) -> np.ndarray:
    """Micro-align an attack without allowing destructive large truncations."""
    if len(y) < 64 or len(ref) < 64:
        return y.astype(np.float32, copy=False)
    yi = onset_peak_index(y, sr)
    ri = onset_peak_index(ref[: len(y)] if len(ref) >= len(y) else ref, sr)
    shift = int(ri - yi)
    if abs(shift) < max(8, int(0.004 * sr)):
        return y.astype(np.float32, copy=False)
    # Segmentation already starts clips near attacks. A larger disagreement is
    # usually an onset-tracker error; shifting by a third of a short tile used
    # to erase note tails and create audible zero-filled gaps.
    max_shift = min(len(y) // 3, max(1, int(round(max_shift_s * sr))))
    shift = int(np.clip(shift, -max_shift, max_shift))
    src = y.astype(np.float32, copy=False)
    out = np.zeros_like(src)
    guard = min(max(8, int(round(0.004 * sr))), max(1, len(src) // 8))
    ramp = np.sin(np.linspace(0.0, 0.5 * np.pi, guard, dtype=np.float32))
    if shift > 0:
        out[shift:] = src[:-shift]
        n = min(guard, len(out) - shift)
        out[shift : shift + n] *= ramp[:n]
    elif shift < 0:
        out[:shift] = src[-shift:]
        edge = len(out) + shift
        n = min(guard, edge)
        if n > 0:
            out[edge - n : edge] *= ramp[:n][::-1]
    else:
        out[:] = src
    return out


def pitch_shift(
    y: np.ndarray,
    sr: int,
    n_steps: float,
    *,
    formant_preserve: bool = True,
) -> np.ndarray:
    """Pitch-shift by fractional semitones (e.g. 2.3 = 2 semitones + 30 cents)."""
    if abs(n_steps) < 1e-4 or len(y) < 64:
        return y.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    require_rubberband()
    import pyrubberband as pyrb

    rbargs: dict = {}
    if formant_preserve:
        rbargs["--formant"] = ""
    try:
        out = pyrb.pitch_shift(y, sr, n_steps, rbargs=rbargs or None)
    except Exception as exc:
        raise RuntimeError(f"Rubber Band pitch shift failed: {exc}") from exc
    return np.asarray(out, dtype=np.float32)


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch by rate (>1 = faster/shorter). Pitch preserved."""
    if abs(rate - 1.0) < 1e-4 or len(y) < 64:
        return y.astype(np.float32, copy=False)
    rate = float(np.clip(rate, 0.5, 2.0))
    y = y.astype(np.float32, copy=False)
    require_rubberband()
    import pyrubberband as pyrb

    try:
        out = pyrb.time_stretch(y, sr, rate)
    except Exception as exc:
        raise RuntimeError(f"Rubber Band time stretch failed: {exc}") from exc
    return np.asarray(out, dtype=np.float32)


def pitch_time_transform(
    y: np.ndarray,
    sr: int,
    *,
    n_steps: float,
    rate: float,
    formant_preserve: bool = True,
) -> np.ndarray:
    """Apply pitch and duration changes in one Rubber Band process."""
    if abs(n_steps) < 1e-4:
        return time_stretch(y, sr, rate)
    if abs(rate - 1.0) < 1e-4:
        return pitch_shift(y, sr, n_steps, formant_preserve=formant_preserve)
    require_rubberband()
    rate = float(np.clip(rate, 0.5, 2.0))
    import pyrubberband.pyrb as pyrb

    rbargs: dict[str, str | float] = {
        "--pitch": float(n_steps),
        "--tempo": rate,
    }
    if formant_preserve:
        rbargs["--formant"] = ""
    try:
        out = pyrb.__rubberband(y.astype(np.float32, copy=False), sr, **rbargs)
    except Exception as exc:
        raise RuntimeError(f"Rubber Band pitch/time transform failed: {exc}") from exc
    return np.asarray(out, dtype=np.float32)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Pad or trim to exactly n samples."""
    if len(y) == n:
        return y.astype(np.float32, copy=False)
    if len(y) > n:
        return y[:n].astype(np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)
    out[: len(y)] = y
    return out


def prepare_clip(
    song: np.ndarray,
    sr: int,
    start_s: float,
    *,
    target_n: int,
    source_n: int | None = None,
    n_steps: float = 0.0,
    pad_s: float = 0.12,
    cache: dict | None = None,
    cache_key: str = "",
    formant_preserve: bool = True,
) -> np.ndarray:
    """Slice one source event → pitch-shift → time-stretch to ``target_n``.

    ``source_n`` is the detected source-note duration. When omitted, the
    historical fixed-window behavior is retained.
    """
    if target_n <= 0:
        return np.zeros(0, dtype=np.float32)

    a = max(0, int(round(start_s * sr)))
    pad = int(round(pad_s * sr))
    raw_n = max(1, int(source_n)) if source_n is not None else target_n
    a0 = max(0, a - pad)
    b0 = min(len(song), a + raw_n + pad)
    key = (
        cache_key,
        a0,
        b0,
        raw_n,
        round(n_steps, 1),
        target_n,
        formant_preserve,
    )
    if cache is not None and key in cache:
        return cache[key]

    chunk = song[a0:b0].astype(np.float32, copy=False)
    if len(chunk) < 64:
        out = fit_length(chunk, target_n)
        if cache is not None:
            cache[key] = out
        return out

    left = a - a0
    rate = raw_n / float(target_n)
    needs_pitch = abs(n_steps) >= 1e-4
    needs_stretch = abs(rate - 1.0) >= 1e-4
    if needs_pitch and needs_stretch:
        transformed = pitch_time_transform(
            chunk,
            sr,
            n_steps=n_steps,
            rate=rate,
            formant_preserve=formant_preserve,
        )
        # Padding was transformed too; map the original event start through the
        # tempo ratio before extracting the target-length core.
        left_out = int(round(left / float(np.clip(rate, 0.5, 2.0))))
        core = transformed[left_out : left_out + target_n]
    else:
        shifted = (
            pitch_shift(chunk, sr, n_steps, formant_preserve=formant_preserve)
            if needs_pitch
            else chunk
        )
        core = shifted[left : left + raw_n]
        if len(core) < max(64, raw_n // 4):
            core = fit_length(shifted, raw_n)
        if needs_stretch and len(core) > 32:
            core = time_stretch(core, sr, rate)
    out = fit_length(core, target_n)
    if cache is not None:
        cache[key] = out
    return out
