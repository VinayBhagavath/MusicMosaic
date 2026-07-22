"""Note-like segmentation: onset / beat aligned units (MIDI-note mental model).

Target and sources are chopped into short events — closer to sequencer notes
than long collage strips — so pitch-shifting and stacking can rebuild the song.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(slots=True)
class Segment:
    song_id: str
    index: int
    start_s: float
    end_s: float
    waveform: np.ndarray  # shape (n_samples,), float32


def _onset_starts(
    y: np.ndarray,
    sr: int,
    *,
    hop_s: float,
    window_s: float,
) -> np.ndarray | None:
    """Snap unit starts to musical onsets (note attacks), densified by hop_s."""
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units="time",
            backtrack=True,
            delta=0.07,
            wait=max(1, int(0.04 * sr / 512)),
        )
        onsets = np.asarray(onsets, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if onsets.size < 4:
        return None

    dur = len(y) / sr
    min_gap = max(0.08, hop_s * 0.35)
    starts: list[float] = [0.0]
    for o in onsets:
        if o <= starts[-1] + min_gap:
            continue
        # Fill large gaps so we don't miss sustained pads / held notes
        while o - starts[-1] > hop_s * 1.6:
            starts.append(starts[-1] + hop_s)
        starts.append(float(o))

    while starts[-1] + hop_s < dur - 1e-6:
        starts.append(starts[-1] + hop_s)

    max_start = max(0.0, dur - window_s * 0.35)
    out = [s for s in starts if s <= max_start]
    return np.asarray(out, dtype=np.float64) if len(out) >= 3 else None


def _phrase_starts(
    y: np.ndarray,
    sr: int,
    *,
    hop_s: float,
    window_s: float,
    bars_per_phrase: int = 2,
    beats_per_bar: int = 4,
) -> np.ndarray | None:
    """Snap starts to beat grid, preferring bar/phrase boundaries when confident."""
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beats = np.asarray(beats, dtype=np.float64).reshape(-1)
        tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    except Exception:
        return None
    if beats.size < 8:
        return None

    phrase_beats = max(1, bars_per_phrase * beats_per_bar)
    preferred = set(range(0, len(beats), phrase_beats))
    preferred |= set(range(0, len(beats), beats_per_bar))

    starts: list[float] = [0.0]
    for i, b in enumerate(beats):
        min_gap = hop_s * (0.28 if i in preferred else 0.45)
        if b <= starts[-1] + min_gap:
            continue
        while b - starts[-1] > hop_s * 1.7:
            starts.append(starts[-1] + hop_s)
        starts.append(float(b))

    dur = len(y) / sr
    while starts[-1] + hop_s < dur - 1e-6:
        starts.append(starts[-1] + hop_s)

    max_start = max(0.0, dur - window_s * 0.35)
    out = [s for s in starts if s <= max_start]
    if tempo < 40 or tempo > 220:
        return np.asarray(out, dtype=np.float64) if len(out) >= 3 else None
    return np.asarray(out, dtype=np.float64) if len(out) >= 3 else None


def _beat_starts(y: np.ndarray, sr: int, *, hop_s: float, window_s: float) -> np.ndarray | None:
    return _phrase_starts(y, sr, hop_s=hop_s, window_s=window_s)


def segment_audio(
    y: np.ndarray,
    sr: int,
    song_id: str,
    *,
    window_s: float = 0.45,
    hop_s: float = 0.22,
    beat_sync: bool = True,
    phrase_sync: bool = False,
    onset_sync: bool = True,
    variable_length: bool = False,
) -> list[Segment]:
    """Split `y` into short note-like units; prefer onsets, then beats, then grid."""
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("window_s and hop_s must be positive")

    n = len(y)
    if n == 0:
        return []

    starts_s: np.ndarray | None = None
    if onset_sync:
        starts_s = _onset_starts(y, sr, hop_s=hop_s, window_s=window_s)
    if starts_s is None and beat_sync:
        if phrase_sync:
            starts_s = _phrase_starts(y, sr, hop_s=hop_s, window_s=window_s)
        else:
            starts_s = _beat_starts(y, sr, hop_s=hop_s, window_s=window_s)

    if starts_s is None:
        starts_s = np.arange(0, max(n - 1, 1), hop) / sr

    segments: list[Segment] = []
    for i, start_s in enumerate(starts_s):
        start = int(round(float(start_s) * sr))
        if start >= n:
            break
        if variable_length and i + 1 < len(starts_s):
            # Preserve the musical event duration, with a short release tail for
            # crossfading. Bound pathological beat/onset tracker gaps.
            next_start = int(round(float(starts_s[i + 1]) * sr))
            release = int(round(min(0.08, hop_s * 0.3) * sr))
            event_n = int(np.clip(next_start - start + release, sr * 0.12, win * 2))
            end = start + event_n
        else:
            end = start + win
        chunk = y[start:end]
        expected = end - start
        if len(chunk) < expected:
            chunk = np.pad(chunk, (0, expected - len(chunk)))
        segments.append(
            Segment(
                song_id=song_id,
                index=i,
                start_s=start / sr,
                end_s=min(end, n) / sr,
                waveform=chunk.astype(np.float32, copy=False),
            )
        )
        if end >= n:
            break
    return segments
