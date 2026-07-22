"""Overlapping window segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Segment:
    song_id: str
    index: int
    start_s: float
    end_s: float
    waveform: np.ndarray  # shape (n_samples,), float32


def segment_audio(
    y: np.ndarray,
    sr: int,
    song_id: str,
    *,
    window_s: float = 0.5,
    hop_s: float = 0.25,
) -> list[Segment]:
    """Split `y` into overlapping windows; zero-pad the final incomplete window."""
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if win <= 0 or hop <= 0:
        raise ValueError("window_s and hop_s must be positive")

    segments: list[Segment] = []
    n = len(y)
    if n == 0:
        return segments

    for i, start in enumerate(range(0, max(n - 1, 1), hop)):
        end = start + win
        chunk = y[start:end]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
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
