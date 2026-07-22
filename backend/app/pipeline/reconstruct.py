"""Overlap-add reconstruction from matched source clips."""

from __future__ import annotations

import numpy as np

from app.pipeline.index import SourceIndex
from app.pipeline.match import TileMatch


def reconstruct_ola(
    tiles: list[TileMatch],
    source: SourceIndex,
    *,
    sr: int,
    window_s: float = 0.5,
    hop_s: float = 0.25,
    target_duration_s: float | None = None,
) -> np.ndarray:
    """Hann-windowed overlap-add of chosen source waveforms onto the target timeline."""
    if not tiles:
        return np.zeros(0, dtype=np.float32)

    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    hann = np.hanning(win).astype(np.float32)

    if target_duration_s is None:
        last = tiles[-1]
        target_duration_s = last.target_start_s + window_s
    out_len = int(round(target_duration_s * sr))
    acc = np.zeros(out_len + win, dtype=np.float32)
    weight = np.zeros(out_len + win, dtype=np.float32)

    for tile in tiles:
        start = int(round(tile.target_start_s * sr))
        clip = source.waveforms[tile.source_id]
        if len(clip) != win:
            # Defensive resize (should already match)
            c = np.zeros(win, dtype=np.float32)
            n = min(win, len(clip))
            c[:n] = clip[:n]
            clip = c
        end = start + win
        acc[start:end] += clip * hann
        weight[start:end] += hann

    nz = weight > 1e-6
    acc[nz] /= weight[nz]
    return acc[:out_len].astype(np.float32)
