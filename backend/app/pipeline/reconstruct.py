"""Reconstruction: contiguous-run playback + equal-power crossfades.

Literature note (Schwarz CATERPILLAR; unit-selection TTS):
natural neighbors should play as continuous audio (join cost ≈ 0),
and only true switches need smoothing. OLA of every hop creates
phasiness; run-collapse is cleaner and faster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.index import SourceIndex
from app.pipeline.match import TileMatch


@dataclass(slots=True)
class _Run:
    song_id: str
    source_start_s: float
    source_end_s: float
    target_start_s: float
    target_end_s: float


def _equal_power_xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Crossfade last n of a into first n of b (equal-power)."""
    if n <= 0 or len(a) == 0:
        return b.astype(np.float32, copy=False)
    if len(b) == 0:
        return a.astype(np.float32, copy=False)
    n = min(n, len(a), len(b))
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]]).astype(np.float32)


def _collapse_runs(tiles: list[TileMatch], *, hop_s: float, window_s: float) -> list[_Run]:
    if not tiles:
        return []
    runs: list[_Run] = []
    cur = _Run(
        song_id=tiles[0].song_id,
        source_start_s=tiles[0].source_start_s,
        source_end_s=tiles[0].source_start_s + window_s,
        target_start_s=tiles[0].target_start_s,
        target_end_s=tiles[0].target_start_s + window_s,
    )
    for t in tiles[1:]:
        contiguous = (
            t.song_id == cur.song_id
            and abs((cur.source_end_s - window_s + hop_s) - t.source_start_s) < hop_s * 0.6
        )
        if contiguous:
            # Extend run: keep source continuous; target advances by hop each tile
            cur.source_end_s = t.source_start_s + window_s
            cur.target_end_s = t.target_start_s + window_s
        else:
            runs.append(cur)
            cur = _Run(
                song_id=t.song_id,
                source_start_s=t.source_start_s,
                source_end_s=t.source_start_s + window_s,
                target_start_s=t.target_start_s,
                target_end_s=t.target_start_s + window_s,
            )
    runs.append(cur)
    return runs


def _slice_song(song: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(round(start_s * sr)))
    b = max(a + 1, int(round(end_s * sr)))
    chunk = song[a:b]
    return chunk.astype(np.float32, copy=False)


def reconstruct_ola(
    tiles: list[TileMatch],
    source: SourceIndex,
    *,
    sr: int,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    target_duration_s: float | None = None,
    crossfade_ms: float = 60.0,
) -> np.ndarray:
    """Prefer contiguous source runs; equal-power crossfade only at switches."""
    if not tiles:
        return np.zeros(0, dtype=np.float32)

    if target_duration_s is None:
        target_duration_s = tiles[-1].target_start_s + window_s
    out_len = int(round(target_duration_s * sr))
    xfade = int(sr * crossfade_ms / 1000.0)

    runs = _collapse_runs(tiles, hop_s=hop_s, window_s=window_s)
    use_songs = bool(source.songs)

    # Build timeline via placing runs with crossfades
    out = np.zeros(out_len + xfade + 8, dtype=np.float32)

    for run in runs:
        target_len = max(1, int(round((run.target_end_s - run.target_start_s) * sr)))
        if use_songs and run.song_id in source.songs:
            clip = _slice_song(
                source.songs[run.song_id], sr, run.source_start_s, run.source_end_s
            )
        else:
            # Fallback: OLA the tiles that fall in this run (should be rare)
            clip = _fallback_ola_run(tiles, source, run, sr, window_s, hop_s)

        # Fit clip length to target span (trim or pad — avoid time-stretch for quality)
        if len(clip) > target_len:
            clip = clip[:target_len]
        elif len(clip) < target_len:
            clip = np.pad(clip, (0, target_len - len(clip)))

        start = int(round(run.target_start_s * sr))
        end = start + len(clip)
        if start >= out_len:
            continue
        if end > len(out):
            clip = clip[: len(out) - start]
            end = start + len(clip)

        if start == 0 or xfade <= 0:
            out[start:end] = clip
        else:
            # Crossfade into existing audio
            overlap = min(xfade, start, len(clip))
            if overlap > 0:
                prev = out[start - overlap : start].copy()
                mixed = _equal_power_xfade(prev, clip[:overlap], overlap)
                # mixed = prev[:-overlap]? _equal_power_xfade returns prev[:-n]+mid+clip[n:]
                # Here prev is only the overlap region — simpler mix:
                t = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
                out[start - overlap : start] = prev * np.cos(t) + clip[:overlap] * np.sin(t)
                out[start:end] = 0.0
                out[start : start + (len(clip) - overlap)] = clip[overlap:]
            else:
                out[start:end] = clip

    return out[:out_len].astype(np.float32)


def _fallback_ola_run(
    tiles: list[TileMatch],
    source: SourceIndex,
    run: _Run,
    sr: int,
    window_s: float,
    hop_s: float,
) -> np.ndarray:
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    hann = np.hanning(win).astype(np.float32)
    dur = max(win, int(round((run.target_end_s - run.target_start_s) * sr)))
    acc = np.zeros(dur + win, dtype=np.float32)
    wsum = np.zeros_like(acc)
    for t in tiles:
        if t.song_id != run.song_id:
            continue
        if t.target_start_s < run.target_start_s - 1e-6:
            continue
        if t.target_start_s > run.target_end_s + 1e-6:
            continue
        local = int(round((t.target_start_s - run.target_start_s) * sr))
        clip = source.waveforms[t.source_id]
        if len(clip) != win:
            c = np.zeros(win, np.float32)
            c[: min(win, len(clip))] = clip[: min(win, len(clip))]
            clip = c
        acc[local : local + win] += clip * hann
        wsum[local : local + win] += hann
    nz = wsum > 1e-6
    acc[nz] /= wsum[nz]
    return acc[:dur]
