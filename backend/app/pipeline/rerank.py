"""Post-Viterbi per-note re-rank by *actual* post-transform spectral distance.

The Viterbi beam scores candidates on cached feature similarity (chroma /
timbre / dynamics) computed *before* any pitch/time transform. That is fast but
approximate: a candidate that looks slightly worse in feature space can, after
being pitch-shifted and length-fit toward the target window, actually be the
better acoustic reconstruction of that note.

This pass keeps the Viterbi path exact and only re-ranks *within* each note's
retained beam: it transforms the top candidates the same way reconstruction
will, measures a real spectral distance to the target window, and swaps to a
clearly better candidate. To bound the extra Rubber Band cost it only touches
ambiguous / weaker notes and caches every transform.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.index import SourceIndex
from app.pipeline.match import LayerMatch, TileMatch
from app.pipeline.transform import (
    estimate_f0_hz,
    fit_length,
    prepare_clip,
)


def _spectral_distance(y: np.ndarray, ref: np.ndarray, sr: int) -> float:
    """Multi-resolution log-magnitude L1 distance; lower is a closer match."""
    import librosa

    n = min(len(y), len(ref))
    if n < 256:
        return float("inf")
    y = y[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)
    dists: list[float] = []
    for n_fft in (512, 1024):
        if n < n_fft:
            continue
        hop = n_fft // 4
        sy = np.log1p(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)))
        sr_ = np.log1p(np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)))
        m = min(sy.shape[1], sr_.shape[1])
        if m < 1:
            continue
        scale = float(np.mean(np.abs(sr_[:, :m]))) + 1e-6
        dists.append(float(np.mean(np.abs(sy[:, :m] - sr_[:, :m])) / scale))
    return float(np.mean(dists)) if dists else float("inf")


def _target_slice(target: np.ndarray, sr: int, start_s: float, n: int) -> np.ndarray | None:
    a = max(0, int(round(start_s * sr)))
    b = min(len(target), a + n)
    if b <= a:
        return None
    out = np.zeros(n, dtype=np.float32)
    out[: b - a] = target[a:b]
    return out


def _candidate_clip(
    source: SourceIndex,
    source_id: int,
    key_shift: float,
    *,
    sr: int,
    win: int,
    ref: np.ndarray,
    shift_cache: dict,
    f0_cache: dict,
    ref_f0: float | None,
) -> np.ndarray | None:
    meta = source.meta[source_id]
    song = source.songs.get(meta.song_id) if source.songs else None
    if song is None:
        if 0 <= source_id < len(source.waveforms):
            song = source.waveforms[source_id]
        else:
            return None
    a = int(round(meta.start_s * sr))
    source_n = max(64, int(round((meta.end_s - meta.start_s) * sr)))
    src_chunk = song[a : a + source_n]
    if len(src_chunk) < source_n:
        src_chunk = fit_length(src_chunk.astype(np.float32), source_n)

    # Mirror reconstruction's pitch policy: prefer per-window F0 match.
    steps = float(key_shift)
    if f0_cache is not None:
        if source_id not in f0_cache:
            f0_cache[source_id] = estimate_f0_hz(src_chunk, sr)
        src_f0 = f0_cache[source_id]
        if src_f0 is not None and ref_f0 is not None and src_f0 > 1e-6:
            steps = float(np.clip(12.0 * np.log2(ref_f0 / src_f0), -12.0, 12.0))
    while steps > 6:
        steps -= 12
    while steps < -6:
        steps += 12

    if song is not None and len(song) > win:
        return prepare_clip(
            song,
            sr,
            meta.start_s,
            target_n=win,
            source_n=source_n,
            n_steps=steps,
            cache=shift_cache,
            cache_key=f"rerank:{meta.song_id}:{round(steps, 1)}",
        )
    return fit_length(src_chunk.astype(np.float32), win)


def rerank_tiles_spectral(
    tiles: list[TileMatch],
    source: SourceIndex,
    target_audio: np.ndarray | None,
    sr: int,
    *,
    window_s: float = 0.45,
    top_m: int = 4,
    sim_ceiling: float = 0.985,
    beam_margin: float = 0.04,
    improve_margin: float = 0.04,
    max_tiles: int = 160,
) -> int:
    """Re-rank each note's beam by post-transform spectral distance in place.

    Returns the number of notes whose primary source was swapped.

    Feature similarity on this corpus often sits in the mid-0.9s even when the
    acoustic match is wrong, so a hard ``sim < 0.9`` gate never fires. Instead
    we re-rank notes where the beam is *ambiguous* (runner-up within
    ``beam_margin`` of the winner) or the absolute score is below
    ``sim_ceiling``, capped at ``max_tiles`` weakest notes for speed.
    """
    if target_audio is None or not tiles:
        return 0

    base_win = int(round(window_s * sr))
    shift_cache: dict = {}
    f0_cache: dict = {}
    ref_f0_cache: dict = {}
    swaps = 0

    # Score how worth re-ranking each tile is (lower = more ambiguous / weaker).
    ranked: list[tuple[float, int]] = []
    for i, tile in enumerate(tiles):
        if len(tile.layers) > 1 or not tile.candidates or len(tile.candidates) < 2:
            continue
        sims = [c[2] for c in tile.candidates if c[0] != tile.source_id]
        runner = max(sims) if sims else -1.0
        margin = tile.similarity - runner
        # Prefer weak or close races.
        priority = min(tile.similarity, 0.5 + 0.5 * max(0.0, margin))
        if tile.similarity < sim_ceiling or margin < beam_margin:
            ranked.append((priority, i))
    ranked.sort()
    consider = {i for _, i in ranked[: max(1, max_tiles)]}

    for ti, tile in enumerate(tiles):
        if ti not in consider:
            continue
        win = int(
            round(
                np.clip(
                    tile.target_duration_s if tile.target_duration_s is not None else window_s,
                    0.12,
                    window_s * 2.0,
                )
                * sr
            )
        )
        win = max(base_win // 2, win)
        ref = _target_slice(target_audio, sr, tile.target_start_s, win)
        if ref is None:
            continue
        key = round(tile.target_start_s, 3)
        if key not in ref_f0_cache:
            ref_f0_cache[key] = estimate_f0_hz(ref, sr)
        ref_f0 = ref_f0_cache[key]

        # Deduplicate candidates by source id, keep beam order (best-sim first).
        seen: set[int] = set()
        ordered: list[tuple[int, float]] = []
        for cid, shift, _sim in tile.candidates:
            if cid in seen:
                continue
            seen.add(cid)
            ordered.append((cid, shift))
        # Always evaluate the current pick as the baseline.
        current = (tile.source_id, tile.key_shift)
        cand_list = [current] + [c for c in ordered if c[0] != tile.source_id]
        cand_list = cand_list[: max(1, top_m)]

        best_id, best_shift, best_d = tile.source_id, tile.key_shift, None
        cur_d = None
        for cid, shift in cand_list:
            try:
                clip = _candidate_clip(
                    source,
                    cid,
                    shift,
                    sr=sr,
                    win=win,
                    ref=ref,
                    shift_cache=shift_cache,
                    f0_cache=f0_cache,
                    ref_f0=ref_f0,
                )
            except Exception:
                clip = None
            if clip is None:
                continue
            d = _spectral_distance(clip, ref, sr)
            if cid == tile.source_id:
                cur_d = d
            if best_d is None or d < best_d:
                best_id, best_shift, best_d = cid, shift, d

        if best_id == tile.source_id or best_d is None or cur_d is None:
            continue
        # Require a meaningful improvement to justify breaking the Viterbi pick.
        if best_d > cur_d * (1.0 - improve_margin):
            continue

        meta = source.meta[best_id]
        prev_role = tile.layers[0].role if tile.layers else "full"
        tile.source_id = best_id
        tile.song_id = meta.song_id
        tile.source_start_s = meta.start_s
        tile.key_shift = best_shift
        # Keep existing secondary layers when possible; refresh primary only.
        if len(tile.layers) <= 1:
            tile.layers = [
                LayerMatch(
                    source_id=best_id,
                    song_id=meta.song_id,
                    source_start_s=meta.start_s,
                    similarity=tile.similarity,
                    weight=1.0,
                    key_shift=best_shift,
                    role=prev_role,
                )
            ]
        else:
            primary_w = float(tile.layers[0].weight)
            rest = [ly for ly in tile.layers[1:] if ly.source_id != best_id]
            tile.layers = [
                LayerMatch(
                    source_id=best_id,
                    song_id=meta.song_id,
                    source_start_s=meta.start_s,
                    similarity=tile.similarity,
                    weight=primary_w,
                    key_shift=best_shift,
                    role=prev_role,
                ),
                *rest,
            ]
        swaps += 1

    return swaps
