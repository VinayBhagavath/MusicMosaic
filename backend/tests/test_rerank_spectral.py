"""Tests for the post-Viterbi per-note spectral re-rank."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta
from app.pipeline.match import TileMatch
from app.pipeline.rerank import _spectral_distance, rerank_tiles_spectral


def _sine(sr: int, hz: float, seconds: float) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_spectral_distance_smaller_for_closer_clip():
    sr = 22050
    ref = _sine(sr, 300.0, 0.5)
    close = _sine(sr, 300.0, 0.5)
    far = _sine(sr, 900.0, 0.5)
    assert _spectral_distance(close, ref, sr) < _spectral_distance(far, ref, sr)


def test_rerank_swaps_to_spectrally_closer_candidate():
    sr = 22050
    dur = 1.0
    # Same fundamental (so no pitch shift / Rubber Band needed), different
    # timbre: A is bright (extra 3 kHz partial), B is a clean 300 Hz tone that
    # matches the target exactly.
    base = _sine(sr, 300.0, dur)
    song_a = (base + 0.3 * _sine(sr, 3000.0, dur)).astype(np.float32)
    song_b = base.copy()
    target = base.copy()  # target matches B

    meta = [SourceMeta("A", 0.0, dur, 0), SourceMeta("B", 0.0, dur, 1)]
    win = int(0.5 * sr)
    waveforms = [song_a[:win].copy(), song_b[:win].copy()]
    chroma = np.eye(2, 12, dtype=np.float32)
    timbre = np.eye(2, 26, dtype=np.float32)
    pack = EmbPack(chroma=chroma, timbre=timbre, energy=np.ones((2, 4), np.float32))
    source = SourceIndex(
        meta=meta, waveforms=waveforms, pack=pack,
        songs={"A": song_a, "B": song_b}, sr=sr,
    )

    # Viterbi picked A, but keeps B in the beam. Similarity is mediocre so the
    # note is eligible for re-rank.
    tile = TileMatch(
        target_idx=0,
        target_start_s=0.0,
        source_id=0,
        song_id="A",
        source_start_s=0.0,
        similarity=0.5,
        key_shift=0.0,
        target_duration_s=0.5,
        candidates=[(0, 0.0, 0.5), (1, 0.0, 0.48)],
    )

    swaps = rerank_tiles_spectral(
        [tile], source, target, sr, window_s=0.5, top_m=4
    )
    assert swaps == 1
    assert tile.source_id == 1
    assert tile.song_id == "B"
    assert tile.layers and tile.layers[0].song_id == "B"


def test_rerank_keeps_confident_tiles_untouched():
    sr = 22050
    dur = 1.0
    song_a = _sine(sr, 300.0, dur)
    song_b = _sine(sr, 305.0, dur)
    target = song_a.copy()
    meta = [SourceMeta("A", 0.0, dur, 0), SourceMeta("B", 0.0, dur, 1)]
    win = int(0.5 * sr)
    pack = EmbPack(
        chroma=np.eye(2, 12, dtype=np.float32),
        timbre=np.eye(2, 26, dtype=np.float32),
        energy=np.ones((2, 4), np.float32),
    )
    source = SourceIndex(
        meta=meta,
        waveforms=[song_a[:win].copy(), song_b[:win].copy()],
        pack=pack,
        songs={"A": song_a, "B": song_b},
        sr=sr,
    )
    # Huge beam margin + near-perfect sim → not worth re-ranking.
    tile = TileMatch(
        target_idx=0, target_start_s=0.0, source_id=0, song_id="A",
        source_start_s=0.0, similarity=0.995, key_shift=0.0,
        target_duration_s=0.5, candidates=[(0, 0.0, 0.995), (1, 0.0, 0.70)],
    )
    swaps = rerank_tiles_spectral([tile], source, target, sr, window_s=0.5)
    assert swaps == 0
    assert tile.source_id == 0
