"""Reconstruction tests: contiguous runs + length."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta
from app.pipeline.match import TileMatch
from app.pipeline.reconstruct import reconstruct_ola


def test_contiguous_run_reconstruction():
    sr = 22050
    hop_s = 0.5
    window_s = 1.0
    # 3s of audio at constant 0.5
    song = np.ones(int(3.0 * sr), dtype=np.float32) * 0.5
    n_tiles = 4
    meta = [
        SourceMeta("A", i * hop_s, i * hop_s + window_s, i) for i in range(n_tiles)
    ]
    win = int(window_s * sr)
    waveforms = [song[int(i * hop_s * sr) : int(i * hop_s * sr) + win].copy() for i in range(n_tiles)]
    for i, w in enumerate(waveforms):
        if len(w) < win:
            waveforms[i] = np.pad(w, (0, win - len(w)))

    chroma = np.zeros((n_tiles, 12), np.float32)
    chroma[:, 0] = 1.0
    timbre = np.zeros((n_tiles, 26), np.float32)
    timbre[:, 0] = 1.0
    pack = EmbPack(chroma=chroma, timbre=timbre, energy=np.ones((n_tiles, 4), np.float32))
    source = SourceIndex(
        meta=meta,
        waveforms=waveforms,
        pack=pack,
        songs={"A": song},
        sr=sr,
    )
    tiles = [
        TileMatch(i, i * hop_s, i, "A", i * hop_s, 0.9) for i in range(n_tiles)
    ]
    target_dur = (n_tiles - 1) * hop_s + window_s
    out = reconstruct_ola(
        tiles, source, sr=sr, window_s=window_s, hop_s=hop_s, target_duration_s=target_dur
    )
    assert len(out) == int(round(target_dur * sr))
    # Continuous constant signal should stay near 0.5 away from edges
    mid = out[len(out) // 2]
    assert 0.35 < mid < 0.65
