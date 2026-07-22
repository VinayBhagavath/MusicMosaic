"""OLA reconstruction tests."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta
from app.pipeline.match import TileMatch
from app.pipeline.reconstruct import reconstruct_ola


def test_ola_length_and_energy():
    sr = 22050
    win = int(0.5 * sr)
    hop_s = 0.25
    n_tiles = 8
    waveforms = [np.ones(win, dtype=np.float32) * 0.5 for _ in range(n_tiles)]
    meta = [
        SourceMeta(song_id="A", start_s=i * hop_s, end_s=i * hop_s + 0.5, segment_idx=i)
        for i in range(n_tiles)
    ]
    pack = EmbPack(
        chroma=np.eye(12, dtype=np.float32)[:n_tiles],
        timbre=np.eye(26, dtype=np.float32)[:n_tiles],
        energy=np.ones((n_tiles, 4), dtype=np.float32),
    )
    # pad chroma if n_tiles < 12
    if n_tiles < 12:
        chroma = np.zeros((n_tiles, 12), np.float32)
        chroma[:, 0] = 1.0
        timbre = np.zeros((n_tiles, 26), np.float32)
        timbre[:, 0] = 1.0
        pack = EmbPack(chroma=chroma, timbre=timbre, energy=np.ones((n_tiles, 4), np.float32))

    source = SourceIndex(meta=meta, waveforms=waveforms, pack=pack)
    tiles = [
        TileMatch(
            target_idx=i,
            target_start_s=i * hop_s,
            source_id=i,
            song_id="A",
            source_start_s=i * hop_s,
            similarity=0.9,
        )
        for i in range(n_tiles)
    ]
    target_dur = (n_tiles - 1) * hop_s + 0.5
    out = reconstruct_ola(
        tiles, source, sr=sr, window_s=0.5, hop_s=hop_s, target_duration_s=target_dur
    )
    assert len(out) == int(round(target_dur * sr))
    mid = out[len(out) // 2]
    assert 0.3 < mid < 0.7
