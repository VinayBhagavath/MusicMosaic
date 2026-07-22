"""Reconstruction tests: per-splice collage OLA."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta
from app.pipeline.match import LayerMatch, TileMatch
from app.pipeline.reconstruct import _compute_seam_xf, reconstruct_ola


def test_contiguous_run_reconstruction():
    """Single-song tiles still produce continuous output via OLA."""
    sr = 22050
    hop_s = 0.5
    window_s = 1.0
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
    mid = out[len(out) // 2]
    assert 0.35 < mid < 0.65


def test_transitions_are_click_free_across_distinct_sources():
    """Alternating loud/quiet source tiles must not leave clicks or dropouts."""
    sr = 22050
    hop_s = 0.25
    window_s = 0.5
    win = int(window_s * sr)
    song_a = np.ones(int(6.0 * sr), dtype=np.float32) * 0.8
    song_b = np.ones(int(6.0 * sr), dtype=np.float32) * 0.15
    n_tiles = 8
    meta = [SourceMeta("A", 0.0, window_s, 0), SourceMeta("B", 0.0, window_s, 1)]
    waveforms = [song_a[:win].copy(), song_b[:win].copy()]
    chroma = np.eye(2, 12, dtype=np.float32)
    timbre = np.eye(2, 26, dtype=np.float32)
    pack = EmbPack(chroma=chroma, timbre=timbre, energy=np.ones((2, 4), np.float32))
    source = SourceIndex(
        meta=meta, waveforms=waveforms, pack=pack,
        songs={"A": song_a, "B": song_b}, sr=sr,
    )
    tiles = []
    for i in range(n_tiles):
        use_a = i % 2 == 0
        tiles.append(
            TileMatch(
                i, i * hop_s, 0 if use_a else 1, "A" if use_a else "B",
                0.0, 0.9, target_duration_s=window_s,
            )
        )
    target_dur = (n_tiles - 1) * hop_s + window_s
    out = reconstruct_ola(
        tiles, source, sr=sr, window_s=window_s, hop_s=hop_s,
        target_duration_s=target_dur, apply_key_shift=False,
    )
    # No dropouts: interior never collapses to silence at a seam.
    interior = out[win : len(out) - win]
    assert float(np.min(np.abs(interior))) > 0.05
    # No clicks: sample-to-sample jumps stay bounded (equal-power crossfades).
    assert float(np.max(np.abs(np.diff(out)))) < 0.05


def test_onset_synchronous_seam_shortens_crossfade_at_attacks():
    """Seams landing on a target onset collapse to the click guard; sustained
    seams keep the full ~30 ms crossfade so only attacks stay punchy."""
    sr = 22050
    spans = [sr // 2, sr // 2, sr // 2]
    positions = [0, sr // 2, sr]
    xf_target = int(round(0.030 * sr))
    click_guard = max(16, int(round(0.004 * sr)))
    onset_tol = int(round(0.020 * sr))

    no_onset = _compute_seam_xf(
        spans, positions,
        xf_target=xf_target, click_guard=click_guard,
        onset_samples=None, onset_tol=onset_tol,
    )
    # With big spans, sustained seams use the full target crossfade.
    assert no_onset[0] == xf_target

    # Onset exactly on the first seam boundary (positions[1]).
    onset = np.array([sr // 2], dtype=np.int64)
    with_onset = _compute_seam_xf(
        spans, positions,
        xf_target=xf_target, click_guard=click_guard,
        onset_samples=onset, onset_tol=onset_tol,
    )
    # Attack seam collapses to (at most) the click guard...
    assert with_onset[0] <= click_guard
    assert with_onset[0] < no_onset[0]
    # ...while the non-onset seam keeps the full crossfade.
    assert with_onset[1] == xf_target


def test_layered_reconstruction_mixes_sources():
    sr = 22050
    hop_s = 0.5
    window_s = 1.0
    win = int(window_s * sr)
    song_a = np.ones(int(2.5 * sr), dtype=np.float32) * 0.6
    song_b = np.ones(int(2.5 * sr), dtype=np.float32) * 0.2
    meta = [
        SourceMeta("A", 0.0, 1.0, 0),
        SourceMeta("B", 0.0, 1.0, 1),
    ]
    waveforms = [song_a[:win].copy(), song_b[:win].copy()]
    chroma = np.eye(2, 12, dtype=np.float32)
    timbre = np.eye(2, 26, dtype=np.float32)
    pack = EmbPack(chroma=chroma, timbre=timbre, energy=np.ones((2, 4), np.float32))
    source = SourceIndex(
        meta=meta,
        waveforms=waveforms,
        pack=pack,
        songs={"A": song_a, "B": song_b},
        sr=sr,
    )
    tile = TileMatch(
        0,
        0.0,
        0,
        "A",
        0.0,
        0.9,
        layers=[
            LayerMatch(0, "A", 0.0, 0.9, 0.7),
            LayerMatch(1, "B", 0.0, 0.6, 0.3),
        ],
    )
    out = reconstruct_ola(
        [tile],
        source,
        sr=sr,
        window_s=window_s,
        hop_s=hop_s,
        target_duration_s=window_s,
        apply_key_shift=False,
    )
    assert len(out) == win
    # Residual fill without a target ref: primary + amount*secondary (> primary alone).
    mid = float(out[win // 2])
    assert mid > 0.55
