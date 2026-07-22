"""Key-shift search + min-run cohesion."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, TileMatch, _enforce_min_runs, match_sequence
from app.pipeline.segment import Segment


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def test_search_returns_key_shift():
    # Source in C (bin 0); query in D (bin 2) → expect shift ≈ 2
    segs = [
        Segment("A", 0, 0.0, 1.0, np.zeros(64, np.float32)),
        Segment("A", 1, 0.5, 1.5, np.zeros(64, np.float32)),
    ]
    pack = EmbPack(
        chroma=np.stack([_unit(np.eye(12, dtype=np.float32)[0])] * 2),
        timbre=np.stack([_unit(np.ones(8, np.float32))] * 2),
        energy=np.stack([_unit(np.ones(4, np.float32))] * 2),
    )
    source = build_source_index(segs, pack, compute_edges=False)

    q_c = _unit(np.eye(12, dtype=np.float32)[2])
    query = EmbPack(
        chroma=q_c[None, :],
        timbre=_unit(np.ones(8, np.float32))[None, :],
        energy=_unit(np.ones(4, np.float32))[None, :],
    )
    sims, ids, shifts = source.search(query, k=2)
    assert sims.shape == (1, 2)
    assert ids.shape == (1, 2)
    assert int(shifts[0, 0]) == 2


def test_search_diverse_spans_songs():
    segs = []
    chroma, timbre, energy = [], [], []
    for song, base in [("A", 0), ("B", 1), ("C", 2)]:
        for i in range(4):
            c = np.zeros(12, np.float32)
            c[base] = 1.0
            chroma.append(_unit(c))
            t = np.zeros(8, np.float32)
            t[base] = 1.0
            timbre.append(_unit(t))
            energy.append(_unit(np.ones(4, np.float32)))
            segs.append(Segment(song, i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    source = build_source_index(segs, pack, compute_edges=False)
    # Query very close to A — without diversity A would fill the beam
    q = EmbPack(
        chroma=_unit(np.eye(12, dtype=np.float32)[0])[None, :],
        timbre=_unit(np.eye(8, dtype=np.float32)[0])[None, :],
        energy=_unit(np.ones(4, np.float32))[None, :],
    )
    sims, ids, shifts = source.search_diverse(q, k=6, per_song=2)
    songs = {source.meta[int(i)].song_id for i in ids[0] if i >= 0}
    assert songs >= {"A", "B", "C"}
    assert shifts.shape == sims.shape


def test_min_run_absorbs_short_island():
    segs = []
    chroma, timbre, energy = [], [], []
    for song, base in [("A", 0), ("B", 5)]:
        for i in range(8):
            c = np.zeros(12, np.float32)
            c[base] = 1.0
            chroma.append(_unit(c))
            t = np.zeros(8, np.float32)
            t[0 if song == "A" else 1] = 1.0
            timbre.append(_unit(t))
            energy.append(_unit(np.ones(4, np.float32)))
            segs.append(Segment(song, i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    source = build_source_index(segs, pack, compute_edges=False)

    tiles = []
    for i in range(6):
        tiles.append(TileMatch(i, i * 0.5, i, "A", i * 0.5, 0.9, 0))
    for i in range(2):
        tiles.append(TileMatch(6 + i, (6 + i) * 0.5, 8 + i, "B", i * 0.5, 0.9, 0))
    for i in range(4):
        tiles.append(TileMatch(8 + i, (8 + i) * 0.5, 4 + i, "A", (4 + i) * 0.5, 0.9, 0))

    fixed = _enforce_min_runs(tiles, source, min_run=4, hop_s=0.5, max_share=1.0)
    assert all(t.song_id == "A" for t in fixed[6:8])


def test_layers_attach_on_tiles_when_enabled():
    """Multi-song layers can attach on any splice that has strong complementary matches."""
    segs = []
    chroma, timbre, energy = [], [], []
    for song, base, tim_i in [("A", 0, 0), ("B", 1, 1), ("C", 2, 2)]:
        for i in range(6):
            c = np.zeros(12, np.float32)
            c[base] = 1.0
            c[0] = 0.85  # all somewhat close to query in C
            chroma.append(_unit(c))
            t = np.zeros(8, np.float32)
            t[tim_i] = 1.0
            timbre.append(_unit(t))
            energy.append(_unit(np.ones(4, np.float32)))
            segs.append(Segment(song, i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    source = build_source_index(segs, pack, compute_edges=False)
    n_t = 8
    q = EmbPack(
        chroma=np.stack([_unit(np.eye(12, dtype=np.float32)[0])] * n_t),
        timbre=np.stack([_unit(np.eye(8, dtype=np.float32)[0])] * n_t),
        energy=np.stack([_unit(np.ones(4, np.float32))] * n_t),
    )
    starts = np.arange(n_t, dtype=np.float64) * 0.5
    res = match_sequence(
        q,
        starts,
        source,
        MatchParams(
            top_k=12,
            min_run_tiles=1,
            n_layers=3,
            layer_primary_weight=0.55,
            lambda_switch=0.3,
            lambda_balance=1.2,
            max_share=0.4,
            per_song_k=3,
            fidelity_first=False,
        ),
    )
    multi = [t for t in res.tiles if len(t.layers) >= 2]
    assert len(multi) >= 1
    assert len({t.song_id for t in res.tiles}) >= 2
    for t in multi:
        assert abs(sum(layer.weight for layer in t.layers) - 1.0) < 1e-5
        assert len({layer.song_id for layer in t.layers}) == len(t.layers)


def test_temporal_chroma_and_register_rerank_mean_ties():
    base_chroma = _unit(np.ones(12, np.float32))
    timbre = _unit(np.ones(4, np.float32))
    energy = _unit(np.ones(3, np.float32))
    rising = _unit(np.concatenate([np.eye(12, dtype=np.float32)[i] for i in range(4)]))
    falling = _unit(np.concatenate([np.eye(12, dtype=np.float32)[i] for i in [3, 2, 1, 0]]))
    low = _unit(np.array([1, 0, 0, 0, 0, 0], np.float32))
    high = _unit(np.array([0, 0, 0, 0, 0, 1], np.float32))
    pack = EmbPack(
        chroma=np.stack([base_chroma, base_chroma]),
        timbre=np.stack([timbre, timbre]),
        energy=np.stack([energy, energy]),
        temporal=np.stack([falling, rising]),
        register=np.stack([high, low]),
    )
    segs = [
        Segment("A", 0, 0.0, 0.5, np.zeros(64, np.float32)),
        Segment("B", 0, 0.0, 0.5, np.zeros(64, np.float32)),
    ]
    source = build_source_index(segs, pack, compute_edges=False)
    query = EmbPack(
        chroma=base_chroma[None, :],
        timbre=timbre[None, :],
        energy=energy[None, :],
        temporal=rising[None, :],
        register=low[None, :],
    )
    _scores, ids, _shifts = source.search(query, k=2)
    assert int(ids[0, 0]) == 1


def test_match_tracks_key_shift_field():
    segs = [Segment("A", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)) for i in range(4)]
    pack = EmbPack(
        chroma=np.stack([_unit(np.eye(12, dtype=np.float32)[0])] * 4),
        timbre=np.stack([_unit(np.ones(8, np.float32))] * 4),
        energy=np.stack([_unit(np.ones(4, np.float32))] * 4),
    )
    source = build_source_index(segs, pack, compute_edges=False)
    q = EmbPack(
        chroma=np.stack([_unit(np.eye(12, dtype=np.float32)[3])] * 3),
        timbre=np.stack([_unit(np.ones(8, np.float32))] * 3),
        energy=np.stack([_unit(np.ones(4, np.float32))] * 3),
    )
    starts = np.arange(3, dtype=np.float64) * 0.5
    res = match_sequence(
        q,
        starts,
        source,
        MatchParams(
            top_k=4,
            min_run_tiles=1,
            hop_s=0.5,
            lambda_switch=0.0,
            lambda_balance=0.0,
            max_share=1.0,
            n_layers=1,
        ),
    )
    assert all(t.key_shift == 3 for t in res.tiles)


def test_search_diverse_per_song_one_takes_top_only():
    """per_song=1 must pick the best clip per song, not every clip."""
    segs = []
    chroma, timbre, energy = [], [], []
    # Song A: 5 clips — timbre alignment to query increases with i
    for i in range(5):
        c = np.zeros(12, np.float32)
        c[0] = 1.0
        chroma.append(_unit(c))
        t = np.zeros(8, np.float32)
        t[0] = 0.1 + 0.2 * i  # best at i=4
        t[2] = 1.0 - t[0]
        timbre.append(_unit(t))
        energy.append(_unit(np.ones(4, np.float32)))
        segs.append(Segment("A", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    for i in range(5):
        c = np.zeros(12, np.float32)
        c[5] = 1.0
        chroma.append(_unit(c))
        t = np.zeros(8, np.float32)
        t[1] = 1.0
        timbre.append(_unit(t))
        energy.append(_unit(np.ones(4, np.float32)))
        segs.append(Segment("B", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    source = build_source_index(segs, pack, compute_edges=False)
    q = EmbPack(
        chroma=_unit(np.eye(12, dtype=np.float32)[0])[None, :],
        timbre=_unit(np.eye(8, dtype=np.float32)[0])[None, :],
        energy=_unit(np.ones(4, np.float32))[None, :],
    )
    sims, ids, _ = source.search_diverse(q, k=2, per_song=1)
    picked = [int(i) for i in ids[0] if i >= 0]
    assert len(picked) == 2
    songs = {source.meta[i].song_id for i in picked}
    assert songs == {"A", "B"}
    assert 4 in picked  # best A clip by timbre