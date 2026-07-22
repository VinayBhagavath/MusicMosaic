"""Viterbi continuity + diversity / multi-layer tests."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, match_sequence
from app.pipeline.segment import Segment


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _build_ab_source():
    segs: list[Segment] = []
    chroma = []
    timbre = []
    energy = []
    for i in range(4):
        c = np.zeros(12, dtype=np.float32)
        c[0] = 1.0
        c[1] = 0.05 * i
        chroma.append(_unit(c))
        t = np.zeros(26, dtype=np.float32)
        t[0] = 1.0
        timbre.append(_unit(t))
        energy.append(_unit(np.array([1.0, 0.1, 0.2, 0.05], np.float32)))
        segs.append(Segment("A", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))
    for i in range(4):
        c = np.zeros(12, dtype=np.float32)
        c[5] = 1.0
        c[6] = 0.05 * i
        chroma.append(_unit(c))
        t = np.zeros(26, dtype=np.float32)
        t[5] = 1.0
        timbre.append(_unit(t))
        energy.append(_unit(np.array([0.9, 0.1, 0.2, 0.05], np.float32)))
        segs.append(Segment("B", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))

    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    return build_source_index(segs, pack)


def test_viterbi_prefers_continuity_over_greedy_switch():
    source = _build_ab_source()

    n_t = 10
    q_chroma, q_timbre, q_energy = [], [], []
    for t in range(n_t):
        c = np.zeros(12, dtype=np.float32)
        tb = np.zeros(26, dtype=np.float32)
        if t % 2 == 0:
            c[0], c[5] = 1.0, 0.9
            tb[0], tb[5] = 1.0, 0.9
        else:
            c[5], c[0] = 1.0, 0.9
            tb[5], tb[0] = 1.0, 0.9
        q_chroma.append(_unit(c))
        q_timbre.append(_unit(tb))
        q_energy.append(_unit(np.array([1.0, 0.1, 0.2, 0.05], np.float32)))

    query = EmbPack(
        chroma=np.stack(q_chroma),
        timbre=np.stack(q_timbre),
        energy=np.stack(q_energy),
    )
    starts = np.arange(n_t, dtype=np.float64) * 0.5

    # Disable variety machinery so this isolates continuity behavior
    continuity = dict(
        lambda_balance=0.0,
        max_share=1.0,
        balance_iters=1,
        n_layers=1,
        min_run_tiles=1,
        per_song_k=8,
    )

    greedyish = match_sequence(
        query,
        starts,
        source,
        MatchParams(
            top_k=8,
            lambda_switch=0.0,
            lambda_jump=0.0,
            lambda_self=0.0,
            lambda_concat=0.0,
            lambda_join=0.0,
            hop_s=0.5,
            **continuity,
        ),
    )
    smooth = match_sequence(
        query,
        starts,
        source,
        MatchParams(
            top_k=8,
            lambda_switch=1.2,
            lambda_jump=0.1,
            lambda_self=0.0,
            lambda_concat=0.0,
            lambda_join=0.0,
            hop_s=0.5,
            **continuity,
        ),
    )

    assert len(smooth.tiles) == n_t
    assert greedyish.transitions_greedy >= 5
    assert smooth.transitions_viterbi < greedyish.transitions_greedy
    assert smooth.transitions_viterbi <= 2


def test_balance_prevents_single_song_domination():
    """When A is a near-perfect match to the target, balance still leaves room for B."""
    segs: list[Segment] = []
    chroma, timbre, energy = [], [], []

    # Song A: identical chroma/timbre family to the query (would otherwise monopolize)
    for i in range(8):
        c = np.zeros(12, dtype=np.float32)
        c[0] = 1.0
        chroma.append(_unit(c))
        t = np.zeros(26, dtype=np.float32)
        t[0] = 1.0
        timbre.append(_unit(t))
        energy.append(_unit(np.array([1.0, 0.1, 0.2, 0.05], np.float32)))
        segs.append(Segment("A", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))

    # Song B: weaker but still plausible
    for i in range(8):
        c = np.zeros(12, dtype=np.float32)
        c[0] = 0.7
        c[1] = 0.3
        chroma.append(_unit(c))
        t = np.zeros(26, dtype=np.float32)
        t[3] = 1.0  # distinct timbre
        timbre.append(_unit(t))
        energy.append(_unit(np.array([0.9, 0.1, 0.2, 0.05], np.float32)))
        segs.append(Segment("B", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))

    # Song C: another underdog
    for i in range(8):
        c = np.zeros(12, dtype=np.float32)
        c[0] = 0.65
        c[2] = 0.35
        chroma.append(_unit(c))
        t = np.zeros(26, dtype=np.float32)
        t[7] = 1.0
        timbre.append(_unit(t))
        energy.append(_unit(np.array([0.85, 0.1, 0.2, 0.05], np.float32)))
        segs.append(Segment("C", i, i * 0.5, i * 0.5 + 1.0, np.zeros(64, np.float32)))

    pack = EmbPack(
        chroma=np.stack(chroma),
        timbre=np.stack(timbre),
        energy=np.stack(energy),
    )
    source = build_source_index(segs, pack)

    n_t = 20
    q_chroma = [_unit(np.array([1.0] + [0.0] * 11, np.float32)) for _ in range(n_t)]
    q_timbre = [_unit(np.array([1.0] + [0.0] * 25, np.float32)) for _ in range(n_t)]
    q_energy = [_unit(np.array([1.0, 0.1, 0.2, 0.05], np.float32)) for _ in range(n_t)]
    query = EmbPack(
        chroma=np.stack(q_chroma),
        timbre=np.stack(q_timbre),
        energy=np.stack(q_energy),
    )
    starts = np.arange(n_t, dtype=np.float64) * 0.5

    dominated = match_sequence(
        query,
        starts,
        source,
        MatchParams(
            top_k=12,
            lambda_switch=1.2,
            lambda_balance=0.0,
            max_share=1.0,
            balance_iters=1,
            n_layers=1,
            min_run_tiles=1,
            per_song_k=8,
            lambda_jump=0.0,
            lambda_concat=0.0,
            lambda_join=0.0,
        ),
    )
    balanced = match_sequence(
        query,
        starts,
        source,
        MatchParams(
            top_k=12,
            lambda_switch=0.7,
            lambda_balance=1.15,
            max_share=0.4,
            balance_iters=3,
            n_layers=3,
            min_run_tiles=2,
            per_song_k=3,
            lambda_jump=0.2,
            lambda_concat=0.1,
            lambda_join=0.1,
                fidelity_first=False,
        ),
    )

    a_dom = sum(1 for t in dominated.tiles if t.song_id == "A") / n_t
    a_bal = sum(1 for t in balanced.tiles if t.song_id == "A") / n_t
    assert a_dom >= 0.85
    assert a_bal <= 0.45
    assert len({t.song_id for t in balanced.tiles}) >= 2

    # Multi-song layers attach on splices with strong complementary matches
    multi = [t for t in balanced.tiles if len(t.layers) >= 2]
    assert len(multi) >= 1
    for t in balanced.tiles:
        assert t.layers
        assert abs(sum(L.weight for L in t.layers) - 1.0) < 1e-5
        songs = {L.song_id for L in t.layers}
        assert len(songs) == len(t.layers)
