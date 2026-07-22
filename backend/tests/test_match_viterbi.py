"""Viterbi continuity tests with synthetic musical embeddings."""

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, match_sequence
from app.pipeline.segment import Segment


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def test_viterbi_prefers_continuity_over_greedy_switch():
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
    source = build_source_index(segs, pack)

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
        ),
    )

    assert len(smooth.tiles) == n_t
    assert greedyish.transitions_greedy >= 5
    assert smooth.transitions_viterbi < greedyish.transitions_greedy
    assert smooth.transitions_viterbi <= 2
