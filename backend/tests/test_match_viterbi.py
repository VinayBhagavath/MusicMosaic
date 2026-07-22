"""Viterbi continuity tests with synthetic embeddings."""

import numpy as np

from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, match_sequence
from app.pipeline.segment import Segment


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def test_viterbi_prefers_continuity_over_greedy_switch():
    """With both songs in top-k, switch penalty should collapse flips into long runs."""
    dim = 4
    # Few segments so top_k includes both songs (not drowned by near-duplicates).
    segs: list[Segment] = []
    emb: list[np.ndarray] = []
    for i in range(4):
        v = np.array([1.0, 0.05 * i, 0.0, 0.0], dtype=np.float32)
        emb.append(_unit(v))
        segs.append(Segment("A", i, i * 0.25, i * 0.25 + 0.5, np.zeros(64, np.float32)))
    for i in range(4):
        v = np.array([0.0, 0.0, 1.0, 0.05 * i], dtype=np.float32)
        emb.append(_unit(v))
        segs.append(Segment("B", i, i * 0.25, i * 0.25 + 0.5, np.zeros(64, np.float32)))

    source = build_source_index(segs, np.stack(emb))

    # Alternate preference A/B/A/B… with a thin margin so both stay in top-k.
    n_t = 10
    t_emb = []
    for t in range(n_t):
        if t % 2 == 0:
            v = np.array([1.0, 0.0, 0.92, 0.0], dtype=np.float32)
        else:
            v = np.array([0.92, 0.0, 1.0, 0.0], dtype=np.float32)
        t_emb.append(_unit(v))
    t_emb = np.stack(t_emb)
    starts = np.arange(n_t, dtype=np.float64) * 0.25

    greedyish = match_sequence(
        t_emb,
        starts,
        source,
        MatchParams(top_k=8, lambda_switch=0.0, lambda_jump=0.0, lambda_self=0.0),
    )
    smooth = match_sequence(
        t_emb,
        starts,
        source,
        MatchParams(top_k=8, lambda_switch=1.0, lambda_jump=0.05, lambda_self=0.0),
    )

    assert len(smooth.tiles) == n_t
    assert greedyish.transitions_viterbi >= 5
    assert smooth.transitions_viterbi < greedyish.transitions_viterbi
    assert smooth.transitions_viterbi <= 2
