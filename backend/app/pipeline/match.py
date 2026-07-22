"""Nearest-neighbor + Viterbi sequence matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.index import SourceIndex, SourceMeta


@dataclass(slots=True)
class MatchParams:
    top_k: int = 8
    lambda_switch: float = 0.35
    lambda_jump: float = 0.25
    jump_norm_s: float = 2.0
    lambda_self: float = 0.05
    hop_s: float = 0.25


@dataclass(slots=True)
class TileMatch:
    target_idx: int
    target_start_s: float
    source_id: int  # FAISS / meta index
    song_id: str
    source_start_s: float
    similarity: float


@dataclass(slots=True)
class MatchResult:
    tiles: list[TileMatch]
    transitions_viterbi: int
    transitions_greedy: int
    avg_similarity: float


def _transition_cost(
    a: SourceMeta,
    a_id: int,
    b: SourceMeta,
    b_id: int,
    params: MatchParams,
) -> float:
    cost = 0.0
    if a.song_id != b.song_id:
        cost += params.lambda_switch
    else:
        expected = a.start_s + params.hop_s
        jump = abs(b.start_s - expected)
        cost += params.lambda_jump * min(1.0, jump / params.jump_norm_s)
    if a_id == b_id:
        cost += params.lambda_self
    return cost


def _count_transitions(song_ids: list[str]) -> int:
    return sum(1 for i in range(1, len(song_ids)) if song_ids[i] != song_ids[i - 1])


def match_sequence(
    target_embeddings: np.ndarray,
    target_starts: np.ndarray,
    source: SourceIndex,
    params: MatchParams | None = None,
) -> MatchResult:
    """Top-k FAISS candidates per frame + Viterbi path with transition penalties."""
    params = params or MatchParams()
    n = len(target_embeddings)
    if n == 0:
        return MatchResult([], 0, 0, 0.0)

    sims, ids = source.search(target_embeddings, k=params.top_k)
    k = sims.shape[1]

    # Local costs: 1 - cosine similarity
    local = 1.0 - sims  # [n, k]

    # DP
    dp = np.full((n, k), np.inf, dtype=np.float64)
    back = np.full((n, k), -1, dtype=np.int32)
    dp[0] = local[0]

    for t in range(1, n):
        for j in range(k):
            b_id = int(ids[t, j])
            if b_id < 0:
                continue
            b_meta = source.meta[b_id]
            best_c, best_i = np.inf, -1
            for i in range(k):
                a_id = int(ids[t - 1, i])
                if a_id < 0 or not np.isfinite(dp[t - 1, i]):
                    continue
                a_meta = source.meta[a_id]
                c = dp[t - 1, i] + _transition_cost(a_meta, a_id, b_meta, b_id, params) + local[t, j]
                if c < best_c:
                    best_c, best_i = c, i
            dp[t, j] = best_c
            back[t, j] = best_i

    # Backtrace
    path_k = np.empty(n, dtype=np.int32)
    path_k[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 2, -1, -1):
        path_k[t] = back[t + 1, path_k[t + 1]]

    tiles: list[TileMatch] = []
    for t in range(n):
        j = int(path_k[t])
        sid = int(ids[t, j])
        meta = source.meta[sid]
        tiles.append(
            TileMatch(
                target_idx=t,
                target_start_s=float(target_starts[t]),
                source_id=sid,
                song_id=meta.song_id,
                source_start_s=meta.start_s,
                similarity=float(sims[t, j]),
            )
        )

    greedy_ids = [source.meta[int(ids[t, 0])].song_id for t in range(n) if ids[t, 0] >= 0]
    viterbi_ids = [t.song_id for t in tiles]
    avg_sim = float(np.mean([t.similarity for t in tiles])) if tiles else 0.0

    return MatchResult(
        tiles=tiles,
        transitions_viterbi=_count_transitions(viterbi_ids),
        transitions_greedy=_count_transitions(greedy_ids),
        avg_similarity=avg_sim,
    )
