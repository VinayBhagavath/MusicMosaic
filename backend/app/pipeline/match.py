"""Nearest-neighbor + Viterbi with literature-style target + join costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta


@dataclass(slots=True)
class MatchParams:
    top_k: int = 12
    # Prefer long same-song runs (CATERPILLAR / Musical Mosaicing continuity)
    lambda_switch: float = 1.15
    lambda_jump: float = 0.55
    jump_norm_s: float = 2.0
    lambda_self: float = 0.02
    # Whole-clip embedding discontinuity
    lambda_concat: float = 0.25
    # Boundary spectral join cost (Vepa & King style MFCC-edge distance)
    lambda_join: float = 0.55
    hop_s: float = 0.5


@dataclass(slots=True)
class TileMatch:
    target_idx: int
    target_start_s: float
    source_id: int
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
    concat_penalty: float,
    join_penalty: float,
) -> float:
    # Natural neighbors in the same song: near-zero join (unit-selection gold standard)
    if (
        a.song_id == b.song_id
        and abs((a.start_s + params.hop_s) - b.start_s) < params.hop_s * 0.55
    ):
        return -0.25 + concat_penalty * 0.15  # strong reward for true continuity

    cost = concat_penalty + join_penalty
    if a.song_id != b.song_id:
        cost += params.lambda_switch
    else:
        expected = a.start_s + params.hop_s
        jump = abs(b.start_s - expected)
        cost += params.lambda_jump * min(1.0, jump / params.jump_norm_s)
        if jump < params.hop_s * 0.6 and params.lambda_jump > 0:
            cost -= 0.15
    if a_id == b_id:
        cost += params.lambda_self
    return cost


def _count_transitions(song_ids: list[str]) -> int:
    return sum(1 for i in range(1, len(song_ids)) if song_ids[i] != song_ids[i - 1])


def _concat_matrix(pack: EmbPack) -> np.ndarray:
    c = 0.55 * (pack.chroma @ pack.chroma.T) + 0.45 * (pack.timbre @ pack.timbre.T)
    return (1.0 - c).astype(np.float32)


def match_sequence(
    query: EmbPack,
    target_starts: np.ndarray,
    source: SourceIndex,
    params: MatchParams | None = None,
) -> MatchResult:
    params = params or MatchParams()
    n = query.chroma.shape[0]
    if n == 0:
        return MatchResult([], 0, 0, 0.0)

    sims, ids = source.search(query, k=params.top_k)
    k = sims.shape[1]
    local = 1.0 - sims

    concat = _concat_matrix(source.pack)
    lam_c = params.lambda_concat
    lam_j = params.lambda_join

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
                cpen = lam_c * float(concat[a_id, b_id])
                jpen = lam_j * source.join_distance(a_id, b_id)
                c = (
                    dp[t - 1, i]
                    + _transition_cost(a_meta, a_id, b_meta, b_id, params, cpen, jpen)
                    + local[t, j]
                )
                if c < best_c:
                    best_c, best_i = c, i
            dp[t, j] = best_c
            back[t, j] = best_i

    path_k = np.empty(n, dtype=np.int32)
    path_k[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 2, -1, -1):
        prev = int(back[t + 1, path_k[t + 1]])
        path_k[t] = prev if prev >= 0 else 0

    tiles: list[TileMatch] = []
    for t in range(n):
        j = int(path_k[t])
        if j < 0 or j >= k:
            j = 0
        sid = int(ids[t, j])
        if sid < 0:
            sid = int(ids[t, 0])
        if sid < 0:
            raise RuntimeError(f"No match for target frame {t}")
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
