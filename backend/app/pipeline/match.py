"""Nearest-neighbor + Viterbi for MIDI-note mosaicing.

Mental model: each unit is a sample/note. Song switches are fine (like changing
MIDI instruments). We care about:
1. Pitch + timbre + dynamics match to the target note
2. Audio-edge join quality (crossfade-friendly), not same-song loyalty
3. Optional stacking of complementary roles (bass / drums / harmonic)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.index import SourceIndex, SourceMeta


@dataclass(slots=True)
class MatchParams:
    top_k: int = 20
    # Free instrument switching — like picking different MIDI samples
    lambda_switch: float = 0.08
    lambda_jump: float = 0.15
    jump_norm_s: float = 2.0
    lambda_self: float = 0.03
    # Join cost = audio edge continuity (not song ID)
    lambda_concat: float = 0.55
    lambda_join: float = 0.70
    hop_s: float = 0.22
    # Allow note-to-note switching; don't force long same-song runs
    min_run_tiles: int = 1
    per_song_k: int = 4
    lambda_balance: float = 0.0
    max_share: float = 1.0
    balance_iters: int = 1
    # Stack complementary roles / chroma-fill like MIDI tracks.
    # Default 1: with mismatched sources, extra layers often muddy more than
    # they help. Set ≥2 to enable residual spectral fill for polyphony.
    n_layers: int = 1
    layer_primary_weight: float = 0.62
    fidelity_first: bool = True


@dataclass(slots=True)
class LayerMatch:
    source_id: int
    song_id: str
    source_start_s: float
    similarity: float
    weight: float
    key_shift: float = 0.0  # semitones (may be fractional / cents)
    role: str = "full"  # bass | drums | harmonic | full


@dataclass(slots=True)
class TileMatch:
    target_idx: int
    target_start_s: float
    source_id: int
    song_id: str
    source_start_s: float
    similarity: float
    key_shift: float = 0.0
    target_duration_s: float | None = None
    layers: list[LayerMatch] = field(default_factory=list)
    # Beam candidates for this frame: (source_id, key_shift, feature_similarity).
    # Retained so a post-Viterbi pass can re-rank by actual post-transform
    # spectral distance to the target window.
    candidates: list[tuple[int, float, float]] = field(default_factory=list)


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
    *,
    step_s: float | None = None,
) -> float:
    """Audio-edge glue. Same-song neighbors get a small bonus; switches are cheap."""
    step = float(step_s) if step_s is not None and step_s > 1e-4 else params.hop_s
    tol = max(0.06, step * 0.55)
    # Natural continuation of the same sample strip (don't shred mid-note)
    if a.song_id == b.song_id and abs((a.start_s + step) - b.start_s) < tol:
        return -0.28 + concat_penalty * 0.10 + join_penalty * 0.10

    cost = concat_penalty + join_penalty
    if a.song_id != b.song_id:
        cost += params.lambda_switch
    else:
        expected = a.start_s + step
        jump = abs(b.start_s - expected)
        cost += params.lambda_jump * min(1.0, jump / params.jump_norm_s)
    if a_id == b_id:
        cost += params.lambda_self
    return cost


def _count_transitions(song_ids: list[str]) -> int:
    return sum(1 for i in range(1, len(song_ids)) if song_ids[i] != song_ids[i - 1])


def _pairwise_concat(
    pack: EmbPack, id_pairs: set[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """Concat discontinuity only for pairs the Viterbi beam will actually ask for."""
    if not id_pairs:
        return {}
    uniq = sorted({i for pair in id_pairs for i in pair})
    if not uniq:
        return {}
    idx = np.array(uniq, dtype=np.int64)
    pos = {int(u): i for i, u in enumerate(uniq)}
    c_sub = pack.chroma[idx]
    t_sub = pack.timbre[idx]
    e_sub = pack.energy[idx]
    # Include dynamics so loudness/onset jumps cost more (classic concat cost)
    sim = 0.40 * (c_sub @ c_sub.T) + 0.40 * (t_sub @ t_sub.T) + 0.20 * (e_sub @ e_sub.T)
    disc = (1.0 - sim).astype(np.float32)
    out: dict[tuple[int, int], float] = {}
    for a, b in id_pairs:
        out[(a, b)] = float(disc[pos[a], pos[b]])
    return out


def _beam_pairs(ids: np.ndarray) -> set[tuple[int, int]]:
    n, k = ids.shape
    pairs: set[tuple[int, int]] = set()
    for t in range(1, n):
        for i in range(k):
            a = int(ids[t - 1, i])
            if a < 0:
                continue
            for j in range(k):
                b = int(ids[t, j])
                if b < 0:
                    continue
                pairs.add((a, b))
    return pairs


def _song_shares(song_ids: list[str]) -> dict[str, float]:
    n = len(song_ids)
    if n == 0:
        return {}
    counts = Counter(song_ids)
    return {s: c / n for s, c in counts.items()}


def _song_clip_ids(source: SourceIndex) -> dict[str, list[int]]:
    by: dict[str, list[int]] = {}
    for i, m in enumerate(source.meta):
        by.setdefault(m.song_id, []).append(i)
    for ids in by.values():
        ids.sort(key=lambda i: source.meta[i].start_s)
    return by


def _viterbi_path(
    local: np.ndarray,
    ids: np.ndarray,
    shifts: np.ndarray,
    source: SourceIndex,
    params: MatchParams,
    concat: dict[tuple[int, int], float],
    target_starts: np.ndarray | None = None,
) -> np.ndarray:
    n, k = local.shape
    lam_c = params.lambda_concat
    lam_j = params.lambda_join

    dp = np.full((n, k), np.inf, dtype=np.float64)
    back = np.full((n, k), -1, dtype=np.int32)
    dp[0] = local[0]

    for t in range(1, n):
        if target_starts is not None:
            step_s = float(target_starts[t] - target_starts[t - 1])
        else:
            step_s = params.hop_s
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
                cpen = lam_c * concat.get((a_id, b_id), 0.5)
                jpen = lam_j * source.join_distance(a_id, b_id)
                # Penalize large key jumps more than ±1 (was binary)
                d_shift = abs(int(shifts[t, j]) - int(shifts[t - 1, i]))
                key_pen = 0.06 * float(min(6, d_shift))
                c = (
                    dp[t - 1, i]
                    + _transition_cost(
                        a_meta,
                        a_id,
                        b_meta,
                        b_id,
                        params,
                        cpen,
                        jpen,
                        step_s=step_s,
                    )
                    + key_pen
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
    return path_k


def _path_to_tiles(
    path_k: np.ndarray,
    sims: np.ndarray,
    ids: np.ndarray,
    shifts: np.ndarray,
    target_starts: np.ndarray,
    source: SourceIndex,
    target_durations: np.ndarray | None = None,
) -> list[TileMatch]:
    n, k = sims.shape
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
        cands: list[tuple[int, float, float]] = []
        for jj in range(k):
            cid = int(ids[t, jj])
            if cid < 0:
                continue
            cands.append((cid, float(shifts[t, jj]), float(sims[t, jj])))
        tiles.append(
            TileMatch(
                target_idx=t,
                target_start_s=float(target_starts[t]),
                source_id=sid,
                song_id=meta.song_id,
                source_start_s=meta.start_s,
                similarity=float(sims[t, j]),
                key_shift=float(shifts[t, j]),
                target_duration_s=(
                    float(target_durations[t]) if target_durations is not None else None
                ),
                candidates=cands,
            )
        )
    return tiles


def _enforce_max_share(
    tiles: list[TileMatch],
    sims: np.ndarray,
    ids: np.ndarray,
    shifts: np.ndarray,
    source: SourceIndex,
    max_share: float,
) -> list[TileMatch]:
    """Swap over-quota tiles to the best underused-song candidate when possible."""
    if not tiles or max_share <= 0:
        return tiles
    n = len(tiles)
    max_count = max(1, int(np.floor(max_share * n)))
    song_ids = [t.song_id for t in tiles]

    for _ in range(n):
        counts = Counter(song_ids)
        over = [s for s, c in counts.items() if c > max_count]
        if not over:
            break
        under = {s for s, c in counts.items() if c < max_count}
        all_songs = {m.song_id for m in source.meta}
        under |= all_songs - set(counts.keys())
        if not under:
            break

        best_swap: tuple[float, int, int, str, float, int] | None = None
        for t, tile in enumerate(tiles):
            if tile.song_id not in over:
                continue
            for j in range(ids.shape[1]):
                cand = int(ids[t, j])
                if cand < 0 or cand == tile.source_id:
                    continue
                meta = source.meta[cand]
                if meta.song_id not in under:
                    continue
                if counts[meta.song_id] >= max_count:
                    continue
                # Prefer the swap that loses the least target similarity
                margin = tile.similarity - float(sims[t, j])
                if best_swap is None or margin < best_swap[0]:
                    best_swap = (
                        margin,
                        t,
                        cand,
                        meta.song_id,
                        float(sims[t, j]),
                        int(shifts[t, j]),
                    )

        if best_swap is None:
            break
        _, t, cand, new_song, new_sim, key_shift = best_swap
        meta = source.meta[cand]
        tiles[t] = TileMatch(
            target_idx=tiles[t].target_idx,
            target_start_s=tiles[t].target_start_s,
            source_id=cand,
            song_id=new_song,
            source_start_s=meta.start_s,
            similarity=new_sim,
            key_shift=key_shift,
        )
        song_ids[t] = new_song

    return tiles


def _enforce_min_runs(
    tiles: list[TileMatch],
    source: SourceIndex,
    *,
    min_run: int,
    hop_s: float,
    max_share: float,
) -> list[TileMatch]:
    """Absorb short song islands into a neighbor, without breaking max_share."""
    if min_run <= 1 or len(tiles) < min_run:
        return tiles

    n = len(tiles)
    max_count = max(1, int(np.floor(max_share * n))) if max_share > 0 else n

    runs: list[tuple[int, int, str]] = []
    s = 0
    for i in range(1, n + 1):
        if i == n or tiles[i].song_id != tiles[s].song_id:
            runs.append((s, i, tiles[s].song_id))
            s = i

    by_song = _song_clip_ids(source)
    out = list(tiles)
    counts = Counter(t.song_id for t in out)

    for ri, (a, b, song) in enumerate(runs):
        length = b - a
        if length >= min_run:
            continue
        left = runs[ri - 1] if ri > 0 else None
        right = runs[ri + 1] if ri + 1 < len(runs) else None
        target_song = None
        if left and right:
            left_len = left[1] - left[0]
            right_len = right[1] - right[0]
            target_song = left[2] if left_len >= right_len else right[2]
        elif left:
            target_song = left[2]
        elif right:
            target_song = right[2]
        else:
            continue

        if counts.get(target_song, 0) + length > max_count:
            continue

        cand = by_song.get(target_song, [])
        if not cand:
            continue

        if left and target_song == left[2]:
            t0 = out[left[1] - 1].source_start_s + hop_s
            key0 = out[left[1] - 1].key_shift
        elif right and target_song == right[2]:
            t0 = out[right[0]].source_start_s - hop_s * length
            key0 = out[right[0]].key_shift
        else:
            t0 = out[a].source_start_s
            key0 = out[a].key_shift

        starts = np.array([source.meta[i].start_s for i in cand], dtype=np.float64)
        for j, ti in enumerate(range(a, b)):
            want = t0 + j * hop_s
            idx = int(np.argmin(np.abs(starts - want)))
            sid = cand[idx]
            meta = source.meta[sid]
            prev = out[ti]
            counts[prev.song_id] -= 1
            counts[meta.song_id] = counts.get(meta.song_id, 0) + 1
            out[ti] = TileMatch(
                target_idx=prev.target_idx,
                target_start_s=prev.target_start_s,
                source_id=sid,
                song_id=meta.song_id,
                source_start_s=meta.start_s,
                similarity=prev.similarity * 0.98,
                key_shift=key0,
            )
    return out


def _chroma_residual_fill(
    target_chroma: np.ndarray,
    primary_chroma: np.ndarray,
    cand_chroma: np.ndarray,
) -> float:
    """How well `cand` covers pitch classes the primary is missing vs target."""
    residual = np.maximum(0.0, target_chroma - primary_chroma)
    rnorm = float(np.linalg.norm(residual))
    if rnorm < 1e-6:
        return 0.0
    return float(np.dot(residual, cand_chroma) / rnorm)


def _complementarity(
    primary_id: int,
    cand_id: int,
    target_sim: float,
    source: SourceIndex,
    *,
    primary_role: str,
    cand_role: str,
    target_chroma: np.ndarray | None = None,
) -> float:
    """Score secondary layers: target fit, timbre distinct, chroma fill, roles."""
    p_tim = source.pack.timbre[primary_id]
    c_tim = source.pack.timbre[cand_id]
    timbre_overlap = float(np.dot(p_tim, c_tim))
    distinct = 1.0 - max(0.0, min(1.0, timbre_overlap))
    role_bonus = 0.0
    if cand_role != primary_role and cand_role != "full":
        role_bonus = 0.18
    if primary_role == "bass" and cand_role == "bass":
        role_bonus = -0.25  # avoid double-bass
    if primary_role == "drums" and cand_role == "drums":
        role_bonus = -0.12
    fill = 0.0
    if target_chroma is not None:
        fill = _chroma_residual_fill(
            target_chroma,
            source.pack.chroma[primary_id],
            source.pack.chroma[cand_id],
        )
    return (
        0.50 * max(0.0, target_sim)
        + 0.18 * distinct
        + 0.22 * fill
        + role_bonus
    )

def _primary_layer(tile: TileMatch, role: str = "full") -> list[LayerMatch]:
    return [
        LayerMatch(
            source_id=tile.source_id,
            song_id=tile.song_id,
            source_start_s=tile.source_start_s,
            similarity=tile.similarity,
            weight=1.0,
            key_shift=tile.key_shift,
            role=role,
        )
    ]


def _role_from_pack(pack: EmbPack, i: int) -> str:
    """Cheap role from mosaic descriptors — avoids HPSS on every source clip."""
    if pack.register is None or i >= pack.energy.shape[0]:
        return "full"
    try:
        reg = pack.register[i]
        en = pack.energy[i]
        low = float(reg[0] + reg[1]) if len(reg) >= 2 else 0.0
        high = float(reg[-1]) if len(reg) else 0.0
        onset = float(en[5]) if len(en) > 5 else 0.0
        flat = float(en[4]) if len(en) > 4 else 0.0
        if low > 0.55 and onset < 0.45:
            return "bass"
        if onset > 0.55 or (high > 0.4 and onset > 0.35) or flat > 0.55:
            return "drums"
        if low < 0.35 and onset < 0.4:
            return "harmonic"
        return "full"
    except Exception:
        return "full"


def _attach_layers(
    tiles: list[TileMatch],
    sims: np.ndarray,
    ids: np.ndarray,
    shifts: np.ndarray,
    source: SourceIndex,
    params: MatchParams,
    *,
    roles: list[str] | None = None,
    query: EmbPack | None = None,
) -> None:
    """Attach multi-song layers; prefer complementary chroma fill + roles."""
    n_layers = max(1, params.n_layers)
    # Keep the primary audible but leave room for chord fill.
    primary_w = float(np.clip(params.layer_primary_weight, 0.45, 0.85))

    if n_layers <= 1:
        for tile in tiles:
            tile.layers = _primary_layer(tile, _role_from_pack(source.pack, tile.source_id))
        return

    if roles is None:
        roles = [_role_from_pack(source.pack, i) for i in range(source.n)]

    for t, tile in enumerate(tiles):
        p_role = roles[tile.source_id] if tile.source_id < len(roles) else "full"
        used_songs = {tile.song_id}
        used_roles = {p_role} if p_role != "full" else set()
        target_chroma = query.chroma[t] if query is not None else None
        scored: list[tuple[float, int, float, float, str]] = []
        for j in range(ids.shape[1]):
            cand = int(ids[t, j])
            if cand < 0 or cand == tile.source_id:
                continue
            meta = source.meta[cand]
            if meta.song_id in used_songs:
                continue
            sim = float(sims[t, j])
            # Fidelity mode used to require near-identical sims (~0.92×), which
            # blocked almost all secondaries when the primary already scored
            # ~0.96. Allow chord-fill layers that are still strong matches.
            fidelity_threshold = 0.82 if params.fidelity_first else 0.55
            if sim < max(0.22, tile.similarity * fidelity_threshold):
                continue
            c_role = roles[cand] if cand < len(roles) else "full"
            comp = _complementarity(
                tile.source_id,
                cand,
                sim,
                source,
                primary_role=p_role,
                cand_role=c_role,
                target_chroma=target_chroma,
            )
            scored.append((comp, cand, sim, float(shifts[t, j]), c_role))

        scored.sort(reverse=True)
        secondaries: list[tuple[int, float, float, str]] = []
        for _comp, cand, sim, shift, c_role in scored:
            meta = source.meta[cand]
            if meta.song_id in used_songs:
                continue
            used_songs.add(meta.song_id)
            if c_role != "full":
                used_roles.add(c_role)
            secondaries.append((cand, sim, shift, c_role))
            if len(secondaries) >= n_layers - 1:
                break

        if not secondaries:
            tile.layers = _primary_layer(tile, p_role)
            continue

        raw = np.array([max(1e-3, s) for _, s, _, _ in secondaries], dtype=np.float64)
        temp = 0.28
        ex = np.exp((raw - raw.max()) / temp)
        sec_frac = ex / ex.sum()
        remain = 1.0 - primary_w

        layers = [
            LayerMatch(
                source_id=tile.source_id,
                song_id=tile.song_id,
                source_start_s=tile.source_start_s,
                similarity=tile.similarity,
                weight=primary_w,
                key_shift=tile.key_shift,
                role=p_role,
            )
        ]
        for (cand, sim, shift, c_role), frac in zip(secondaries, sec_frac):
            meta = source.meta[cand]
            layers.append(
                LayerMatch(
                    source_id=cand,
                    song_id=meta.song_id,
                    source_start_s=meta.start_s,
                    similarity=sim,
                    weight=float(remain * frac),
                    key_shift=shift,
                    role=c_role,
                )
            )
        tile.layers = layers


def match_sequence(
    query: EmbPack,
    target_starts: np.ndarray,
    source: SourceIndex,
    params: MatchParams | None = None,
    *,
    target_durations: np.ndarray | None = None,
) -> MatchResult:
    params = params or MatchParams()
    n = query.chroma.shape[0]
    if n == 0:
        return MatchResult([], 0, 0, 0.0)

    if params.fidelity_first:
        sims, ids, shifts = source.search(query, k=params.top_k)
    else:
        sims, ids, shifts = source.search_diverse(
            query, k=params.top_k, per_song=params.per_song_k
        )
    k = sims.shape[1]
    if k == 0:
        return MatchResult([], 0, 0, 0.0)

    concat = _pairwise_concat(source.pack, _beam_pairs(ids))
    unique_songs = sorted({m.song_id for m in source.meta})
    n_songs = max(1, len(unique_songs))
    target_share = (
        min(params.max_share, 1.0 / n_songs) if params.max_share > 0 else 1.0 / n_songs
    )

    bias_by_song = {s: 0.0 for s in unique_songs}
    path_k = np.zeros(n, dtype=np.int32)

    iters = max(1, params.balance_iters if params.lambda_balance > 0 else 1)
    for it in range(iters):
        local = (1.0 - sims).astype(np.float64)
        if params.lambda_balance > 0:
            for t in range(n):
                for j in range(k):
                    sid = int(ids[t, j])
                    if sid < 0:
                        continue
                    local[t, j] += bias_by_song[source.meta[sid].song_id]

        path_k = _viterbi_path(
            local, ids, shifts, source, params, concat, target_starts=target_starts
        )
        song_ids = [
            source.meta[int(ids[t, int(path_k[t])])].song_id for t in range(n)
        ]
        shares = _song_shares(song_ids)
        for s in unique_songs:
            excess = max(0.0, shares.get(s, 0.0) - target_share)
            bias_by_song[s] = params.lambda_balance * (excess**1.25) * (1.0 + 0.55 * it)

    tiles = _path_to_tiles(
        path_k,
        sims,
        ids,
        shifts,
        target_starts,
        source,
        target_durations=target_durations,
    )
    if not params.fidelity_first:
        tiles = _enforce_max_share(tiles, sims, ids, shifts, source, params.max_share)
    if params.min_run_tiles > 1:
        tiles = _enforce_min_runs(
            tiles,
            source,
            min_run=params.min_run_tiles,
            hop_s=params.hop_s,
            max_share=params.max_share,
        )
        if not params.fidelity_first:
            tiles = _enforce_max_share(tiles, sims, ids, shifts, source, params.max_share)
    _attach_layers(tiles, sims, ids, shifts, source, params, query=query)

    greedy_ids = [
        source.meta[int(ids[t, 0])].song_id for t in range(n) if ids[t, 0] >= 0
    ]
    viterbi_ids = [t.song_id for t in tiles]
    avg_sim = float(np.mean([t.similarity for t in tiles])) if tiles else 0.0

    return MatchResult(
        tiles=tiles,
        transitions_viterbi=_count_transitions(viterbi_ids),
        transitions_greedy=_count_transitions(greedy_ids),
        avg_similarity=avg_sim,
    )
