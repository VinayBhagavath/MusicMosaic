"""Source index with multi-metric search, join features, and key-shift tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.segment import Segment


@dataclass(slots=True)
class SourceMeta:
    song_id: str
    start_s: float
    end_s: float
    segment_idx: int
    role: str = "full"
    f0_hz: float | None = None


def _edge_vec(y: np.ndarray, sr: int, *, which: str, edge_ms: float = 80.0) -> np.ndarray:
    """Spectral fingerprint of clip head/tail for concatenation / join cost.

    Slightly longer than a click (~80 ms) so joins reflect timbre continuity,
    not just a transient spike. Log-spaced bands ≈ MFCC-ish envelope, cheap.
    """
    n = max(256, int(sr * edge_ms / 1000.0))
    chunk = y[:n] if which == "head" else y[-n:]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    chunk = chunk.astype(np.float32, copy=False)
    windowed = chunk * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(windowed)).astype(np.float32)
    edges = np.geomspace(1, max(2, len(mag) - 1), 14).astype(np.int64)
    bands = np.empty(13, dtype=np.float32)
    for i in range(13):
        a, b = int(edges[i]), int(edges[i + 1])
        bands[i] = float(np.mean(mag[a:b])) if b > a else 0.0
    bands = np.log1p(bands)
    rms = np.array([float(np.sqrt(np.mean(chunk * chunk) + 1e-12))], dtype=np.float32)
    # First-difference of bands → rough spectral flux at the edge
    flux = np.array([float(np.mean(np.abs(np.diff(bands))))], dtype=np.float32)
    vec = np.concatenate([bands, rms, flux])
    norm = float(np.linalg.norm(vec)) + 1e-8
    return (vec / norm).astype(np.float32)


def _weights(
    backend: str,
    w_chroma: float | None,
    w_timbre: float | None,
    w_energy: float | None,
) -> tuple[float, float, float]:
    # MIDI-note model: pitch class matters (we'll Rubber-Band to target), but
    # the perceived "is this the right sound" is dominated by timbre, so weight
    # timbre nearly as much as pitch. Dynamics/onsets (velocity/attack) last.
    if backend in ("mosaic", "handcrafted"):
        return (
            0.42 if w_chroma is None else w_chroma,
            0.40 if w_timbre is None else w_timbre,
            0.18 if w_energy is None else w_energy,
        )
    if backend == "clap-hybrid":
        return (
            0.42 if w_chroma is None else w_chroma,
            0.40 if w_timbre is None else w_timbre,
            0.18 if w_energy is None else w_energy,
        )
    return (
        0.48 if w_chroma is None else w_chroma,
        0.32 if w_timbre is None else w_timbre,
        0.20 if w_energy is None else w_energy,
    )


def _score_matrix(
    query: EmbPack,
    pack: EmbPack,
    *,
    w_chroma: float,
    w_timbre: float,
    w_energy: float,
    chroma_rolls: np.ndarray | None = None,
    temporal_rolls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (score [n_q,n_s], best_shift [n_q,n_s])."""
    src_c = pack.chroma
    rolls = (
        chroma_rolls
        if chroma_rolls is not None
        else np.stack([np.roll(src_c, s, axis=1) for s in range(12)], axis=0)
    )
    sim_all = np.einsum("qd,ksd->kqs", query.chroma, rolls, optimize=True)
    has_temporal = (
        query.temporal is not None
        and pack.temporal is not None
        and query.temporal.shape[1] % 12 == 0
        and pack.temporal.shape[1] == query.temporal.shape[1]
    )
    if has_temporal:
        bins = query.temporal.shape[1] // 12
        q_temporal = query.temporal.reshape(len(query.temporal), bins, 12)
        if temporal_rolls is None:
            src_temporal = pack.temporal.reshape(len(pack.temporal), bins, 12)
            temporal_rolls = np.stack(
                [np.roll(src_temporal, s, axis=2) for s in range(12)], axis=0
            )
        temporal_sim = np.einsum(
            "qbd,ksbd->kqs", q_temporal, temporal_rolls, optimize=True
        )
        # Rerank the key-shift hypotheses by their short-time harmonic path.
        sim_all = 0.68 * sim_all + 0.32 * temporal_sim
    best_shift = sim_all.argmax(axis=0).astype(np.int8)
    best_chroma = sim_all.max(axis=0).astype(np.float32)
    timbre = query.timbre @ pack.timbre.T
    energy = query.energy @ pack.energy.T
    if (
        query.level is not None
        and pack.level is not None
        and query.level.shape[1] == pack.level.shape[1]
    ):
        # Row-normalizing the compound dynamics descriptor preserves its shape
        # but largely hides absolute note velocity. Blend in the peak-normalized
        # track-relative dB level so quiet and loud target notes choose
        # correspondingly quiet and loud source events.
        level_sim = 1.0 - np.mean(
            np.abs(query.level[:, None, :] - pack.level[None, :, :]),
            axis=2,
        )
        energy = 0.72 * energy + 0.28 * np.clip(level_sim, 0.0, 1.0)
    register = 0.0
    register_weight = 0.0
    if (
        query.register is not None
        and pack.register is not None
        and query.register.shape[1] == pack.register.shape[1]
    ):
        register = query.register @ pack.register.T
        register_weight = 0.14
    base_scale = 1.0 - register_weight
    score = (
        base_scale
        * (w_chroma * best_chroma + w_timbre * timbre + w_energy * energy)
        + register_weight * register
    ).astype(np.float32)
    return score, best_shift


@dataclass(slots=True)
class SourceIndex:
    meta: list[SourceMeta]
    waveforms: list[np.ndarray]
    pack: EmbPack
    songs: dict[str, np.ndarray] = field(default_factory=dict)
    sr: int = 22_050
    head_feat: np.ndarray | None = None
    tail_feat: np.ndarray | None = None
    chroma_rolls: np.ndarray | None = None
    temporal_rolls: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.meta)

    def search(
        self,
        query: EmbPack,
        *,
        k: int = 12,
        w_chroma: float | None = None,
        w_timbre: float | None = None,
        w_energy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (scores [n_q,k], ids [n_q,k], key_shifts [n_q,k])."""
        w_chroma, w_timbre, w_energy = _weights(
            query.backend, w_chroma, w_timbre, w_energy
        )
        n_q = query.chroma.shape[0]
        n_s = self.n
        if n_q == 0 or n_s == 0:
            z = np.zeros((n_q, 0), np.float32)
            return z, np.zeros((n_q, 0), np.int64), np.zeros((n_q, 0), np.int8)

        k = min(k, n_s)
        score, best_shift = _score_matrix(
            query,
            self.pack,
            w_chroma=w_chroma,
            w_timbre=w_timbre,
            w_energy=w_energy,
            chroma_rolls=self.chroma_rolls,
            temporal_rolls=self.temporal_rolls,
        )
        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n_q)[:, None]
        top_scores = score[row, idx]
        order = np.argsort(-top_scores, axis=1)
        ids = idx[row, order]
        sims = top_scores[row, order]
        shifts = best_shift[row, ids]
        return sims, ids.astype(np.int64), shifts

    def search_diverse(
        self,
        query: EmbPack,
        *,
        k: int = 15,
        per_song: int = 3,
        w_chroma: float | None = None,
        w_timbre: float | None = None,
        w_energy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-k search that keeps multiple songs in the candidate beam."""
        w_chroma, w_timbre, w_energy = _weights(
            query.backend, w_chroma, w_timbre, w_energy
        )
        n_q = query.chroma.shape[0]
        n_s = self.n
        if n_q == 0 or n_s == 0:
            z = np.zeros((n_q, 0), np.float32)
            return z, np.zeros((n_q, 0), np.int64), np.zeros((n_q, 0), np.int8)

        k = min(k, n_s)
        per_song = max(1, per_song)
        score, best_shift = _score_matrix(
            query,
            self.pack,
            w_chroma=w_chroma,
            w_timbre=w_timbre,
            w_energy=w_energy,
            chroma_rolls=self.chroma_rolls,
            temporal_rolls=self.temporal_rolls,
        )

        song_of = np.array([m.song_id for m in self.meta], dtype=object)
        unique_songs = list(dict.fromkeys(song_of.tolist()))
        # Precompute per-song index masks once
        song_masks = {
            song: np.where(song_of == song)[0] for song in unique_songs
        }

        out_ids = np.full((n_q, k), -1, dtype=np.int64)
        out_sims = np.full((n_q, k), -np.inf, dtype=np.float32)
        out_shifts = np.zeros((n_q, k), dtype=np.int8)

        for qi in range(n_q):
            row = score[qi]
            picked: list[tuple[float, int]] = []
            used: set[int] = set()
            for song in unique_songs:
                mask = song_masks[song]
                if mask.size == 0:
                    continue
                take = min(per_song, mask.size)
                if take >= mask.size:
                    local = mask
                else:
                    part = np.argpartition(-row[mask], kth=take - 1)[:take]
                    local = mask[part]
                for idx in local:
                    ii = int(idx)
                    picked.append((float(row[ii]), ii))
                    used.add(ii)
            if len(picked) < k:
                # Fill remaining from global top scores
                order = np.argpartition(-row, kth=min(k, n_s) - 1)[: min(k * 3, n_s)]
                order = order[np.argsort(-row[order])]
                for idx in order:
                    ii = int(idx)
                    if ii in used:
                        continue
                    picked.append((float(row[ii]), ii))
                    used.add(ii)
                    if len(picked) >= k:
                        break
            picked.sort(reverse=True)
            picked = picked[:k]
            for j, (sc, idx) in enumerate(picked):
                out_ids[qi, j] = idx
                out_sims[qi, j] = sc
                out_shifts[qi, j] = best_shift[qi, idx]

        return out_sims, out_ids, out_shifts

    def join_distance(self, a_id: int, b_id: int) -> float:
        if self.tail_feat is None or self.head_feat is None:
            return 0.0
        if a_id == b_id:
            return 0.0
        return float(max(0.0, 1.0 - np.dot(self.tail_feat[a_id], self.head_feat[b_id])))


def build_source_index(
    segments: list[Segment],
    pack: EmbPack,
    *,
    songs: dict[str, np.ndarray] | None = None,
    sr: int = 22_050,
    compute_edges: bool = True,
) -> SourceIndex:
    if len(segments) != pack.chroma.shape[0]:
        raise ValueError("segments/embeddings length mismatch")
    meta = [
        SourceMeta(
            song_id=s.song_id,
            start_s=s.start_s,
            end_s=s.end_s,
            segment_idx=s.index,
        )
        for s in segments
    ]
    waveforms = [s.waveform for s in segments]
    head = tail = None
    if compute_edges and waveforms:
        head = np.stack([_edge_vec(w, sr, which="head") for w in waveforms])
        tail = np.stack([_edge_vec(w, sr, which="tail") for w in waveforms])
    chroma_rolls = np.stack(
        [np.roll(pack.chroma, shift, axis=1) for shift in range(12)], axis=0
    )
    temporal_rolls = None
    if pack.temporal is not None and pack.temporal.shape[1] % 12 == 0:
        bins = pack.temporal.shape[1] // 12
        temporal = pack.temporal.reshape(len(pack.temporal), bins, 12)
        temporal_rolls = np.stack(
            [np.roll(temporal, shift, axis=2) for shift in range(12)], axis=0
        )
    return SourceIndex(
        meta=meta,
        waveforms=waveforms,
        pack=pack,
        songs=songs or {},
        sr=sr,
        head_feat=head,
        tail_feat=tail,
        chroma_rolls=chroma_rolls,
        temporal_rolls=temporal_rolls,
    )
