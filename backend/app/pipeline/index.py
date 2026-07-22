"""Source index with multi-metric musical search (no FAISS needed at this scale)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.segment import Segment


@dataclass(slots=True)
class SourceMeta:
    song_id: str
    start_s: float
    end_s: float
    segment_idx: int


@dataclass(slots=True)
class SourceIndex:
    meta: list[SourceMeta]
    waveforms: list[np.ndarray]
    pack: EmbPack

    @property
    def n(self) -> int:
        return len(self.meta)

    def search(
        self,
        query: EmbPack,
        *,
        k: int = 12,
        w_chroma: float = 0.55,
        w_timbre: float = 0.35,
        w_energy: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores [n_q,k], ids [n_q,k]) — higher is better.

        Chroma similarity is key-invariant (best of 12 pitch-class rotations).
        """
        n_q = query.chroma.shape[0]
        n_s = self.n
        if n_q == 0 or n_s == 0:
            return np.zeros((n_q, 0), np.float32), np.zeros((n_q, 0), np.int64)

        k = min(k, n_s)
        # Key-invariant chroma: max over cyclic shifts of source chroma
        best_chroma = np.full((n_q, n_s), -1.0, dtype=np.float32)
        src_c = self.pack.chroma
        for shift in range(12):
            rolled = np.roll(src_c, shift, axis=1)
            sim = query.chroma @ rolled.T
            np.maximum(best_chroma, sim, out=best_chroma)

        timbre = query.timbre @ self.pack.timbre.T
        energy = query.energy @ self.pack.energy.T
        score = (w_chroma * best_chroma + w_timbre * timbre + w_energy * energy).astype(np.float32)

        # Top-k per query row
        # argpartition then sort the k
        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n_q)[:, None]
        top_scores = score[row, idx]
        order = np.argsort(-top_scores, axis=1)
        ids = idx[row, order]
        sims = top_scores[row, order]
        return sims, ids.astype(np.int64)


def build_source_index(segments: list[Segment], pack: EmbPack) -> SourceIndex:
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
    return SourceIndex(meta=meta, waveforms=waveforms, pack=pack)
