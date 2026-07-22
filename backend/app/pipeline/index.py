"""FAISS index over source segments."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from app.pipeline.segment import Segment


@dataclass(slots=True)
class SourceMeta:
    song_id: str
    start_s: float
    end_s: float
    segment_idx: int


@dataclass(slots=True)
class SourceIndex:
    index: faiss.IndexFlatIP
    meta: list[SourceMeta]
    waveforms: list[np.ndarray]
    embeddings: np.ndarray

    def search(self, queries: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities [n,k], ids [n,k]). Queries must be L2-normalized."""
        k = min(k, self.index.ntotal)
        sims, ids = self.index.search(queries.astype(np.float32), k)
        return sims, ids


def build_source_index(
    segments: list[Segment],
    embeddings: np.ndarray,
) -> SourceIndex:
    if len(segments) != len(embeddings):
        raise ValueError("segments/embeddings length mismatch")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    xb = np.ascontiguousarray(embeddings.astype(np.float32))
    # Guard against zero vectors
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / np.maximum(norms, 1e-8)
    index.add(xb)

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
    return SourceIndex(index=index, meta=meta, waveforms=waveforms, embeddings=xb)
