"""Source index with multi-metric search + boundary (join) features."""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

from app.pipeline.features import EmbPack
from app.pipeline.segment import Segment


@dataclass(slots=True)
class SourceMeta:
    song_id: str
    start_s: float
    end_s: float
    segment_idx: int


def _edge_vec(y: np.ndarray, sr: int, *, which: str, edge_ms: float = 50.0) -> np.ndarray:
    """Compact spectral fingerprint of clip head or tail (join-cost feature).

    Classic unit-selection join cost (Vepa & King; Schwarz CATERPILLAR):
    compare trailing frame of unit A to leading frame of unit B.
    """
    n = max(256, int(sr * edge_ms / 1000.0))
    if which == "head":
        chunk = y[:n]
    else:
        chunk = y[-n:]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    mfcc = librosa.feature.mfcc(y=chunk.astype(np.float32), sr=sr, n_mfcc=13, n_fft=512, hop_length=128)
    rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=128)
    vec = np.concatenate([mfcc.mean(axis=1), rms.mean(axis=1)]).astype(np.float32)
    norm = float(np.linalg.norm(vec)) + 1e-8
    return vec / norm


@dataclass(slots=True)
class SourceIndex:
    meta: list[SourceMeta]
    waveforms: list[np.ndarray]
    pack: EmbPack
    # Full tracks for contiguous-run reconstruction (song_id → mono float32)
    songs: dict[str, np.ndarray] = field(default_factory=dict)
    sr: int = 22_050
    head_feat: np.ndarray | None = None  # [n, D]
    tail_feat: np.ndarray | None = None

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
    ) -> tuple[np.ndarray, np.ndarray]:
        if query.backend == "clap-hybrid":
            w_chroma = 0.40 if w_chroma is None else w_chroma
            w_timbre = 0.50 if w_timbre is None else w_timbre
            w_energy = 0.10 if w_energy is None else w_energy
        else:
            w_chroma = 0.55 if w_chroma is None else w_chroma
            w_timbre = 0.35 if w_timbre is None else w_timbre
            w_energy = 0.10 if w_energy is None else w_energy

        n_q = query.chroma.shape[0]
        n_s = self.n
        if n_q == 0 or n_s == 0:
            return np.zeros((n_q, 0), np.float32), np.zeros((n_q, 0), np.int64)

        k = min(k, n_s)
        src_c = self.pack.chroma
        # Vectorized key-invariant chroma (12 shifts in one stack)
        rolls = np.stack([np.roll(src_c, s, axis=1) for s in range(12)], axis=0)  # [12,n_s,12]
        # [12, n_q, n_s]
        sim_all = np.einsum("qd,ksd->kqs", query.chroma, rolls, optimize=True)
        best_chroma = sim_all.max(axis=0).astype(np.float32)

        timbre = query.timbre @ self.pack.timbre.T
        energy = query.energy @ self.pack.energy.T
        score = (w_chroma * best_chroma + w_timbre * timbre + w_energy * energy).astype(np.float32)

        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        row = np.arange(n_q)[:, None]
        top_scores = score[row, idx]
        order = np.argsort(-top_scores, axis=1)
        ids = idx[row, order]
        sims = top_scores[row, order]
        return sims, ids.astype(np.int64)

    def join_distance(self, a_id: int, b_id: int) -> float:
        """Perceptual join cost in ~[0, 2]: 0 = seamless spectral match at boundary."""
        if self.tail_feat is None or self.head_feat is None:
            return 0.0
        if a_id == b_id:
            return 0.0
        # 1 - cosine (features are L2-normalized)
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
    return SourceIndex(
        meta=meta,
        waveforms=waveforms,
        pack=pack,
        songs=songs or {},
        sr=sr,
        head_feat=head,
        tail_feat=tail,
    )
