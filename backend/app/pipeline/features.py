"""Feature extraction: handcrafted + optional pretrained CLAP hybrid.

Hybrid design (best listening quality for mosaics):
- Chroma (handcrafted, key-invariant) → harmony / key feel
- CLAP music embedding (pretrained) → timbre, genre, "what it sounds like"
- Energy (RMS/centroid) → loudness contour

CLAP alone is weak on key transposition; chroma alone is weak on instrument identity.
Together they beat either alone for instrumental collage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np

from app.pipeline.segment import Segment

ProgressCb = Callable[[float, str], None]

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48_000


def _l2_rows(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, eps)).astype(np.float32)


def _mean_std(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or x.shape[1] == 0:
        return np.zeros(x.shape[0] * 2, dtype=np.float32)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


@dataclass(slots=True)
class EmbPack:
    chroma: np.ndarray  # [n, 12]
    timbre: np.ndarray  # [n, D] MFCC or CLAP
    energy: np.ndarray  # [n, 4]
    backend: str = "handcrafted"  # or "clap-hybrid"


class MusicalExtractor:
    """Handcrafted chroma + MFCC timbre + energy."""

    def __init__(self, *, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.name = "handcrafted"

    def _frames(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, **kw)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, **kw)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = min(chroma.shape[1], mfcc.shape[1], cent.shape[1], rms.shape[1])
        return chroma[:, :n], mfcc[:, :n], cent[:, :n], rms[:, :n]

    def _chroma_energy(
        self, y: np.ndarray, sr: int, segments: list[Segment], on_progress: ProgressCb | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns chroma [n,12], energy [n,4], optional mfcc-timbre [n,26]."""
        n = len(segments)
        if on_progress:
            on_progress(0.05, "frame features")
        chroma_f, mfcc_f, cent_f, rms_f = self._frames(y, sr)
        n_frames = chroma_f.shape[1]
        chroma = np.empty((n, 12), np.float32)
        timbre = np.empty((n, self.n_mfcc * 2), np.float32)
        energy = np.empty((n, 4), np.float32)
        report_every = max(1, n // 5)
        for i, seg in enumerate(segments):
            f0 = int(seg.start_s * sr) // self.hop_length
            f1 = int(seg.end_s * sr) // self.hop_length
            f0 = max(0, min(f0, n_frames - 1))
            f1 = max(f0 + 1, min(f1, n_frames))
            chroma[i] = chroma_f[:, f0:f1].mean(axis=1)
            timbre[i] = _mean_std(mfcc_f[:, f0:f1])
            energy[i] = _mean_std(np.vstack([rms_f[:, f0:f1], cent_f[:, f0:f1]]))
            if on_progress and (i % report_every == 0 or i == n - 1):
                on_progress(0.1 + 0.45 * (i + 1) / n, f"windows {i + 1}/{n}")
        return _l2_rows(chroma), _l2_rows(energy), _l2_rows(timbre)

    def embed_segments(
        self,
        y: np.ndarray,
        sr: int,
        segments: list[Segment],
        *,
        on_progress: ProgressCb | None = None,
    ) -> EmbPack:
        n = len(segments)
        if n == 0:
            return EmbPack(
                np.zeros((0, 12), np.float32),
                np.zeros((0, self.n_mfcc * 2), np.float32),
                np.zeros((0, 4), np.float32),
                backend=self.name,
            )
        chroma, energy, timbre = self._chroma_energy(y, sr, segments, on_progress)
        assert timbre is not None
        return EmbPack(chroma=chroma, timbre=timbre, energy=energy, backend=self.name)


class ClapHybridExtractor:
    """Key-invariant chroma + LAION CLAP music embeddings + energy."""

    def __init__(self, *, batch_size: int = 16):
        self.base = MusicalExtractor()
        self.batch_size = batch_size
        self.name = "clap-hybrid"
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._dim = 512

    def _to_device(self, inputs: dict):
        return {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

    @staticmethod
    def _as_emb(out) -> "torch.Tensor":
        import torch

        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
            return out.audio_embeds
        raise RuntimeError(f"Unexpected CLAP output type: {type(out)}")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        self._model = ClapModel.from_pretrained(CLAP_MODEL_ID)
        self._model.eval()
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        self._model.to(self._device)
        with torch.no_grad():
            dummy = np.zeros(CLAP_SR, dtype=np.float32)
            inputs = self._processor(
                audio=dummy, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
            inputs = self._to_device(inputs)
            out = self._as_emb(self._model.get_audio_features(**inputs))
            self._dim = int(out.shape[-1])

    def _clap_batch(self, waves: list[np.ndarray], sr: int) -> np.ndarray:
        import torch

        self._ensure_model()
        assert self._processor is not None and self._model is not None
        audio = []
        for w in waves:
            if sr != CLAP_SR:
                w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=CLAP_SR)
            if len(w) < CLAP_SR // 2:
                w = np.pad(w, (0, CLAP_SR // 2 - len(w)))
            audio.append(w.astype(np.float32))

        inputs = self._processor(
            audio=audio if len(audio) > 1 else audio[0],
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            emb = self._as_emb(self._model.get_audio_features(**inputs))
        arr = emb.detach().float().cpu().numpy().astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def embed_segments(
        self,
        y: np.ndarray,
        sr: int,
        segments: list[Segment],
        *,
        on_progress: ProgressCb | None = None,
    ) -> EmbPack:
        n = len(segments)
        if n == 0:
            return EmbPack(
                np.zeros((0, 12), np.float32),
                np.zeros((0, self._dim), np.float32),
                np.zeros((0, 4), np.float32),
                backend=self.name,
            )

        chroma, energy, _mfcc = self.base._chroma_energy(y, sr, segments, on_progress)
        self._ensure_model()
        timbre = np.empty((n, self._dim), np.float32)
        waves = [s.waveform for s in segments]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            timbre[start:end] = self._clap_batch(waves[start:end], sr)
            if on_progress:
                on_progress(0.55 + 0.45 * end / n, f"CLAP {end}/{n}")
        return EmbPack(
            chroma=chroma,
            timbre=_l2_rows(timbre),
            energy=energy,
            backend=self.name,
        )


_extractor_singleton: MusicalExtractor | ClapHybridExtractor | None = None
_extractor_name: str | None = None


def get_extractor(*, prefer_clap: bool = True) -> MusicalExtractor | ClapHybridExtractor:
    """Lazy singleton. Tries CLAP hybrid first; falls back to handcrafted."""
    global _extractor_singleton, _extractor_name
    if _extractor_singleton is not None:
        return _extractor_singleton

    if prefer_clap:
        try:
            ext = ClapHybridExtractor()
            ext._ensure_model()
            _extractor_singleton = ext
            _extractor_name = ext.name
            return ext
        except Exception as e:
            print(f"[musicmosaic] CLAP unavailable ({e}); using handcrafted features", flush=True)

    _extractor_singleton = MusicalExtractor()
    _extractor_name = _extractor_singleton.name
    return _extractor_singleton


HandcraftedExtractor = MusicalExtractor
