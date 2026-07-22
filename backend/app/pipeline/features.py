"""Feature extraction for acoustic mosaicing (concatenative synthesis).

Default path uses classic MIR descriptors proven in CataRT / AudioGuide-style
mosaicing — not semantic neural embeddings:

- Chroma (CQT when available) → harmony / key
- MFCC + spectral contrast → timbre / spectral envelope
- Loudness, centroid, flatness, onset strength → dynamics & articulation

CLAP hybrid remains available via prefer_clap=True / MUSICMOSAIC_USE_CLAP=1
but is demoted: it optimizes semantic similarity, which is the wrong objective
for frame-level mosaicing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np

from app.pipeline.segment import Segment

ProgressCb = Callable[[float, str], None]

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48_000

# Mosaic descriptor layout (handcrafted)
N_MFCC = 13
N_CONTRAST = 7  # librosa default spectral_contrast bands
TIMBRE_DIM = N_MFCC * 2 + N_CONTRAST  # mean+std MFCC + contrast means
ENERGY_DIM = 6  # rms μ/σ, centroid μ/σ, flatness μ, onset μ
TEMPORAL_BINS = 4
REGISTER_BANDS = 6


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
    timbre: np.ndarray  # [n, D] MFCC+/CLAP
    energy: np.ndarray  # [n, E]
    backend: str = "mosaic"  # mosaic | handcrafted | clap-hybrid
    temporal: np.ndarray | None = None  # [n, TEMPORAL_BINS * 12] chroma trajectory
    register: np.ndarray | None = None  # [n, REGISTER_BANDS] octave/register profile


def _temporal_pool(x: np.ndarray, bins: int = TEMPORAL_BINS) -> np.ndarray:
    """Pool feature frames into a fixed trajectory without discarding ordering."""
    if x.size == 0 or x.shape[1] == 0:
        return np.zeros(x.shape[0] * bins, dtype=np.float32)
    chunks = np.array_split(np.arange(x.shape[1]), bins)
    pooled = [
        x[:, idx].mean(axis=1) if len(idx) else np.zeros(x.shape[0], dtype=np.float32)
        for idx in chunks
    ]
    return np.concatenate(pooled).astype(np.float32)


class MusicalExtractor:
    """Mosaicing descriptors: chroma + MFCC/contrast timbre + dynamics."""

    def __init__(
        self,
        *,
        n_mfcc: int = N_MFCC,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_cqt_chroma: bool = True,
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_cqt_chroma = use_cqt_chroma
        self.name = "mosaic"
        self.timbre_dim = n_mfcc * 2 + N_CONTRAST
        self.energy_dim = ENERGY_DIM

    def _frames(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, ...]:
        kw = dict(n_fft=self.n_fft, hop_length=self.hop_length)
        if self.use_cqt_chroma:
            try:
                chroma = librosa.feature.chroma_cqt(
                    y=y, sr=sr, hop_length=self.hop_length, n_chroma=12
                )
            except Exception:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, **kw)
        else:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, **kw)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, **kw)
        # Spectral contrast: band-wise peak-to-valley — strong timbre cue beyond MFCC
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, **kw)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, **kw)
        flat = librosa.feature.spectral_flatness(y=y, **kw)
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset = np.atleast_2d(onset)
        # Coarse log-frequency energy retains register/octave information that
        # pitch-class chroma intentionally discards.
        register = np.log1p(
            librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=REGISTER_BANDS,
                fmin=32.0,
                fmax=min(sr / 2, 8000.0),
                power=1.0,
            )
        )

        n = min(
            chroma.shape[1],
            mfcc.shape[1],
            contrast.shape[1],
            cent.shape[1],
            flat.shape[1],
            rms.shape[1],
            onset.shape[1],
            register.shape[1],
        )
        return (
            chroma[:, :n],
            mfcc[:, :n],
            contrast[:, :n],
            cent[:, :n],
            flat[:, :n],
            rms[:, :n],
            onset[:, :n],
            register[:, :n],
        )

    def _chroma_energy(
        self, y: np.ndarray, sr: int, segments: list[Segment], on_progress: ProgressCb | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Legacy helper used by CLAP path: chroma + energy + optional mfcc-timbre."""
        n = len(segments)
        if on_progress:
            on_progress(0.05, "frame features")
        chroma_f, mfcc_f, _contrast, cent_f, flat_f, rms_f, onset_f, _register = (
            self._frames(y, sr)
        )
        n_frames = chroma_f.shape[1]
        chroma = np.empty((n, 12), np.float32)
        timbre = np.empty((n, self.n_mfcc * 2), np.float32)
        energy = np.empty((n, ENERGY_DIM), np.float32)
        report_every = max(1, n // 5)
        for i, seg in enumerate(segments):
            f0 = int(seg.start_s * sr) // self.hop_length
            f1 = int(seg.end_s * sr) // self.hop_length
            f0 = max(0, min(f0, n_frames - 1))
            f1 = max(f0 + 1, min(f1, n_frames))
            chroma[i] = chroma_f[:, f0:f1].mean(axis=1)
            timbre[i] = _mean_std(mfcc_f[:, f0:f1])
            energy[i] = np.concatenate(
                [
                    _mean_std(np.vstack([rms_f[:, f0:f1], cent_f[:, f0:f1]])),
                    np.array(
                        [
                            float(flat_f[:, f0:f1].mean()),
                            float(onset_f[:, f0:f1].mean()),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
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
                np.zeros((0, self.timbre_dim), np.float32),
                np.zeros((0, self.energy_dim), np.float32),
                backend=self.name,
            )

        if on_progress:
            on_progress(0.05, "mosaic descriptors")
        (
            chroma_f,
            mfcc_f,
            contrast_f,
            cent_f,
            flat_f,
            rms_f,
            onset_f,
            register_f,
        ) = self._frames(y, sr)
        n_frames = chroma_f.shape[1]
        chroma = np.empty((n, 12), np.float32)
        timbre = np.empty((n, self.timbre_dim), np.float32)
        energy = np.empty((n, self.energy_dim), np.float32)
        temporal = np.empty((n, TEMPORAL_BINS * 12), np.float32)
        register = np.empty((n, REGISTER_BANDS), np.float32)
        report_every = max(1, n // 5)

        for i, seg in enumerate(segments):
            f0 = int(round(seg.start_s * sr)) // self.hop_length
            f1 = int(round(seg.end_s * sr)) // self.hop_length
            f0 = max(0, min(f0, n_frames - 1))
            f1 = max(f0 + 1, min(f1, n_frames))

            chroma[i] = chroma_f[:, f0:f1].mean(axis=1)
            temporal[i] = _temporal_pool(chroma_f[:, f0:f1])
            register[i] = register_f[:, f0:f1].mean(axis=1)
            mfcc_ms = _mean_std(mfcc_f[:, f0:f1])
            # Contrast: mean over time per band (timbre shape beyond MFCC)
            n_c = min(N_CONTRAST, contrast_f.shape[0])
            contrast_m = contrast_f[:n_c, f0:f1].mean(axis=1)
            if n_c < N_CONTRAST:
                contrast_m = np.pad(contrast_m, (0, N_CONTRAST - n_c))
            timbre[i] = np.concatenate([mfcc_ms, contrast_m.astype(np.float32)])

            energy[i] = np.array(
                [
                    float(rms_f[:, f0:f1].mean()),
                    float(rms_f[:, f0:f1].std()),
                    float(cent_f[:, f0:f1].mean()),
                    float(cent_f[:, f0:f1].std()),
                    float(flat_f[:, f0:f1].mean()),
                    float(onset_f[:, f0:f1].mean()),
                ],
                dtype=np.float32,
            )

            if on_progress and (i % report_every == 0 or i == n - 1):
                on_progress(0.1 + 0.85 * (i + 1) / n, f"windows {i + 1}/{n}")

        return EmbPack(
            chroma=_l2_rows(chroma),
            timbre=_l2_rows(timbre),
            energy=_l2_rows(energy),
            backend=self.name,
            temporal=_l2_rows(temporal),
            register=_l2_rows(register),
        )


class ClapHybridExtractor:
    """Optional semantic path — kept for experiments, not default mosaicing."""

    def __init__(self, *, batch_size: int = 48):
        self.base = MusicalExtractor()
        self.batch_size = batch_size
        self.name = "clap-hybrid"
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._use_amp = False
        self._amp_dtype = None
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
            self._use_amp = True
            self._amp_dtype = torch.float16
        elif torch.cuda.is_available():
            self._device = "cuda"
            self._use_amp = True
            self._amp_dtype = torch.float16
        else:
            self._device = "cpu"
            self._use_amp = False
            self._amp_dtype = None
        self._model.to(self._device)
        with torch.no_grad():
            dummy = np.zeros(CLAP_SR, dtype=np.float32)
            inputs = self._processor(
                audio=dummy, sampling_rate=CLAP_SR, return_tensors="pt", padding=True
            )
            inputs = self._to_device(inputs)
            try:
                out = self._forward_features(inputs)
            except Exception:
                self._use_amp = False
                self._amp_dtype = None
                out = self._forward_features(inputs)
            self._dim = int(out.shape[-1])

    def _forward_features(self, inputs: dict):
        import torch

        assert self._model is not None
        if self._use_amp and self._device in ("mps", "cuda"):
            device_type = "cuda" if self._device == "cuda" else "mps"
            with torch.autocast(device_type=device_type, dtype=self._amp_dtype):
                return self._as_emb(self._model.get_audio_features(**inputs))
        return self._as_emb(self._model.get_audio_features(**inputs))

    def _clap_batch(
        self,
        waves: list[np.ndarray],
        *,
        already_48k: bool = True,
        orig_sr: int | None = None,
    ) -> np.ndarray:
        import torch

        self._ensure_model()
        assert self._processor is not None and self._model is not None
        audio = []
        for w in waves:
            w = w.astype(np.float32, copy=False)
            if not already_48k:
                if orig_sr is None:
                    raise ValueError("orig_sr required when already_48k=False")
                w = librosa.resample(w, orig_sr=orig_sr, target_sr=CLAP_SR)
            if len(w) < CLAP_SR // 2:
                w = np.pad(w, (0, CLAP_SR // 2 - len(w)))
            audio.append(w)

        inputs = self._processor(
            audio=audio if len(audio) > 1 else audio[0],
            sampling_rate=CLAP_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            try:
                emb = self._forward_features(inputs)
            except Exception:
                self._use_amp = False
                self._amp_dtype = None
                emb = self._forward_features(inputs)
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
                np.zeros((0, ENERGY_DIM), np.float32),
                backend=self.name,
            )

        chroma, energy, _mfcc = self.base._chroma_energy(y, sr, segments, on_progress)
        self._ensure_model()

        if sr != CLAP_SR:
            if on_progress:
                on_progress(0.52, "resample for CLAP")
            y48 = librosa.resample(
                y.astype(np.float32), orig_sr=sr, target_sr=CLAP_SR
            ).astype(np.float32)
            ratio = CLAP_SR / float(sr)
        else:
            y48 = y.astype(np.float32, copy=False)
            ratio = 1.0

        def _slice48(seg: Segment) -> np.ndarray:
            a = int(round(seg.start_s * sr * ratio))
            b = int(round(seg.end_s * sr * ratio))
            if b <= a:
                b = a + max(1, int(CLAP_SR // 2))
            return y48[a:b]

        stride = 2 if n >= 4 else 1
        anchor_idx = list(range(0, n, stride))
        if anchor_idx[-1] != n - 1:
            anchor_idx.append(n - 1)
        waves = [_slice48(segments[i]) for i in anchor_idx]
        anchor_emb = np.empty((len(anchor_idx), self._dim), np.float32)
        for start in range(0, len(anchor_idx), self.batch_size):
            end = min(start + self.batch_size, len(anchor_idx))
            anchor_emb[start:end] = self._clap_batch(
                waves[start:end], already_48k=True
            )
            if on_progress:
                on_progress(
                    0.55 + 0.45 * end / len(anchor_idx),
                    f"CLAP {end}/{len(anchor_idx)} anchors",
                )

        timbre = np.empty((n, self._dim), np.float32)
        for a in range(len(anchor_idx) - 1):
            i0, i1 = anchor_idx[a], anchor_idx[a + 1]
            e0, e1 = anchor_emb[a], anchor_emb[a + 1]
            span = max(1, i1 - i0)
            for i in range(i0, i1 + 1):
                alpha = (i - i0) / span
                timbre[i] = (1.0 - alpha) * e0 + alpha * e1
        timbre[anchor_idx[-1]] = anchor_emb[-1]

        return EmbPack(
            chroma=chroma,
            timbre=_l2_rows(timbre),
            energy=energy,
            backend=self.name,
        )


_extractor_singleton: MusicalExtractor | ClapHybridExtractor | None = None


def reset_extractor() -> None:
    """Test helper / force reload."""
    global _extractor_singleton
    _extractor_singleton = None


def get_extractor(*, prefer_clap: bool | None = None) -> MusicalExtractor | ClapHybridExtractor:
    """Lazy singleton. Default: mosaic (handcrafted). CLAP only if explicitly requested."""
    global _extractor_singleton

    if prefer_clap is None:
        prefer_clap = os.environ.get("MUSICMOSAIC_USE_CLAP", "").strip() in (
            "1",
            "true",
            "yes",
        )

    want = "clap-hybrid" if prefer_clap else "mosaic"
    if _extractor_singleton is not None:
        if _extractor_singleton.name == want or (
            not prefer_clap and _extractor_singleton.name in ("mosaic", "handcrafted")
        ):
            return _extractor_singleton
        # Preference changed — rebuild
        _extractor_singleton = None

    if prefer_clap:
        try:
            ext = ClapHybridExtractor()
            ext._ensure_model()
            _extractor_singleton = ext
            print("[musicmosaic] using CLAP-hybrid embeddings (opt-in)", flush=True)
            return ext
        except Exception as e:
            print(f"[musicmosaic] CLAP unavailable ({e}); using mosaic descriptors", flush=True)

    _extractor_singleton = MusicalExtractor()
    print(
        "[musicmosaic] using mosaic descriptors (chroma+MFCC+contrast+dynamics)",
        flush=True,
    )
    return _extractor_singleton


HandcraftedExtractor = MusicalExtractor
