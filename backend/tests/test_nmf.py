import numpy as np

from app.pipeline.metrics import candidate_improves, quality_metrics
from app.pipeline.nmf import NMFParams, reconstruct_nmf


def _tone(sr: int, hz: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.25 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_sparse_nmf_reconstructs_target_from_source_spectra():
    sr = 8000
    a = _tone(sr, 220.0)
    b = _tone(sr, 330.0)
    target = np.concatenate([a[:sr], b[sr:]])

    result = reconstruct_nmf(
        target,
        {"A": a, "B": b},
        sr,
        params=NMFParams(
            n_fft=512,
            hop_length=128,
            n_mels=32,
            candidate_k=8,
            nearest_k=4,
            iterations=4,
            polyphony=2,
        ),
    )

    assert len(result.audio) == len(target)
    assert float(np.max(np.abs(result.audio))) > 0.01
    assert result.n_source_frames > 0
    assert result.n_target_frames > 0
    assert 1.0 <= result.active_polyphony <= 2.0
    assert abs(sum(result.contribution_pct.values()) - 100.0) < 0.2
    metrics = quality_metrics(target, result.audio, sr)
    assert metrics["chroma_similarity"] > 0.80


def test_nmf_can_use_source_only_phase_reference():
    sr = 8000
    source = _tone(sr, 220.0)
    target = source.copy()
    result = reconstruct_nmf(
        target,
        {"A": source},
        sr,
        params=NMFParams(
            n_fft=512,
            hop_length=128,
            n_mels=24,
            candidate_k=4,
            nearest_k=2,
            iterations=2,
            polyphony=1,
        ),
        phase_reference=source,
    )
    assert quality_metrics(target, result.audio, sr)["chroma_similarity"] > 0.95


def test_auto_gate_accepts_real_improvement_and_rejects_tradeoff():
    baseline = {
        "log_mel_distance": 0.73,
        "chroma_similarity": 0.73,
        "onset_correlation": 0.26,
        "boundary_discontinuity": 0.05,
    }
    improved = {
        "log_mel_distance": 0.54,
        "chroma_similarity": 0.80,
        "onset_correlation": 0.64,
        "boundary_discontinuity": 0.02,
    }
    lost_harmony = {
        "log_mel_distance": 0.40,
        "chroma_similarity": 0.60,
        "onset_correlation": 0.70,
        "boundary_discontinuity": 0.01,
    }
    assert candidate_improves(baseline, improved)
    assert not candidate_improves(baseline, lost_harmony)
