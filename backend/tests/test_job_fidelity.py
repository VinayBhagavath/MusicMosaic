import json

import numpy as np

from app.pipeline.job import JobConfig, run_job


def test_run_job_emits_fidelity_metrics_and_variable_tiles(tmp_path, monkeypatch):
    sr = 22050

    def fake_load(path):
        name = str(path)
        hz = 440.0 if "target" in name else 220.0 + 30.0 * int(name[-1])
        t = np.arange(sr, dtype=np.float32) / sr
        y = 0.2 * np.sin(2 * np.pi * hz * t)
        y[:: sr // 4] += 0.5
        return y.astype(np.float32), sr

    monkeypatch.setattr("app.pipeline.job.load_and_normalize", fake_load)
    monkeypatch.setattr("app.pipeline.job.write_wav", lambda *args, **kwargs: None)

    result = run_job(
        "target",
        ["source0", "source1", "source2", "source3", "source4"],
        tmp_path,
        config=JobConfig(
            window_s=0.3,
            hop_s=0.25,
            top_k=5,
            beat_sync=False,
            onset_sync=False,
            apply_key_shift=False,
            spectral_match=False,
            transient_match=False,
            n_layers=1,
        ),
    )

    stats = result.mosaic["stats"]
    assert stats["fidelity_first"] is True
    assert set(stats["quality"]) == {
        "log_mel_distance",
        "chroma_similarity",
        "onset_correlation",
        "boundary_discontinuity",
    }
    assert result.mosaic["tiles"]
    assert all(tile["target_duration_s"] is not None for tile in result.mosaic["tiles"])
    persisted = json.loads((tmp_path / "mosaic.json").read_text())
    assert persisted["stats"]["quality"] == stats["quality"]
