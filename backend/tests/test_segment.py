"""Segmentation tests."""

import numpy as np

from app.pipeline.segment import segment_audio


def test_segment_count_and_pad():
    sr = 22050
    y = np.zeros(int(1.2 * sr), dtype=np.float32)
    segs = segment_audio(y, sr, "A", window_s=0.5, hop_s=0.25)
    # starts: 0.0, 0.25, 0.5, 0.75 — last window pads past end, then stop
    assert len(segs) == 4
    assert all(len(s.waveform) == int(0.5 * sr) for s in segs)
    assert segs[0].start_s == 0.0
    assert abs(segs[1].start_s - 0.25) < 1e-4


def test_empty():
    assert segment_audio(np.zeros(0, dtype=np.float32), 22050, "A") == []
