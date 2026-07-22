"""Segmentation tests."""

import numpy as np

from app.pipeline.segment import segment_audio


def test_segment_count_and_pad():
    sr = 22050
    y = np.zeros(int(1.2 * sr), dtype=np.float32)
    segs = segment_audio(y, sr, "A", window_s=0.5, hop_s=0.25, beat_sync=False)
    # starts: 0.0, 0.25, 0.5, 0.75 — last window pads past end, then stop
    assert len(segs) == 4
    assert all(len(s.waveform) == int(0.5 * sr) for s in segs)
    assert segs[0].start_s == 0.0
    assert abs(segs[1].start_s - 0.25) < 1e-4


def test_empty():
    assert segment_audio(np.zeros(0, dtype=np.float32), 22050, "A") == []


def test_beat_sync_falls_back_on_silence():
    sr = 22050
    y = np.zeros(int(2.0 * sr), dtype=np.float32)
    segs = segment_audio(
        y, sr, "A", window_s=1.0, hop_s=0.5, beat_sync=True, onset_sync=False
    )
    # Silence → beat track unusable → fixed-hop fallback
    assert len(segs) >= 2
    assert abs(segs[1].start_s - segs[0].start_s - 0.5) < 0.05


def test_onset_sync_on_clicks():
    sr = 22050
    y = np.zeros(int(2.0 * sr), dtype=np.float32)
    # Impulse every 0.4s → clear onsets
    for t in [0.4, 0.8, 1.2, 1.6]:
        i = int(t * sr)
        y[i : i + 32] = 0.9
    segs = segment_audio(
        y, sr, "A", window_s=0.35, hop_s=0.2, onset_sync=True, beat_sync=False
    )
    assert len(segs) >= 3
    # At least one start near an onset (not only the fixed grid)
    starts = [s.start_s for s in segs]
    assert any(abs(s - 0.4) < 0.08 or abs(s - 0.8) < 0.08 for s in starts)


def test_variable_length_follows_event_spacing():
    sr = 22050
    y = np.zeros(int(2.0 * sr), dtype=np.float32)
    for t in [0.3, 0.65, 1.2, 1.55]:
        y[int(t * sr) : int(t * sr) + 64] = 1.0
    segs = segment_audio(
        y,
        sr,
        "A",
        window_s=0.4,
        hop_s=0.2,
        onset_sync=True,
        beat_sync=False,
        variable_length=True,
    )
    lengths = {round(s.end_s - s.start_s, 2) for s in segs[:-1]}
    assert len(lengths) > 1
