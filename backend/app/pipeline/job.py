"""End-to-end mosaic job orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.pipeline.audio_io import load_and_normalize, write_wav
from app.pipeline.features import HandcraftedExtractor
from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, match_sequence
from app.pipeline.reconstruct import reconstruct_ola
from app.pipeline.segment import segment_audio

ProgressCb = Callable[[str, float, str], None]  # stage, pct 0-100, message

SONG_COLORS = ["#E85D4C", "#F0A202", "#3DDC97", "#4C6EF5", "#E599F7"]
SONG_IDS = ["A", "B", "C", "D", "E"]


@dataclass
class JobConfig:
    window_s: float = 0.5
    hop_s: float = 0.25
    top_k: int = 8
    lambda_switch: float = 0.35
    lambda_jump: float = 0.25
    jump_norm_s: float = 2.0
    lambda_self: float = 0.05


@dataclass
class JobResult:
    mosaic: dict
    duration_s: float
    elapsed_s: float
    paths: dict[str, str] = field(default_factory=dict)


def _contrib(tiles: list, song_ids: list[str]) -> dict[str, float]:
    if not tiles:
        return {s: 0.0 for s in song_ids}
    counts = {s: 0 for s in song_ids}
    for t in tiles:
        counts[t.song_id] = counts.get(t.song_id, 0) + 1
    n = len(tiles)
    return {s: round(100.0 * counts.get(s, 0) / n, 1) for s in song_ids}


def run_job(
    target_path: str | Path,
    source_paths: list[str | Path],
    out_dir: str | Path,
    *,
    config: JobConfig | None = None,
    source_names: list[str] | None = None,
    on_progress: ProgressCb | None = None,
) -> JobResult:
    """Load → segment → embed → index → match → reconstruct → write artifacts."""
    cfg = config or JobConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def prog(stage: str, pct: float, msg: str) -> None:
        if on_progress:
            on_progress(stage, pct, msg)

    if len(source_paths) != 5:
        raise ValueError(f"Expected 5 source songs, got {len(source_paths)}")

    prog("load", 2, "Loading target")
    target_y, sr = load_and_normalize(target_path)
    target_duration_s = len(target_y) / sr
    write_wav(out / "target.wav", target_y, sr)

    sources_y: list[tuple[str, np.ndarray]] = []
    names = source_names or [Path(p).name for p in source_paths]
    for i, path in enumerate(source_paths):
        prog("load", 5 + i * 3, f"Loading source {SONG_IDS[i]}")
        y, _ = load_and_normalize(path)
        sources_y.append((SONG_IDS[i], y))

    prog("segment", 22, "Segmenting")
    target_segs = segment_audio(
        target_y, sr, "T", window_s=cfg.window_s, hop_s=cfg.hop_s
    )
    source_segs = []
    for sid, y in sources_y:
        source_segs.extend(
            segment_audio(y, sr, sid, window_s=cfg.window_s, hop_s=cfg.hop_s)
        )

    extractor = HandcraftedExtractor()

    prog("features", 30, "Embedding target")
    target_emb = extractor.embed_segments(
        target_y,
        sr,
        target_segs,
        on_progress=lambda f, m: prog("features", 30 + f * 15, m),
    )
    prog("features", 48, "Embedding sources")
    source_emb_parts: list[np.ndarray] = []
    for i, (sid, y) in enumerate(sources_y):
        segs_i = [s for s in source_segs if s.song_id == sid]
        part = extractor.embed_segments(
            y,
            sr,
            segs_i,
            on_progress=lambda f, m, i=i: prog(
                "features", 48 + (i + f) / 5 * 25, f"source {sid}: {m}"
            ),
        )
        source_emb_parts.append(part)
    source_emb = (
        np.vstack(source_emb_parts) if source_emb_parts else np.zeros((0, 54), np.float32)
    )

    prog("index", 75, "Building FAISS index")
    source_index = build_source_index(source_segs, source_emb)

    prog("match", 80, "Matching sequence")
    target_starts = np.array([s.start_s for s in target_segs], dtype=np.float64)
    match_params = MatchParams(
        top_k=cfg.top_k,
        lambda_switch=cfg.lambda_switch,
        lambda_jump=cfg.lambda_jump,
        jump_norm_s=cfg.jump_norm_s,
        lambda_self=cfg.lambda_self,
        hop_s=cfg.hop_s,
    )
    match = match_sequence(target_emb, target_starts, source_index, match_params)

    prog("reconstruct", 90, "Overlap-add reconstruction")
    recon = reconstruct_ola(
        match.tiles,
        source_index,
        sr=sr,
        window_s=cfg.window_s,
        hop_s=cfg.hop_s,
        target_duration_s=target_duration_s,
    )
    audio_path = out / "reconstructed.wav"
    write_wav(audio_path, recon, sr)

    songs = [
        {"id": SONG_IDS[i], "name": names[i], "color": SONG_COLORS[i]}
        for i in range(5)
    ]
    mosaic = {
        "window_s": cfg.window_s,
        "hop_s": cfg.hop_s,
        "duration_s": target_duration_s,
        "sr": sr,
        "songs": songs,
        "tiles": [
            {
                "i": t.target_idx,
                "target_start_s": t.target_start_s,
                "song_id": t.song_id,
                "source_start_s": t.source_start_s,
                "similarity": round(t.similarity, 4),
            }
            for t in match.tiles
        ],
        "stats": {
            "contribution_pct": _contrib(match.tiles, SONG_IDS),
            "avg_similarity": round(match.avg_similarity, 4),
            "num_transitions": match.transitions_viterbi,
            "transitions_viterbi": match.transitions_viterbi,
            "transitions_greedy": match.transitions_greedy,
            "num_tiles": len(match.tiles),
        },
    }
    mosaic_path = out / "mosaic.json"
    mosaic_path.write_text(json.dumps(mosaic, indent=2))

    elapsed = time.perf_counter() - t0
    prog("done", 100, f"Done in {elapsed:.1f}s")
    return JobResult(
        mosaic=mosaic,
        duration_s=target_duration_s,
        elapsed_s=elapsed,
        paths={
            "audio": str(audio_path),
            "target": str(out / "target.wav"),
            "mosaic": str(mosaic_path),
        },
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="MusicMosaic CLI")
    p.add_argument("target")
    p.add_argument("sources", nargs=5)
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--window", type=float, default=0.5)
    p.add_argument("--hop", type=float, default=0.25)
    args = p.parse_args()

    def on_progress(stage: str, pct: float, msg: str) -> None:
        print(f"[{pct:5.1f}%] {stage}: {msg}", flush=True)

    cfg = JobConfig(window_s=args.window, hop_s=args.hop)
    result = run_job(args.target, args.sources, args.out, config=cfg, on_progress=on_progress)
    print(json.dumps({"elapsed_s": result.elapsed_s, "stats": result.mosaic["stats"]}, indent=2))


if __name__ == "__main__":
    main()
