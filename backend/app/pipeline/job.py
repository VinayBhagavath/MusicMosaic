"""End-to-end mosaic job orchestration."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from app.pipeline.audio_io import load_and_normalize, write_wav
from app.pipeline.features import EmbPack, get_extractor
from app.pipeline.index import build_source_index
from app.pipeline.match import MatchParams, match_sequence
from app.pipeline.metrics import (
    boundary_discontinuity,
    candidate_improves,
    quality_metrics,
)
from app.pipeline.palette import SONG_COLORS, SONG_IDS
from app.pipeline.reconstruct import reconstruct_ola
from app.pipeline.segment import segment_audio

ProgressCb = Callable[[str, float, str], None]


@dataclass
class JobConfig:
    # MIDI-note mosaicing: short onset units, free sample switching, pitch-to-target
    window_s: float = 0.45
    hop_s: float = 0.22
    top_k: int = 20
    lambda_switch: float = 0.08
    lambda_jump: float = 0.15
    jump_norm_s: float = 2.0
    lambda_self: float = 0.03
    lambda_concat: float = 0.55
    lambda_join: float = 0.70
    min_run_tiles: int = 1
    per_song_k: int = 4
    lambda_balance: float = 0.0
    max_share: float = 1.0
    balance_iters: int = 1
    n_layers: int = 1
    layer_primary_weight: float = 0.62
    fidelity_first: bool = True
    beat_sync: bool = True
    beat_sync_sources: bool = True
    phrase_sync: bool = False
    onset_sync: bool = True
    apply_key_shift: bool = True
    spectral_match: bool = True
    loudness_match: bool = True
    transient_match: bool = True
    spectral_strength: float = 0.7
    harmonic_match: bool = True
    harmonic_strength: float = 0.42
    onset_sync_xf: bool = True
    rerank_spectral: bool = True
    rerank_top_m: int = 3
    # unit: existing concatenative renderer; nmf: force sparse diagonal NMF;
    # auto: render both and accept NMF only through the objective quality gate.
    reconstruction_backend: str = "auto"
    nmf_iterations: int = 8
    nmf_polyphony: int = 3
    use_stems: bool = False
    prefer_clap: bool = False


@dataclass
class JobResult:
    mosaic: dict
    duration_s: float
    elapsed_s: float
    paths: dict[str, str] = field(default_factory=dict)


def _contrib(tiles: list, song_ids: list[str]) -> dict[str, float]:
    """Weighted contribution across primary + secondary layers."""
    weights = {s: 0.0 for s in song_ids}
    total = 0.0
    for t in tiles:
        layers = getattr(t, "layers", None) or []
        if layers:
            for layer in layers:
                sid = layer.song_id
                w = float(layer.weight)
                weights[sid] = weights.get(sid, 0.0) + w
                total += w
        else:
            weights[t.song_id] = weights.get(t.song_id, 0.0) + 1.0
            total += 1.0
    if total <= 0:
        return {s: 0.0 for s in song_ids}
    return {s: round(100.0 * weights.get(s, 0.0) / total, 1) for s in song_ids}


def _stack_packs(parts: list[EmbPack]) -> EmbPack:
    def optional_stack(name: str) -> np.ndarray | None:
        values = [getattr(p, name) for p in parts]
        return np.vstack(values) if values and all(v is not None for v in values) else None

    return EmbPack(
        chroma=np.vstack([p.chroma for p in parts]),
        timbre=np.vstack([p.timbre for p in parts]),
        energy=np.vstack([p.energy for p in parts]),
        backend=parts[0].backend if parts else "handcrafted",
        temporal=optional_stack("temporal"),
        register=optional_stack("register"),
        level=optional_stack("level"),
    )


def run_job(
    target_path: str | Path,
    source_paths: list[str | Path],
    out_dir: str | Path,
    *,
    config: JobConfig | None = None,
    source_names: list[str] | None = None,
    on_progress: ProgressCb | None = None,
) -> JobResult:
    cfg = config or JobConfig()
    if cfg.apply_key_shift:
        from app.pipeline.transform import require_rubberband

        require_rubberband()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stage_t0 = t0
    stage_timings: dict[str, float] = {}

    def prog(stage: str, pct: float, msg: str) -> None:
        if on_progress:
            on_progress(stage, pct, msg)

    def _mark(stage: str) -> None:
        nonlocal stage_t0
        now = time.perf_counter()
        stage_timings[stage] = round(now - stage_t0, 2)
        stage_t0 = now

    if len(source_paths) != 5:
        raise ValueError(f"Expected 5 source songs, got {len(source_paths)}")

    prog("load", 2, "Loading target")
    target_y, sr = load_and_normalize(target_path)
    target_duration_s = len(target_y) / sr
    write_wav(out / "target.wav", target_y, sr)

    names = source_names or [Path(p).name for p in source_paths]

    def _load_one(i_path: tuple[int, str | Path]) -> tuple[int, np.ndarray]:
        i, path = i_path
        y, _ = load_and_normalize(path)
        return i, y

    prog("load", 6, "Loading sources")
    sources_y: list[np.ndarray | None] = [None] * 5
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, y in pool.map(_load_one, list(enumerate(source_paths))):
            sources_y[i] = y
            prog("load", 8 + i * 2, f"Loaded source {SONG_IDS[i]}")
    sources_typed = [(SONG_IDS[i], sources_y[i]) for i in range(5)]
    assert all(y is not None for _, y in sources_typed)
    _mark("load")

    prog("segment", 22, "Segmenting notes")
    target_segs = segment_audio(
        target_y,
        sr,
        "T",
        window_s=cfg.window_s,
        hop_s=cfg.hop_s,
        beat_sync=cfg.beat_sync,
        phrase_sync=cfg.phrase_sync,
        onset_sync=cfg.onset_sync,
        variable_length=cfg.fidelity_first,
    )
    def _segment_source(item: tuple[str, np.ndarray]) -> list:
        sid, y = item
        return segment_audio(
            y,
            sr,
            sid,
            window_s=cfg.window_s,
            hop_s=cfg.hop_s,
            beat_sync=cfg.beat_sync and cfg.beat_sync_sources,
            phrase_sync=cfg.phrase_sync and cfg.beat_sync_sources,
            onset_sync=cfg.onset_sync,
            variable_length=cfg.fidelity_first,
        )

    source_segs = []
    with ThreadPoolExecutor(max_workers=min(5, len(sources_typed))) as pool:
        for segs in pool.map(_segment_source, sources_typed):  # type: ignore[arg-type]
            source_segs.extend(segs)
    _mark("segment")

    extractor = get_extractor(prefer_clap=cfg.prefer_clap)
    prog("features", 26, f"Embedding target ({extractor.name})")
    target_pack = extractor.embed_segments(
        target_y,
        sr,
        target_segs,
        on_progress=lambda f, m: prog("features", 28 + f * 12, m),
    )

    prog("features", 42, "Embedding sources")

    def _embed_source(item: tuple[int, str, np.ndarray]) -> tuple[int, EmbPack]:
        i, sid, y = item
        segs_i = [s for s in source_segs if s.song_id == sid]
        pack = extractor.embed_segments(y, sr, segs_i)
        return i, pack

    packs: list[EmbPack | None] = [None] * 5
    items = [(i, sid, y) for i, (sid, y) in enumerate(sources_typed)]  # type: ignore[misc]
    # Parallelize mosaic descriptors across sources (CPU-bound, no GPU contention)
    if extractor.name != "clap-hybrid":
        with ThreadPoolExecutor(max_workers=min(5, len(items))) as pool:
            for i, pack in pool.map(_embed_source, items):
                packs[i] = pack
                prog("features", 45 + i * 5, f"Embedded source {SONG_IDS[i]}")
    else:
        for item in items:
            i, pack = _embed_source(item)
            packs[i] = pack
            prog("features", 45 + i * 5, f"CLAP source {SONG_IDS[i]}")
    source_pack = _stack_packs([p for p in packs if p is not None])
    _mark("features")

    prog("index", 72, "Indexing source clips")
    songs_map = {sid: y for sid, y in sources_typed}  # type: ignore[misc]
    source_index = build_source_index(
        source_segs,
        source_pack,
        songs=songs_map,
        sr=sr,
        compute_edges=True,
    )
    _mark("index")

    prog("match", 78, "Matching sequence")
    target_starts = np.array([s.start_s for s in target_segs], dtype=np.float64)
    target_durations = np.array(
        [max(0.12, s.end_s - s.start_s) for s in target_segs], dtype=np.float64
    )
    match_params = MatchParams(
        top_k=cfg.top_k,
        lambda_switch=cfg.lambda_switch,
        lambda_jump=cfg.lambda_jump,
        jump_norm_s=cfg.jump_norm_s,
        lambda_self=cfg.lambda_self,
        lambda_concat=cfg.lambda_concat,
        lambda_join=cfg.lambda_join,
        hop_s=cfg.hop_s,
        min_run_tiles=cfg.min_run_tiles,
        per_song_k=cfg.per_song_k,
        lambda_balance=cfg.lambda_balance,
        max_share=cfg.max_share,
        balance_iters=cfg.balance_iters,
        n_layers=cfg.n_layers,
        layer_primary_weight=cfg.layer_primary_weight,
        fidelity_first=cfg.fidelity_first,
    )
    match = match_sequence(
        target_pack,
        target_starts,
        source_index,
        match_params,
        target_durations=target_durations,
    )
    _mark("match")

    rerank_swaps = 0
    if cfg.rerank_spectral and cfg.apply_key_shift:
        from app.pipeline.rerank import rerank_tiles_spectral

        prog("match", 84, "Re-ranking notes by post-transform spectral fit")
        rerank_swaps = rerank_tiles_spectral(
            match.tiles,
            source_index,
            target_y,
            sr,
            window_s=cfg.window_s,
            top_m=cfg.rerank_top_m,
        )
        _mark("rerank")

    stems_map = None
    if cfg.use_stems:
        from app.pipeline.stems import demucs_available, demucs_mlx_available, separate_many

        if demucs_available():
            backend = "MLX" if demucs_mlx_available() else "CPU"
            prog("stems", 86, f"Separating stems ({backend})")
            songs_only = {sid: y for sid, y in sources_typed}  # type: ignore[misc]
            stems_map = separate_many(
                songs_only,
                sr,
                on_progress=lambda f, m: prog("stems", 86 + f * 3, m),
            )
            if not stems_map:
                stems_map = None
                prog("stems", 89, "Stem separation skipped")
        else:
            prog("reconstruct", 88, "Demucs not installed — full-mix layers")
        _mark("stems")

    prog("reconstruct", 90, "Cohesive unit reconstruction")
    unit_recon = reconstruct_ola(
        match.tiles,
        source_index,
        sr=sr,
        window_s=cfg.window_s,
        hop_s=cfg.hop_s,
        target_duration_s=target_duration_s,
        apply_key_shift=cfg.apply_key_shift,
        target_audio=target_y,
        stems=stems_map,
        spectral_match=cfg.spectral_match,
        loudness_match=cfg.loudness_match,
        transient_match=cfg.transient_match,
        spectral_strength=cfg.spectral_strength,
        harmonic_match=cfg.harmonic_match,
        harmonic_strength=cfg.harmonic_strength,
        onset_sync_xf=cfg.onset_sync_xf,
    )
    _mark("reconstruct_unit")

    requested_backend = cfg.reconstruction_backend.lower().strip()
    if requested_backend not in {"auto", "unit", "nmf"}:
        raise ValueError(
            "reconstruction_backend must be one of: auto, unit, nmf"
        )

    recon = unit_recon
    selected_backend = "unit"
    unit_quality: dict[str, float] | None = None
    nmf_quality: dict[str, float] | None = None
    nmf_result = None
    nmf_accepted = False
    selected_contribution = _contrib(match.tiles, SONG_IDS)

    if requested_backend != "unit":
        from app.pipeline.nmf import NMFParams, reconstruct_nmf

        prog("reconstruct", 92, "Sparse diagonal NMF spectral mosaic")
        nmf_result = reconstruct_nmf(
            target_y,
            songs_map,
            sr,
            params=NMFParams(
                iterations=cfg.nmf_iterations,
                polyphony=cfg.nmf_polyphony,
            ),
            # The unit renderer is source-only and provides temporally coherent
            # phase; NMF supplies the more target-legible source-basis magnitude.
            phase_reference=unit_recon,
            on_progress=lambda f, m: prog("reconstruct", 92 + 5 * f, m),
        )
        _mark("reconstruct_nmf")
        nmf_quality = quality_metrics(
            target_y,
            nmf_result.audio,
            sr,
            boundaries_s=target_starts[1:],
        )
        if requested_backend == "nmf":
            nmf_accepted = True
        else:
            unit_quality = quality_metrics(
                target_y,
                unit_recon,
                sr,
                boundaries_s=target_starts[1:],
            )
            nmf_accepted = candidate_improves(unit_quality, nmf_quality)
        if nmf_accepted:
            recon = nmf_result.audio
            selected_backend = "nmf"
            selected_contribution = nmf_result.contribution_pct
            # Keep the baseline for direct local A/B and debugging.
            write_wav(out / "reconstructed_unit.wav", unit_recon, sr)

    audio_path = out / "reconstructed.wav"
    write_wav(audio_path, recon, sr)
    prog("reconstruct", 98, "Measuring reconstruction quality")
    if selected_backend == "nmf" and nmf_quality is not None:
        audio_quality = {
            **nmf_quality,
            "boundary_discontinuity": round(
                boundary_discontinuity(recon, sr, target_starts[1:]), 4
            ),
        }
    elif unit_quality is not None:
        audio_quality = {
            **unit_quality,
            "boundary_discontinuity": round(
                boundary_discontinuity(recon, sr, target_starts[1:]), 4
            ),
        }
    else:
        audio_quality = quality_metrics(
            target_y,
            recon,
            sr,
            boundaries_s=target_starts[1:],
        )
    _mark("quality")

    songs = [
        {"id": SONG_IDS[i], "name": names[i], "color": SONG_COLORS[i]}
        for i in range(5)
    ]

    def _tile_payload(t) -> dict:
        if (
            selected_backend == "nmf"
            and nmf_result is not None
            and nmf_result.frame_song_ids
        ):
            frame = int(
                np.clip(
                    round(t.target_start_s * sr / nmf_result.hop_length),
                    0,
                    len(nmf_result.frame_song_ids) - 1,
                )
            )
            song_id = nmf_result.frame_song_ids[frame]
            source_start_s = float(nmf_result.frame_source_times_s[frame])
            similarity = float(nmf_result.frame_weights[frame])
            layers = [
                {
                    "song_id": song_id,
                    "source_start_s": source_start_s,
                    "similarity": round(similarity, 4),
                    "weight": 1.0,
                    "key_shift": 0.0,
                    "role": "spectral",
                }
            ]
            key_shift = 0.0
        else:
            song_id = t.song_id
            source_start_s = t.source_start_s
            similarity = t.similarity
            key_shift = float(t.key_shift)
            layers = [
                {
                    "song_id": ly.song_id,
                    "source_start_s": ly.source_start_s,
                    "similarity": round(ly.similarity, 4),
                    "weight": round(ly.weight, 4),
                    "key_shift": round(float(ly.key_shift), 3),
                    "role": ly.role,
                }
                for ly in t.layers
            ]
        return {
            "i": t.target_idx,
            "target_start_s": t.target_start_s,
            "target_duration_s": t.target_duration_s,
            "song_id": song_id,
            "source_start_s": source_start_s,
            "similarity": round(similarity, 4),
            "key_shift": round(key_shift, 3),
            "layers": layers,
        }

    mosaic = {
        "window_s": cfg.window_s,
        "hop_s": cfg.hop_s,
        "duration_s": target_duration_s,
        "sr": sr,
        "songs": songs,
        "tiles": [_tile_payload(t) for t in match.tiles],
        "stats": {
            "contribution_pct": selected_contribution,
            "avg_similarity": round(match.avg_similarity, 4),
            "num_transitions": match.transitions_viterbi,
            "transitions_viterbi": match.transitions_viterbi,
            "transitions_greedy": match.transitions_greedy,
            "num_tiles": len(match.tiles),
            "embedding_backend": target_pack.backend,
            "quality": audio_quality,
            "fidelity_first": cfg.fidelity_first,
            "beat_sync": cfg.beat_sync,
            "phrase_sync": cfg.phrase_sync,
            "onset_sync": cfg.onset_sync,
            "min_run_tiles": cfg.min_run_tiles,
            "apply_key_shift": cfg.apply_key_shift,
            "max_share": cfg.max_share,
            "n_layers": cfg.n_layers,
            "spectral_match": cfg.spectral_match,
            "loudness_match": cfg.loudness_match,
            "transient_match": cfg.transient_match,
            "harmonic_match": cfg.harmonic_match,
            "harmonic_strength": cfg.harmonic_strength,
            "onset_sync_xf": cfg.onset_sync_xf,
            "rerank_spectral": cfg.rerank_spectral,
            "rerank_swaps": rerank_swaps,
            "reconstruction_backend_requested": requested_backend,
            "reconstruction_backend": selected_backend,
            "nmf_accepted": nmf_accepted,
            "unit_quality": unit_quality,
            "nmf_quality": nmf_quality,
            "nmf": (
                {
                    "spectral_error": round(nmf_result.spectral_error, 6),
                    "active_polyphony": round(nmf_result.active_polyphony, 3),
                    "source_frames": nmf_result.n_source_frames,
                    "target_frames": nmf_result.n_target_frames,
                }
                if nmf_result is not None
                else None
            ),
            "use_stems": bool(stems_map),
            "stage_timings_s": stage_timings,
        },
    }
    mosaic_path = out / "mosaic.json"
    mosaic_path.write_text(json.dumps(mosaic, indent=2))

    elapsed = time.perf_counter() - t0
    timing_msg = " ".join(f"{k}={v:.1f}s" for k, v in stage_timings.items())
    prog("done", 100, f"Done in {elapsed:.1f}s ({timing_msg})")
    print(f"[musicmosaic] job timings: total={elapsed:.1f}s {timing_msg}", flush=True)
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
    p.add_argument("--window", type=float, default=0.45)
    p.add_argument("--hop", type=float, default=0.22)
    args = p.parse_args()

    def on_progress(stage: str, pct: float, msg: str) -> None:
        print(f"[{pct:5.1f}%] {stage}: {msg}", flush=True)

    cfg = JobConfig(window_s=args.window, hop_s=args.hop)
    result = run_job(args.target, args.sources, args.out, config=cfg, on_progress=on_progress)
    print(json.dumps({"elapsed_s": result.elapsed_s, "stats": result.mosaic["stats"]}, indent=2))


if __name__ == "__main__":
    main()
