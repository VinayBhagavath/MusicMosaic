"""Reconstruction: synthesize selected notes toward the target.

After a match is chosen, each note is transformed hard toward the target window:
1. Pitch — per-window F0 match (fallback: chroma key shift) via Rubber Band
2. Rhythm — onset-align attack to the target, then length-fit
3. Dynamics — morph loudness envelope to the target shape
4. Timbre — transfer the target's tonal envelope (cepstral) + transient
5. Stack complementary roles; stitch with guaranteed equal-power crossfades
   over real overlapping audio (no fades into silence, DC-blocked seams)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import local

import numpy as np

from app.pipeline.cohesion import (
    harmonic_reconstruct,
    highpass,
    lowpass,
    match_envelope,
    match_loudness,
    match_spectrum,
    match_transient,
    residual_fill,
    soft_limit,
)
from app.pipeline.index import SourceIndex
from app.pipeline.match import LayerMatch, TileMatch
from app.pipeline.stems import StemBundle
from app.pipeline.transform import (
    align_onset_to_ref,
    estimate_f0_hz,
    estimate_tuning_cents,
    f0_semitone_delta,
    fit_length,
    pitch_shift,
    prepare_clip,
)


def _tile_layers(tile: TileMatch) -> list[LayerMatch]:
    if tile.layers:
        return tile.layers
    return [
        LayerMatch(
            source_id=tile.source_id,
            song_id=tile.song_id,
            source_start_s=tile.source_start_s,
            similarity=tile.similarity,
            weight=1.0,
            key_shift=tile.key_shift,
            role="full",
        )
    ]


def _normalize_shift(shift: float, *, role: str) -> float:
    s = float(shift)
    while s > 6:
        s -= 12
    while s < -6:
        s += 12
    if role == "drums":
        if abs(s) >= 1.5:
            return 0.0
        return float(np.clip(s, -0.35, 0.35))
    return s


def _target_slice(
    target: np.ndarray | None,
    sr: int,
    start_s: float,
    n: int,
) -> np.ndarray | None:
    if target is None or n <= 0:
        return None
    a = max(0, int(round(start_s * sr)))
    b = min(len(target), a + n)
    if b <= a:
        return None
    out = np.zeros(n, dtype=np.float32)
    out[: b - a] = target[a:b]
    return out


def _song_for_layer(
    layer: LayerMatch,
    source: SourceIndex,
    stems: dict[str, StemBundle] | None,
) -> np.ndarray | None:
    if stems and layer.song_id in stems:
        bundle = stems[layer.song_id]
        role = layer.role or "full"
        if role == "drums":
            return bundle.drums
        if role == "bass":
            return bundle.bass
        if role == "harmonic":
            return (bundle.other * 0.65 + bundle.vocals * 0.35).astype(np.float32)
        return (
            bundle.drums + bundle.bass + bundle.other + bundle.vocals
        ).astype(np.float32) * 0.5

    if source.songs and layer.song_id in source.songs:
        return source.songs[layer.song_id]
    if 0 <= layer.source_id < len(source.waveforms):
        return source.waveforms[layer.source_id]
    return None


def _apply_role_filter(
    clip: np.ndarray,
    sr: int,
    role: str,
    *,
    multi_layer: bool,
    has_stems: bool,
) -> np.ndarray:
    if not multi_layer or role in ("", "full"):
        return clip
    if has_stems:
        return clip
    if role == "bass":
        return lowpass(clip, sr, cutoff_hz=220.0)
    if role == "drums":
        try:
            import librosa

            _h, p = librosa.effects.hpss(clip.astype(np.float32))
            return highpass(p.astype(np.float32), sr, cutoff_hz=120.0)
        except Exception:
            return highpass(clip, sr, cutoff_hz=180.0)
    if role == "harmonic":
        return highpass(clip, sr, cutoff_hz=160.0)
    return clip


def _equal_power_envelope(n: int, fade_in: int, fade_out: int) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    fade_in = int(np.clip(fade_in, 0, n))
    fade_out = int(np.clip(fade_out, 0, n))
    if fade_in > 0:
        t = np.linspace(0.0, 0.5 * np.pi, fade_in, endpoint=False, dtype=np.float32)
        env[:fade_in] = np.sin(t)
    if fade_out > 0:
        t = np.linspace(0.0, 0.5 * np.pi, fade_out, endpoint=True, dtype=np.float32)
        env[n - fade_out :] *= np.cos(t)
    return env


def _target_onset_samples(target: np.ndarray | None, sr: int) -> np.ndarray | None:
    """Sample positions of note attacks in the target (for onset-synced seams)."""
    if target is None or len(target) < sr // 4:
        return None
    try:
        import librosa

        env = librosa.onset.onset_strength(y=target.astype(np.float32), sr=sr, hop_length=256)
        onsets = librosa.onset.onset_detect(
            onset_envelope=env,
            sr=sr,
            hop_length=256,
            units="samples",
            backtrack=True,
        )
        arr = np.asarray(onsets, dtype=np.int64).reshape(-1)
        return arr if arr.size else None
    except Exception:
        return None


def _compute_seam_xf(
    spans: list[int],
    positions: list[int],
    *,
    xf_target: int,
    click_guard: int,
    onset_samples: np.ndarray | None,
    onset_tol: int,
) -> list[int]:
    """Per-seam crossfade length.

    Target attacks use a shorter fade to stay punchy, but retain enough overlap
    to avoid the grainy near-hard splices produced by a click-guard-only fade.
    Sustained seams keep the full ~30 ms equal-power crossfade.
    """
    n = len(spans)
    seam_xf = [0] * n
    for i in range(n - 1):
        xf = min(xf_target, spans[i] // 2, spans[i + 1] // 2)
        seam = max(xf, min(click_guard, spans[i], spans[i + 1]))
        if onset_samples is not None and onset_samples.size:
            boundary = positions[i + 1]
            nearest = int(np.min(np.abs(onset_samples - boundary)))
            if nearest <= onset_tol:
                # Keep about one third of the normal overlap at attacks. At
                # 22.05 kHz this is ~10 ms rather than the previous ~4 ms.
                onset_xf = max(click_guard, int(round(xf * 0.35)))
                seam = min(seam, max(1, min(onset_xf, spans[i], spans[i + 1])))
        seam_xf[i] = seam
    return seam_xf


def _resolve_pitch_steps(
    *,
    role: str,
    key_shift: float,
    fine_cents: float,
    src_chunk: np.ndarray | None,
    ref: np.ndarray | None,
    sr: int,
    apply_key_shift: bool,
    source_f0: float | None = None,
    ref_f0: float | None = None,
) -> float:
    """Prefer per-window F0 match; fall back to chroma key_shift + tuning."""
    if not apply_key_shift:
        return 0.0
    if role == "drums":
        return _normalize_shift(key_shift, role=role)

    if source_f0 is not None and ref_f0 is not None and source_f0 > 1e-6:
        delta = 12.0 * np.log2(ref_f0 / source_f0)
        return float(np.clip(delta, -12.0, 12.0))

    if src_chunk is not None and ref is not None and len(src_chunk) > sr // 5:
        f0_delta = f0_semitone_delta(src_chunk, ref, sr)
        if f0_delta is not None:
            return float(np.clip(f0_delta, -12.0, 12.0))

    steps = _normalize_shift(key_shift, role=role)
    if abs(fine_cents) > 0.5:
        steps = steps + fine_cents / 100.0
    return steps


def _layer_audio(
    layer: LayerMatch,
    source: SourceIndex,
    *,
    sr: int,
    win: int,
    apply_key_shift: bool,
    fine_cents: float,
    shift_cache: dict,
    stems: dict[str, StemBundle] | None,
    multi_layer: bool,
    ref: np.ndarray | None,
    f0_cache: dict,
    ref_f0: float | None,
) -> np.ndarray:
    song = _song_for_layer(layer, source, stems)
    role = layer.role or "full"
    if song is None:
        return np.zeros(win, dtype=np.float32)

    # Use the detected source event duration instead of blindly taking a
    # target-sized slice. This lets Rubber Band map the source note's attack and
    # release onto the target note length.
    source_n = win
    meta = source.meta[layer.source_id] if 0 <= layer.source_id < len(source.meta) else None
    if meta is not None:
        source_n = max(64, int(round((meta.end_s - meta.start_s) * sr)))

    # Raw source event for F0 estimate (before pitch)
    a = int(round(layer.source_start_s * sr))
    src_chunk = song[a : a + source_n]
    if len(src_chunk) < source_n:
        src_chunk = fit_length(src_chunk.astype(np.float32), source_n)
    f0_key = (layer.source_id, source_n, role, bool(stems))
    if role != "drums" and f0_key not in f0_cache:
        f0_cache[f0_key] = estimate_f0_hz(src_chunk, sr)

    steps = _resolve_pitch_steps(
        role=role,
        key_shift=layer.key_shift,
        fine_cents=fine_cents,
        src_chunk=src_chunk,
        ref=ref,
        sr=sr,
        apply_key_shift=apply_key_shift,
        source_f0=f0_cache.get(f0_key),
        ref_f0=ref_f0,
    )

    if source.songs and layer.song_id in source.songs and len(song) > win:
        clip = prepare_clip(
            song,
            sr,
            layer.source_start_s,
            target_n=win,
            source_n=source_n,
            n_steps=steps,
            cache=shift_cache,
            cache_key=f"{layer.song_id}:{layer.role}:{round(steps, 1)}",
            formant_preserve=role not in ("drums", "bass"),
        )
    else:
        from app.pipeline.transform import time_stretch

        raw = fit_length(song.astype(np.float32), max(len(song), 64))
        if abs(steps) >= 1e-4:
            raw = pitch_shift(raw, sr, steps)
        if len(raw) != win and len(raw) > 32:
            raw = time_stretch(raw, sr, len(raw) / float(win))
        clip = fit_length(raw, win)

    # Rhythm: lock attack to target onset inside the window
    if ref is not None and role != "drums":
        clip = align_onset_to_ref(clip, ref, sr)
    elif ref is not None and role == "drums":
        # Drums: still align hits — critical for groove
        clip = align_onset_to_ref(clip, ref, sr)

    clip = _apply_role_filter(
        clip,
        sr,
        role,
        multi_layer=multi_layer,
        has_stems=bool(stems),
    )
    return clip.astype(np.float32)


def reconstruct_ola(
    tiles: list[TileMatch],
    source: SourceIndex,
    *,
    sr: int,
    window_s: float = 0.45,
    hop_s: float = 0.22,
    target_duration_s: float | None = None,
    apply_key_shift: bool = True,
    target_audio: np.ndarray | None = None,
    stems: dict[str, StemBundle] | None = None,
    spectral_match: bool = True,
    loudness_match: bool = True,
    transient_match: bool = True,
    spectral_strength: float = 0.6,
    harmonic_match: bool = True,
    harmonic_strength: float = 0.35,
    onset_sync_xf: bool = True,
) -> np.ndarray:
    """Synthesize each matched note toward the target, then OLA."""
    if not tiles:
        return np.zeros(0, dtype=np.float32)

    if target_duration_s is None:
        target_duration_s = tiles[-1].target_start_s + window_s

    base_win = int(round(window_s * sr))
    tile_wins = [
        int(
            round(
                np.clip(
                    t.target_duration_s if t.target_duration_s is not None else window_s,
                    0.12,
                    window_s * 2.0,
                )
                * sr
            )
        )
        for t in tiles
    ]
    positions = [int(round(t.target_start_s * sr)) for t in tiles]

    # Advance from each tile to the next (samples). The last tile keeps its own
    # duration. Spans drive both coverage and how long each crossfade can be.
    spans = [
        max(1, positions[i + 1] - positions[i]) if i + 1 < len(positions) else tile_wins[i]
        for i in range(len(tiles))
    ]

    # Target crossfade ~30 ms; a hard click-guard floor keeps every seam smooth
    # even when onsets are dense. Each seam uses one length on both sides so the
    # equal-power sin/cos pair sums to constant power (no dips, no bumps).
    xf_target = int(round(0.030 * sr))
    click_guard = max(16, int(round(0.004 * sr)))
    onset_samples = (
        _target_onset_samples(target_audio, sr) if onset_sync_xf else None
    )
    onset_tol = int(round(0.020 * sr))
    seam_xf = _compute_seam_xf(
        spans,
        positions,
        xf_target=xf_target,
        click_guard=click_guard,
        onset_samples=onset_samples,
        onset_tol=onset_tol,
    )

    # Preserve the detected note/release duration. Previously interior tiles
    # always stopped just after the next onset, even when segmentation had
    # measured a longer target event, which chopped instrumental sustain tails.
    # Keep at least enough audio for the paired seam crossfade.
    render_lens = [
        max(
            tile_wins[i],
            spans[i] + (seam_xf[i] if i + 1 < len(tiles) else 0),
        )
        for i in range(len(tiles))
    ]
    max_win = max([base_win, *render_lens])

    out_len = int(round(target_duration_s * sr))
    acc = np.zeros(out_len + max_win + 8, dtype=np.float32)
    wsum = np.zeros_like(acc)
    power_sum = np.zeros_like(acc)
    target_cents = 0.0
    song_cents: dict[str, float] = {}
    if target_audio is not None and apply_key_shift:
        target_cents = estimate_tuning_cents(
            target_audio[: min(len(target_audio), sr * 20)], sr
        )
        if source.songs:
            for sid, song in source.songs.items():
                song_cents[sid] = estimate_tuning_cents(
                    song[: min(len(song), sr * 20)], sr
                )

    refs = [
        _target_slice(target_audio, sr, tile.target_start_s, render_lens[i])
        for i, tile in enumerate(tiles)
    ]
    def _ref_f0(ref: np.ndarray | None) -> float | None:
        return estimate_f0_hz(ref, sr) if ref is not None and apply_key_shift else None

    if apply_key_shift and len(refs) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(refs))) as pool:
            ref_f0s = list(pool.map(_ref_f0, refs))
    else:
        ref_f0s = [_ref_f0(ref) for ref in refs]

    thread_state = local()

    def _render_tile(item: tuple[int, TileMatch]) -> np.ndarray:
        ti, tile = item
        if not hasattr(thread_state, "shift_cache"):
            thread_state.shift_cache = {}
            thread_state.f0_cache = {}
        win = render_lens[ti]
        layers = _tile_layers(tile)
        multi_layer = len(layers) > 1
        ref = refs[ti]

        mix = np.zeros(win, dtype=np.float32)
        for li, layer in enumerate(layers):
            src_cents = song_cents.get(layer.song_id, 0.0) if apply_key_shift else 0.0
            fine = target_cents - src_cents
            clip = _layer_audio(
                layer,
                source,
                sr=sr,
                win=win,
                apply_key_shift=apply_key_shift,
                fine_cents=fine,
                shift_cache=thread_state.shift_cache,
                stems=stems,
                multi_layer=multi_layer,
                ref=ref,
                f0_cache=thread_state.f0_cache,
                ref_f0=ref_f0s[ti],
            )
            if li == 0:
                mix = clip.astype(np.float32, copy=True)
            else:
                # Residual fill: secondary only supplies missing target energy.
                mix = residual_fill(
                    mix,
                    clip,
                    ref,
                    sr,
                    amount=float(np.clip(layer.weight * 1.35, 0.12, 0.55)),
                )

        # Always synthesize toward target when we have a reference window
        if ref is not None:
            # Envelope first (rhythm/dynamics), then spectrum, transient, loudness
            mix = match_envelope(mix, ref, sr, blend=0.74)
            if spectral_match:
                # Push tone harder when the match is weaker; the envelope
                # transfer is gentle so this stays natural.
                adaptive_strength = spectral_strength * float(
                    np.clip(1.35 - tile.similarity, 0.6, 1.0)
                )
                mix = match_spectrum(mix, ref, sr, strength=adaptive_strength)
            if harmonic_match:
                # Legibility: push the target's harmonic detail (which notes
                # sound) harder when the acoustic match is weaker, so the tune
                # stays followable even when no source clip fits well.
                adaptive_harm = harmonic_strength * float(
                    np.clip(1.5 - tile.similarity, 0.7, 1.15)
                )
                # Extra minutia: if the mix still misses the target's pitch
                # classes, lean harder into harmonic reconstruction.
                try:
                    import librosa

                    cm = librosa.feature.chroma_stft(
                        y=mix, sr=sr, n_fft=1024, hop_length=256
                    ).mean(axis=1)
                    cr = librosa.feature.chroma_stft(
                        y=ref, sr=sr, n_fft=1024, hop_length=256
                    ).mean(axis=1)
                    denom = (np.linalg.norm(cm) * np.linalg.norm(cr)) + 1e-8
                    gap = 1.0 - float(np.dot(cm, cr) / denom)
                    adaptive_harm *= float(np.clip(1.0 + 0.65 * gap, 1.0, 1.55))
                except Exception:
                    pass
                mix = harmonic_reconstruct(mix, ref, sr, strength=min(0.95, adaptive_harm))
            if transient_match:
                mix = match_transient(mix, ref, sr, amount=0.32)
            if loudness_match:
                mix = match_loudness(mix, ref, blend=0.82)

        # Remove a genuine DC offset (a classic seam-click source) without
        # gutting legitimately low-frequency / bass-heavy clips.
        m = float(np.mean(mix))
        peak = float(np.max(np.abs(mix))) + 1e-9
        if abs(m) < 0.2 * peak:
            mix = mix - m
        return mix.astype(np.float32, copy=False)

    items = list(enumerate(tiles))
    # Rubber Band runs out-of-process and the spectral transforms release the
    # GIL, so tile rendering scales well across a small worker pool. Keeping
    # caches thread-local avoids races while still reusing work on each worker.
    workers = min(4, len(items))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            mixes = list(pool.map(_render_tile, items))
    else:
        mixes = [_render_tile(item) for item in items]

    for ti, mix in enumerate(mixes):
        win = render_lens[ti]
        pos = positions[ti]
        if pos >= out_len:
            continue

        # Fade lengths come from the shared seam crossfades so neighboring tiles
        # ramp against each other with matched equal-power curves. Absolute
        # start/end fades are applied globally after normalization (a one-sided
        # fade here would be divided back out by the weight sum).
        fade_in = seam_xf[ti - 1] if ti > 0 else 0
        fade_out = seam_xf[ti] if ti + 1 < len(tiles) else 0

        env = _equal_power_envelope(win, fade_in, fade_out)
        end = min(pos + win, len(acc))
        span = end - pos
        acc[pos:end] += mix[:span] * env[:span]
        wsum[pos:end] += env[:span]
        power_sum[pos:end] += env[:span] * env[:span]

    # Preserve linear/correlation-safe fades for natural continuations, but use
    # true constant-power normalization when unrelated source clips meet. The
    # previous linear normalization attenuated independent signals around every
    # switch, which was audible as repeated muffled level dips.
    denom = wsum.copy()
    for i in range(len(tiles) - 1):
        xf = seam_xf[i]
        if xf <= 0:
            continue
        step_s = max(1, positions[i + 1] - positions[i]) / sr
        expected = tiles[i].source_start_s + step_s
        natural = (
            tiles[i].song_id == tiles[i + 1].song_id
            and abs(expected - tiles[i + 1].source_start_s)
            <= max(0.06, step_s * 0.55)
        )
        if natural:
            continue
        a = max(0, positions[i + 1])
        b = min(len(denom), a + xf)
        # Ease into/out of constant-power normalization. With release tails,
        # more than two tiles may overlap; switching denominator formulas
        # abruptly at the seam edge creates a zipper click.
        n = b - a
        if n > 0:
            t = np.linspace(0.0, np.pi, n, endpoint=True, dtype=np.float32)
            blend = np.sin(t) ** 2
            power_denom = np.sqrt(np.maximum(power_sum[a:b], 1e-12))
            denom[a:b] = (1.0 - blend) * denom[a:b] + blend * power_denom
    nz = denom > 1e-6
    acc[nz] /= denom[nz]
    out = acc[:out_len]

    # Global click-guard so the very first/last samples ramp from/to zero.
    guard = min(click_guard, len(out) // 2)
    if guard > 1:
        ramp = np.sin(np.linspace(0.0, 0.5 * np.pi, guard, dtype=np.float32))
        out[:guard] *= ramp
        out[-guard:] *= ramp[::-1]
    return soft_limit(out)
