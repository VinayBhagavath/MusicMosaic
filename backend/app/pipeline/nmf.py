"""Sparse diagonal NMF audio mosaicing.

This is a bounded implementation of the core idea from Driedger, Prätzlich,
and Müller, "Let It Bee" (ISMIR 2015):

    target magnitude V ~= fixed source-frame dictionary W @ activations H

A literal full H for five multi-minute songs is too large for an interactive
app.  We retain the full-song source dictionary but keep only a small nearest
source-frame beam per target frame.  Multiplicative KL updates are followed by
the paper's important structural constraints:

* repetition suppression: do not stutter the same source frame;
* polyphony limiting: retain only a few simultaneous source activations;
* diagonal continuity: reward advancing through adjacent source frames.

The magnitude is therefore assembled only from source spectra.  Phase can come
from the weighted source atoms (pure Driedger-style output) or from the existing
unit renderer, which is also source-only and gives substantially more coherent
overlap-add phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np
from scipy.spatial import cKDTree

ProgressCb = Callable[[float, str], None]


@dataclass(slots=True)
class NMFParams:
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 48
    candidate_k: int = 16
    nearest_k: int = 8
    # Maximum simultaneous source atoms. The effective value is estimated from
    # the target chroma, so monophonic vocals stay sparse while chords may use
    # more atoms.
    polyphony: int = 5
    iterations: int = 8
    continuity: float = 0.42
    repetition: int = 3
    repetition_suppression: float = 0.18
    temporal_smoothing: float = 0.28
    phase_coherence_blend: float = 0.16


@dataclass(slots=True)
class NMFResult:
    audio: np.ndarray
    contribution_pct: dict[str, float]
    spectral_error: float
    active_polyphony: float
    n_source_frames: int
    n_target_frames: int
    hop_length: int
    frame_song_ids: list[str]
    frame_source_times_s: np.ndarray
    frame_weights: np.ndarray


def _safe_unit_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def _safe_l1_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sums = np.sum(x, axis=1, keepdims=True)
    return (x / np.maximum(sums, eps)).astype(np.float32)


def _frame_features(
    y: np.ndarray,
    sr: int,
    *,
    params: NMFParams,
    mel_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex STFT, L1-normalized linear mel, and KD-tree features."""
    X = librosa.stft(
        y.astype(np.float32),
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        window="hann",
    ).astype(np.complex64)
    mel = (mel_basis @ np.abs(X)).T.astype(np.float32)
    mel_shape = _safe_l1_rows(mel)
    # Mel shape finds similar timbre, while chroma makes the retrieval robust
    # across instrumentation and preserves the actual notes/chords. This is
    # especially important for instrumentals: vocal frames already have a
    # dominant harmonic track, but a chord can otherwise be matched mostly by
    # broad spectral color.
    mel_search = _safe_unit_rows(np.log1p(80.0 * mel_shape))
    chroma = librosa.feature.chroma_stft(
        S=np.abs(X) ** 2,
        sr=sr,
        n_fft=params.n_fft,
        hop_length=params.hop_length,
    ).T.astype(np.float32)
    chroma_shape = _safe_l1_rows(chroma)
    chroma = _safe_unit_rows(chroma)
    n = min(len(mel_search), len(chroma))
    search = _safe_unit_rows(
        np.concatenate(
            [
                np.sqrt(0.65) * mel_search[:n],
                np.sqrt(0.35) * chroma[:n],
            ],
            axis=1,
        )
    )
    # KL-NMF must optimize the notes as well as broad color. A mel-only
    # objective can approximate a whole chord with one middle-frequency atom,
    # which is why vocal melodies worked while instrumental harmony vanished.
    objective_shape = np.concatenate(
        [
            0.42 * mel_shape[:n],
            0.58 * chroma_shape[:n],
        ],
        axis=1,
    ).astype(np.float32)
    return X[:, :n], objective_shape, search


def _source_dictionary(
    songs: dict[str, np.ndarray],
    sr: int,
    *,
    params: NMFParams,
    mel_basis: np.ndarray,
    on_progress: ProgressCb | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the searchable source dictionary without retaining all STFTs."""
    shapes: list[np.ndarray] = []
    searches: list[np.ndarray] = []
    song_indices: list[np.ndarray] = []
    local_frames: list[np.ndarray] = []
    song_ids = list(songs)
    for si, song_id in enumerate(song_ids):
        _X, shape, search = _frame_features(
            songs[song_id], sr, params=params, mel_basis=mel_basis
        )
        n = shape.shape[0]
        shapes.append(shape)
        searches.append(search)
        song_indices.append(np.full(n, si, dtype=np.int16))
        local_frames.append(np.arange(n, dtype=np.int32))
        if on_progress:
            on_progress(0.05 + 0.18 * (si + 1) / max(1, len(song_ids)), f"NMF source {song_id}")
    return (
        np.vstack(shapes).astype(np.float32),
        np.vstack(searches).astype(np.float32),
        np.concatenate(song_indices),
        np.concatenate(local_frames),
        song_ids,
    )


def _candidate_beam(
    source_search: np.ndarray,
    target_search: np.ndarray,
    source_song: np.ndarray,
    source_frame: np.ndarray,
    *,
    params: NMFParams,
) -> np.ndarray:
    """Nearest source spectra plus local neighbors for diagonal paths."""
    tree = cKDTree(source_search)
    nearest_k = min(params.nearest_k, len(source_search))
    _dist, nearest = tree.query(target_search, k=nearest_k, workers=-1)
    nearest = np.asarray(nearest, dtype=np.int64)
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    # A global nearest-neighbor list can be monopolized by many adjacent frames
    # from one song. Reserve candidates from every source so polyphonic targets
    # can combine complementary notes/timbres across songs.
    per_song_nearest: list[np.ndarray] = []
    for song in np.unique(source_song):
        song_ids = np.flatnonzero(source_song == song)
        if not len(song_ids):
            continue
        take = min(2, len(song_ids))
        song_tree = cKDTree(source_search[song_ids])
        _song_dist, song_local = song_tree.query(
            target_search,
            k=take,
            workers=-1,
        )
        song_local = np.asarray(song_local, dtype=np.int64)
        if song_local.ndim == 1:
            song_local = song_local[:, None]
        per_song_nearest.append(song_ids[song_local])

    # Global-id lookup for source-frame neighbors. Each song's frames are
    # contiguous in the dictionary, but use an explicit map to guard boundaries.
    lookup = {
        (int(song), int(frame)): i
        for i, (song, frame) in enumerate(zip(source_song, source_frame))
    }
    k = max(nearest_k, params.candidate_k)
    out = np.empty((len(target_search), k), dtype=np.int64)
    for t, row in enumerate(nearest):
        picked: list[int] = []
        used: set[int] = set()

        def add(idx: int | None) -> None:
            if idx is not None and idx not in used and len(picked) < k:
                picked.append(idx)
                used.add(idx)

        for idx in row:
            add(int(idx))
        for song_candidates in per_song_nearest:
            for idx in song_candidates[t]:
                add(int(idx))
        # Neighboring source frames make diagonal continuation available even
        # when only the center frame was returned by spectral nearest-neighbor.
        for idx in row[: max(1, nearest_k // 2)]:
            sid = int(source_song[idx])
            frame = int(source_frame[idx])
            add(lookup.get((sid, frame - 1)))
            add(lookup.get((sid, frame + 1)))
        # Propagate previous paths one frame forward.
        if t > 0:
            for idx in out[t - 1, : min(4, k)]:
                sid = int(source_song[idx])
                frame = int(source_frame[idx])
                add(lookup.get((sid, frame + 1)))
        # Fill any remaining slots with nearest atoms (duplicates are harmless
        # only as a last resort when the source dictionary is tiny).
        fill = 0
        while len(picked) < k:
            # A tiny dictionary may not provide k unique atoms. Duplicates are
            # safe here (their activations simply split) and, importantly, this
            # guarantees termination after unique neighbors are exhausted.
            picked.append(int(row[fill % len(row)]))
            fill += 1
        out[t] = np.asarray(picked[:k], dtype=np.int64)
    return out


def _limit_polyphony(
    H: np.ndarray,
    polyphony: int,
    *,
    candidates: np.ndarray | None = None,
    source_song: np.ndarray | None = None,
) -> np.ndarray:
    p = int(np.clip(polyphony, 1, H.shape[1]))
    values = H
    if candidates is not None and source_song is not None:
        # Adjacent frames from one source often have nearly identical scores.
        # Keeping several of them wastes all polyphony slots on one note/timbre
        # and creates phasey combing. Retain the strongest atom per source song
        # before selecting the final cross-song mixture.
        song_at_candidate = source_song[candidates]
        diverse_mask = np.zeros_like(H, dtype=bool)
        rows = np.arange(len(H))
        for song in np.unique(source_song):
            eligible = song_at_candidate == song
            if not np.any(eligible):
                continue
            best = np.argmax(np.where(eligible, H, -np.inf), axis=1)
            valid = np.any(eligible, axis=1)
            diverse_mask[rows[valid], best[valid]] = True
        values = np.where(diverse_mask, H, 0.0)
    if p >= H.shape[1]:
        return values.astype(np.float32)
    keep = np.argpartition(values, -p, axis=1)[:, -p:]
    mask = np.zeros_like(H, dtype=bool)
    np.put_along_axis(mask, keep, True, axis=1)
    return np.where(mask, values, 0.0).astype(np.float32)


def _estimate_polyphony(target_X: np.ndarray, sr: int, params: NMFParams) -> int:
    """Estimate simultaneous pitch classes, bounded by the configured maximum."""
    maximum = max(1, int(params.polyphony))
    if maximum <= 1 or target_X.shape[1] < 2:
        return maximum
    try:
        chroma = librosa.feature.chroma_stft(
            S=np.abs(target_X) ** 2,
            sr=sr,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
        )
        peaks = np.max(chroma, axis=0, keepdims=True)
        active = np.sum(chroma >= np.maximum(1e-6, peaks * 0.32), axis=0)
        voiced = peaks.reshape(-1) > 1e-6
        if not np.any(voiced):
            return 1
        # The upper-middle frame is representative of sustained chords without
        # letting broadband drum hits force every frame to maximum polyphony.
        estimate = int(round(float(np.percentile(active[voiced], 65))))
        return int(np.clip(estimate, 1, maximum))
    except Exception:
        return min(3, maximum)


def _diagonal_continuity(
    H: np.ndarray,
    candidates: np.ndarray,
    source_song: np.ndarray,
    source_frame: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Convolve sparse activations along source-time/target-time diagonals."""
    if len(H) < 2 or amount <= 0:
        return H
    song = source_song[candidates]
    frame = source_frame[candidates]
    prev_match = (
        (song[1:, :, None] == song[:-1, None, :])
        & (frame[1:, :, None] == frame[:-1, None, :] + 1)
    )
    prev = np.max(
        np.where(prev_match, H[:-1, None, :], 0.0),
        axis=2,
    )
    continuity = np.zeros_like(H)
    continuity[1:] = prev
    out = (1.0 - amount) * H + amount * (H + continuity)
    return out.astype(np.float32)


def _suppress_repetition(
    H: np.ndarray,
    candidates: np.ndarray,
    target_shapes: np.ndarray,
    radius: int,
    suppression: float,
) -> np.ndarray:
    """Suppress stutter only while the target spectrum is actually changing."""
    if radius <= 0 or len(H) < 2:
        return H
    out = H.copy()
    for lag in range(1, min(radius, len(H) - 1) + 1):
        same = candidates[lag:, :, None] == candidates[:-lag, None, :]
        prior = np.max(np.where(same, out[:-lag, None, :], 0.0), axis=2)
        target_change = np.sum(
            np.abs(target_shapes[lag:] - target_shapes[:-lag]),
            axis=1,
        )
        # Reusing a source frame is correct for held vowels and sustained
        # instrumental chords. It sounds like stutter only when the target has
        # moved on but the same atom remains pinned.
        changing = target_change > 0.08
        repeated = (prior >= out[lag:]) & changing[:, None]
        out[lag:] = np.where(repeated, out[lag:] * suppression, out[lag:])
    return out.astype(np.float32)


def _learn_activations(
    source_shapes: np.ndarray,
    target_shapes: np.ndarray,
    candidates: np.ndarray,
    source_song: np.ndarray,
    source_frame: np.ndarray,
    *,
    params: NMFParams,
    on_progress: ProgressCb | None,
) -> tuple[np.ndarray, float]:
    """Sparse KL-NMF with progressively stronger Driedger constraints."""
    W = source_shapes[candidates]  # [target frame, candidate, mel]
    V = target_shapes
    H = np.full(candidates.shape, 1.0 / candidates.shape[1], dtype=np.float32)
    denom = np.sum(W, axis=2) + 1e-8

    for iteration in range(max(1, params.iterations)):
        estimate = np.einsum("tkf,tk->tf", W, H, optimize=True) + 1e-8
        ratio = V / estimate
        numer = np.einsum("tkf,tf->tk", W, ratio, optimize=True)
        H *= numer / denom

        progress = (iteration + 1) / max(1, params.iterations)
        strength = params.continuity * progress
        H = _suppress_repetition(
            H,
            candidates,
            V,
            params.repetition,
            1.0 - progress * (1.0 - params.repetition_suppression),
        )
        H = _diagonal_continuity(
            H, candidates, source_song, source_frame, strength
        )
        # Apply strict polyphony only in the latter half, as in the paper's
        # progressive constraints; early iterations may explore more atoms.
        if progress >= 0.5:
            H = _limit_polyphony(
                H,
                params.polyphony,
                candidates=candidates,
                source_song=source_song,
            )
        H /= np.maximum(np.sum(H, axis=1, keepdims=True), 1e-8)
        if on_progress:
            on_progress(
                0.32 + 0.30 * progress,
                f"NMF activations {iteration + 1}/{params.iterations}",
            )

    estimate = np.einsum("tkf,tk->tf", W, H, optimize=True)
    error = float(np.mean(np.abs(V - estimate)))
    return H.astype(np.float32), error


def _selected_complex_atoms(
    songs: dict[str, np.ndarray],
    song_ids: list[str],
    source_song: np.ndarray,
    source_frame: np.ndarray,
    candidates: np.ndarray,
    sr: int,
    *,
    params: NMFParams,
    on_progress: ProgressCb | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute and retain only complex source frames selected by the beam."""
    unique, inverse = np.unique(candidates, return_inverse=True)
    inverse = inverse.reshape(candidates.shape)
    atoms = np.zeros(
        (params.n_fft // 2 + 1, len(unique)), dtype=np.complex64
    )
    for si, song_id in enumerate(song_ids):
        positions = np.flatnonzero(source_song[unique] == si)
        if not len(positions):
            continue
        local = source_frame[unique[positions]]
        X = librosa.stft(
            songs[song_id].astype(np.float32),
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            window="hann",
        ).astype(np.complex64)
        valid = local < X.shape[1]
        atoms[:, positions[valid]] = X[:, local[valid]]
        if on_progress:
            on_progress(
                0.64 + 0.10 * (si + 1) / max(1, len(song_ids)),
                f"NMF phases {song_id}",
            )
    norms = np.sum(np.abs(atoms), axis=0) + 1e-8
    atoms /= norms[None, :]
    return atoms, inverse


def _synthesize(
    target_X: np.ndarray,
    atoms: np.ndarray,
    inverse_candidates: np.ndarray,
    H: np.ndarray,
    length: int,
    *,
    params: NMFParams,
    phase_reference: np.ndarray | None,
) -> np.ndarray:
    """Synthesize source-basis magnitudes with source-only coherent phase."""
    n_frames = min(target_X.shape[1], H.shape[0])
    target_energy = np.sum(np.abs(target_X[:, :n_frames]), axis=0)
    out_X = np.zeros((target_X.shape[0], n_frames), dtype=np.complex64)

    reference_X = None
    if phase_reference is not None:
        reference_X = librosa.stft(
            phase_reference.astype(np.float32),
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            window="hann",
        )

    def smooth_shape(magnitude: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Suppress frame-to-frame atom flicker without blurring target dynamics."""
        amount = float(np.clip(params.temporal_smoothing, 0.0, 1.0))
        if amount <= 0 or magnitude.shape[1] < 3:
            return magnitude
        shape = magnitude / np.maximum(np.sum(magnitude, axis=0, keepdims=True), 1e-8)
        padded = np.pad(shape, ((0, 0), (1, 1)), mode="edge")
        smooth = 0.25 * padded[:, :-2] + 0.50 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
        blended = (1.0 - amount) * shape + amount * smooth
        blended /= np.maximum(np.sum(blended, axis=0, keepdims=True), 1e-8)
        return (blended * energy[None, :]).astype(np.float32)

    block = 256
    for start in range(0, n_frames, block):
        end = min(n_frames, start + block)
        # One-frame halo keeps temporal smoothing continuous across blocks.
        halo_start = max(0, start - 1)
        halo_end = min(n_frames, end + 1)
        inv = inverse_candidates[halo_start:halo_end]
        # [frequency, block, candidate]
        selected = atoms[:, inv]
        weights = H[halo_start:halo_end]
        magnitude = np.einsum(
            "fbk,bk->fb", np.abs(selected), weights, optimize=True
        )
        halo_energy = target_energy[halo_start:halo_end]
        magnitude = smooth_shape(magnitude, halo_energy)
        left = start - halo_start
        right = left + (end - start)
        magnitude = magnitude[:, left:right]
        if reference_X is not None and reference_X.shape[1] >= end:
            # NMF magnitude paired with unrelated phase can sound metallic or
            # static. Blend in a small amount of the source-only unit
            # reference magnitude, scaled to the same frame energy, with more
            # correction only when the two spectral shapes disagree.
            ref_magnitude = np.abs(reference_X[:, start:end]).astype(np.float32)
            dot = np.sum(magnitude * ref_magnitude, axis=0)
            norm = (
                np.linalg.norm(magnitude, axis=0)
                * np.linalg.norm(ref_magnitude, axis=0)
                + 1e-8
            )
            mismatch = 1.0 - np.clip(dot / norm, 0.0, 1.0)
            base_blend = float(np.clip(params.phase_coherence_blend, 0.0, 1.0))
            blend = base_blend * (0.35 + 0.65 * mismatch)
            # Multiplicative shaping keeps zero source bins at zero: the phase
            # reference can improve compatibility but can never inject its own
            # magnitude or become an audio carrier.
            source_shape = magnitude / np.maximum(
                np.sum(magnitude, axis=0, keepdims=True),
                1e-8,
            )
            ref_shape = ref_magnitude / np.maximum(
                np.sum(ref_magnitude, axis=0, keepdims=True),
                1e-8,
            )
            shape_ratio = np.clip(
                ref_shape / np.maximum(source_shape, 1e-8),
                0.5,
                2.0,
            )
            magnitude *= np.exp(blend[None, :] * np.log(shape_ratio))
            magnitude *= target_energy[start:end][None, :] / np.maximum(
                np.sum(magnitude, axis=0, keepdims=True),
                1e-8,
            )
            phase = np.angle(reference_X[:, start:end])
        else:
            mixed = np.einsum("fbk,bk->fb", selected, weights, optimize=True)
            phase = np.angle(mixed[:, left:right])
        out_X[:, start:end] = magnitude * np.exp(1j * phase)

    audio = librosa.istft(
        out_X,
        hop_length=params.hop_length,
        window="hann",
        length=length,
    )
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.99:
        audio *= 0.99 / peak
    return audio.astype(np.float32)


def reconstruct_nmf(
    target: np.ndarray,
    songs: dict[str, np.ndarray],
    sr: int,
    *,
    params: NMFParams | None = None,
    phase_reference: np.ndarray | None = None,
    on_progress: ProgressCb | None = None,
) -> NMFResult:
    """Render the complete target through a sparse source-spectrogram NMF."""
    params = params or NMFParams()
    if len(target) < params.n_fft or not songs:
        return NMFResult(
            np.zeros_like(target),
            {song_id: 0.0 for song_id in songs},
            1.0,
            0.0,
            0,
            0,
            params.hop_length,
            [],
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=params.n_fft,
        n_mels=params.n_mels,
        fmin=30.0,
        fmax=sr / 2,
        norm="slaney",
    ).astype(np.float32)
    source_shapes, source_search, source_song, source_frame, song_ids = (
        _source_dictionary(
            songs,
            sr,
            params=params,
            mel_basis=mel_basis,
            on_progress=on_progress,
        )
    )
    target_X, target_shapes, target_search = _frame_features(
        target, sr, params=params, mel_basis=mel_basis
    )
    effective_polyphony = _estimate_polyphony(target_X, sr, params)
    effective_params = NMFParams(
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        n_mels=params.n_mels,
        candidate_k=params.candidate_k,
        nearest_k=params.nearest_k,
        polyphony=effective_polyphony,
        iterations=params.iterations,
        continuity=params.continuity,
        repetition=params.repetition,
        repetition_suppression=params.repetition_suppression,
        temporal_smoothing=params.temporal_smoothing,
        phase_coherence_blend=params.phase_coherence_blend,
    )
    if on_progress:
        on_progress(0.26, "NMF nearest source spectra")
    candidates = _candidate_beam(
        source_search,
        target_search,
        source_song,
        source_frame,
        params=params,
    )
    H, error = _learn_activations(
        source_shapes,
        target_shapes,
        candidates,
        source_song,
        source_frame,
        params=effective_params,
        on_progress=on_progress,
    )
    atoms, inverse = _selected_complex_atoms(
        songs,
        song_ids,
        source_song,
        source_frame,
        candidates,
        sr,
        params=effective_params,
        on_progress=on_progress,
    )
    if on_progress:
        on_progress(0.78, "NMF source-only resynthesis")
    audio = _synthesize(
        target_X,
        atoms,
        inverse,
        H,
        len(target),
        params=effective_params,
        phase_reference=phase_reference,
    )

    activation_by_song = np.zeros(len(song_ids), dtype=np.float64)
    candidate_songs = source_song[candidates]
    for si in range(len(song_ids)):
        activation_by_song[si] = float(np.sum(H[candidate_songs == si]))
    total = float(np.sum(activation_by_song)) + 1e-8
    contribution = {
        song_id: float(round(100.0 * activation_by_song[i] / total, 1))
        for i, song_id in enumerate(song_ids)
    }
    active = float(np.mean(np.sum(H > 1e-4, axis=1)))
    best_k = np.argmax(H, axis=1)
    rows = np.arange(len(H))
    best_source = candidates[rows, best_k]
    frame_song_ids = [song_ids[int(si)] for si in source_song[best_source]]
    frame_source_times = (
        source_frame[best_source].astype(np.float32) * params.hop_length / sr
    )
    frame_weights = H[rows, best_k].astype(np.float32)
    if on_progress:
        on_progress(1.0, "NMF mosaic ready")
    return NMFResult(
        audio=audio,
        contribution_pct=contribution,
        spectral_error=error,
        active_polyphony=active,
        n_source_frames=len(source_shapes),
        n_target_frames=len(target_shapes),
        hop_length=params.hop_length,
        frame_song_ids=frame_song_ids,
        frame_source_times_s=frame_source_times,
        frame_weights=frame_weights,
    )
