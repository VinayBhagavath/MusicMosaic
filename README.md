# MusicMosaic

Reconstruct one song using only short clips from five other songs.

## Quick start

```bash
brew install rubberband # required for high-quality pitch/time transforms
npm install
cd backend && uv sync --extra dev && cd ..
npm run dev
```

Open **http://localhost:5173**.

Each slot accepts an **MP3 upload** or a **YouTube URL**. In-app search uses top results for `\"{query} songs\"` (e.g. piano → piano songs).

**Matching:** fidelity-first concatenative synthesis. Target and source tracks are split into
short onset-aligned events. Exact candidate search compares temporal chroma, pitch register,
MFCC/contrast timbre, dynamics, and splice edges. After the Viterbi path is fixed, each note's
beam is **re-ranked by the actual post-transform spectral distance** to its target window (not
just pre-transform feature similarity). The best sequence is pitch/time transformed with Rubber
Band and morphed toward the target envelope and spectrum.

**Legibility:** to make the result recognizable as the target song — not just similar in texture —
each note gets a **harmonic-reconstruction pass** (inspired by Driedger et al., *Let It Bee*,
ISMIR 2015). Where `match_spectrum` transfers only the smooth tonal envelope (keeping timbre but
leaving the source's own pitches), this pass keeps the source envelope but morphs the harmonic
fine structure — *which notes are sounding* — toward the target, so melody and chords read
through. Seams are **onset-synchronous**: crossfades collapse to a click-guard minimum on target
attacks (punchy transients) and keep the full ~30 ms equal-power fade on sustained boundaries.

Fidelity mode deliberately does **not** force equal use of all five songs: a source may dominate
when it is the closest acoustic match. Turn Fidelity first off in Advanced parameters for a more
visually balanced collage. Multi-layer mode (`n_layers` ≥ 2) stacks a residual spectral fill from
a second source to approximate polyphony; it helps when sources actually contain the missing
chord tones, but with mismatched instrumentation it can muddy the result — leave it at 1 for
piano-like targets unless you have measured an improvement. Stem (Demucs) mode is experimental.

Tracks must be **5 seconds–8 minutes** at download/process time. Repeated YouTube
URLs are cached under `backend/data/cache/youtube/` (the Interstellar demo preset
is pre-seeded) so re-runs skip network downloads.

### CLI (no UI)

```bash
cd backend
uv run python -m app.pipeline.job target.mp3 s1.mp3 s2.mp3 s3.mp3 s4.mp3 s5.mp3 -o ./out
```

### Tests

```bash
npm test
```

Each completed job reports listening-oriented diagnostics in `mosaic.json`: frame chroma
similarity and onset correlation (higher is better), plus multi-resolution log-mel distance and
boundary discontinuity (lower is better). These are comparison aids, not replacements for A/B
listening.

### Optional models

The default installation avoids large neural dependencies. Install only what you use:

```bash
cd backend
uv sync --extra clap   # then MUSICMOSAIC_USE_CLAP=1; semantic, not default
uv sync --extra stems  # Demucs / Demucs-MLX stem experiments
```

## Demo

1. Drop MP3s **or** paste YouTube links for 1 target + 5 sources.
2. Watch the splice + quilt fill animation.
3. Play the mosaic; toggle target vs reconstruction.
4. Click a tile for source timestamp, pitch shift, role, and similarity.

## Fidelity limits

MusicMosaic only synthesizes from audio present in the five source songs. It can reshape pitch,
timing, loudness, and spectral envelope, but it cannot perfectly recover missing instruments,
polyphony, vocals, room acoustics, or phase. Sources with similar instrumentation, tempo,
register, and production to the target will produce the closest result.

## Stack

FastAPI · librosa · Rubber Band · Viterbi · yt-dlp · React · Vite
