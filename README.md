# MusicMosaic

Reconstruct one song using only short clips from five other songs.

Upload a **target** track and five **source** tracks. MusicMosaic chops everything into overlapping 0.5 s windows, embeds each clip with handcrafted acoustic features (MFCC, chroma, centroid, RMS), searches with FAISS, then picks a globally smooth sequence via Viterbi (penalizing song switches). The result is a new WAV built only from source audio, plus an interactive mosaic of which song contributed each tile.

## Quick start

**Requirements:** Python 3.11+, Node 20+, [uv](https://github.com/astral-sh/uv), ffmpeg.

```bash
# Backend
cd backend
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### CLI (no UI)

```bash
cd backend
uv run python -m app.pipeline.job target.mp3 s1.mp3 s2.mp3 s3.mp3 s4.mp3 s5.mp3 -o ./out
```

Writes `out/reconstructed.wav` and `out/mosaic.json`.

### Tests

```bash
cd backend && uv run pytest -q
```

## Demo script (~60s)

1. Pick a well-known target and five sources in a related genre/tempo.
2. Drop them on the upload screen → **Build mosaic**.
3. Watch progress (load → features → match → reconstruct).
4. Hit play: mosaic tiles light in sync; timeline shows source-song runs.
5. Click a tile to A/B the original target window vs the matched source clip.
6. Check stats: contribution %, avg similarity, greedy vs Viterbi transitions.

## Architecture

| Layer | Stack |
|---|---|
| API | FastAPI, background jobs on disk under `backend/data/jobs/` |
| DSP | librosa, pyloudnorm, soundfile |
| Search | FAISS `IndexFlatIP` (exact cosine on L2-normalized vectors) |
| Sequence | Viterbi with switch / temporal-jump costs |
| Reconstruct | Hann overlap-add |
| UI | React + Vite + TypeScript |

## Defaults

- Sample rate: 22.05 kHz mono
- Window / hop: 0.5 s / 0.25 s
- Loudness: −14 LUFS
- Transition weights: λ_switch=0.35, λ_jump=0.25

## License

MIT
