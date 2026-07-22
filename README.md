# MusicMosaic

Reconstruct one song using only short clips from five other songs.

## Quick start

```bash
npm install
cd backend && uv sync --extra dev && cd ..
npm run dev
```

Open **http://localhost:5173**.

Each slot accepts an **MP3 upload** or a **YouTube URL**. In-app search uses top results for `\"{query} songs\"` (e.g. piano → piano songs).

**Embeddings:** prefers **CLAP** (`laion/larger_clap_music_and_speech`) hybridized with key-invariant chroma; falls back to handcrafted MFCC if torch/transformers aren't available.

Tracks must be **5 seconds–8 minutes** at download/process time.

### CLI (no UI)

```bash
cd backend
uv run python -m app.pipeline.job target.mp3 s1.mp3 s2.mp3 s3.mp3 s4.mp3 s5.mp3 -o ./out
```

### Tests

```bash
npm test
```

## Demo

1. Drop MP3s **or** paste YouTube links for 1 target + 5 sources.
2. Watch the splice + quilt fill animation.
3. Play the mosaic; toggle target vs reconstruction.
4. Click a tile for source timestamp + similarity.

## Stack

FastAPI · librosa · FAISS · Viterbi · yt-dlp · React · Vite
