# MusicMosaic

Reconstruct one song using only short clips from five other songs.

## Quick start

```bash
npm install
npm run dev
```

That starts the FastAPI backend and the Vite frontend together. Open **http://localhost:5173**.

**Requirements:** Node 20+, Python 3.11+, [uv](https://github.com/astral-sh/uv), ffmpeg.

First time only — sync Python deps:

```bash
cd backend && uv sync --extra dev && cd ..
```

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

1. Drop one **target** MP3 and five **source** MP3s.
2. Watch the splice + quilt fill animation while it reconstructs.
3. Play the mosaic — tiles light in sync; toggle target vs reconstruction.
4. Click a tile for source timestamp + similarity.

## Stack

FastAPI · librosa · FAISS · Viterbi matching · React · Vite
