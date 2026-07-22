"""YouTube / yt-dlp audio download helpers."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from app.pipeline.audio_io import MAX_DURATION_S, MIN_DURATION_S

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
_YT_RE = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/",
    re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    u = url.strip()
    if not u or not _YT_RE.match(u):
        return False
    try:
        host = urlparse(u if "://" in u else f"https://{u}").hostname or ""
    except Exception:
        return False
    return host.lower() in _YT_HOSTS


def normalize_youtube_url(url: str) -> str:
    u = url.strip()
    if not u.startswith("http"):
        u = "https://" + u
    if not is_youtube_url(u):
        raise ValueError("Only YouTube links are supported")
    return u


def download_youtube_audio(url: str, dest_dir: Path, *, stem: str) -> tuple[Path, str]:
    """Download best audio as mp3 into dest_dir/{stem}.mp3. Returns (path, title)."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed") from e

    url = normalize_youtube_url(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / f"{stem}.%(ext)s")

    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "socket_timeout": 30,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            raise ValueError("Could not read YouTube video info")
        duration = float(info.get("duration") or 0)
        if duration and duration < MIN_DURATION_S:
            raise ValueError(f"Video too short ({duration:.0f}s; need ≥{MIN_DURATION_S:.0f}s)")
        if duration and duration > MAX_DURATION_S:
            raise ValueError(f"Video too long ({duration:.0f}s; max {MAX_DURATION_S / 60:.0f} min)")
        title = str(info.get("title") or stem)
        ydl.download([url])

    path = dest_dir / f"{stem}.mp3"
    if not path.exists():
        # Rare: already-mp3 without postprocess rename
        matches = list(dest_dir.glob(f"{stem}.*"))
        if not matches:
            raise RuntimeError("Download finished but audio file missing (is ffmpeg installed?)")
        path = matches[0]
    return path, title
