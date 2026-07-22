"""YouTube / yt-dlp audio download + search helpers."""

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
_SONG_WORDS = ("song", "songs", "music", "track", "tracks", "audio", "official")


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
        matches = list(dest_dir.glob(f"{stem}.*"))
        if not matches:
            raise RuntimeError("Download finished but audio file missing (is ffmpeg installed?)")
        path = matches[0]
    return path, title


def search_youtube(query: str, *, limit: int = 10) -> list[dict]:
    """Top YouTube song results for a query (no instrumental bias).

    \"piano\" → searches \"piano songs\". Only keeps clips in [5s, 5min].
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed") from e

    q = query.strip()
    if not q:
        raise ValueError("Empty search query")
    ql = q.lower()
    if not any(w in ql for w in _SONG_WORDS):
        q = f"{q} songs"

    # Over-fetch then filter by duration so we still return ~limit valid songs
    fetch_n = max(1, min(int(limit) * 4, 40))
    want = max(1, min(int(limit), 20))
    min_s = MIN_DURATION_S  # 5s
    max_s = 5 * 60.0  # 5 minutes — search filter (stricter than process max)

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{fetch_n}:{q}", download=False)

    results: list[dict] = []
    for entry in (info or {}).get("entries") or []:
        if not entry or len(results) >= want:
            break
        vid = entry.get("id") or ""
        title = entry.get("title") or "Untitled"
        url = entry.get("url") or entry.get("webpage_url")
        if not url and vid:
            url = f"https://www.youtube.com/watch?v={vid}"
        if not url:
            continue
        dur = entry.get("duration")
        if dur is None:
            continue  # unknown length — skip so only valid songs appear
        try:
            d = float(dur)
        except (TypeError, ValueError):
            continue
        if d < min_s or d > max_s:
            continue
        results.append(
            {
                "id": str(vid or url),
                "title": str(title),
                "url": str(url),
                "duration_s": d,
                "channel": entry.get("uploader") or entry.get("channel"),
            }
        )
    return results
