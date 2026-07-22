"""YouTube / yt-dlp audio download + search helpers."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.pipeline.audio_io import MAX_DURATION_S, MIN_DURATION_S

_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
_YT_RE = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/",
    re.IGNORECASE,
)
_SONG_WORDS = ("song", "songs", "music", "track", "tracks", "audio", "official")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")

# Persistent cache so repeated Interstellar/demo runs skip yt-dlp entirely.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "youtube"


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


def youtube_video_id(url: str) -> str | None:
    """Extract a stable video id for cache keys."""
    try:
        u = normalize_youtube_url(url)
        parsed = urlparse(u)
        host = (parsed.hostname or "").lower()
        if host in {"youtu.be", "www.youtu.be"}:
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid if _ID_RE.match(vid) else None
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            vid = qs["v"][0]
            return vid if _ID_RE.match(vid) else None
        # /shorts/ID or /embed/ID
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1] if _ID_RE.match(parts[1]) else None
    except Exception:
        return None
    return None


def _cache_index_path() -> Path:
    return _CACHE_DIR / "index.json"


def _load_cache_index() -> dict:
    path = _cache_index_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache_index(index: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_index_path().write_text(json.dumps(index, indent=2))


def _cached_mp3(video_id: str) -> Path | None:
    path = _CACHE_DIR / f"{video_id}.mp3"
    return path if path.exists() and path.stat().st_size > 1024 else None


def download_youtube_audio(url: str, dest_dir: Path, *, stem: str) -> tuple[Path, str]:
    """Download best audio as mp3 into dest_dir/{stem}.mp3. Returns (path, title).

    Hits a content-addressed cache under ``backend/data/cache/youtube/{videoId}.mp3``
    so repeated demo runs (same Interstellar + sources) skip the network.
    """
    url = normalize_youtube_url(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}.mp3"
    video_id = youtube_video_id(url)
    index = _load_cache_index()

    if video_id:
        cached = _cached_mp3(video_id)
        if cached is not None:
            shutil.copy2(cached, dest)
            title = str((index.get(video_id) or {}).get("title") or stem)
            print(f"[musicmosaic] youtube cache hit {video_id}", flush=True)
            return dest, title

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed") from e

    # Download once into the cache (or dest if we have no video id), then copy.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if video_id:
        outtmpl = str(_CACHE_DIR / f"{video_id}.%(ext)s")
    else:
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
        # Single network round-trip: extract + download together.
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise ValueError("Could not read YouTube video info")
        duration = float(info.get("duration") or 0)
        if duration and duration < MIN_DURATION_S:
            raise ValueError(f"Video too short ({duration:.0f}s; need ≥{MIN_DURATION_S:.0f}s)")
        if duration and duration > MAX_DURATION_S:
            raise ValueError(f"Video too long ({duration:.0f}s; max {MAX_DURATION_S / 60:.0f} min)")
        title = str(info.get("title") or stem)
        if not video_id:
            video_id = str(info.get("id") or "") or None

    if video_id:
        cached = _CACHE_DIR / f"{video_id}.mp3"
        if not cached.exists():
            matches = list(_CACHE_DIR.glob(f"{video_id}.*"))
            if matches:
                cached = matches[0]
        if cached.exists():
            shutil.copy2(cached, dest)
            index[video_id] = {"title": title}
            _save_cache_index(index)
            return dest, title

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
