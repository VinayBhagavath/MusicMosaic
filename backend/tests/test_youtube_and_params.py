"""YouTube URL / search helpers (no network for unit tests)."""

import pytest
from pydantic import ValidationError

from app.pipeline.youtube import is_youtube_url, normalize_youtube_url
from app.schemas import JobParams


def test_youtube_url_accepts():
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("youtube.com/watch?v=abc")


def test_youtube_url_rejects_ssrf():
    assert not is_youtube_url("http://127.0.0.1/secret")
    assert not is_youtube_url("https://evil.com/?u=youtube.com")
    assert not is_youtube_url("")


def test_normalize_adds_https():
    u = normalize_youtube_url("youtu.be/dQw4w9WgXcQ")
    assert u.startswith("https://")


def test_hop_must_not_exceed_window():
    with pytest.raises(ValidationError):
        JobParams(window_s=0.25, hop_s=0.5)


def test_songs_bias_query_helper():
    # mirror search_youtube bias without network
    from app.pipeline import youtube as yt

    q = "piano"
    ql = q.lower()
    if not any(w in ql for w in yt._SONG_WORDS):
        q = f"{q} songs"
    assert q == "piano songs"
