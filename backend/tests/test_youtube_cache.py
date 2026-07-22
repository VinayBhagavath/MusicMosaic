"""YouTube download cache helpers."""

from pathlib import Path

from app.pipeline.youtube import download_youtube_audio, youtube_video_id


def test_youtube_video_id_parses_watch_and_short():
    assert youtube_video_id("https://www.youtube.com/watch?v=4y33h81phKU") == "4y33h81phKU"
    assert youtube_video_id("https://youtu.be/4y33h81phKU") == "4y33h81phKU"


def test_download_uses_seeded_cache(tmp_path):
    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / "youtube"
    cached = cache / "4y33h81phKU.mp3"
    if not cached.exists():
        import pytest

        pytest.skip("demo youtube cache not seeded")

    path, title = download_youtube_audio(
        "https://www.youtube.com/watch?v=4y33h81phKU",
        tmp_path,
        stem="yt_target",
    )
    assert path.exists()
    assert path.stat().st_size == cached.stat().st_size
    assert title
