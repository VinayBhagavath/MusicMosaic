"""FastAPI routes for mosaic jobs."""

from __future__ import annotations

import re
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.pipeline.job import JobConfig, run_job
from app.pipeline.youtube import download_youtube_audio, is_youtube_url
from app.schemas import JobCreateResponse, JobParams, JobStatus

router = APIRouter(prefix="/api")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB per file
_JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def _require_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(400, "Invalid job id")
    return job_id


def _job_dir(job_id: str) -> Path:
    return DATA_DIR / _require_job_id(job_id)


def _set(job_id: str, **kwargs) -> None:
    with _lock:
        cur = _jobs.get(job_id)
        if cur is None:
            return
        _jobs[job_id] = cur.model_copy(update=kwargs)


async def _save_upload(upload: UploadFile, dest: Path) -> str:
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
        chunks.append(chunk)
    dest.write_bytes(b"".join(chunks))
    return upload.filename or dest.name


def _run(
    job_id: str,
    target_path: Path | None,
    target_url: str | None,
    source_paths: list[Path | None],
    source_urls: list[str | None],
    names: list[str | None],
    params: JobParams,
) -> None:
    out = _job_dir(job_id)
    try:
        def on_progress(stage: str, pct: float, msg: str) -> None:
            _set(job_id, stage=stage, pct=pct, message=msg)  # type: ignore[arg-type]

        # Resolve YouTube URLs first
        resolved_target = target_path
        resolved_sources: list[Path] = []
        resolved_names: list[str] = []

        if target_url:
            on_progress("download", 3, "Downloading target from YouTube")
            path, title = download_youtube_audio(target_url, out, stem="yt_target")
            resolved_target = path
            target_name = title
        else:
            assert resolved_target is not None
            target_name = names[0] if names and names[0] else resolved_target.name

        for i in range(5):
            url = source_urls[i]
            path = source_paths[i]
            if url:
                on_progress("download", 5 + i * 3, f"Downloading source {chr(65 + i)} from YouTube")
                sp, title = download_youtube_audio(url, out, stem=f"yt_source_{i}")
                resolved_sources.append(sp)
                resolved_names.append(title)
            else:
                assert path is not None
                resolved_sources.append(path)
                resolved_names.append(names[i + 1] if names and names[i + 1] else path.name)

        assert resolved_target is not None
        cfg = JobConfig(
            window_s=params.window_s,
            hop_s=params.hop_s,
            top_k=params.top_k,
            lambda_switch=params.lambda_switch,
            lambda_jump=params.lambda_jump,
            jump_norm_s=params.jump_norm_s,
            lambda_self=params.lambda_self,
        )
        result = run_job(
            resolved_target,
            resolved_sources,
            out,
            config=cfg,
            source_names=resolved_names,
            on_progress=on_progress,
        )
        _set(
            job_id,
            stage="done",
            pct=100,
            message="Complete",
            stats=result.mosaic.get("stats"),
            elapsed_s=result.elapsed_s,
        )
    except Exception as e:
        _set(job_id, stage="error", pct=100, message="Failed", error=str(e))


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    target: UploadFile | None = File(None),
    source_0: UploadFile | None = File(None),
    source_1: UploadFile | None = File(None),
    source_2: UploadFile | None = File(None),
    source_3: UploadFile | None = File(None),
    source_4: UploadFile | None = File(None),
    target_url: str | None = Form(None),
    source_0_url: str | None = Form(None),
    source_1_url: str | None = Form(None),
    source_2_url: str | None = Form(None),
    source_3_url: str | None = Form(None),
    source_4_url: str | None = Form(None),
    window_s: float = Form(0.5),
    hop_s: float = Form(0.25),
    top_k: int = Form(8),
    lambda_switch: float = Form(0.35),
    lambda_jump: float = Form(0.25),
    jump_norm_s: float = Form(2.0),
    lambda_self: float = Form(0.05),
) -> JobCreateResponse:
    try:
        params = JobParams(
            window_s=window_s,
            hop_s=hop_s,
            top_k=top_k,
            lambda_switch=lambda_switch,
            lambda_jump=lambda_jump,
            jump_norm_s=jump_norm_s,
            lambda_self=lambda_self,
        )
    except ValidationError as e:
        raise HTTPException(422, e.errors()) from e

    t_url = (target_url or "").strip() or None
    s_urls = [
        (source_0_url or "").strip() or None,
        (source_1_url or "").strip() or None,
        (source_2_url or "").strip() or None,
        (source_3_url or "").strip() or None,
        (source_4_url or "").strip() or None,
    ]
    uploads = [source_0, source_1, source_2, source_3, source_4]

    def has_file(u: UploadFile | None) -> bool:
        return u is not None and bool(u.filename)

    if t_url:
        if not is_youtube_url(t_url):
            raise HTTPException(400, "target_url must be a YouTube link")
    elif not has_file(target):
        raise HTTPException(400, "Provide target file or target_url")

    for i, (url, up) in enumerate(zip(s_urls, uploads)):
        if url:
            if not is_youtube_url(url):
                raise HTTPException(400, f"source_{i}_url must be a YouTube link")
        elif not has_file(up):
            raise HTTPException(400, f"Provide source_{i} file or source_{i}_url")

    job_id = uuid.uuid4().hex[:12]
    out = DATA_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)

    target_path: Path | None = None
    names: list[str | None] = [None] * 6
    if has_file(target) and not t_url:
        assert target is not None
        target_path = out / f"upload_target{Path(target.filename or 't.mp3').suffix or '.mp3'}"
        names[0] = await _save_upload(target, target_path)

    source_paths: list[Path | None] = [None] * 5
    for i, (url, up) in enumerate(zip(s_urls, uploads)):
        if has_file(up) and not url:
            assert up is not None
            sp = out / f"upload_source_{i}{Path(up.filename or 's.mp3').suffix or '.mp3'}"
            names[i + 1] = await _save_upload(up, sp)
            source_paths[i] = sp

    with _lock:
        _jobs[job_id] = JobStatus(job_id=job_id, stage="queued", pct=0, message="Queued")

    background_tasks.add_task(
        _run, job_id, target_path, t_url, source_paths, s_urls, names, params
    )
    return JobCreateResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    _require_job_id(job_id)
    with _lock:
        status = _jobs.get(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@router.get("/jobs/{job_id}/mosaic")
def get_mosaic(job_id: str) -> dict:
    path = _job_dir(job_id) / "mosaic.json"
    if not path.exists():
        raise HTTPException(404, "Mosaic not ready")
    import json

    return json.loads(path.read_text())


@router.get("/jobs/{job_id}/audio")
def get_audio(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "reconstructed.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not ready")
    return FileResponse(path, media_type="audio/wav", filename="reconstructed.wav")


@router.get("/jobs/{job_id}/target")
def get_target(job_id: str) -> FileResponse:
    path = _job_dir(job_id) / "target.wav"
    if not path.exists():
        raise HTTPException(404, "Target not ready")
    return FileResponse(path, media_type="audio/wav", filename="target.wav")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    _require_job_id(job_id)
    with _lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(DATA_DIR / job_id, ignore_errors=True)
    return {"ok": True}
