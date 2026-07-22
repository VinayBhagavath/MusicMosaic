"""FastAPI routes for mosaic jobs."""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.pipeline.job import JobConfig, run_job
from app.schemas import JobCreateResponse, JobParams, JobStatus

router = APIRouter(prefix="/api")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (single-process local demo)
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def _set(job_id: str, **kwargs) -> None:
    with _lock:
        cur = _jobs[job_id]
        _jobs[job_id] = cur.model_copy(update=kwargs)


def _run(job_id: str, target: Path, sources: list[Path], names: list[str], params: JobParams) -> None:
    out = DATA_DIR / job_id
    try:
        def on_progress(stage: str, pct: float, msg: str) -> None:
            _set(job_id, stage=stage, pct=pct, message=msg)  # type: ignore[arg-type]

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
            target,
            sources,
            out,
            config=cfg,
            source_names=names,
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
    target: UploadFile = File(...),
    source_0: UploadFile = File(...),
    source_1: UploadFile = File(...),
    source_2: UploadFile = File(...),
    source_3: UploadFile = File(...),
    source_4: UploadFile = File(...),
    window_s: float = Form(0.5),
    hop_s: float = Form(0.25),
    top_k: int = Form(8),
    lambda_switch: float = Form(0.35),
    lambda_jump: float = Form(0.25),
    jump_norm_s: float = Form(2.0),
    lambda_self: float = Form(0.05),
) -> JobCreateResponse:
    params = JobParams(
        window_s=window_s,
        hop_s=hop_s,
        top_k=top_k,
        lambda_switch=lambda_switch,
        lambda_jump=lambda_jump,
        jump_norm_s=jump_norm_s,
        lambda_self=lambda_self,
    )
    job_id = uuid.uuid4().hex[:12]
    out = DATA_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)

    async def save(upload: UploadFile, dest: Path) -> str:
        dest.write_bytes(await upload.read())
        return upload.filename or dest.name

    target_path = out / f"upload_target{Path(target.filename or 't.mp3').suffix}"
    source_files = [source_0, source_1, source_2, source_3, source_4]
    source_paths: list[Path] = []
    names: list[str] = []
    await save(target, target_path)
    for i, up in enumerate(source_files):
        sp = out / f"upload_source_{i}{Path(up.filename or 's.mp3').suffix}"
        name = await save(up, sp)
        source_paths.append(sp)
        names.append(name)

    with _lock:
        _jobs[job_id] = JobStatus(
            job_id=job_id, stage="queued", pct=0, message="Queued"
        )

    background_tasks.add_task(_run, job_id, target_path, source_paths, names, params)
    return JobCreateResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    with _lock:
        status = _jobs.get(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@router.get("/jobs/{job_id}/mosaic")
def get_mosaic(job_id: str) -> dict:
    path = DATA_DIR / job_id / "mosaic.json"
    if not path.exists():
        raise HTTPException(404, "Mosaic not ready")
    import json

    return json.loads(path.read_text())


@router.get("/jobs/{job_id}/audio")
def get_audio(job_id: str) -> FileResponse:
    path = DATA_DIR / job_id / "reconstructed.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not ready")
    return FileResponse(path, media_type="audio/wav", filename="reconstructed.wav")


@router.get("/jobs/{job_id}/target")
def get_target(job_id: str) -> FileResponse:
    path = DATA_DIR / job_id / "target.wav"
    if not path.exists():
        raise HTTPException(404, "Target not ready")
    return FileResponse(path, media_type="audio/wav", filename="target.wav")


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    with _lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(DATA_DIR / job_id, ignore_errors=True)
    return {"ok": True}
