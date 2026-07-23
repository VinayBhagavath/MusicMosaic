"""Pydantic API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class JobParams(BaseModel):
    window_s: float = Field(0.45, ge=0.25, le=2.0)
    hop_s: float = Field(0.22, ge=0.1, le=1.0)
    top_k: int = Field(20, ge=1, le=32)
    lambda_switch: float = Field(0.08, ge=0.0, le=2.0)
    lambda_jump: float = Field(0.15, ge=0.0, le=2.0)
    jump_norm_s: float = Field(2.0, ge=0.1, le=30.0)
    lambda_self: float = Field(0.03, ge=0.0, le=1.0)
    lambda_concat: float = Field(0.55, ge=0.0, le=2.0)
    lambda_join: float = Field(0.70, ge=0.0, le=2.0)
    per_song_k: int = Field(4, ge=1, le=8)
    lambda_balance: float = Field(0.0, ge=0.0, le=3.0)
    max_share: float = Field(1.0, ge=0.15, le=1.0)
    balance_iters: int = Field(1, ge=1, le=6)
    min_run_tiles: int = Field(1, ge=1, le=12)
    n_layers: int = Field(1, ge=1, le=5)
    layer_primary_weight: float = Field(0.62, ge=0.35, le=1.0)
    fidelity_first: bool = True
    harmonic_match: bool = True
    harmonic_strength: float = Field(0.42, ge=0.0, le=1.0)
    onset_sync_xf: bool = True
    rerank_spectral: bool = True
    reconstruction_backend: Literal["auto", "unit", "nmf"] = "auto"
    use_stems: bool = False

    @model_validator(mode="after")
    def hop_not_longer_than_window(self) -> JobParams:
        if self.hop_s > self.window_s:
            raise ValueError("hop_s must be ≤ window_s (otherwise reconstruction has gaps)")
        return self


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    stage: Literal[
        "queued",
        "download",
        "load",
        "segment",
        "features",
        "index",
        "match",
        "stems",
        "reconstruct",
        "done",
        "error",
    ]
    pct: float
    message: str
    error: str | None = None
    stats: dict[str, Any] | None = None
    elapsed_s: float | None = None


class YouTubeHit(BaseModel):
    id: str
    title: str
    url: str
    duration_s: float | None = None
    channel: str | None = None


class YouTubeSearchResponse(BaseModel):
    query: str
    results: list[YouTubeHit]
