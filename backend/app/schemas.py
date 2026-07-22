"""Pydantic API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class JobParams(BaseModel):
    window_s: float = Field(0.5, ge=0.1, le=2.0)
    hop_s: float = Field(0.25, ge=0.05, le=1.0)
    top_k: int = Field(8, ge=1, le=32)
    lambda_switch: float = Field(0.35, ge=0.0, le=2.0)
    lambda_jump: float = Field(0.25, ge=0.0, le=2.0)
    jump_norm_s: float = Field(2.0, ge=0.1, le=30.0)
    lambda_self: float = Field(0.05, ge=0.0, le=1.0)

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
        "reconstruct",
        "done",
        "error",
    ]
    pct: float
    message: str
    error: str | None = None
    stats: dict[str, Any] | None = None
    elapsed_s: float | None = None
