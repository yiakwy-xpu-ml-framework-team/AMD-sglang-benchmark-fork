# SPDX-License-Identifier: Apache-2.0
"""Direct-curl HTTP route for the YuE2 multimodal pipeline.

This module is intentionally kept out of the core multimodal API until the
audio-generation request schema is upstreamed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.srt.observability.trace import extract_trace_headers
from sglang.srt.models.yue2.protocol import SongRequest


router = APIRouter(prefix="/v1/audio", tags=["audio"])


class Yue2AudioGenerationsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    style: str = Field(..., description="style/tags")
    lyrics: str = Field("", description="lyrics text")
    cot: Literal["off", "melody", "full"] = "full"
    abc: Optional[str] = Field(None, description="optional ABC score")
    seed: int = Field(831001, ge=0)
    ode_steps: int = Field(32, ge=1, le=128)
    semantic_max_tokens: int = Field(9000, ge=1)
    abc_max_tokens: int = Field(4096, ge=1)
    vae_core_frames: int = Field(1024, ge=1)
    vae_halo_frames: int = Field(16, ge=0)
    output_path: Optional[str] = Field(
        None,
        description="server-side output directory; remote clients should omit it",
    )
    output_file_name: Optional[str] = Field(None, description="output file name")
    cfg_scale: Optional[float] = Field(None, description="classifier-free guidance scale")
    artifacts_dir: Optional[str] = Field(
        None, description="export plan/semantic/latent/request/config artifacts here")


def _validate_song_request(request: "Yue2AudioGenerationsRequest",
                           request_id: str) -> str:
    """Return a filename-safe song id or raise HTTP 400.

    The scheduler builds ``SongRequest`` inside a worker; validation errors
    there surface as an opaque HTTP 500.  Validate at the edge instead so a
    malformed ABC/empty output name is a 400 with an actionable message.
    """
    if request.abc is not None and not request.abc.strip():
        raise HTTPException(
            400,
            "abc was provided but is empty; omit it (null) for a full song, "
            "or send non-empty ABC text for cot=melody/full",
        )
    if request.abc is not None and request.cot == "off":
        raise HTTPException(400, "abc cannot be combined with cot=off")
    song_id = request.output_file_name or request_id
    try:
        SongRequest(
            style=request.style,
            lyrics=request.lyrics,
            cot=request.cot,
            seed=request.seed,
            abc=request.abc,
            cfg_scale=request.cfg_scale,
            id=song_id,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return song_id


def _server_output_root(server_args) -> Path:
    """Resolve the server-side directory that is safe to serve to clients."""
    raw = getattr(server_args, "output_path", None) or "outputs/"
    root = Path(os.path.expandvars(str(raw))).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _canonical_audio(server_args, request_id: str, audio_path: str) -> Path:
    """Copy the generated file to a stable, request-id-addressed path.

    The pipeline writes to ``batch.output_path`` on the *server*.  A remote
    client must not receive that absolute path as a local filename.  Keeping a
    canonical ``<request_id>.wav`` under the server output root lets a later
    ``GET .../content`` work even after the original filename is gone.
    """
    source = Path(os.path.expandvars(audio_path)).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.is_file():
        raise HTTPException(500, f"generated audio is missing: {source}")
    target = _server_output_root(server_args) / f"{request_id}.wav"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _seconds(value) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _number(value):
    return float(value) if isinstance(value, (int, float)) else None


def _timing_payload(wall_seconds: float, result, extra: dict) -> dict:
    """Client-facing timing breakdown: HTTP wall, pipeline stages, YuE2 steps."""
    metrics = getattr(result, "metrics", None)
    stage_ms = dict(getattr(metrics, "stages", None) or {})
    inference = getattr(metrics, "total_duration_s", None) if metrics else None
    yue2 = (extra or {}).get("yue2_timing") or {}
    abc = yue2.get("abc") or {}
    semantic = yue2.get("semantic") or {}
    stages = {
        "prepare": _seconds((extra or {}).get("yue2_prepare_seconds")),
        "ar": _seconds(yue2.get("total_seconds")),
        "artifact_export": _seconds(
            (extra or {}).get("yue2_artifact_seconds")),
        "nar": _seconds((extra or {}).get("yue2_nar_seconds")),
        "vae": _seconds((extra or {}).get("yue2_vae_seconds")),
    }

    # In the YuE2 AR timing, abc.seconds and semantic.seconds are cumulative
    # from the shared prefill start, not per-phase durations.  Convert them to
    # additive decode times so a client can chart prefill -> ABC -> transition
    # -> semantic -> post without double counting.
    prefill = _number(abc.get("prefill_seconds"))
    abc_cum = _number(abc.get("seconds"))
    sem_cum = _number(semantic.get("seconds"))
    transition = _number(semantic.get("transition_seconds"))
    abc_decode = None
    sem_decode = None
    ar_post = None
    if prefill is not None and abc_cum is not None:
        abc_decode = max(0.0, abc_cum - prefill)
    if abc_cum is not None and sem_cum is not None:
        sem_decode = max(0.0, sem_cum - abc_cum - (transition or 0.0))
    ar_total = _number(yue2.get("total_seconds"))
    if ar_total is not None and sem_cum is not None:
        ar_post = max(0.0, ar_total - sem_cum)

    return {
        "request_wall_seconds": _seconds(wall_seconds),
        "inference_seconds": _seconds(inference),
        "overhead_seconds": (
            _seconds(max(0.0, wall_seconds - inference))
            if inference is not None else None
        ),
        "stages": {name: value for name, value in stages.items()
                   if value is not None},
        "scheduler_stages": {
            name: _seconds(ms / 1000.0) for name, ms in stage_ms.items()
        },
        "yue2": {
            "ar_seconds": _seconds(ar_total),
            "prefill_seconds": _seconds(prefill),
            "abc_decode_seconds": _seconds(abc_decode),
            "abc_cumulative_seconds": _seconds(abc_cum),
            "abc_output_tokens": abc.get("output_tokens"),
            "abc_content_tokens": abc.get("content_tokens"),
            "abc_truncated": abc.get("truncated"),
            "semantic_decode_seconds": _seconds(sem_decode),
            "semantic_cumulative_seconds": _seconds(sem_cum),
            "semantic_transition_seconds": _seconds(transition),
            "semantic_output_tokens": semantic.get("output_tokens"),
            "semantic_content_tokens": semantic.get("content_tokens"),
            "semantic_truncated": semantic.get("truncated"),
            "ar_post_seconds": _seconds(ar_post),
            "nar_seconds": _seconds((extra or {}).get("yue2_nar_seconds")),
            "vae_seconds": _seconds((extra or {}).get("yue2_vae_seconds")),
            "audio_duration_seconds": _seconds(
                (extra or {}).get("yue2_audio_seconds")),
            "nar_fast": (extra or {}).get("yue2_nar_fast"),
        },
    }


@router.post("/generations")
@router.post("/music/generations")
async def generate_audio(
    request: Yue2AudioGenerationsRequest,
    raw_request: Request,
):
    request_started = time.perf_counter()
    server_args = get_global_server_args()
    request_id = generate_request_id()
    song_id = _validate_song_request(request, request_id)
    prompt_payload = {
        "style": request.style,
        "lyrics": request.lyrics,
        "cot": request.cot,
        "seed": request.seed,
        "abc": request.abc,
        "cfg_scale": request.cfg_scale,
        "id": song_id,
    }
    if request.artifacts_dir:
        prompt_payload["artifacts_dir"] = request.artifacts_dir
    sampling = build_sampling_params(
        request_id,
        prompt=json.dumps(prompt_payload, ensure_ascii=False),
        style=request.style,
        lyrics=request.lyrics,
        cot=request.cot,
        request_abc=request.abc,
        seed=request.seed,
        cfg_scale=request.cfg_scale,
        num_inference_steps=request.ode_steps,
        ode_steps=request.ode_steps,
        abc_max_tokens=request.abc_max_tokens,
        semantic_max_tokens=request.semantic_max_tokens,
        vae_core_frames=request.vae_core_frames,
        vae_halo_frames=request.vae_halo_frames,
        output_sample_rate=48000,
        # The client's output_path is a *server-side* path.  A remote Mac
        # cannot write here, so always use the server's configured output
        # directory; the response exposes a download URL instead.
        output_path=server_args.output_path,
        output_file_name=request.output_file_name or f"{request_id}.wav",
    )
    trace_headers = extract_trace_headers(raw_request.headers)
    batch = prepare_request(
        server_args=server_args,
        sampling_params=sampling,
        external_trace_header=trace_headers,
    )
    save_file_path_list, result = await process_generation_batch(
        async_scheduler_client, batch
    )
    if not save_file_path_list:
        raise HTTPException(status_code=500, detail="YuE2 returned no audio output")
    audio_path = save_file_path_list[0]
    served_path = _canonical_audio(server_args, request_id, audio_path)
    audio_filename = request.output_file_name or served_path.name
    if not audio_filename.lower().endswith(".wav"):
        audio_filename = f"{audio_filename}.wav"
    extra = getattr(result, "extra", None) or {}
    timings = _timing_payload(
        time.perf_counter() - request_started, result, extra
    )
    return {
        "id": request_id,
        "object": "audio.generation",
        # Server-side path (debug only).  Remote clients must use audio_url.
        "audio": str(served_path),
        "audio_url": f"/v1/audio/generations/{request_id}/content",
        "audio_filename": audio_filename,
        "sample_rate": 48000,
        "inference_time_s": timings.get("inference_seconds"),
        "timings": timings,
        "peak_memory_mb": result.peak_memory_mb,
        "yue2_timing": extra.get("yue2_timing"),
        "yue2_stage_seconds": {
            key: extra.get(key)
            for key in ("yue2_nar_seconds", "yue2_vae_seconds", "yue2_audio_seconds")
        },
        "yue2_nar_fast": extra.get("yue2_nar_fast"),
    }


@router.get("/generations/{request_id}/content")
async def download_audio(request_id: str):
    """Serve generated audio by request id (remote clients download here)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", request_id):
        raise HTTPException(400, "invalid request id")
    server_args = get_global_server_args()
    path = _server_output_root(server_args) / f"{request_id}.wav"
    if not path.is_file():
        raise HTTPException(404, f"audio for request {request_id} not found")
    return FileResponse(
        str(path), media_type="audio/wav", filename=f"{request_id}.wav"
    )
