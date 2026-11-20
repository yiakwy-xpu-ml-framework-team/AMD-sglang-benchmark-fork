# SPDX-License-Identifier: Apache-2.0
"""Rotated per-layer NVFP4 quantizer for sm_120/121a attention.

The NVFP4 rotated attention recipe has the same block-H64 rotation and
per layer offset as the FP8 rotated attention recipe.

Q/K/V use per-16-element E4M3 block scales and are fed to the
sm_120/121a NVFP4 MMA kernel.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


HEAD_DIM = 128

_REPO_ENV = "SGLANG_ROT_NVFP4_REPO"
_SO_DIR_ENV = "SGLANG_ROT_NVFP4_SO_DIR"


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NvFp4Env:
    """All ``SGLANG_ROT_NVFP4_*`` knobs, read once per process.

    ``MIX``: all (default), none, layers:0-27,47,49, last:N, every:N.
    ``MIX_BACKEND``: sdpa (default) or fp8 for the non-NVFP4 layers.
    ``CALIB_MODE``: offline (default), online, hybrid.
    ``SENSITIVE_LAYERS``: none (default), all, or 0-27,47,49.
    ``TOTAL_LAYERS``: only needed by last:/every: when the model does not
    pass its DiT depth to the backend; H3 passes it automatically.
    ``OFFSETS``: cache path; ``SAVE_OFFSETS``/``AUDIT``/``CALIB_DEBUG``:
    ``1`` enables; ``AUDIT_MIN_SEQ``/``BOUND``: numeric thresholds.
    """

    mix: str
    mix_backend: str
    calib_mode: str
    sensitive_layers: str
    total_layers: int | None
    offsets_path: str | None
    save_offsets: bool
    audit: bool
    audit_min_seq: int
    calib_debug: bool
    fp4_bound: float


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


@lru_cache(maxsize=1)
def nvfp4_env() -> NvFp4Env:
    calib_mode = (
        os.environ.get("SGLANG_ROT_NVFP4_CALIB_MODE", "offline").strip().lower()
    )
    if calib_mode not in ("offline", "online", "hybrid"):
        raise ValueError(
            "SGLANG_ROT_NVFP4_CALIB_MODE must be offline, online, or hybrid; "
            f"got {calib_mode!r}"
        )
    mix_backend = (
        os.environ.get("SGLANG_ROT_NVFP4_MIX_BACKEND", "sdpa").strip().lower()
    )
    if not mix_backend:
        mix_backend = "sdpa"
    if mix_backend not in ("sdpa", "fp8"):
        raise ValueError(
            "SGLANG_ROT_NVFP4_MIX_BACKEND must be 'sdpa' or 'fp8'; "
            f"got {mix_backend!r}"
        )
    total = os.environ.get("SGLANG_ROT_NVFP4_TOTAL_LAYERS", "").strip()
    return NvFp4Env(
        mix=os.environ.get("SGLANG_ROT_NVFP4_MIX", "all").strip().lower(),
        mix_backend=mix_backend,
        calib_mode=calib_mode,
        sensitive_layers=os.environ.get(
            "SGLANG_ROT_NVFP4_SENSITIVE_LAYERS", ""
        ).strip().lower(),
        total_layers=int(total) if total else None,
        offsets_path=os.environ.get("SGLANG_ROT_NVFP4_OFFSETS") or None,
        save_offsets=_env_flag("SGLANG_ROT_NVFP4_SAVE_OFFSETS"),
        audit=_env_flag("SGLANG_ROT_NVFP4_AUDIT"),
        audit_min_seq=int(os.environ.get("SGLANG_ROT_NVFP4_AUDIT_MIN_SEQ", "0")),
        calib_debug=_env_flag("SGLANG_ROT_NVFP4_CALIB_DEBUG"),
        fp4_bound=float(os.environ.get("SGLANG_ROT_NVFP4_BOUND", "4.0")),
    )


class _RequestEpoch:
    """Monotonic request counter shared by the denoising hook and the backend."""

    def __init__(self) -> None:
        self._serial = itertools.count(1)
        self._current = 0

    def bump(self) -> int:
        self._current = next(self._serial)
        return self._current

    @property
    def current(self) -> int:
        return self._current


_EPOCH = _RequestEpoch()


def begin_request() -> int:
    """Mark a new generation request; online calibration uses this boundary."""
    epoch = _EPOCH.bump()
    if nvfp4_env().calib_debug:
        logger.info("rot_nvfp4 begin_request epoch=%d", epoch)
    return epoch


def current_epoch() -> int:
    return _EPOCH.current


# ---------------------------------------------------------------------------
# kernel discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Kernel:
    """Resolved entry points of the external NVFP4 extension."""

    source: str
    sageattn_blackwell: Callable[..., torch.Tensor]
    calibrate_offsets_h64: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    make_pad_raw: Callable[[torch.Tensor], torch.Tensor]
    rotate_h64: Callable[[torch.Tensor], torch.Tensor] | None = None


def _unavailable_message(exc: BaseException) -> str:
    return (
        "NVFP4 attention kernel is unavailable: install the kernel package "
        "(`scripts/install.py` in HKUST-Rotate_fp4_attn_qat-fork) so "
        "'attn_qat_infer', 'fp4attn_cuda' and 'fp4preprocess_cuda' import from "
        "site-packages, or "
        f"set {_REPO_ENV} to that checkout and {_SO_DIR_ENV} to a matching "
        "prebuilt .so directory for the active torch. "
        f"Last error: {exc!r}"
    )


def _import_kernel() -> _Kernel:
    # SO_DIR first: a prebuilt .so directory must win over .so files that may
    # also sit in the source checkout (they target a different torch build).
    repo = os.environ.get(_REPO_ENV)
    if repo:
        sys.path.insert(0, str(Path(repo).expanduser()))
    so_dir = os.environ.get(_SO_DIR_ENV)
    if so_dir:
        sys.path.insert(0, str(Path(so_dir).expanduser()))

    import fp4attn_cuda
    import fp4preprocess_cuda  # noqa: F401
    from attn_qat_infer.api import sageattn_blackwell
    from attn_qat_infer.hopper.prefill import (
        calibrate_offsets_h64,
        rotate_h64,
    )
    from attn_qat_infer.quantization import make_pad_raw

    return _Kernel(
        source=str(Path(fp4attn_cuda.__file__).resolve().parent),
        sageattn_blackwell=sageattn_blackwell,
        calibrate_offsets_h64=calibrate_offsets_h64,
        rotate_h64=rotate_h64,
        make_pad_raw=make_pad_raw,
    )


@lru_cache(maxsize=1)
def _load() -> _Kernel | ImportError:
    """Import the extension once; a failed lookup is cached with its reason."""
    try:
        return _import_kernel()
    except Exception as exc:  # ImportError or a CUDA ABI mismatch
        return ImportError(_unavailable_message(exc))


def nvfp4_status() -> tuple[bool, str]:
    loaded = _load()
    if isinstance(loaded, ImportError):
        return False, str(loaded)
    return True, ""


def has_nvfp4_kernel() -> bool:
    return not isinstance(_load(), ImportError)


def nvfp4_kernel_source() -> str:
    loaded = _load()
    return "unavailable" if isinstance(loaded, ImportError) else loaded.source


def _api() -> _Kernel:
    loaded = _load()
    if isinstance(loaded, ImportError):
        raise loaded
    return loaded


def prepare_offsets(
    q_offset: torch.Tensor,
    k_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast calibrated rotated-domain offsets and build the raw pad tails."""
    if not q_offset.is_cuda or not k_offset.is_cuda:
        raise ValueError("NVFP4 offsets must be CUDA tensors")

    q_offset = q_offset.to(torch.float16).contiguous()
    k_offset = k_offset.to(torch.float16).contiguous()

    if q_offset.shape != k_offset.shape or q_offset.shape[-1] != HEAD_DIM:
        raise ValueError(
            "NVFP4 offsets must share shape [H,128] or [B,H,128], got "
            f"{tuple(q_offset.shape)} and {tuple(k_offset.shape)}"
        )

    make_pad_raw = _api().make_pad_raw
    return (
        q_offset,
        k_offset,
        make_pad_raw(q_offset),
        make_pad_raw(k_offset),
    )


def calibrate_offsets(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrate ``(q_offset, k_offset, q_pad_raw, k_pad_raw)`` from BSHD Q/K.

    H64 is linear, so ``mean(H64(x)) == H64(mean(x))``.  Reduce first and
    rotate the tiny ``[H,128]`` mean: this keeps online per-request
    calibration at ~8 ms / layer instead of rotating the full sequence.
    """
    kernel = _api()
    rotate = kernel.rotate_h64
    if rotate is not None:
        q_offset = rotate(q.mean(dim=(0, 1), dtype=torch.float32)).contiguous()
        k_offset = rotate(k.mean(dim=(0, 1), dtype=torch.float32)).contiguous()
    else:  # older kernel tree without rotate_h64
        q_offset, k_offset = kernel.calibrate_offsets_h64(q, k)
    return prepare_offsets(q_offset, k_offset)


def _safe_prefix(prefix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "attention"


def _prefix_file(path: Path, prefix: str) -> Path:
    return path / f"{_safe_prefix(prefix)}.pt"


def save_offsets(
    path: str | os.PathLike,
    offsets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Persist FP16 offsets keyed by attention prefix for reproducible runs."""
    target = Path(path).expanduser()
    payload = {
        prefix: {
            "q_offset": pair[0].detach().cpu(),
            "k_offset": pair[1].detach().cpu(),
        }
        for prefix, pair in offsets.items()
    }

    if target.suffix == ".pt":
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        return

    target.mkdir(parents=True, exist_ok=True)
    for prefix, entry in payload.items():
        torch.save(entry, _prefix_file(target, prefix))


def load_offsets(
    path: str | os.PathLike,
    prefix: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load persisted offsets for one prefix, or ``None`` when absent."""
    target = Path(path).expanduser()
    entry = None

    if target.is_dir():
        candidate = _prefix_file(target, prefix)
        if candidate.is_file():
            entry = torch.load(candidate, map_location="cpu", weights_only=True)
    elif target.is_file():
        payload = torch.load(target, map_location="cpu", weights_only=True)
        entry = payload.get(prefix) if isinstance(payload, dict) else None
    if entry is None or "q_offset" not in entry or "k_offset" not in entry:
        return None

    q_offset = entry["q_offset"].to(device=device, dtype=torch.float16)
    k_offset = entry["k_offset"].to(device=device, dtype=torch.float16)

    return prepare_offsets(q_offset, k_offset)


def nvfp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    softmax_scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run one dense BSHD NVFP4 attention call with calibrated offsets."""
    q_offset, k_offset, q_pad_raw, k_pad_raw = offsets
    return _api().sageattn_blackwell(
        q,
        k,
        v,
        is_causal=is_causal,
        sm_scale=softmax_scale,
        q_offset=q_offset,
        k_offset=k_offset,
        q_pad_raw=q_pad_raw,
        k_pad_raw=k_pad_raw,
        fp4_bound=nvfp4_env().fp4_bound,
        output_layout="BSHD",
        output_dtype=q.dtype,
    )


__all__ = [
    "HEAD_DIM",
    "NvFp4Env",
    "begin_request",
    "calibrate_offsets",
    "current_epoch",
    "has_nvfp4_kernel",
    "load_offsets",
    "nvfp4_attention",
    "nvfp4_env",
    "nvfp4_kernel_source",
    "nvfp4_status",
    "prepare_offsets",
    "save_offsets",
]
