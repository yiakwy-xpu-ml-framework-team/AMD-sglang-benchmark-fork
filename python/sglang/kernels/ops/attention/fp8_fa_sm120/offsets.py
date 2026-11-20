# SPDX-License-Identifier: Apache-2.0
"""H64 rotation and K-only offset helpers for the SM120a FP8 reference.

The rotated FP8 recipe is the same as ``rot_nvfp4`` on the Q/K side:

    q' = H64(q),    k' = H64(k) - mean_{b,s} H64(k)

V is not rotated.  Only K is offset: ``(Rq) . (Rk - ok)`` differs from
``(Rq) . (Rk)`` by a per-query row constant, which softmax removes.  A Q offset
is *not* applied because it adds a per-key bias and needs a per-key ``corr``
epilogue that this kernel does not have.

Offsets are per layer, per head and per channel, fp32 ``[H, 128]``.  The cache
format is intentionally K-only; do not reuse ``rot_nvfp4``'s Q+K files.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch

HEAD_DIM = 128
_H64_BLOCK = 64


# ---------------------------------------------------------------------------
# H64 rotation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def h64_matrix(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Two 64x64 Sylvester blocks, normalized by 1/8."""
    block = torch.ones(1, 1, device=device, dtype=dtype)
    while block.shape[0] < _H64_BLOCK:
        block = torch.cat(
            (torch.cat((block, block), dim=1), torch.cat((block, -block), dim=1)),
            dim=0,
        )
    return torch.block_diag(block / 8.0, block / 8.0)


def rotate_h64(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last, 128-wide dimension of ``x`` in ``x``'s dtype."""
    if x.shape[-1] != HEAD_DIM:
        raise ValueError(f"rotate_h64 expects last dim {HEAD_DIM}, got {x.shape[-1]}")
    return x @ h64_matrix(x.device, x.dtype)


def calibrate_k_offset(k: torch.Tensor) -> torch.Tensor:
    """Per-layer rotated-domain K offset, fp32 ``[H, 128]``.

    ``k`` is ``[S, H, 128]`` or ``[B, S, H, 128]``.  H64 is linear, so this
    reduces first and rotates the tiny ``[H, 128]`` mean instead of rotating the
    whole sequence.
    """
    if k.ndim == 4:
        mean = k.mean(dim=(0, 1), dtype=torch.float32)
    elif k.ndim == 3:
        mean = k.mean(dim=0, dtype=torch.float32)
    else:
        raise ValueError(f"expected [S,H,128] or [B,S,H,128], got {tuple(k.shape)}")
    return rotate_h64(mean).contiguous()


def validate_k_offset(
    offset: torch.Tensor,
    device: torch.device,
    heads: int | None = None,
) -> torch.Tensor:
    """Validate the ``[H, 128]`` fp32 K offset consumed by the prep path."""
    if offset.device != device:
        raise ValueError(
            f"k_offset must be on {device}, got {offset.device}"
        )
    if offset.dtype != torch.float32:
        raise ValueError(f"k_offset must be fp32, got {offset.dtype}")
    if offset.ndim != 2 or offset.shape[-1] != HEAD_DIM:
        raise ValueError(f"k_offset must be [H,{HEAD_DIM}], got {tuple(offset.shape)}")
    if heads is not None and offset.shape[0] != heads:
        raise ValueError(
            f"k_offset has {offset.shape[0]} heads, expected {heads}"
        )
    return offset.contiguous()


# ---------------------------------------------------------------------------
# Per-prefix cache
# ---------------------------------------------------------------------------


def _safe_prefix(prefix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "attention"


def _prefix_file(path: Path, prefix: str) -> Path:
    return path / f"{_safe_prefix(prefix)}.pt"


def save_offsets(
    path: str | Path,
    offsets: dict[str, torch.Tensor],
) -> None:
    """Persist fp32 K-only offsets keyed by attention prefix."""
    target = Path(path).expanduser()
    payload = {
        prefix: {"k_offset": offset.detach().cpu()}
        for prefix, offset in offsets.items()
    }

    if target.suffix == ".pt":
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        return

    target.mkdir(parents=True, exist_ok=True)
    for prefix, entry in payload.items():
        torch.save(entry, _prefix_file(target, prefix))


def load_offsets(
    path: str | Path,
    prefix: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Load one fp32 ``[H,128]`` K offset, or ``None`` when absent."""
    target = Path(path).expanduser()
    entry = None

    if target.is_dir():
        candidate = _prefix_file(target, prefix)
        if candidate.is_file():
            entry = torch.load(candidate, map_location="cpu", weights_only=True)
    elif target.is_file():
        payload = torch.load(target, map_location="cpu", weights_only=True)
        entry = payload.get(prefix) if isinstance(payload, dict) else None

    if entry is None or "k_offset" not in entry:
        return None
    return entry["k_offset"].to(device=device, dtype=torch.float32).contiguous()


__all__ = [
    "HEAD_DIM",
    "calibrate_k_offset",
    "h64_matrix",
    "load_offsets",
    "rotate_h64",
    "save_offsets",
    "validate_k_offset",
]
