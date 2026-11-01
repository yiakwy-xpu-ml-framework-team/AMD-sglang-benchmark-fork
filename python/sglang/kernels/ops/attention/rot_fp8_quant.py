# SPDX-License-Identifier: Apache-2.0
"""Rotated per-head E4M3 quantizer for FP8 KV / FP8 QKV attention.

This is the preprocessing half of the "rotated FP8 attention" path that ships
in ``Rotate_fp4_attn_qat`` (``legacy/quantization/fa3_f8_head_quant.cu``): the
FA3-SOL recipe for Hopper.

What it does, per (token, head) 128-wide vector:

1. rotate the 128 channels by an orthogonal block-H64 Hadamard transform
   (``R = blkdiag(H64, H64) / 8``). Q and K share the same ``R``, so the score
   matrix is unchanged in exact arithmetic (``RR^T = I``) while activation
   outliers, which live in a handful of channels, are spread over 64 channels
   before the E4M3 cast.
2. take one amax per (batch, head) over the whole sequence -- the scale unit the
   FA3 fp8 kernel consumes as ``q/k/v_descale`` of shape ``[B, H]``.
3. quantize to E4M3 with ``448 / amax``.

V is quantized *without* the rotation: ``O = P V`` is invariant to a rotation of
V as well (it cancels against the inverse basis of the output), so the output of
the attention kernel already lives in the un-rotated basis and no inverse
transform pass is needed.

The CUDA kernel is either an already-built ``fp8_head_quant_cuda`` extension or
a JIT build of the vendored source (compiled once, cached by torch). A pure
torch implementation is kept as the last resort so the backend still runs on
machines without nvcc; it is numerically equivalent but slower.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

_HEAD_DIM = 128
_H64_BLOCK = 64
_SRC = Path(__file__).resolve().parent / "rot_fp8_quant" / "fa3_f8_head_quant.cu"


# ---------------------------------------------------------------------------
# kernel discovery
# ---------------------------------------------------------------------------


def _import_prebuilt():
    """An already-built ``fp8_head_quant_cuda`` (repo .so or pip package)."""
    try:
        import fp8_head_quant_cuda as lib  # noqa: F401

        return lib
    except Exception:
        return None


def _jit_build():
    """Compile the vendored kernel once; torch caches it per source hash."""
    if not _SRC.exists() or os.environ.get("SGLANG_ROT_FP8_DISABLE_JIT") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import load

    return load(
        name="fp8_head_quant_cuda",
        sources=[str(_SRC)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
        ],
        extra_include_paths=[str(_SRC.parent)],
        verbose=False,
    )


@lru_cache(maxsize=1)
def _lib():
    lib = _import_prebuilt()
    if lib is not None:
        return lib
    try:
        return _jit_build()
    except Exception:
        return None


def has_cuda_quantizer() -> bool:
    return _lib() is not None


def quantizer_source() -> str:
    return "cuda" if has_cuda_quantizer() else "torch"


# ---------------------------------------------------------------------------
# torch fallback
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _h64(device, dtype=torch.float32) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < _H64_BLOCK:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return torch.block_diag(h / 8.0, h / 8.0)


def _quant_torch(x: torch.Tensor, rotate: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B,S,H,128] -> (e4m3 [B,S,H,128], fp32 [B,H] descale)."""
    b, s, h, d = x.shape
    xf = x.float()
    if rotate:
        xf = (xf.reshape(-1, d) @ _h64(x.device)).reshape(b, s, h, d)
    amax = xf.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    scale = amax / 448.0
    q8 = (xf / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q8, scale.contiguous()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def quantize_q_k_v(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotated per-head E4M3 quantization of a BSHD Q/K/V triple.

    ``q``/``k``/``v``: contiguous BF16 or FP16 ``[B, S, H, 128]``.
    Returns ``(q8, q_scale, k8, k_scale, v8, v_scale)`` where the ``*8`` tensors
    are E4M3 and the scales are FP32 ``[B, H]`` descales (amax / 448).
    """
    lib = _lib()
    if lib is not None:
        empty = torch.empty(0, device=q.device, dtype=torch.float32)
        q8, sq = lib.preprocess_q_head_fp8_cuda(q, empty)
        k8, sk, _corr = lib.preprocess_k_head_fp8_cuda(k, empty, empty)
        v8, sv = lib.preprocess_v_head_fp8_cuda_norot(v)
        return (
            q8.view(torch.float8_e4m3fn),
            sq,
            k8.view(torch.float8_e4m3fn),
            sk,
            v8.view(torch.float8_e4m3fn),
            sv,
        )
    q8, sq = _quant_torch(q, rotate=True)
    k8, sk = _quant_torch(k, rotate=True)
    v8, sv = _quant_torch(v, rotate=False)
    return q8, sq, k8, sk, v8, sv


__all__ = [
    "has_cuda_quantizer",
    "quantize_q_k_v",
    "quantizer_source",
]
