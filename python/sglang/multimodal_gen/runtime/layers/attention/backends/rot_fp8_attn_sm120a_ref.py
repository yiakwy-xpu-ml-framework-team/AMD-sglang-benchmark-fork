# SPDX-License-Identifier: Apache-2.0
"""Rotated FP8 SM120a reference attention backend.

Q/K are rotated by block-H64 and K is centered by a calibrated per-layer
``mean_{b,s} H64(K)`` offset before per-head E4M3 quantization.  V is neither
rotated nor offset.  This mirrors ``rot_nvfp4``'s Q/K calibration on the K side,
but keeps E4M3 values and per-head scales; it is not an E2M1/block-scaled
NVFP4 kernel.

A Q offset is deliberately not applied: it adds a per-key bias and needs a
per-key ``corr`` epilogue that this CuTe-DSL kernel does not have.

Opt in with ``--attention-backend fp8_fa_sm120``. Q/K/V are quantized per head
to E4M3 on the fly (two fused Triton passes), attention runs in FP8 with FP32
accumulation, and the output is BF16. Expect about 5% relative RMS against cuDNN
BF16 per call on normally distributed inputs.

Scope: dense, non-causal, batch 1, head_dim 128, BF16 inputs on an SM120 device.
Everything else goes to cuDNN SDPA. Each distinct sequence length compiles once
(about 10 s); later calls with the same shape reuse the compiled kernel. The FP8 and
output buffers are allocated per call and belong to the caller.

Calibration knobs:

``SGLANG_ROT_FP8_SM120A_ROTATE``         1 (default) or 0 for the plain baseline
``SGLANG_ROT_FP8_SM120A_CALIB_MODE``     offline (default), online, or hybrid
``SGLANG_ROT_FP8_SM120A_OFFSETS``        per-prefix fp32 K-offset cache
``SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS``   1 writes the cache on first calibration
``SGLANG_ROT_FP8_SM120A_SENSITIVE_LAYERS`` hybrid-mode online layer list
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import torch

from sglang.kernels.ops.attention.fp8_fa_sm120 import (
    HEAD_DIM,
    calibrate_k_offset,
    fp8_attention,
    load_offsets,
    save_offsets,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import CudnnSDPAImpl
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DIT_BLOCK = re.compile(r"^blocks\.(\d+)\.attn(?:\.impl)?$")


# TODO(yiakwy): K-only is implemented.  Q offset remains unsupported: it needs
# a per-key ``corr`` added after QK and masking, i.e. a kernel epilogue change.
# Full NVFP4 (E2M1 + per-16 block scales) is a different kernel; use
# ``rot_nvfp4`` for that.  Runtime H64 currently runs in ``plan.py``; fold it
# into ``fused_prep`` or the Q/K projection weights if the prep cost shows up.


@dataclass(frozen=True)
class Sm120aFp8Env:
    """All ``SGLANG_ROT_FP8_SM120A_*`` knobs, read once per process."""

    rotate: bool
    calib_mode: str
    offsets_path: str | None
    save_offsets: bool
    sensitive_layers: str


@lru_cache(maxsize=1)
def sm120a_fp8_env() -> Sm120aFp8Env:
    calib_mode = (
        os.environ.get("SGLANG_ROT_FP8_SM120A_CALIB_MODE", "offline")
        .strip()
        .lower()
    )
    if calib_mode not in ("offline", "online", "hybrid"):
        raise ValueError(
            "SGLANG_ROT_FP8_SM120A_CALIB_MODE must be offline, online, or "
            f"hybrid; got {calib_mode!r}"
        )
    return Sm120aFp8Env(
        rotate=os.environ.get("SGLANG_ROT_FP8_SM120A_ROTATE", "1") != "0",
        calib_mode=calib_mode,
        offsets_path=os.environ.get("SGLANG_ROT_FP8_SM120A_OFFSETS") or None,
        save_offsets=os.environ.get("SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS") == "1",
        sensitive_layers=(
            os.environ.get("SGLANG_ROT_FP8_SM120A_SENSITIVE_LAYERS", "")
            .strip()
            .lower()
        ),
    )


def _layer_index(prefix: str) -> int | None:
    match = _DIT_BLOCK.match(prefix)
    return int(match.group(1)) if match else None


def _indices(text: str) -> frozenset[int]:
    indices: set[int] = set()
    for token in text.replace(",", " ").split():
        if "-" in token:
            start, end = token.split("-", 1)
            low, high = int(start), int(end)
            if low < 0 or high < low:
                raise ValueError(f"invalid layer range {token!r}")
            indices.update(range(low, high + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"invalid layer index {token!r}")
            indices.add(value)
    return frozenset(indices)


def _sensitive_layers(value: str) -> frozenset[int] | None:
    """``None`` means every layer; an empty set means no layer."""
    value = value.strip().lower()
    if not value or value == "none":
        return frozenset()
    if value == "all":
        return None
    return _indices(value)


def _request_epoch() -> int:
    try:
        from sglang.kernels.ops.attention.rot_nvfp4_quant import current_epoch

        return current_epoch()
    except Exception:
        return 0


class FP8FlashAttentionSM120Backend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FP8_FA_SM120

    @staticmethod
    def get_impl_cls() -> type["FP8FlashAttentionSM120Impl"]:
        return FP8FlashAttentionSM120Impl


class FP8FlashAttentionSM120Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self._env = sm120a_fp8_env()
        self._sensitive = _sensitive_layers(self._env.sensitive_layers)
        index = _layer_index(prefix)
        if self._env.calib_mode == "online":
            self._online = True
        elif self._env.calib_mode == "hybrid":
            self._online = self._sensitive is None or (
                index is not None and index in self._sensitive
            )
        else:
            self._online = False
        self._k_offset: torch.Tensor | None = None
        self._offset_epoch = -1
        self.fallback = CudnnSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        logger.info_once(
            "rot_fp8_attn_sm120a_ref: H64=%s calib=%s online=%s "
            "offsets=%s head_dim=%d heads=%d causal=%s",
            "on" if self._env.rotate else "off",
            self._env.calib_mode,
            "yes" if self._online else "no",
            self._env.offsets_path or "memory-only",
            head_size,
            num_heads,
            causal,
        )

    # --- calibration ----------------------------------------------------------

    def _ensure_k_offset(self, key: torch.Tensor) -> torch.Tensor | None:
        """Per-layer fp32 ``[H,128]`` K offset, or ``None`` when rotation is off."""
        if not self._env.rotate:
            return None

        epoch = _request_epoch()
        if self._k_offset is not None:
            if not self._online or self._offset_epoch == epoch:
                return self._k_offset

        if self._online:
            self._k_offset = calibrate_k_offset(key)
            self._offset_epoch = epoch
            return self._k_offset

        if self._env.offsets_path:
            loaded = load_offsets(self._env.offsets_path, self.prefix, key.device)
            if loaded is not None:
                self._k_offset = loaded
                self._offset_epoch = epoch
                return self._k_offset

        self._k_offset = calibrate_k_offset(key)
        if self._env.offsets_path and self._env.save_offsets:
            save_offsets(
                self._env.offsets_path,
                {self.prefix: self._k_offset},
            )
        self._offset_epoch = epoch
        return self._k_offset

    # --- dispatch -------------------------------------------------------------

    def _fallback_reason(self, query, key, value) -> str | None:
        """None when the kernel applies to these [S, H, D] tensors, else why not."""
        if self.causal:
            return "causal attention"
        if query.shape != key.shape or key.shape != value.shape:
            return "Q/K/V shapes differ"
        if query.ndim != 3 or query.shape[-1] != HEAD_DIM:
            return f"head_dim {query.shape[-1]} (kernel is head_dim {HEAD_DIM})"
        if query.dtype != torch.bfloat16:
            return f"dtype {query.dtype} (kernel takes BF16)"
        if not query.is_cuda:
            return "non-CUDA tensors"
        if torch.cuda.get_device_capability(query.device) not in ((12, 0), (12, 1)):
            return "device is not SM120/SM121"
        return None

    def _report_fallback(self, reason: str) -> None:
        logger.warning_once(
            "fp8_fa_sm120 attention: %s; using cuDNN SDPA for these calls.", reason
        )

    # --- kernel path ----------------------------------------------------------

    def _run(self, query, key, value):
        """[S, H, 128] strided BF16 in, a new contiguous [S, H, 128] BF16 out."""
        output, _ = fp8_attention(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            k_offset=self._ensure_k_offset(key),
            rotate=self._env.rotate,
        )
        return output

    # --- AttentionImpl --------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # [B, S, H, D]; the kernel handles one dense sequence at a time.
        if query.shape[0] != 1:
            self._report_fallback(f"batch {query.shape[0]}")
            return self.fallback.forward(query, key, value, attn_metadata)
        reason = self._fallback_reason(query[0], key[0], value[0])
        if reason is not None:
            self._report_fallback(reason)
            return self.fallback.forward(query, key, value, attn_metadata)
        try:
            output = self._run(query[0], key[0], value[0])
        except ValueError as error:
            self._report_fallback(str(error))
            return self.fallback.forward(query, key, value, attn_metadata)
        return output.unsqueeze(0)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        reason = self._fallback_reason(query, key, value)
        if reason is not None:
            self._report_fallback(reason)
            return self.fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=bounds,
            )

        # MiniMax-H3 packs one live document as (0, used, total); the tail is
        # 64-aligned padding that downstream masks, so it stays zero.
        used = trailing_padding_used_len(
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
            bounds=bounds,
        )
        try:
            if used is not None:
                live_output = self._run(query[:used], key[:used], value[:used])
                if used == query.shape[0]:
                    return live_output
                output = torch.zeros_like(query)
                output[:used].copy_(live_output)
                return output

            output = torch.empty_like(query)
            for start, stop in zip(bounds[:-1], bounds[1:]):
                if start == stop:
                    continue
                segment_output = self._run(
                    query[start:stop], key[start:stop], value[start:stop]
                )
                output[start:stop].copy_(segment_output)
            return output
        except ValueError as error:
            self._report_fallback(str(error))
            return self.fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=bounds,
            )
