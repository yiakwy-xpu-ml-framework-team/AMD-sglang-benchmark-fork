# SPDX-License-Identifier: Apache-2.0
"""Rotated NVFP4 attention backend (Blackwell sm_120/121a).

Opt in with ``--attention-backend rot_nvfp4``. Q/K use the same block-H64
rotation as ``rot_fp8_attn``, are centered by calibrated per-layer offsets, and
are consumed as NVFP4 with per-16-element E4M3 scales by the Blackwell MMA
kernel. V is quantized without rotation or offset.

Unsupported calls
  - head_dim != 128,
  - GQA/MQA,
  - non-BF16/FP16,
  - non-SM120/121
fall back to cuDNN SDPA. All knobs are read once by
``rot_nvfp4_quant.nvfp4_env``.

Measured on GB10 (S=70339): 883 ms vs 1573 ms for cuDNN SDPA (1.78x).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.rot_nvfp4_quant import (
    calibrate_offsets,
    current_epoch,
    has_nvfp4_kernel,
    load_offsets,
    nvfp4_attention,
    nvfp4_env,
    nvfp4_kernel_source,
    save_offsets,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.rot_fp8_attn import (
    RotFp8AttentionBackend,
    RotFp8AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    CudnnSDPAImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SUPPORTED_CAPABILITIES = {(12, 0), (12, 1)}
_DIT_BLOCK = re.compile(r"^blocks\.(\d+)\.attn(?:\.impl)?$")


def _layer_index(prefix: str) -> int | None:
    match = _DIT_BLOCK.match(prefix)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class LayerSet:
    """DiT block selection; ``all`` matches every block, including the refiner."""

    all: bool = False
    layers: frozenset[int] = frozenset()

    def contains(self, index: int | None) -> bool:
        return self.all or (index is not None and index in self.layers)

    def __bool__(self) -> bool:
        return self.all or bool(self.layers)


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


def _selected_layers(mode: str, total_layers: int | None = None) -> LayerSet:
    """Parse ``SGLANG_ROT_NVFP4_MIX`` into a layer predicate."""
    mode = mode.strip().lower()
    if mode in ("", "all"):
        return LayerSet(all=True)
    if mode == "none":
        return LayerSet()

    head, _, arg = mode.partition(":")
    if head == "layers":
        layers = _indices(arg)
        if not layers:
            raise ValueError(f"SGLANG_ROT_NVFP4_MIX={mode!r}: empty layer list")
        return LayerSet(layers=layers)
    if head not in ("last", "every"):
        raise ValueError(
            "SGLANG_ROT_NVFP4_MIX must be all, none, layers:..., last:N, or "
            f"every:N; got {mode!r}"
        )

    if total_layers is None:
        raise ValueError(
            f"SGLANG_ROT_NVFP4_MIX={mode!r} needs the DiT depth; use all, "
            "none, or layers:..., or set SGLANG_ROT_NVFP4_TOTAL_LAYERS."
        )

    count = int(arg)
    if head == "last":
        if not 0 <= count <= total_layers:
            raise ValueError(
                f"SGLANG_ROT_NVFP4_MIX={mode!r}: last count must be in "
                f"[0,{total_layers}]"
            )
        return LayerSet(layers=frozenset(range(total_layers - count, total_layers)))
    if count < 1:
        raise ValueError(f"SGLANG_ROT_NVFP4_MIX={mode!r}: stride must be >= 1")
    if count == 1:
        return LayerSet(all=True)
    return LayerSet(layers=frozenset(range(0, total_layers, count)))


def _sensitive_layers(value: str) -> LayerSet:
    """Parse ``SGLANG_ROT_NVFP4_SENSITIVE_LAYERS``: none, all, or indices."""
    value = value.strip().lower()
    if not value or value == "none":
        return LayerSet()
    if value == "all":
        return LayerSet(all=True)
    return LayerSet(layers=_indices(value))


class RotNvfp4AttentionBackend(RotFp8AttentionBackend):
    """mma.nvfp4 attention over rotated, offset-centered Q/K/V."""

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.ROT_NVFP4

    @staticmethod
    def get_impl_cls() -> type["RotNvfp4AttentionImpl"]:
        return RotNvfp4AttentionImpl


class RotNvfp4AttentionImpl(RotFp8AttentionImpl):
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
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
        self.fallback = CudnnSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self._env = nvfp4_env()
        self._other = self._build_mix_backend(extra_impl_args)
        self._offsets = None
        self._offsets_path = self._env.offsets_path
        self._save_offsets = self._env.save_offsets
        self._audit = self._env.audit
        self._audit_min_seq = self._env.audit_min_seq
        self._audited = False
        self._reported = False

        index = _layer_index(prefix)
        total_layers = extra_impl_args.get("total_layers") or self._env.total_layers
        self._mix = _selected_layers(self._env.mix, total_layers)
        self._use_nvfp4 = self._mix.contains(index)

        self._calib_mode = self._env.calib_mode
        self._sensitive = _sensitive_layers(self._env.sensitive_layers)
        if self._calib_mode == "online":
            self._online_calib = True
        elif self._calib_mode == "hybrid":
            self._online_calib = (
                index is not None and self._sensitive.contains(index)
            )
        else:
            self._online_calib = False
        self._epoch = -1

        logger.info_once(
            "rot_nvfp4 attention: per-layer H64 calibration + mma.nvfp4 Q/K/V "
            "[kernel=%s, head_dim=%d, heads=%d, causal=%s, mix=%s, "
            "calib=%s, sensitive=%s, online=%s, backend=%s]",
            nvfp4_kernel_source() if has_nvfp4_kernel() else "unavailable",
            head_size,
            num_heads,
            causal,
            self._env.mix,
            self._calib_mode,
            (
                "all"
                if self._sensitive.all
                else sorted(self._sensitive.layers) or "none"
            ),
            "yes" if self._online_calib else "no",
            type(self._other).__name__,
        )

    def _build_mix_backend(self, extra_impl_args: dict):
        if self._env.mix_backend == "fp8":
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.rot_fp8_attn_sm120a_ref import (
                    FP8FlashAttentionSM120Impl,
                )

                return FP8FlashAttentionSM120Impl(
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    causal=self.causal,
                    softmax_scale=self.softmax_scale,
                    num_kv_heads=self.num_kv_heads,
                    prefix=self.prefix,
                    **extra_impl_args,
                )
            except ImportError as error:
                logger.warning_once(
                    "rot_nvfp4 mix backend=fp8 is unavailable (%s); "
                    "non-NVFP4 layers use exact cuDNN SDPA.",
                    error,
                )
        return self.fallback

    def _unsupported_reason(self, query: torch.Tensor, key: torch.Tensor) -> str | None:
        if self.head_size != 128:
            return f"head_dim {self.head_size} (kernel is head_dim 128)"
        if self.num_heads != self.num_kv_heads:
            return "GQA/MQA"
        if key.shape[-2] != query.shape[-2] or key.shape[-1] != query.shape[-1]:
            return "GQA/MQA at call time (q heads != kv heads)"
        if query.dtype not in (torch.bfloat16, torch.float16):
            return f"dtype {query.dtype} (kernel takes BF16/FP16)"
        if not query.is_cuda:
            return "non-CUDA tensors"
        if torch.cuda.get_device_capability(query.device) not in SUPPORTED_CAPABILITIES:
            return "device is not consumer Blackwell"
        return None

    def _report(self, reason: str) -> None:
        if not self._reported:
            logger.warning_once(
                "rot_nvfp4 attention: %s; using the dense fallback for these calls.",
                reason,
            )
            self._reported = True

    def _ensure_offsets(self, query: torch.Tensor, key: torch.Tensor):
        epoch = current_epoch()
        if self._offsets is not None:
            if not self._online_calib or self._epoch == epoch:
                return self._offsets

        if self._online_calib:
            if self._env.calib_debug:
                logger.info(
                    "rot_nvfp4 online calib %s epoch=%d shape=%s",
                    self.prefix,
                    epoch,
                    tuple(query.shape),
                )
            self._offsets = calibrate_offsets(query, key)
            self._epoch = epoch
            return self._offsets

        if self._offsets_path:
            loaded = load_offsets(self._offsets_path, self.prefix, query.device)
            if loaded is not None:
                self._offsets = loaded
                self._epoch = epoch
                return self._offsets
        self._offsets = calibrate_offsets(query, key)
        if self._offsets_path and self._save_offsets:
            save_offsets(
                self._offsets_path,
                {self.prefix: (self._offsets[0], self._offsets[1])},
            )
        self._epoch = epoch
        return self._offsets

    def _run_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """One dense call over a single contiguous [T, H, 128] segment.

        forward_varlen splits packed documents and slices the alignment
        tail before calling here.  The NVFP4 kernel has no varlen or mask, so
        B=1 is deliberate: a second segment must not become a second batch row
        or it would attend across documents.
        """
        if not self._use_nvfp4:
            return self._other.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=(0, query.shape[0]),
            )
        if cu_seqlens.numel() != 2:
            raise ValueError(
                "rot_nvfp4's dense kernel takes one contiguous segment per call "
                f"(got {cu_seqlens.numel() - 1}); forward_varlen must split "
                "multi-document packed batches"
            )
        t, h, d = query.shape

        # NOTE (yiakwy) : forward_varlen only process 1xt segments per step
        # TODO (yiakwy) : enable multiple segments processed simutaneously
        q = query.reshape(1, t, h, d).contiguous()
        k = key.reshape(1, t, h, d).contiguous()
        v = value.reshape(1, t, h, d).contiguous()
        output = nvfp4_attention(
            q,
            k,
            v,
            self._ensure_offsets(q, k),
            softmax_scale=self.softmax_scale,
            is_causal=self.causal,
        )[0]
        if self._audit and not self._audited and t >= self._audit_min_seq:
            self._audited = True
            try:
                exact = self.fallback.forward_varlen(
                    query,
                    key,
                    value,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    cu_seqlens_host=(0, t),
                )
                relative = (
                    (output.float() - exact.float()).norm()
                    / exact.float().norm().clamp_min(1e-12)
                ).item()
                logger.info(
                    "rot_nvfp4 audit[%s]: S=%d relL2=%.4f",
                    self.prefix,
                    t,
                    relative,
                )
            except Exception as error:  # diagnostics must not break generation
                logger.warning("rot_nvfp4 audit[%s] failed: %r", self.prefix, error)
        return output
