# SPDX-License-Identifier: Apache-2.0
"""Rotated FP8 attention backend (Hopper / SM90), FP8 QKV with an FP8 KV path.

Opt in with ``--attention-backend rot_fp8`` (or
``--component-attention-backends transformer=rot_fp8``).

Per call the backend

1. rotates Q and K by an orthogonal block-H64 Hadamard transform and quantizes
   Q, K and V to E4M3 with one scale per (batch, head) -- the rotation makes the
   quantizer robust to the activation-outlier channels that dominate DiT
   attention, so a single scale per head is enough (see
   ``sglang/kernels/ops/attention/rot_fp8_quant.py``);
2. runs the same FlashAttention-3 fp8 kernel the FA backend uses, but reading
   E4M3 K/V instead of BF16 -- i.e. a real FP8 KV path: half the K/V bytes of
   the BF16 call and 2x the tensor-core rate.

Because ``R R^T = I`` and Q/K share ``R``, the scores and the output are in the
original basis; no inverse rotation or rescale pass is needed. Any call the
kernel cannot serve (non-128 head dim, non-SM90, GQA/MQA, non-BF16/FP16 input,
head-group pipelining) is delegated to the dense FA implementation.

Measured on H800 with MiniMax-H3 shapes (MHA, head_dim 128, non-causal):
attention-only 1.25-1.35x over the BF16 FA3 kernel at S >= 8k, 2.3x on
outlier-heavy activations, with a per-call relative L2 of ~5e-2 against the BF16
kernel (single-side E4M3 floor).
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.ops.attention.rot_fp8_quant import (
    has_cuda_quantizer,
    quantize_q_k_v,
    quantizer_source,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
    cu_seqlens_on_device,
    fa_ver,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

SUPPORTED_HEAD_SIZES = [128]
SUPPORTED_CAPABILITIES = {(9, 0)}
_DUMPED = []

class RotFp8AttentionBackend(AttentionBackend):
    """FA3 fp8 attention over rotated, per-head-quantized Q/K/V."""

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return SUPPORTED_HEAD_SIZES

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.ROT_FP8

    @staticmethod
    def get_impl_cls() -> type["RotFp8AttentionImpl"]:
        return RotFp8AttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError


class RotFp8AttentionImpl(AttentionImpl):
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
        self.fallback = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self._reported = False
        logger.info_once(
            "rot_fp8 attention: per-head block-H64 rotation + E4M3 Q/K/V "
            "(FP8 KV) over the FA%d kernel [quantizer=%s, head_dim=%d, "
            "heads=%d, causal=%s]",
            fa_ver,
            quantizer_source(),
            head_size,
            num_heads,
            causal,
        )


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
            return "device is not SM90"
        if fa_ver != 3:
            return f"FA{fa_ver} (fp8 descales need the FA3 kernel)"
        return None

    def _report(self, reason: str) -> None:
        if not self._reported:
            logger.warning_once(
                "rot_fp8 attention: %s; using the dense FA backend for these calls.",
                reason,
            )
            self._reported = True


    def _run_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """[T, H, 128] packed tensors -> [T, H, 128] BF16 output."""
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_varlen_func,
        )

        h = query.shape[1]
        d = query.shape[2]
        t = query.shape[0]
        q = query.reshape(1, t, h, d).contiguous()
        k = key.reshape(1, t, h, d).contiguous()
        v = value.reshape(1, t, h, d).contiguous()
        q8, sq, k8, sk, v8, sv = quantize_q_k_v(q, k, v)
        output = flash_attn_varlen_func(
            q8.view(t, h, d),
            k8.view(t, h, d),
            v8.view(t, h, d),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            q_descale=sq,
            k_descale=sk,
            v_descale=sv,
            ver=fa_ver,
        )
        if isinstance(output, tuple):
            output = output[0]

        return output


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """Dense BSHD [B, S, H, D] or packed [T, H, D] call."""
        if query.dim() == 3:  # packed single document
            cu = cu_seqlens_on_device((0, query.shape[0]), query.device)
            return self._run_varlen(query, key, value, cu, query.shape[0])

        reason = self._unsupported_reason(query[0], key[0])
        if reason is not None:
            self._report(reason)
            return self.fallback.forward(query, key, value, attn_metadata)

        b, s = query.shape[0], query.shape[1]
        if b == 1:
            cu = cu_seqlens_on_device((0, s), query.device)
            return self._run_varlen(
                query[0], key[0], value[0], cu, s
            ).unsqueeze(0)

        output = torch.empty_like(query)
        for index in range(b):
            cu = cu_seqlens_on_device((0, s), query.device)
            output[index] = self._run_varlen(
                query[index], key[index], value[index], cu, s
            )
        return output

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
        reason = self._unsupported_reason(query, key)
        if reason is not None:
            self._report(reason)
            return self.fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        # MiniMax-H3 packs one live document as (0, used, total): the tail is
        # alignment padding that downstream code masks, so it never needs the
        # kernel run over it.
        used = (
            trailing_padding_used_len(
                total_tokens=query.shape[0],
                max_seqlen=max_seqlen,
                bounds=bounds,
            )
            if self.packed_trailing_padding
            else None
        )
        if used is not None:
            live_cu = cu_seqlens[:2]
            live = self._run_varlen(query[:used], key[:used], value[:used], live_cu, used)
            if used == query.shape[0]:
                return live
            output = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
            output[:used] = live
            output[used:] = 0
            return output

        if len(bounds) == 2 and bounds[0] == 0:
            return self._run_varlen(
                query, key, value, cu_seqlens, max_seqlen
            )

        output = query.new_empty(query.shape[0], query.shape[1], value.shape[2])
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            segment_cu = cu_seqlens_on_device((0, stop - start), query.device)
            output[start:stop] = self._run_varlen(
                query[start:stop],
                key[start:stop],
                value[start:stop],
                segment_cu,
                stop - start,
            )
        return output
