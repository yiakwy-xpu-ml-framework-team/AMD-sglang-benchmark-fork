# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, yiakwy@gmail.com for rot attention
"""SM120 FP8 attention: CuTe-DSL kernel, fused Triton prep and the per-call entry point."""

from sglang.kernels.ops.attention.fp8_fa_sm120.offsets import (
    calibrate_k_offset,
    load_offsets,
    rotate_h64,
    save_offsets,
)
from sglang.kernels.ops.attention.fp8_fa_sm120.plan import (
    HEAD_DIM,
    Workspace,
    fp8_attention,
    kernel_key,
    validate_inputs,
)

__all__ = [
    "HEAD_DIM",
    "Workspace",
    "calibrate_k_offset",
    "fp8_attention",
    "kernel_key",
    "load_offsets",
    "rotate_h64",
    "save_offsets",
    "validate_inputs",
]
