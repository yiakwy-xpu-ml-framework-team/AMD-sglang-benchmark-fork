# SPDX-License-Identifier: Apache-2.0
"""rot_fp8 attention backend: rotated per-head E4M3 Q/K/V over FA3.

Checks the pieces that are easy to get wrong at MiniMax-H3's call shapes:

* packed-varlen output matches the dense FA backend within the E4M3 floor,
* the trailing-padding tail is zero-initialised and never affects the live rows,
* the per-(batch, head) descale contract (``[B, H]``) reaches the kernel,
* GQA, causal and non-128 head sizes delegate to the dense FA backend.
"""

import pytest
import torch

H3_HEADS = 14
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5

# E4M3 with one scale per (batch, head) has a ~4% relative L2 floor against the
# BF16 kernel (measured 4.10-4.25e-2 on H3 shapes); a per-token scale is not
# expressible through the FA3 fp8 descale contract.
FP8_REL_L2_MAX = 6e-2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="rot_fp8 attention needs an SM90 (Hopper) GPU",
)


def _impls(num_heads=H3_HEADS, causal=False, trailing_padding=True):
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        FlashAttentionImpl,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_fp8_attn import (
        RotFp8AttentionImpl,
    )

    kwargs = dict(
        num_heads=num_heads,
        head_size=HEAD_DIM,
        causal=causal,
        softmax_scale=SOFTMAX_SCALE,
    )
    ours = RotFp8AttentionImpl(**kwargs, packed_trailing_padding=trailing_padding)
    reference = FlashAttentionImpl(**kwargs)
    return ours, reference


def _make_qkv(sequence, heads, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        sequence, 3 * heads * HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    q, k, v = qkv.split(heads * HEAD_DIM, dim=-1)
    return (
        q.view(sequence, heads, HEAD_DIM).contiguous(),
        k.view(sequence, heads, HEAD_DIM).contiguous(),
        v.view(sequence, heads, HEAD_DIM).contiguous(),
    )


def _rel_l2(candidate, reference):
    candidate = candidate.float()
    reference = reference.float()
    return ((candidate - reference).norm() / reference.norm()).item()


def _varlen(impl, q, k, v, used, total):
    cu = torch.tensor([0, used, total], dtype=torch.int32, device=q.device)
    return impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
    )


@pytest.mark.parametrize("used,total", [(4096, 4096), (4000, 4096)])
def test_matches_bf16_fa_within_fp8_floor(used, total):
    q, k, v = _make_qkv(total, H3_HEADS)
    pad = total - used
    if pad:
        q[used:] = 0
        k[used:] = 0
        v[used:] = 0

    ours, reference = _impls()
    got = _varlen(ours, q, k, v, used, total)
    want = _varlen(reference, q, k, v, used, total)

    assert got.shape == want.shape == (total, H3_HEADS, HEAD_DIM)
    assert got.dtype == want.dtype == torch.bfloat16
    assert _rel_l2(got[:used], want[:used]) < FP8_REL_L2_MAX
    if pad:
        # the padding tail is never attended and must stay exactly zero
        assert got[used:].abs().max().item() == 0.0


def test_padding_tail_does_not_change_live_rows():
    q, k, v = _make_qkv(4096, H3_HEADS)
    ours, _ = _impls()

    padded = _varlen(ours, q, k, v, 4000, 4096)
    tight = _varlen(ours, q[:4000], k[:4000], v[:4000], 4000, 4000)

    assert _rel_l2(padded[:4000], tight) == 0.0


def test_dense_batch_path_matches_varlen():
    q, k, v = _make_qkv(1024, H3_HEADS)
    ours, _ = _impls()

    dense = ours.forward(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        attn_metadata=None,
    )
    packed = _varlen(ours, q, k, v, 1024, 1024)

    assert dense.shape == (1, 1024, H3_HEADS, HEAD_DIM)
    assert _rel_l2(dense[0], packed) == 0.0


def test_quantizer_reports_per_head_descales():
    from sglang.kernels.ops.attention.rot_fp8_quant import quantize_q_k_v

    q, k, v = (t.unsqueeze(0) for t in _make_qkv(2048, H3_HEADS))
    q8, sq, k8, sk, v8, sv = quantize_q_k_v(q, k, v)

    for tensor, scale in ((q8, sq), (k8, sk), (v8, sv)):
        assert tensor.dtype == torch.float8_e4m3fn
        assert tensor.shape == q.shape
        assert scale.shape == (1, H3_HEADS)
        assert scale.dtype == torch.float32
        assert torch.isfinite(scale).all() and (scale > 0).all()


def test_gqa_and_non_128_heads_fall_back():
    ours, reference = _impls()
    q, k, v = _make_qkv(512, H3_HEADS)
    # MQA-style call that the fp8 kernel cannot serve
    got = ours.forward(
        q.unsqueeze(0).contiguous(),
        k[:, :1].contiguous().unsqueeze(0),
        v[:, :1].contiguous().unsqueeze(0),
        attn_metadata=None,
    )
    want = reference.forward(
        q.unsqueeze(0).contiguous(),
        k[:, :1].contiguous().unsqueeze(0),
        v[:, :1].contiguous().unsqueeze(0),
        attn_metadata=None,
    )
    assert torch.equal(got, want)
