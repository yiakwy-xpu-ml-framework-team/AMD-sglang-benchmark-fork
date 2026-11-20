# SPDX-License-Identifier: Apache-2.0
"""rot_nvfp4 attention backend: Blackwell NVFP4 Q/K/V with H64 calibration.

Covers the contract that is easy to break at MiniMax-H3's call shapes:

* packed-varlen live rows match the exact cuDNN SDPA path within the NVFP4
  floor, and the trailing alignment tail stays zero;
* the padding tail cannot change the live rows;
* the BSHD batch path matches the packed path;
* unsupported calls (GQA / non-128 head dim) delegate to dense SDPA;
* ``SGLANG_ROT_NVFP4_MIX=none`` is byte-identical to dense SDPA;
* calibrated offsets survive a save/load round-trip exactly.
"""

import os

import pytest
import torch

H3_HEADS = 56
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5

# Measured 0.198 relative L2 for isotropic Gaussian Q/K/V at H3 shapes; real
# post-projection activations are better conditioned after H64 rotation.
NVFP4_REL_L2_MAX = 0.25


def _nvfp4_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        return False
    from sglang.kernels.ops.attention.rot_nvfp4_quant import has_nvfp4_kernel

    return has_nvfp4_kernel()


pytestmark = pytest.mark.skipif(
    not _nvfp4_available(),
    reason="rot_nvfp4 attention needs SM120/SM121 and the HKUST NVFP4 kernel",
)


@pytest.fixture(autouse=True)
def _fresh_nvfp4_env():
    from sglang.kernels.ops.attention.rot_nvfp4_quant import nvfp4_env

    nvfp4_env.cache_clear()
    yield
    nvfp4_env.cache_clear()


def _impls(num_heads=H3_HEADS):
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        RotNvfp4AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
        CudnnSDPAImpl,
    )

    kwargs = dict(
        num_heads=num_heads,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SOFTMAX_SCALE,
        num_kv_heads=num_heads,
        packed_trailing_padding=True,
    )
    return RotNvfp4AttentionImpl(prefix="blocks.0.attn", **kwargs), CudnnSDPAImpl(
        prefix="blocks.0.attn", **kwargs
    )


def _make_qkv(sequence, heads=H3_HEADS, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sequence, heads, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    k = torch.randn(
        sequence, heads, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    v = torch.randn(
        sequence, heads, HEAD_DIM, device="cuda", dtype=dtype, generator=generator
    )
    return q, k, v


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


@pytest.mark.parametrize("used,total", [(2048, 2048), (2000, 2048)])
def test_matches_exact_sdpa_within_nvfp4_floor(used, total):
    q, k, v = _make_qkv(total)
    if used < total:
        q[used:] = 0
        k[used:] = 0
        v[used:] = 0

    ours, reference = _impls()
    got = _varlen(ours, q, k, v, used, total)
    want = _varlen(reference, q, k, v, used, total)

    assert got.shape == want.shape == (total, H3_HEADS, HEAD_DIM)
    assert got.dtype == want.dtype == torch.bfloat16
    assert _rel_l2(got[:used], want[:used]) < NVFP4_REL_L2_MAX
    if used < total:
        assert got[used:].abs().max().item() == 0.0


def test_dense_nvfp4_rejects_multi_segment_varlen():
    q, k, v = _make_qkv(512)
    ours, _ = _impls()
    cu = torch.tensor([0, 256, 512], dtype=torch.int32, device=q.device)

    with pytest.raises(ValueError, match="one contiguous segment"):
        ours._run_varlen(q, k, v, cu, 256)


def test_padding_tail_does_not_change_live_rows():
    q, k, v = _make_qkv(2048)
    ours, _ = _impls()

    padded = _varlen(ours, q, k, v, 2000, 2048)
    tight = _varlen(ours, q[:2000], k[:2000], v[:2000], 2000, 2000)

    assert _rel_l2(padded[:2000], tight) == 0.0


def test_dense_batch_path_matches_varlen():
    q, k, v = _make_qkv(1024)
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


def test_gqa_and_non_128_heads_fall_back():
    ours, reference = _impls()
    q, k, v = _make_qkv(512)
    kv_heads = 14
    got = ours.forward(
        q.unsqueeze(0).contiguous(),
        k[:, :kv_heads].contiguous().unsqueeze(0),
        v[:, :kv_heads].contiguous().unsqueeze(0),
        attn_metadata=None,
    )
    want = reference.forward(
        q.unsqueeze(0).contiguous(),
        k[:, :kv_heads].contiguous().unsqueeze(0),
        v[:, :kv_heads].contiguous().unsqueeze(0),
        attn_metadata=None,
    )
    assert torch.equal(got, want)


def test_mix_none_is_exact_sdpa(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        RotNvfp4AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
        CudnnSDPAImpl,
    )

    monkeypatch.setenv("SGLANG_ROT_NVFP4_MIX", "none")
    impl = RotNvfp4AttentionImpl(
        num_heads=H3_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SOFTMAX_SCALE,
        num_kv_heads=H3_HEADS,
        prefix="blocks.0.attn",
        packed_trailing_padding=True,
    )
    q, k, v = _make_qkv(512)
    got = impl.forward(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        attn_metadata=None,
    )
    reference = CudnnSDPAImpl(
        num_heads=H3_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SOFTMAX_SCALE,
        num_kv_heads=H3_HEADS,
        prefix="blocks.0.attn",
        packed_trailing_padding=True,
    ).forward(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
        attn_metadata=None,
    )
    assert torch.equal(got, reference)


def test_layer_selection_ignores_refiner_and_accepts_impl_suffix():
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        LayerSet,
        _layer_index,
        _selected_layers,
        _sensitive_layers,
    )

    assert _layer_index("blocks.0.attn") == 0
    assert _layer_index("blocks.49.attn.impl") == 49
    # The refiner is not part of the DiT and stays on the fallback.
    assert _layer_index("token_refiner.blocks.0.attn") is None
    assert _layer_index("token_refiner.blocks.1.attn.impl") is None

    assert _selected_layers("all").all
    assert _selected_layers("none") == LayerSet()
    assert _selected_layers("last:6", 50).layers == set(range(44, 50))
    assert _selected_layers("every:2", 50).layers == set(range(0, 50, 2))
    assert _selected_layers("layers:0-2,49").layers == {0, 1, 2, 49}
    assert _sensitive_layers("0-27,47,49").layers == set(range(28)) | {47, 49}
    assert _sensitive_layers("all").all
    assert _sensitive_layers("none") == LayerSet()

    # last:/every: need the DiT depth; no hardcoded 50 fallback.
    with pytest.raises(ValueError, match="needs the DiT depth"):
        _selected_layers("last:6")


def test_last_layers_uses_model_depth(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        RotNvfp4AttentionImpl,
    )

    monkeypatch.setenv("SGLANG_ROT_NVFP4_MIX", "last:2")
    kwargs = dict(
        num_heads=H3_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SOFTMAX_SCALE,
        num_kv_heads=H3_HEADS,
        packed_trailing_padding=True,
    )
    first = RotNvfp4AttentionImpl(
        prefix="blocks.0.attn", total_layers=50, **kwargs
    )
    last = RotNvfp4AttentionImpl(
        prefix="blocks.49.attn", total_layers=50, **kwargs
    )
    assert not first._use_nvfp4
    assert last._use_nvfp4


def test_mix_selects_only_requested_blocks(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        RotNvfp4AttentionImpl,
    )

    monkeypatch.setenv("SGLANG_ROT_NVFP4_MIX", "layers:0")
    kwargs = dict(
        num_heads=H3_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=SOFTMAX_SCALE,
        num_kv_heads=H3_HEADS,
        packed_trailing_padding=True,
    )
    selected = RotNvfp4AttentionImpl(prefix="blocks.0.attn", **kwargs)
    other = RotNvfp4AttentionImpl(prefix="blocks.1.attn", **kwargs)
    assert selected._use_nvfp4
    assert not other._use_nvfp4


def test_offset_roundtrip(tmp_path):
    from sglang.kernels.ops.attention.rot_nvfp4_quant import (
        calibrate_offsets,
        load_offsets,
        save_offsets,
    )

    q, k, _ = _make_qkv(512)
    offsets = calibrate_offsets(q.unsqueeze(0), k.unsqueeze(0))
    save_offsets(tmp_path, {"blocks.0.attn": offsets[:2]})
    loaded = load_offsets(tmp_path, "blocks.0.attn", q.device)

    assert loaded is not None
    assert torch.equal(offsets[0], loaded[0])
    assert torch.equal(offsets[1], loaded[1])
