# SPDX-License-Identifier: Apache-2.0
"""H64 rotation and K-only offset helpers for the SM120a FP8 reference."""

import pytest
import torch

from sglang.kernels.ops.attention.fp8_fa_sm120.offsets import (
    calibrate_k_offset,
    load_offsets,
    rotate_h64,
    save_offsets,
    validate_k_offset,
)


@pytest.fixture(autouse=True)
def _reset_sm120a_env():
    yield
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            rot_fp8_attn_sm120a_ref as backend,
        )

        backend.sm120a_fp8_env.cache_clear()
    except Exception:
        pass


def test_rotate_h64_is_involutive():
    x = torch.randn(17, 3, 128, dtype=torch.float32)
    rotated = rotate_h64(x)
    restored = rotate_h64(rotated)
    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


def test_calibrate_k_offset_is_mean_first():
    torch.manual_seed(0)
    k = torch.randn(64, 3, 128, dtype=torch.float32)
    expected = rotate_h64(k.mean(dim=0))
    got = calibrate_k_offset(k)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_validate_k_offset_rejects_batched_shape():
    offset = torch.zeros(2, 3, 128, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[H,128\]"):
        validate_k_offset(offset, torch.device("cpu"), heads=3)


def test_offsets_cache_round_trip(tmp_path):
    offset = torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128)
    save_offsets(tmp_path, {"blocks.7.attn": offset})

    loaded = load_offsets(tmp_path, "blocks.7.attn", torch.device("cpu"))

    assert loaded is not None
    assert loaded.dtype == torch.float32
    torch.testing.assert_close(loaded, offset)


def _make_backend(
    monkeypatch,
    mode="offline",
    sensitive="",
    offsets_path=None,
    save=False,
    prefix="blocks.0.attn",
):
    pytest.importorskip("cutlass.cute")
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        rot_fp8_attn_sm120a_ref as backend,
    )

    monkeypatch.setenv("SGLANG_ROT_FP8_SM120A_ROTATE", "1")
    monkeypatch.setenv("SGLANG_ROT_FP8_SM120A_CALIB_MODE", mode)
    monkeypatch.setenv("SGLANG_ROT_FP8_SM120A_SENSITIVE_LAYERS", sensitive)
    if offsets_path is None:
        monkeypatch.delenv("SGLANG_ROT_FP8_SM120A_OFFSETS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ROT_FP8_SM120A_OFFSETS", str(offsets_path))
    if save:
        monkeypatch.setenv("SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS", "1")
    else:
        monkeypatch.delenv("SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS", raising=False)
    backend.sm120a_fp8_env.cache_clear()
    impl = backend.FP8FlashAttentionSM120Impl(
        num_heads=8,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
        prefix=prefix,
    )
    return backend, impl


def test_backend_offline_calibrates_once(monkeypatch):
    backend, impl = _make_backend(monkeypatch, "offline")
    calls = []

    def fake_calibrate(key):
        calls.append(tuple(key.shape))
        return torch.zeros(8, 128)

    monkeypatch.setattr(backend, "calibrate_k_offset", fake_calibrate)
    key = torch.zeros(16, 8, 128)

    first = impl._ensure_k_offset(key)
    second = impl._ensure_k_offset(key)

    assert first is second
    assert calls == [(16, 8, 128)]


def test_backend_hybrid_online_only_for_sensitive_layers(monkeypatch):
    _, sensitive = _make_backend(
        monkeypatch, "hybrid", sensitive="0-3", prefix="blocks.1.attn"
    )
    _, stable = _make_backend(
        monkeypatch, "hybrid", sensitive="0-3", prefix="blocks.7.attn"
    )

    assert sensitive._online is True
    assert stable._online is False


def test_backend_offline_cache_round_trip(monkeypatch, tmp_path):
    backend, impl = _make_backend(
        monkeypatch, "offline", offsets_path=tmp_path, save=True
    )
    key = torch.zeros(16, 8, 128)

    first = impl._ensure_k_offset(key)
    assert first is not None
    assert (tmp_path / "blocks.0.attn.pt").is_file()

    _, cached = _make_backend(monkeypatch, "offline", offsets_path=tmp_path)
    monkeypatch.setattr(
        backend,
        "calibrate_k_offset",
        lambda _key: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )
    second = cached._ensure_k_offset(key)

    torch.testing.assert_close(second, first)


def test_backend_online_recalibrates_per_epoch(monkeypatch):
    backend, impl = _make_backend(monkeypatch, "online")
    epochs = iter([1, 2])
    monkeypatch.setattr(backend, "_request_epoch", lambda: next(epochs))
    calls = []

    def fake_calibrate(key):
        calls.append(1)
        return torch.zeros(8, 128)

    monkeypatch.setattr(backend, "calibrate_k_offset", fake_calibrate)
    key = torch.zeros(16, 8, 128)

    impl._ensure_k_offset(key)
    impl._ensure_k_offset(key)

    assert calls == [1, 1]
