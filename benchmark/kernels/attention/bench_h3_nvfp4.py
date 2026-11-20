"""MiniMax-H3 attention microbenchmark: cuDNN SDPA, NVFP4, SM120a FP8.

The benchmark calls the same backend implementations the server uses, so the
timings include rotation and calibration bookkeeping.  The NVFP4 offsets are
calibrated once and then reused, matching a served request.

    python benchmark/kernels/attention/bench_h3_nvfp4.py \
        --seq 8192 16384 27216 --heads 56
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch


def timeit(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def rel_l2(candidate, reference) -> float:
    return (
        (candidate.float() - reference.float()).norm() / reference.float().norm()
    ).item()


def _cell(value: float | None, digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def print_table(sequences: list[int], rows: list[tuple[str, list[str]]]) -> None:
    label_width = max(len(label) for label, _ in rows)
    column_width = max(10, *(len(str(seq)) for seq in sequences))
    header = " " * label_width + "".join(
        f"{'S=' + str(seq):>{column_width}}" for seq in sequences
    )
    print(header)
    print("-" * len(header))
    for label, values in rows:
        print(
            f"{label:<{label_width}}"
            + "".join(f"{value:>{column_width}}" for value in values)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, nargs="+", default=[8192, 16384, 27216])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    from sglang.multimodal_gen.runtime.layers.attention.backends.rot_nvfp4_attn import (
        RotNvfp4AttentionImpl,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
        CudnnSDPAImpl,
    )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = "cuda"
    capability = torch.cuda.get_device_capability()
    print(f"GPU {torch.cuda.get_device_name(0)}  CC {capability}")
    print(f"heads {args.heads}  head_dim 128  dtype {dtype}")

    kwargs = dict(
        num_heads=args.heads,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
        num_kv_heads=args.heads,
        packed_trailing_padding=True,
    )
    exact = CudnnSDPAImpl(prefix="blocks.0.attn", **kwargs)
    nvfp4 = RotNvfp4AttentionImpl(prefix="blocks.0.attn", **kwargs)

    fp8 = None
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.rot_fp8_attn_sm120a_ref import (
            FP8FlashAttentionSM120Impl,
        )

        fp8 = FP8FlashAttentionSM120Impl(prefix="blocks.0.attn", **kwargs)
    except ImportError:
        pass

    columns = {
        "exact_ms": [],
        "nvfp4_ms": [],
        "fp8_ms": [],
        "exact_tflops": [],
        "nvfp4_tflops": [],
        "speedup": [],
        "nvfp4_rel_l2": [],
        "fp8_rel_l2": [],
        "nvfp4_vs_fp8": [],
        "calibration_ms": [],
        "exact_layers_s": [],
        "nvfp4_layers_s": [],
        "saving_s": [],
    }

    for seq in args.seq:
        torch.manual_seed(0)
        q = torch.randn(seq, args.heads, 128, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = torch.tensor([0, seq], dtype=torch.int32, device=device)

        def run_exact():
            return exact.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=seq, cu_seqlens_host=(0, seq)
            )

        def run_nvfp4():
            return nvfp4.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=seq, cu_seqlens_host=(0, seq)
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        nvfp4._ensure_offsets(q.unsqueeze(0), k.unsqueeze(0))
        torch.cuda.synchronize()
        calibration_ms = (time.perf_counter() - start) * 1e3

        exact_out = run_exact()
        nvfp4_out = run_nvfp4()
        exact_ms = timeit(run_exact, args.iters)
        nvfp4_ms = timeit(run_nvfp4, args.iters)
        flops = 4 * seq * seq * args.heads * 128

        fp8_ms = None
        fp8_rel = None
        nvfp4_vs_fp8 = None
        if fp8 is not None:
            fp8_out = fp8.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=seq, cu_seqlens_host=(0, seq)
            )
            fp8_ms = timeit(
                lambda: fp8.forward_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens=cu,
                    max_seqlen=seq,
                    cu_seqlens_host=(0, seq),
                ),
                args.iters,
            )
            fp8_rel = rel_l2(fp8_out, exact_out)
            nvfp4_vs_fp8 = rel_l2(nvfp4_out, fp8_out)

        columns["exact_ms"].append(_cell(exact_ms, 2))
        columns["nvfp4_ms"].append(_cell(nvfp4_ms, 2))
        columns["fp8_ms"].append(_cell(fp8_ms, 2))
        columns["exact_tflops"].append(_cell(flops / exact_ms / 1e9, 0))
        columns["nvfp4_tflops"].append(_cell(flops / nvfp4_ms / 1e9, 0))
        columns["speedup"].append(_cell(exact_ms / nvfp4_ms, 2))
        columns["nvfp4_rel_l2"].append(_cell(rel_l2(nvfp4_out, exact_out), 4))
        columns["fp8_rel_l2"].append(_cell(fp8_rel, 4))
        columns["nvfp4_vs_fp8"].append(_cell(nvfp4_vs_fp8, 4))
        columns["calibration_ms"].append(_cell(calibration_ms))
        columns["exact_layers_s"].append(_cell(exact_ms * args.layers / 1e3, 2))
        columns["nvfp4_layers_s"].append(_cell(nvfp4_ms * args.layers / 1e3, 2))
        columns["saving_s"].append(
            _cell((exact_ms - nvfp4_ms) * args.layers / 1e3, 2)
        )

    rows = [
        ("exact ms", columns["exact_ms"]),
        ("nvfp4 ms", columns["nvfp4_ms"]),
        ("fp8 ms", columns["fp8_ms"]),
        ("exact TFLOP/s", columns["exact_tflops"]),
        ("nvfp4 TFLOP/s", columns["nvfp4_tflops"]),
        ("speedup exact/nvfp4", columns["speedup"]),
        ("nvfp4 relL2", columns["nvfp4_rel_l2"]),
        ("fp8 relL2", columns["fp8_rel_l2"]),
        ("nvfp4 vs fp8 relL2", columns["nvfp4_vs_fp8"]),
        ("calibration ms", columns["calibration_ms"]),
        (f"exact {args.layers} layers s", columns["exact_layers_s"]),
        (f"nvfp4 {args.layers} layers s", columns["nvfp4_layers_s"]),
        ("saving s/step", columns["saving_s"]),
    ]
    print()
    print_table(args.seq, rows)
    print()
    print("Calibration is measured once; later sequence lengths reuse the offsets.")


if __name__ == "__main__":
    main()
