"""Generate one video with a layer mix selected by rank-layers."""

from __future__ import annotations

from pathlib import Path

from runner import Case, RunConfig
from tasks.common import DEFAULT_GENERIC_OFFSETS, DEFAULT_PROMPT


def add_arguments(parser) -> None:
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--offsets", default=DEFAULT_GENERIC_OFFSETS)
    parser.add_argument("--mix-file", type=Path, default=None)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--mix-backend", default="sdpa", choices=["sdpa", "fp8"])
    parser.add_argument("--tag", default="mix")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)


def build(args, config: RunConfig):
    if args.mix_file is not None:
        mix = args.mix_file.read_text().strip()
    elif args.mix is not None:
        mix = args.mix
    else:
        raise SystemExit("pass --mix or --mix-file")

    prompt = args.prompt or config.root / "calibration" / "prompts" / DEFAULT_PROMPT
    reference = args.reference or (
        config.root
        / "outputs"
        / "tests"
        / f"bullettime_fp16_{args.steps}st_seed{args.seed}.mp4"
    )
    return [
        Case(
            name=args.tag,
            out_dir=args.task_dir,
            prompt=prompt,
            tag=args.tag,
            steps=args.steps,
            seed=args.seed,
            seconds=args.seconds,
            reference=reference,
            server_env={
                "WARMUP_MODE": "request",
                "NVFP4_OFFSETS": str(args.offsets),
                "NVFP4_MIX": mix,
                "NVFP4_MIX_BACKEND": args.mix_backend,
                "SGLANG_ROT_NVFP4_MIX": mix,
                "SGLANG_ROT_NVFP4_MIX_BACKEND": args.mix_backend,
                "SGLANG_ROT_NVFP4_SAVE_OFFSETS": "0",
            },
        )
    ]
