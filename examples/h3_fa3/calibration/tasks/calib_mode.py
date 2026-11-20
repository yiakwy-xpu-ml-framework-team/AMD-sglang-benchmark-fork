"""Compare one NVFP4 calibration mode against an fp16 reference."""

from __future__ import annotations

from pathlib import Path

from runner import RunConfig
from tasks.common import (
    DEFAULT_GENERIC_OFFSETS,
    DEFAULT_PROMPT,
    calib_case,
)


def add_arguments(parser) -> None:
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--offsets", default=DEFAULT_GENERIC_OFFSETS)
    parser.add_argument(
        "--mode", default="offline", choices=["offline", "online", "hybrid"]
    )
    parser.add_argument("--sensitive-layers", default="")
    parser.add_argument("--sensitive-file", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")


def build(args, config: RunConfig):
    prompt = args.prompt or config.root / "calibration" / "prompts" / DEFAULT_PROMPT
    tag = args.tag or args.mode
    reference = args.reference
    if reference is None:
        reference = (
            config.root
            / "outputs"
            / "tests"
            / f"bullettime_fp16_{args.steps}st_seed{args.seed}.mp4"
        )

    sensitive = args.sensitive_layers
    if args.sensitive_file is not None:
        sensitive = args.sensitive_file.read_text().strip()

    return [
        calib_case(
            task_dir=args.task_dir,
            offsets=args.offsets,
            mode=args.mode,
            tag=tag,
            seed=args.seed,
            steps=args.steps,
            prompt=prompt,
            reference=reference,
            sensitive=sensitive,
            seconds=args.seconds,
            debug=args.debug,
        )
    ]
