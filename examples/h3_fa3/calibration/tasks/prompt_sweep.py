"""Calibrate offsets for every prompt in calibration/prompts."""

from __future__ import annotations

import time
from pathlib import Path

from runner import Case, RunConfig


def add_arguments(parser) -> None:
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--expect-offsets", type=int, default=52)


def build(args, config: RunConfig) -> list[Case]:
    prompt_dir = args.prompt_dir or config.root / "calibration" / "prompts"
    prefix = args.prefix or time.strftime("%Y%m%d_%H%M%S")
    cases = []
    for prompt in sorted(prompt_dir.glob("*.txt")):
        out_dir = config.root / "outputs" / "calibration" / f"{prefix}_{prompt.stem}"
        cases.append(
            Case(
                name=f"prompt-{prompt.stem}",
                out_dir=out_dir,
                prompt=prompt,
                tag=prompt.stem,
                steps=args.steps,
                seconds=args.seconds,
                server_env={
                    "WARMUP_MODE": "request",
                    "SGLANG_ROT_NVFP4_SAVE_OFFSETS": "1",
                    "NVFP4_OFFSETS": str(out_dir / "offsets"),
                },
                expect_offsets=args.expect_offsets,
            )
        )
    if not cases:
        raise SystemExit(f"no prompts in {prompt_dir}")
    return cases
