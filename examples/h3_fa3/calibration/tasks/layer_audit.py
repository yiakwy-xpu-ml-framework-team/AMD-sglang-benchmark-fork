"""Audit every NVFP4 attention layer against the exact SDPA fallback."""

from __future__ import annotations

from pathlib import Path

from runner import Case, RunConfig
from tasks.common import DEFAULT_PROMPT


def add_arguments(parser) -> None:
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=4.0)


def build(args, config: RunConfig):
    prompt = args.prompt or config.root / "calibration" / "prompts" / DEFAULT_PROMPT
    offsets = args.offsets or (args.task_dir / "offsets")
    return [
        Case(
            name="layer-audit",
            out_dir=args.task_dir,
            prompt=prompt,
            tag="audit",
            steps=args.steps,
            seconds=args.seconds,
            server_env={
                "WARMUP_MODE": "request",
                "NVFP4_OFFSETS": str(offsets),
                "SGLANG_ROT_NVFP4_AUDIT": "1",
                "SGLANG_ROT_NVFP4_AUDIT_MIN_SEQ": "0",
                "SGLANG_ROT_NVFP4_SAVE_OFFSETS": "0",
            },
            post="audit",
        )
    ]
