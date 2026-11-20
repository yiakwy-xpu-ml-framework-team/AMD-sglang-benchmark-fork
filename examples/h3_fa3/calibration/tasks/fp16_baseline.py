"""Generate the fp16/SDPA reference video for one seed."""

from __future__ import annotations

from pathlib import Path

from runner import RunConfig
from tasks.common import DEFAULT_PROMPT, fp16_case


def add_arguments(parser) -> None:
    parser.add_argument("--task-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--prompt", type=Path, default=None)


def build(args, config: RunConfig):
    prompt = args.prompt or config.root / "calibration" / "prompts" / DEFAULT_PROMPT
    task_dir = args.task_dir or (
        config.root / "outputs" / "calibration" / f"seed{args.seed}_fp16"
    )
    return [
        fp16_case(
            task_dir=task_dir,
            seed=args.seed,
            steps=args.steps,
            prompt=prompt,
            seconds=args.seconds,
        )
    ]
