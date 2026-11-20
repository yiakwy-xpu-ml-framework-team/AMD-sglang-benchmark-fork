"""Run the fp16 / all-load / hybrid / prompt-cache comparison for one seed."""

from __future__ import annotations

from pathlib import Path

from runner import RunConfig
from tasks.common import (
    DEFAULT_GENERIC_OFFSETS,
    DEFAULT_PROMPT,
    calib_case,
    fp16_case,
)


def add_arguments(parser) -> None:
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--task-root", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--generic-offsets", default=DEFAULT_GENERIC_OFFSETS)
    parser.add_argument(
        "--sensitivity-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--orig-offsets", type=Path, default=None)


def build(args, config: RunConfig):
    root = config.root
    task_root = args.task_root or (root / "outputs" / "calibration")
    prompt = args.prompt or root / "calibration" / "prompts" / DEFAULT_PROMPT

    sensitivity = args.sensitivity_file or (
        task_root
        / "20261116_fl2va_prompt_sensitivity"
        / "sensitive_layers_t0.40.txt"
    )
    if not sensitivity.is_file():
        raise SystemExit(f"sensitivity file not found: {sensitivity}")
    sensitive = sensitivity.read_text().strip()

    orig_offsets = args.orig_offsets or (
        task_root / "20261116_fl2va_prompt_orig" / "offsets"
    )
    if not orig_offsets.is_dir():
        raise SystemExit(f"prompt offset cache not found: {orig_offsets}")

    fp16 = fp16_case(
        task_dir=task_root / f"seed{args.seed}_fp16",
        seed=args.seed,
        steps=args.steps,
        prompt=prompt,
        seconds=args.seconds,
    )
    reference = (
        fp16.out_dir / f"calib_fp16_{args.steps}st_seed{args.seed}.mp4"
    )

    return [
        fp16,
        calib_case(
            task_dir=task_root / f"seed{args.seed}_allload",
            offsets=args.generic_offsets,
            mode="offline",
            tag="allload",
            seed=args.seed,
            steps=args.steps,
            prompt=prompt,
            reference=reference,
            seconds=args.seconds,
        ),
        calib_case(
            task_dir=task_root / f"seed{args.seed}_hybrid",
            offsets=args.generic_offsets,
            mode="hybrid",
            tag="hybrid",
            seed=args.seed,
            steps=args.steps,
            prompt=prompt,
            reference=reference,
            sensitive=sensitive,
            seconds=args.seconds,
        ),
        calib_case(
            task_dir=task_root / f"seed{args.seed}_origcache",
            offsets=str(orig_offsets),
            mode="offline",
            tag="origcache",
            seed=args.seed,
            steps=args.steps,
            prompt=prompt,
            reference=reference,
            seconds=args.seconds,
        ),
    ]
