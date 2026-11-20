#!/usr/bin/env python3
"""H3 calibration entry point.

Run a declared experiment:

    python calibration/calib.py run prompt-sweep --prefix 20261120

Inspect results:

    python calibration/calib.py status

The task declarations live in calibration/tasks.  Analysis commands reuse the
modules in calibration/analysis.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from runner import add_common_args, config_from_args, run_cases  # noqa: E402

TASKS = {
    "prompt-sweep": ("tasks.prompt_sweep", "calibrate offsets for every prompt"),
    "fp16-baseline": ("tasks.fp16_baseline", "generate the fp16 reference"),
    "calib-mode": ("tasks.calib_mode", "compare one calibration mode"),
    "layer-audit": ("tasks.layer_audit", "audit layers against SDPA"),
    "mix-test": ("tasks.mix_test", "generate one layer mix"),
    "seed-study": ("tasks.seed_study", "run the four-way seed comparison"),
}

ANALYSIS = {
    "offsets": "analysis.offsets",
    "layer-groups": "analysis.layer_groups",
    "consensus": "analysis.consensus",
    "rank-layers": "analysis.rank_layers",
    "prompt-sensitivity": "analysis.prompt_sensitivity",
    "bench-offsets": "analysis.bench_offsets",
}


def _run_task(task: str, argv: list[str]) -> int:
    if task not in TASKS:
        choices = ", ".join(TASKS)
        raise SystemExit(f"unknown task {task!r}; choose one of: {choices}")

    module_path, _ = TASKS[task]
    module = importlib.import_module(module_path)
    parser = argparse.ArgumentParser(prog=f"calib.py run {task}")
    add_common_args(parser)
    module.add_arguments(parser)
    args = parser.parse_args(argv)

    config = config_from_args(args, ROOT)
    cases = module.build(args, config)
    run_cases(cases, config)
    return 0


def _status(args: argparse.Namespace) -> int:
    base = args.root / "outputs" / "calibration"
    for task_dir in sorted(base.glob(args.pattern)):
        if not task_dir.is_dir():
            continue
        offsets = len(list((task_dir / "offsets").glob("*.pt")))
        videos = len(list(task_dir.glob("*.mp4")))
        if not offsets and not videos:
            if (task_dir / "logs").is_dir():
                print(
                    f"{task_dir.name:48s} offsets= 0 videos=0 [pending]"
                )
            continue
        state = "video" if videos else "offsets"
        print(f"{task_dir.name:48s} offsets={offsets:2d} videos={videos} [{state}]")
    return 0


def _analyze(name: str, argv: list[str]) -> int:
    if name not in ANALYSIS:
        choices = ", ".join(ANALYSIS)
        raise SystemExit(f"unknown analysis {name!r}; choose one of: {choices}")
    module = importlib.import_module(ANALYSIS[name])
    module.main(argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a declared task")
    run.add_argument("task")
    run.add_argument("task_args", nargs=argparse.REMAINDER)

    sub.add_parser("list", help="list task declarations")

    status = sub.add_parser("status", help="show calibration outputs")
    status.add_argument("--root", type=Path, default=ROOT)
    status.add_argument("--pattern", default="*")

    analyze = sub.add_parser("analyze", help="run an analysis command")
    analyze.add_argument("analysis")
    analyze.add_argument("analysis_args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_task(args.task, args.task_args)
    if args.command == "list":
        for name, (_, help_text) in TASKS.items():
            print(f"{name:18s} {help_text}")
        print()
        print("analysis: " + ", ".join(ANALYSIS))
        return 0
    if args.command == "status":
        return _status(args)
    if args.command == "analyze":
        return _analyze(args.analysis, args.analysis_args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
