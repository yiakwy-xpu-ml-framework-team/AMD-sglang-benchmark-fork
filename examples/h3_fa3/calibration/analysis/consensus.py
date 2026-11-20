"""Build a consensus NVFP4 offset cache from several prompt-specific runs.

    python calibration/calib.py analyze consensus \
        --pattern '20261116_fl2va_prompt_*' \
        --out outputs/calibration/20261116_fl2va_prompt_consensus

For every per-layer .pt present in all selected tasks, average q_offset and
k_offset in fp32, cast back to fp16, and write a cache that the SGLang
backend can load with SGLANG_ROT_NVFP4_OFFSETS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_ROOT = ROOT / "outputs" / "calibration"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", default="consensus")
    args = parser.parse_args(argv)

    tasks = sorted(
        path for path in args.root.glob(args.pattern)
        if path.is_dir()
        and (path / "offsets").is_dir()
        and args.exclude not in path.name
    )
    if len(tasks) < 2:
        raise SystemExit(f"need >=2 tasks, found {len(tasks)}: {tasks}")

    out_offsets = args.out / "offsets"
    out_offsets.mkdir(parents=True, exist_ok=True)

    first = tasks[0] / "offsets"
    prefixes = sorted(path.stem for path in first.glob("*.pt"))
    if not prefixes:
        raise SystemExit(f"no .pt files in {first}")

    report = {"sources": [task.name for task in tasks], "layers": {}}
    for prefix in prefixes:
        payloads = [
            torch.load(task / "offsets" / f"{prefix}.pt", map_location="cpu",
                       weights_only=True)
            for task in tasks
        ]
        q = torch.stack([p["q_offset"].float() for p in payloads]).mean(0)
        k = torch.stack([p["k_offset"].float() for p in payloads]).mean(0)
        torch.save(
            {"q_offset": q.to(torch.float16), "k_offset": k.to(torch.float16)},
            out_offsets / f"{prefix}.pt",
        )
        report["layers"][prefix] = {
            "q_relL2_to_consensus": [
                ((p["q_offset"].float() - q).norm() / q.norm().clamp_min(1e-12)).item()
                for p in payloads
            ],
            "k_relL2_to_consensus": [
                ((p["k_offset"].float() - k).norm() / k.norm().clamp_min(1e-12)).item()
                for p in payloads
            ],
        }

    with (args.out / "meta.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"wrote {len(prefixes)} layers to {out_offsets}")
    for prefix in prefixes[:3]:
        entry = report["layers"][prefix]
        print(
            prefix,
            "q", ["%.4f" % x for x in entry["q_relL2_to_consensus"]],
            "k", ["%.4f" % x for x in entry["k_relL2_to_consensus"]],
        )


if __name__ == "__main__":
    main()
