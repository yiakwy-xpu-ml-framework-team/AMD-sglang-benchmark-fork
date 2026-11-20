"""Rank layers by how much their offsets move across prompt rewrites.

    python calibration/calib.py analyze prompt-sensitivity \
        --pattern '20261116_fl2va_prompt_*' --threshold 0.4 --out <task_dir>

For every layer, load q_offset/k_offset from the per-prompt caches, compute
the maximum pairwise relL2 across prompts and tensors, and write:
  <out>/prompt_sensitivity.csv
  <out>/sensitive_layers_t0.40.txt   # comma-separated DiT indices
  <out>/stable_layers_t0.40.txt
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

import torch

from analysis.labels import prompt_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_ROOT = ROOT / "outputs" / "calibration"


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
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
        raise SystemExit(f"need >=2 tasks, found {len(tasks)}")
    labels = [prompt_label(task.name) for task in tasks]

    prefixes = sorted(
        path.stem for path in (tasks[0] / "offsets").glob("*.pt")
    )
    rows = []
    for prefix in prefixes:
        payloads = [
            torch.load(task / "offsets" / f"{prefix}.pt", map_location="cpu",
                       weights_only=True)
            for task in tasks
        ]
        values = []
        for name in ("q_offset", "k_offset"):
            tensors = [payload[name].float() for payload in payloads]
            values.extend(
                rel_l2(tensors[i], tensors[j])
                for i, j in itertools.combinations(range(len(tasks)), 2)
            )
        rows.append({
            "prefix": prefix,
            "max_pairwise_relL2": max(values),
            "mean_pairwise_relL2": sum(values) / len(values),
        })

    rows.sort(key=lambda row: row["max_pairwise_relL2"], reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "prompt_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "prefix", "max_pairwise_relL2",
                                "mean_pairwise_relL2"]
        )
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})

    dit = []
    for row in rows:
        match = re.match(r"^blocks\.(\d+)\.attn$", row["prefix"])
        if match:
            dit.append((int(match.group(1)), row["max_pairwise_relL2"]))

    sensitive = sorted(
        layer for layer, value in dit if value >= args.threshold
    )
    stable = sorted(
        layer for layer, value in dit if value < args.threshold
    )
    tag = f"t{args.threshold:.2f}"
    (args.out / f"sensitive_layers_{tag}.txt").write_text(
        ",".join(str(layer) for layer in sensitive) + "\n"
    )
    (args.out / f"stable_layers_{tag}.txt").write_text(
        ",".join(str(layer) for layer in stable) + "\n"
    )

    print(f"prompts: {labels}")
    print(f"threshold: {args.threshold}")
    print(f"DiT sensitive: {len(sensitive)} / {len(dit)} -> {sensitive}")
    print(f"DiT stable:    {len(stable)} / {len(dit)}")
    print("\nrank  prefix                          max      mean")
    for rank, row in enumerate(rows[:20], 1):
        print(
            f"{rank:4d}  {row['prefix']:30s}  "
            f"{row['max_pairwise_relL2']:.4f}  "
            f"{row['mean_pairwise_relL2']:.4f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
