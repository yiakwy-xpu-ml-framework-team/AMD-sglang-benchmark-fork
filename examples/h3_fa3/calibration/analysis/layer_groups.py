"""Compare prompt sensitivity for the sensitive/stable layer split.

Sensitive layers are blocks 0-27, 47 and 49.  Stable layers are blocks
28-46 and 48.

    python calibration/calib.py analyze layer-groups \
        --pattern '20261116_fl2va_prompt_*'

Writes layer_group_summary.csv, layer_group_per_layer.csv, and two plots
under calibration/plots/<prefix>/.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.labels import prompt_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_ROOT = ROOT / "outputs" / "calibration"

SENSITIVE = set(range(28)) | {47, 49}
STABLE = set(range(28, 47)) | {48}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().ravel()
    b = b.float().ravel()
    return (a.dot(b) / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def group_of(prefix: str) -> tuple[str, int | None]:
    match = re.match(r"^blocks\.(\d+)\.attn$", prefix)
    if not match:
        return "refiner", None
    index = int(match.group(1))
    if index in SENSITIVE:
        return "sensitive", index
    if index in STABLE:
        return "stable", index
    return "other", index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--exclude", default="consensus")
    parser.add_argument("--out", type=Path, default=None)
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
    pairs = list(itertools.combinations(range(len(tasks)), 2))
    pair_labels = [f"{labels[i]}-{labels[j]}" for i, j in pairs]

    prefixes = sorted(
        path.stem for path in (tasks[0] / "offsets").glob("*.pt")
    )
    per_layer = []
    for prefix in prefixes:
        payloads = [
            torch.load(task / "offsets" / f"{prefix}.pt", map_location="cpu",
                       weights_only=True)
            for task in tasks
        ]
        group, index = group_of(prefix)
        for name in ("q_offset", "k_offset"):
            tensors = [payload[name].float() for payload in payloads]
            rels = [rel_l2(tensors[i], tensors[j]) for i, j in pairs]
            cosines = [cosine(tensors[i], tensors[j]) for i, j in pairs]
            max_abs = max(
                (tensors[i] - tensors[j]).abs().max().item()
                for i, j in pairs
            )
            per_layer.append({
                "group": group,
                "layer": index if index is not None else -1,
                "prefix": prefix,
                "tensor": name,
                "rel_l2_mean": float(np.mean(rels)),
                "rel_l2_max": float(np.max(rels)),
                "cosine_mean": float(np.mean(cosines)),
                "max_abs_max": float(max_abs),
            })

    out = args.out or (HERE.parent / "plots" / args.pattern.replace("*", "").strip("_"))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "layer_group_per_layer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_layer[0]))
        writer.writeheader()
        writer.writerows(per_layer)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_layer:
        groups[(row["group"], row["tensor"])].append(row)
    summary = []
    for (group, tensor), rows in sorted(groups.items()):
        rel_mean = np.asarray([row["rel_l2_mean"] for row in rows])
        rel_max = np.asarray([row["rel_l2_max"] for row in rows])
        cos_mean = np.asarray([row["cosine_mean"] for row in rows])
        summary.append({
            "group": group,
            "tensor": tensor,
            "layers": len(rows),
            "rel_l2_mean": float(rel_mean.mean()),
            "rel_l2_median": float(np.median(rel_mean)),
            "rel_l2_max": float(rel_max.max()),
            "cosine_mean": float(cos_mean.mean()),
            "max_abs_max": float(max(row["max_abs_max"] for row in rows)),
        })
    with (out / "layer_group_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"prompts: {labels}")
    print(f"pairs: {pair_labels}")
    print(f"sensitive: {len(SENSITIVE)} layers, stable: {len(STABLE)} layers")
    print(f"\n{'group':10s} {'tensor':10s} {'n':>3s} "
          f"{'relL2 mean':>10s} {'relL2 med':>10s} {'relL2 max':>10s} "
          f"{'cosine':>8s}")
    for row in summary:
        print(
            f"{row['group']:10s} {row['tensor']:10s} {row['layers']:3d} "
            f"{row['rel_l2_mean']:10.4f} {row['rel_l2_median']:10.4f} "
            f"{row['rel_l2_max']:10.4f} {row['cosine_mean']:8.5f}"
        )

    # group bar chart
    group_names = ["sensitive", "stable"]
    tensors = ["q_offset", "k_offset"]
    x = np.arange(len(tensors))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, group in zip((-width / 2, width / 2), group_names):
        values = [
            next(row["rel_l2_mean"] for row in summary
                 if row["group"] == group and row["tensor"] == tensor)
            for tensor in tensors
        ]
        ax.bar(x + offset, values, width, label=group)
        for idx, value in enumerate(values):
            ax.text(x[idx] + offset, value + 0.01, f"{value:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x, ["q_offset", "k_offset"])
    ax.set_ylabel("mean pairwise relL2 across prompts")
    ax.set_title("Prompt sensitivity: sensitive vs stable layers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "layer_group_relL2.png", dpi=160)
    plt.close(fig)

    # per-layer sensitivity
    by_layer: dict[int, float] = {}
    for row in per_layer:
        if row["group"] not in ("sensitive", "stable"):
            continue
        by_layer[row["layer"]] = max(
            by_layer.get(row["layer"], 0.0), row["rel_l2_max"]
        )
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for group, color in (("sensitive", "#d62728"), ("stable", "#1f77b4")):
        xs = sorted(layer for layer in by_layer
                    if group_of(f"blocks.{layer}.attn")[0] == group)
        ys = [by_layer[layer] for layer in xs]
        ax.scatter(xs, ys, s=22, color=color, label=group)
    ax.plot(sorted(by_layer), [by_layer[layer] for layer in sorted(by_layer)],
            color="#888888", linewidth=0.6, alpha=0.6)
    ax.axhline(0.4, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("DiT block index")
    ax.set_ylabel("max pairwise relL2 (q/k, prompts)")
    ax.set_title("Per-layer prompt sensitivity (threshold 0.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "layer_sensitivity_by_group.png", dpi=160)
    plt.close(fig)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
