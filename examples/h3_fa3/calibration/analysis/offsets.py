"""Analyze prompt-dependent NVFP4 offsets and draw 3D surfaces.

    python calibration/calib.py analyze offsets \
        --pattern '20261116_fl2va_prompt_*' \
        --layers blocks.0.attn blocks.25.attn blocks.49.attn

Reads outputs/calibration/<task>/offsets/<layer>.pt, compares the per-prompt
q_offset/k_offset [H,128] tensors, and writes:
  calibration/plots/<prefix>/offset_3d_<layer>_<qk>_mean.png
  calibration/plots/<prefix>/offset_3d_<layer>_<qk>_std.png
  calibration/plots/<prefix>/offset_heatmap_<layer>_<qk>_std.png
  calibration/plots/<prefix>/offset_hist_<layer>_<qk>.png
  calibration/plots/<prefix>/offset_stats.json
  calibration/plots/<prefix>/offset_prompt_relL2.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from analysis.labels import prompt_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_ROOT = ROOT / "outputs" / "calibration"
DEFAULT_LAYERS = ("blocks.0.attn", "blocks.25.attn", "blocks.49.attn")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def load_one(task: Path, layer: str, name: str) -> torch.Tensor:
    payload = torch.load(task / "offsets" / f"{layer}.pt", map_location="cpu",
                         weights_only=True)
    return payload[name].float()


def surface(mean: np.ndarray, title: str, path: Path) -> None:
    heads, channels = mean.shape
    x, y = np.meshgrid(np.arange(channels), np.arange(heads))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, mean, cmap="coolwarm", linewidth=0,
                    antialiased=True, rstride=1, cstride=2)
    ax.set_xlabel("channel")
    ax.set_ylabel("head")
    ax.set_zlabel("offset")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def heatmap(std: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(std, aspect="auto", cmap="magma")
    ax.set_xlabel("channel")
    ax.set_ylabel("head")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="std across prompts")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def histogram(values: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values.ravel(), bins=120, color="#4477aa", alpha=0.85)
    ax.set_xlabel("offset value")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


PROMPT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
PROMPT_MARKERS = ("o", "^", "s", "D", "v")


def prompt_scatter(
    values: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    head_stride: int = 2,
    channel_stride: int = 4,
) -> None:
    """3D scatter with one color+marker per prompt."""
    prompts, heads, channels = values.shape
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    heads_idx = np.arange(0, heads, head_stride)
    channels_idx = np.arange(0, channels, channel_stride)
    hh, cc = np.meshgrid(heads_idx, channels_idx, indexing="ij")
    for i, label in enumerate(labels):
        z = values[i][hh, cc]
        ax.scatter(
            hh.ravel(),
            cc.ravel(),
            z.ravel(),
            c=PROMPT_COLORS[i % len(PROMPT_COLORS)],
            marker=PROMPT_MARKERS[i % len(PROMPT_MARKERS)],
            s=9,
            alpha=0.55,
            depthshade=False,
            label=label,
        )
    ax.set_xlabel("head")
    ax.set_ylabel("channel")
    ax.set_zlabel("offset")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prompt_lines(
    values: np.ndarray,
    labels: list[str],
    head: int,
    title: str,
    path: Path,
) -> None:
    """3D parallel view for one head: one line per prompt, color+marker coded."""
    prompts, heads, channels = values.shape
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(channels)
    for i, label in enumerate(labels):
        ax.plot(
            x,
            np.full_like(x, i),
            values[i, head, :],
            color=PROMPT_COLORS[i % len(PROMPT_COLORS)],
            marker=PROMPT_MARKERS[i % len(PROMPT_MARKERS)],
            markersize=2,
            linewidth=1.2,
            label=label,
        )
    ax.set_xlabel("channel")
    ax.set_ylabel("prompt")
    ax.set_zlabel("offset")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prompt_panels(
    values: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    head_stride: int = 2,
    channel_stride: int = 2,
) -> None:
    """One 3D surface per prompt, same axes/color range, side by side."""
    prompts, heads, channels = values.shape
    heads_idx = np.arange(0, heads, head_stride)
    channels_idx = np.arange(0, channels, channel_stride)
    hh, cc = np.meshgrid(heads_idx, channels_idx, indexing="ij")
    zmin = float(values.min())
    zmax = float(values.max())

    fig = plt.figure(figsize=(18, 5.5))
    for i, label in enumerate(labels):
        ax = fig.add_subplot(1, prompts, i + 1, projection="3d")
        z = values[i][hh, cc]
        ax.plot_surface(
            hh,
            cc,
            z,
            cmap="viridis",
            linewidth=0,
            rstride=1,
            cstride=1,
            vmin=zmin,
            vmax=zmax,
            antialiased=False,
        )
        ax.set_xlabel("head")
        ax.set_ylabel("channel")
        ax.set_zlabel("offset")
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=24, azim=-62)
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """relL2, cosine, Pearson, max|diff| between two flat offset tensors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = a - b
    rel = np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12)
    cosine = float(
        np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    )
    va = a - a.mean()
    vb = b - b.mean()
    pearson = float(
        np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12)
    )
    return float(rel), cosine, pearson, float(np.abs(diff).max())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pattern", required=True,
                        help="glob for task directories, e.g. '20261116_*'")
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--exclude", default="consensus")
    parser.add_argument("--metrics-only", action="store_true",
                        help="skip plot generation; useful with --layers all")
    args = parser.parse_args(argv)

    tasks = sorted(
        path for path in args.root.glob(args.pattern)
        if path.is_dir()
        and (path / "offsets").is_dir()
        and args.exclude not in path.name
    )
    if len(tasks) < 2:
        raise SystemExit(f"need >=2 tasks, found {len(tasks)}: {tasks}")

    prefix = args.pattern.replace("*", "").strip("_") or "offsets"
    out_dir = args.out or (HERE.parent / "plots" / prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [prompt_label(task.name) for task in tasks]

    summary: dict = {"tasks": [task.name for task in tasks], "layers": {}}
    rel_rows: list[dict] = []
    sim_rows: list[dict] = []
    layers = (
        sorted(path.stem for path in (tasks[0] / "offsets").glob("*.pt"))
        if args.layers == ["all"]
        else args.layers
    )

    for layer in layers:
        layer_stats = {}
        for name in ("q_offset", "k_offset"):
            offsets = torch.stack([
                load_one(task, layer, name) for task in tasks
            ])  # [P,H,D]
            mean = offsets.mean(dim=0)
            std = offsets.std(dim=0, unbiased=False)

            rel_to_mean = [rel_l2(offsets[i], mean) for i in range(len(tasks))]
            pairwise = np.zeros((len(tasks), len(tasks)), dtype=np.float64)
            for i in range(len(tasks)):
                for j in range(len(tasks)):
                    pairwise[i, j] = rel_l2(offsets[i], offsets[j])
            for i, j in itertools.combinations(range(len(tasks)), 2):
                rel, cosine, pearson, max_abs = similarity(
                    offsets[i].numpy(), offsets[j].numpy()
                )
                sim_rows.append({
                    "layer": layer,
                    "tensor": name,
                    "pair": f"{labels[i]}-{labels[j]}",
                    "rel_l2": rel,
                    "cosine": cosine,
                    "pearson": pearson,
                    "max_abs": max_abs,
                })

            tag = f"{layer}_{name}"
            if not args.metrics_only:
                surface(
                    mean.numpy(),
                    f"{tag}: mean offset across {len(tasks)} prompts",
                    out_dir / f"offset_3d_{tag}_mean.png",
                )
                surface(
                    std.numpy(),
                    f"{tag}: std across {len(tasks)} prompts",
                    out_dir / f"offset_3d_{tag}_std.png",
                )
                heatmap(
                    std.numpy(),
                    f"{tag}: per-(head,channel) std across prompts",
                    out_dir / f"offset_heatmap_{tag}_std.png",
                )
                histogram(
                    offsets.numpy(),
                    f"{tag}: offset distribution over prompts",
                    out_dir / f"offset_hist_{tag}.png",
                )
                prompt_panels(
                    offsets.numpy(),
                    labels,
                    f"{tag}: per-prompt offsets (one panel per prompt)",
                    out_dir / f"offset_3d_{tag}_prompts.png",
                )
                head = int(std.sum(dim=1).argmax().item())
                prompt_lines(
                    offsets.numpy(),
                    labels,
                    head,
                    f"{tag}: per-prompt offsets, head {head}",
                    out_dir / f"offset_3d_{tag}_prompts_head{head}.png",
                )

            layer_stats[name] = {
                "mean_abs": mean.abs().mean().item(),
                "max_abs": mean.abs().max().item(),
                "std_across_prompts_mean": std.mean().item(),
                "std_across_prompts_max": std.max().item(),
                "relative_std": (std.mean() / mean.abs().mean().clamp_min(1e-12)).item(),
                "relL2_to_mean": rel_to_mean,
                "pairwise_relL2": pairwise.tolist(),
            }
            for i, task in enumerate(tasks):
                rel_rows.append({
                    "layer": layer,
                    "tensor": name,
                    "task": task.name,
                    "relL2_to_mean": rel_to_mean[i],
                })
            print(
                f"{tag:32s} mean|off|={mean.abs().mean():.4f} "
                f"max|off|={mean.abs().max():.4f} "
                f"std={std.mean():.4f} (max {std.max():.4f}) "
                f"relL2_to_mean={['%.4f' % x for x in rel_to_mean]}"
            )
        summary["layers"][layer] = layer_stats

    with (out_dir / "offset_prompt_relL2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["layer", "tensor", "task", "relL2_to_mean"]
        )
        writer.writeheader()
        writer.writerows(rel_rows)

    with (out_dir / "offset_similarity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer", "tensor", "pair", "rel_l2", "cosine",
                        "pearson", "max_abs"],
        )
        writer.writeheader()
        writer.writerows(sim_rows)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in sim_rows:
        groups[(row["tensor"], row["pair"])].append(row)
    sim_summary = []
    for (tensor, pair), items in sorted(groups.items()):
        rels = np.asarray([row["rel_l2"] for row in items])
        cosines = np.asarray([row["cosine"] for row in items])
        pearsons = np.asarray([row["pearson"] for row in items])
        sim_summary.append({
            "tensor": tensor,
            "pair": pair,
            "layers": len(items),
            "rel_l2_mean": float(rels.mean()),
            "rel_l2_median": float(np.median(rels)),
            "rel_l2_max": float(rels.max()),
            "cosine_mean": float(cosines.mean()),
            "pearson_mean": float(pearsons.mean()),
        })
    with (out_dir / "offset_similarity_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sim_summary[0]))
        writer.writeheader()
        writer.writerows(sim_summary)

    summary["similarity_summary"] = sim_summary
    with (out_dir / "offset_stats.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nsimilarity across all audited layers (mean over layers):")
    print(f"{'tensor':10s} {'pair':22s} {'relL2 med':>10s} {'relL2 max':>10s} "
          f"{'cosine':>8s} {'Pearson':>8s}")
    for row in sim_summary:
        print(
            f"{row['tensor']:10s} {row['pair']:22s} "
            f"{row['rel_l2_median']:10.4f} {row['rel_l2_max']:10.4f} "
            f"{row['cosine_mean']:8.5f} {row['pearson_mean']:8.5f}"
        )

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
