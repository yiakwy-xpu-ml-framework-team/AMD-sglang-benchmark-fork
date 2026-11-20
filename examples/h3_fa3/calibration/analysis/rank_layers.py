"""Rank fl2va NVFP4 layers by audit relL2 and build a `layers:` mix string.

    python calibration/calib.py analyze rank-layers --log <audit_raw.txt> \
        --out <task_dir> --top-k 5 --total 50
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--total", type=int, default=50)
    args = parser.parse_args(argv)

    text = args.log.read_text(errors="replace")
    rows = []
    for prefix, seq, rel in re.findall(
        r"rot_nvfp4 audit\[(.*?)\]: S=(\d+) relL2=([0-9.]+)", text
    ):
        match = re.match(r"^blocks\.(\d+)\.attn$", prefix)
        if not match:
            continue
        rows.append({
            "layer": int(match.group(1)),
            "prefix": prefix,
            "seq": int(seq),
            "relL2": float(rel),
        })

    if not rows:
        raise SystemExit(f"no layer audit lines in {args.log}")

    rows.sort(key=lambda row: row["relL2"], reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "layer_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["rank", "layer", "prefix", "seq", "relL2"]
        )
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})

    skip = {row["layer"] for row in rows[: args.top_k]}
    selected = [index for index in range(args.total) if index not in skip]
    mix = "layers:" + ",".join(str(index) for index in selected)
    (args.out / f"mix_skip_top{args.top_k}.txt").write_text(mix + "\n")

    print(f"audited {len(rows)} DiT layers")
    print(f"worst {args.top_k}: {sorted(skip)}")
    print(f"relL2 range: {rows[0]['relL2']:.4f} .. {rows[-1]['relL2']:.4f}")
    print("\nrank  layer  relL2   prefix")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:4d}  {row['layer']:5d}  {row['relL2']:.4f}  {row['prefix']}")
    print(f"\nwrote {args.out/'layer_ranking.csv'}")
    print(f"wrote {args.out/f'mix_skip_top{args.top_k}.txt'}")


if __name__ == "__main__":
    main()
