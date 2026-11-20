#!/usr/bin/env python3
"""MiniMax-H3 generation client.

The client submits one request, waits for the video job, and downloads the
result.  It covers the three H3 tasks used by the examples:

    python gen_video.py ref2va --tag ref --reference clip.mp4
    python gen_video.py fl2va --tag keyframes --first-frame first.png
    python gen_video.py t2va --tag calib --prompt-file prompt.txt --steps 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import requests

DEFAULT_MODEL = "/home/yiakwy/workspace/Github/models/MiniMax-H3"
DEFAULT_REF2VA_PROMPT = (
    "Follow the motion and appearance of <Video 1>, preserving the setting, "
    "the camera motion and the timing."
)
DEFAULT_REFERENCE = (
    "file:///home/yiakwy/workspace/Github/models/MiniMax-H3/assets/ref2va.mp4"
)


def _uri(value: str) -> str:
    if "://" in value:
        return value
    path = pathlib.Path(value).expanduser()
    return path.resolve().as_uri() if path.exists() else value


def _target(args) -> dict:
    return {
        "short_edge": args.short_edge,
        "aspect_ratio": args.aspect,
        "duration_seconds": args.seconds,
    }


def build_payload(args, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "seconds": args.seconds,
        "task": args.command,
        "conditions": [],
        "target": _target(args),
        "num_outputs_per_prompt": 1,
        "num_inference_steps": args.steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": args.seed,
    }

    if args.command == "ref2va":
        payload["conditions"] = [
            {
                "type": "video",
                "uri": _uri(args.reference),
                "role": "reference",
                "start_time_seconds": 0.0,
            }
        ]
    elif args.command == "fl2va":
        if args.first_frame:
            payload["conditions"].append(
                {
                    "type": "image",
                    "uri": _uri(args.first_frame),
                    "role": "keyframe",
                    "frame_index": 0,
                }
            )
        if args.last_frame:
            payload["conditions"].append(
                {
                    "type": "image",
                    "uri": _uri(args.last_frame),
                    "role": "keyframe",
                    "frame_index": -1,
                }
            )
    else:
        payload["size"] = args.size

    return payload


def _read_prompt(args) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text().strip()
    if args.prompt:
        return args.prompt
    if args.command == "ref2va":
        return DEFAULT_REF2VA_PROMPT
    raise SystemExit(f"{args.command} needs --prompt or --prompt-file")


def _download(url: str, out: pathlib.Path, timeout: int) -> None:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", required=True)
    common.add_argument("--port", type=int, default=30010)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--prompt")
    common.add_argument("--prompt-file", type=pathlib.Path)
    common.add_argument("--seconds", type=float, default=4.0)
    common.add_argument("--steps", type=int, default=8)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--timeout", type=int, default=3600)
    common.add_argument("--out", type=pathlib.Path)
    common.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("/tmp/opencode/e2e"))
    common.add_argument("--size", default="1344x768")
    common.add_argument("--aspect", default="16:9")
    common.add_argument("--short-edge", type=int, default=768)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    ref2va = sub.add_parser("ref2va", parents=[common])
    ref2va.add_argument("--reference", default=DEFAULT_REFERENCE)

    fl2va = sub.add_parser("fl2va", parents=[common])
    fl2va.add_argument("--first-frame")
    fl2va.add_argument("--last-frame")

    sub.add_parser("t2va", parents=[common])

    args = parser.parse_args(argv)
    if args.command == "fl2va" and not (args.first_frame or args.last_frame):
        parser.error("fl2va needs --first-frame, --last-frame, or both")

    prompt = _read_prompt(args)
    payload = build_payload(args, prompt)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    base = f"http://localhost:{args.port}"
    start = time.time()
    response = requests.post(f"{base}/v1/videos", json=payload, timeout=120)
    if not response.ok:
        print(f"POST failed {response.status_code}: {response.text[:2000]}")
        return 1

    video_id = response.json()["id"]
    print(f"[{args.tag}] queued id={video_id}")

    deadline = start + args.timeout
    while time.time() < deadline:
        time.sleep(5)
        info = requests.get(f"{base}/v1/videos/{video_id}", timeout=60).json()
        status = info.get("status")
        if status in ("completed", "succeeded", "done"):
            break
        if status in ("failed", "error"):
            print(f"[{args.tag}] failed: {json.dumps(info)[:2000]}")
            return 1
        if int(time.time() - start) % 60 < 5:
            stage = info.get("stage") or info.get("progress") or ""
            print(f"[{args.tag}] {time.time() - start:7.1f}s {status} {stage}", flush=True)
    else:
        print(f"[{args.tag}] timed out after {args.timeout}s")
        return 1

    out = args.out or (args.out_dir / f"{args.command}_{args.tag}.mp4")
    _download(f"{base}/v1/videos/{video_id}/content", out, timeout=1200)
    print(f"[{args.tag}] done in {time.time() - start:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
