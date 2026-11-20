"""Helpers shared by the task declarations."""

from __future__ import annotations

from pathlib import Path

from runner import Case

DEFAULT_PROMPT = "bullettime_v1.txt"
DEFAULT_GENERIC_OFFSETS = "/tmp/opencode/nvfp4_offsets_fl2va"


def prompt_path(root: Path, name: str = DEFAULT_PROMPT) -> Path:
    return root / "calibration" / "prompts" / name


def fp16_case(
    *,
    task_dir: Path,
    seed: int,
    steps: int,
    prompt: Path,
    seconds: float = 4.0,
    tag: str = "fp16",
) -> Case:
    return Case(
        name=f"fp16-seed{seed}",
        out_dir=task_dir,
        prompt=prompt,
        tag=tag,
        steps=steps,
        seed=seed,
        seconds=seconds,
        backend="torch_sdpa",
    )


def calib_case(
    *,
    task_dir: Path,
    offsets: str,
    mode: str,
    tag: str,
    seed: int,
    steps: int,
    prompt: Path,
    reference: Path | None = None,
    sensitive: str = "",
    seconds: float = 4.0,
    debug: bool = False,
) -> Case:
    env = {
        "NVFP4_OFFSETS": str(offsets),
        "SGLANG_ROT_NVFP4_CALIB_MODE": mode,
    }
    if sensitive:
        env["SGLANG_ROT_NVFP4_SENSITIVE_LAYERS"] = sensitive
    if debug:
        env["SGLANG_ROT_NVFP4_CALIB_DEBUG"] = "1"
    return Case(
        name=tag,
        out_dir=task_dir,
        prompt=prompt,
        tag=tag,
        steps=steps,
        seed=seed,
        seconds=seconds,
        reference=reference,
        server_env=env,
    )
