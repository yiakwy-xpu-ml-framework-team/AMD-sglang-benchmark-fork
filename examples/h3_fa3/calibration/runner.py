"""Run one H3 calibration case against a fresh server."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PYTHON = "/home/yiakwy/venvs/sglang-h3/bin/python"
DEFAULT_MODEL = "/home/yiakwy/workspace/Github/models/MiniMax-H3"
DEFAULT_KERNEL_REPO = (
    "/home/yiakwy/workspace/Github/HKUST-Rotate_fp4_attn_qat-fork"
)
DEFAULT_SGL = str(Path(__file__).resolve().parents[3])
DEFAULT_RESIDENCY = (
    "transformer=resident text_encoder=layerwise-offload vae=resident"
)


@dataclass
class Case:
    """One server, one request, and the optional comparison."""

    name: str
    out_dir: Path
    prompt: Path | None = None
    prompt_text: str | None = None
    tag: str = "calib"
    steps: int = 1
    seed: int = 42
    seconds: float = 4.0
    size: str = "1344x768"
    aspect: str = "9:16"
    short_edge: int = 768
    task: str = "t2va"
    backend: str = "rot_nvfp4"
    variant: str = "fl2va"
    reference_video: str | None = None
    first_frame: str | None = None
    last_frame: str | None = None
    out_file: Path | None = None
    reference: Path | None = None
    server_env: dict[str, str] = field(default_factory=dict)
    expect_offsets: int = 0
    post: str | None = None


@dataclass
class RunConfig:
    root: Path
    python: str = DEFAULT_PYTHON
    model: str = DEFAULT_MODEL
    kernel_repo: str = DEFAULT_KERNEL_REPO
    sgl: str = DEFAULT_SGL
    port: int = 30110
    cuda_visible_devices: str = "0"
    ng: int = 1
    tp: int = 1
    uly: int = 1
    residency: str = DEFAULT_RESIDENCY
    dit_offload: str = "0"
    layerwise: str = "text_encoder,vae"
    keep_server: bool = False
    dry_run: bool = False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--kernel-repo", default=DEFAULT_KERNEL_REPO)
    parser.add_argument("--sgl", default=None)
    parser.add_argument("--port", type=int, default=30110)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--ng", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--uly", type=int, default=1)
    parser.add_argument("--residency", default=DEFAULT_RESIDENCY)
    parser.add_argument("--dit-offload", default="0")
    parser.add_argument("--layerwise", default="text_encoder,vae")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace, default_root: Path) -> RunConfig:
    root = (args.root or default_root).expanduser().resolve()
    return RunConfig(
        root=root,
        python=args.python,
        model=args.model,
        kernel_repo=args.kernel_repo,
        sgl=args.sgl or DEFAULT_SGL,
        port=args.port,
        cuda_visible_devices=args.cuda_visible_devices,
        ng=args.ng,
        tp=args.tp,
        uly=args.uly,
        residency=args.residency,
        dit_offload=args.dit_offload,
        layerwise=args.layerwise,
        keep_server=args.keep_server,
        dry_run=args.dry_run,
    )


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_port_free(port: int, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(2)
    raise RuntimeError(f"port {port} is still busy")


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=2
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed ({code}); see {log_path}")


CASE_ENV_KEYS = (
    "NVFP4_OFFSETS",
    "NVFP4_MIX",
    "NVFP4_MIX_BACKEND",
    "NVFP4_REPO",
    "NVFP4_SO_DIR",
    "SGLANG_ROT_NVFP4_OFFSETS",
    "SGLANG_ROT_NVFP4_MIX",
    "SGLANG_ROT_NVFP4_MIX_BACKEND",
    "SGLANG_ROT_NVFP4_CALIB_MODE",
    "SGLANG_ROT_NVFP4_SENSITIVE_LAYERS",
    "SGLANG_ROT_NVFP4_SAVE_OFFSETS",
    "SGLANG_ROT_NVFP4_AUDIT",
    "SGLANG_ROT_NVFP4_AUDIT_MIN_SEQ",
    "SGLANG_ROT_NVFP4_CALIB_DEBUG",
)


class Server:
    def __init__(self, case: Case, config: RunConfig) -> None:
        self.case = case
        self.config = config
        self.log_path = case.out_dir / "logs" / f"server_{case.tag}.log"
        self.process: subprocess.Popen | None = None
        self.log = None

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in CASE_ENV_KEYS:
            env.pop(key, None)
        env.update(self.case.server_env)
        env.update(
            {
                "PY": self.config.python,
                "SGL": self.config.sgl,
                "REPO": self.config.kernel_repo,
                "MODEL": self.config.model,
                "VARIANT": self.case.variant,
                "PORT": str(self.config.port),
                "BACKEND": self.case.backend,
                "NG": str(self.config.ng),
                "TP": str(self.config.tp),
                "ULY": str(self.config.uly),
                "CUDA_VISIBLE_DEVICES": self.config.cuda_visible_devices,
                "DIT_OFFLOAD": self.config.dit_offload,
                "LAYERWISE": self.config.layerwise,
                "RESIDENCY": self.config.residency,
            }
        )
        env["NVFP4_REPO"] = self.case.server_env.get(
            "NVFP4_REPO", self.config.kernel_repo
        )
        env["NVFP4_SO_DIR"] = self.case.server_env.get(
            "NVFP4_SO_DIR", f"{self.config.kernel_repo}/prebuilt/torch214"
        )
        return env

    def __enter__(self) -> "Server":
        if self.config.dry_run:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _wait_port_free(self.config.port)
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(
            ["bash", str(self.config.root / "launch_fp8_server.sh")],
            cwd=self.config.root,
            env=self._env(),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(240):
            if self.process.poll() is not None:
                tail = _tail(self.log_path)
                raise RuntimeError(f"server exited early; see {self.log_path}\n{tail}")
            if _healthy(self.config.port):
                print(f"[{self.case.name}] server ready on {self.config.port}")
                return self
            time.sleep(5)
        self._stop()
        raise RuntimeError(f"server did not become healthy; see {self.log_path}")

    def _stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=30)
        if self.log is not None:
            self.log.close()
            self.log = None

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.config.dry_run or self.process is None:
            return
        if self.config.keep_server:
            print(f"[{self.case.name}] server kept on {self.config.port}")
            return
        self._stop()
        _wait_port_free(self.config.port)


def _generation_command(case: Case, config: RunConfig, out_file: Path) -> list[str]:
    cmd = [
        config.python,
        str(config.root / "gen_video.py"),
        case.task,
        "--tag",
        case.tag,
        "--port",
        str(config.port),
        "--model",
        config.model,
        "--seconds",
        str(case.seconds),
        "--steps",
        str(case.steps),
        "--seed",
        str(case.seed),
        "--size",
        case.size,
        "--aspect",
        case.aspect,
        "--short-edge",
        str(case.short_edge),
        "--out",
        str(out_file),
    ]
    if case.prompt is not None:
        cmd.extend(["--prompt-file", str(case.prompt)])
    elif case.prompt_text is not None:
        cmd.extend(["--prompt", case.prompt_text])
    else:
        raise ValueError(f"{case.name}: either prompt or prompt_text is required")

    if case.task == "ref2va":
        if case.reference_video:
            cmd.extend(["--reference", case.reference_video])
    elif case.task == "fl2va":
        if case.first_frame:
            cmd.extend(["--first-frame", case.first_frame])
        if case.last_frame:
            cmd.extend(["--last-frame", case.last_frame])
    return cmd


def _audit(case: Case) -> None:
    log_path = case.out_dir / "logs" / f"server_{case.tag}.log"
    lines = [
        line
        for line in log_path.read_text(errors="replace").splitlines()
        if "rot_nvfp4 audit" in line
    ]
    raw = case.out_dir / "audit_raw.txt"
    raw.write_text("\n".join(lines) + "\n")
    from analysis import rank_layers

    rank_layers.main(["--log", str(raw), "--out", str(case.out_dir)])


def run_case(case: Case, config: RunConfig) -> Path:
    case.out_dir.mkdir(parents=True, exist_ok=True)
    (case.out_dir / "logs").mkdir(exist_ok=True)
    if case.prompt is not None:
        shutil.copyfile(case.prompt, case.out_dir / "prompt.txt")

    out_file = case.out_file or (
        case.out_dir / f"calib_{case.tag}_{case.steps}st_seed{case.seed}.mp4"
    )
    print(f"[{case.name}] output {out_file}")

    with Server(case, config):
        _run_logged(
            _generation_command(case, config, out_file),
            case.out_dir / "logs" / f"client_{case.tag}.log",
        )

    if case.expect_offsets:
        offsets = list((case.out_dir / "offsets").glob("*.pt"))
        if len(offsets) < case.expect_offsets:
            raise RuntimeError(
                f"{case.name}: expected {case.expect_offsets} offsets, "
                f"found {len(offsets)}"
            )
        print(f"[{case.name}] offsets {len(offsets)}")

    if case.post == "audit":
        _audit(case)

    if case.reference is not None:
        if not case.reference.is_file():
            raise FileNotFoundError(f"reference video not found: {case.reference}")
        _run_logged(
            [
                config.python,
                str(config.root / "compare_video.py"),
                "--reference",
                str(case.reference),
                "--candidate",
                str(out_file),
            ],
            case.out_dir / f"compare_{case.tag}.txt",
        )
    return out_file


def run_cases(cases: list[Case], config: RunConfig) -> None:
    if not cases:
        raise SystemExit("no cases declared")
    if config.dry_run:
        for case in cases:
            print(
                f"{case.name}: backend={case.backend} steps={case.steps} "
                f"seed={case.seed} out={case.out_dir}"
            )
        return
    for case in cases:
        run_case(case, config)
