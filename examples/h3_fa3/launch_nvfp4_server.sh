#!/bin/bash
# Start MiniMax-H3 with the Blackwell rotated-NVFP4 backend.
#
#   ./launch_nvfp4_server.sh
#   VARIANT=ref2va PORT=30010 ./launch_nvfp4_server.sh
#
# The script forwards to launch_fp8_server.sh.  SGL points at this tree,
# KERNEL_REPO at HKUST-Rotate_fp4_attn_qat-fork, and NVFP4_OFFSETS at an
# optional per-layer offset cache.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)

# SGLang tree: examples/h3_fa3 -> repo root.
SGL=${SGL:-$(cd "$HERE/../.." && pwd)}
# Kernel research tree: attn_qat_infer + fp4attn/fp4preprocess.
KERNEL_REPO=${KERNEL_REPO:-/home/yiakwy/workspace/Github/HKUST-Rotate_fp4_attn_qat-fork}
# launch_fp8_server.sh calls the kernel tree REPO.  `infer.sh` uses REPO for
# the SGLang tree, so do not let an exported REPO leak into this launcher.
REPO=$KERNEL_REPO

PY=${PY:-/home/yiakwy/venvs/sglang-h3/bin/python}
MODEL=${MODEL:-/home/yiakwy/workspace/Github/models/MiniMax-H3}
VARIANT=${VARIANT:-fl2va}
PORT=${PORT:-30110}
NG=${NG:-1}
TP=${TP:-1}
ULY=${ULY:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DIT_OFFLOAD=${DIT_OFFLOAD:-0}
LAYERWISE=${LAYERWISE:-text_encoder,vae}
RESIDENCY=${RESIDENCY:-transformer=resident text_encoder=layerwise-offload vae=resident}

NVFP4_REPO=${NVFP4_REPO:-$KERNEL_REPO}
NVFP4_SO_DIR=${NVFP4_SO_DIR:-$NVFP4_REPO/prebuilt/torch214}
NVFP4_OFFSETS=${NVFP4_OFFSETS:-/tmp/opencode/nvfp4_offsets_h3}
NVFP4_MIX=${NVFP4_MIX:-all}
NVFP4_MIX_BACKEND=${NVFP4_MIX_BACKEND:-sdpa}

export SGL REPO PY MODEL VARIANT PORT NG TP ULY CUDA_VISIBLE_DEVICES
export DIT_OFFLOAD LAYERWISE RESIDENCY
export BACKEND=rot_nvfp4
export NVFP4_REPO NVFP4_SO_DIR NVFP4_OFFSETS NVFP4_MIX NVFP4_MIX_BACKEND
# Set to 1 (and point NVFP4_OFFSETS at an empty dir) to regenerate the cache.
export SGLANG_ROT_NVFP4_SAVE_OFFSETS=${SGLANG_ROT_NVFP4_SAVE_OFFSETS:-0}

echo "== MiniMax-H3 / rotated-NVFP4 =="
echo "SGL=$SGL"
echo "KERNEL_REPO=$KERNEL_REPO"
echo "PY=$PY"
echo "MODEL=$MODEL VARIANT=$VARIANT PORT=$PORT"
echo "GPU=$CUDA_VISIBLE_DEVICES NG=$NG TP=$TP ULY=$ULY"
echo "NVFP4_SO_DIR=$NVFP4_SO_DIR"
echo "NVFP4_OFFSETS=$NVFP4_OFFSETS MIX=$NVFP4_MIX"
echo

exec bash "$HERE/launch_fp8_server.sh"
