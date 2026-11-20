#!/bin/bash
# MiniMax-H3 server for the fp16, FP8 and NVFP4 attention backends.
#
#   BACKEND=fa            ./launch_fp8_server.sh
#   BACKEND=rot_fp8       ./launch_fp8_server.sh
#   BACKEND=fp8_fa_sm120  ./launch_fp8_server.sh
#   BACKEND=rot_nvfp4     ./launch_fp8_server.sh
#
# rot_fp8 is the Hopper backend, fp8_fa_sm120 is the SM120a reference, and
# rot_nvfp4 is the Blackwell NVFP4 backend.  Calls the selected kernel cannot
# serve fall back to the dense SDPA implementation.
#
# PORT defaults to 30110.  TP, ULY and NG default to 2, 2 and 4.
set -e

PY=${PY:-/opt/sglang/bin/python3}
REPO=${REPO:-/home/yiakwy/workspace/Github/Rotate_fp4_attn_qat}
SGL=${SGL:-$REPO/3rdparty/sglang}
MODEL=${MODEL:-/raid/yiakwy/MiniMax-H3}
VARIANT=${VARIANT:-fl2va}
PORT=${PORT:-30110}
BACKEND=${BACKEND:-rot_fp8}
TP=${TP:-2}
ULY=${ULY:-2}
NG=${NG:-4}
DIT_OFFLOAD=${DIT_OFFLOAD:-0}
LAYERWISE=${LAYERWISE:-text_encoder,vae}
# GB10 example: RESIDENCY='transformer=resident text_encoder=layerwise-offload vae=resident'
# keeps the 62 GB DiT resident instead of letting the auto policy stream it per step.
RESIDENCY=${RESIDENCY:-}
NVFP4_REPO=${NVFP4_REPO:-/home/yiakwy/workspace/Github/HKUST-Rotate_fp4_attn_qat-fork}
NVFP4_SO_DIR=${NVFP4_SO_DIR:-$NVFP4_REPO/prebuilt/torch214}
NVFP4_OFFSETS=${NVFP4_OFFSETS:-}
NVFP4_MIX=${NVFP4_MIX:-all}
NVFP4_MIX_BACKEND=${NVFP4_MIX_BACKEND:-sdpa}
FP8_OFFSETS=${FP8_OFFSETS:-}
FP8_CALIB_MODE=${FP8_CALIB_MODE:-offline}
FP8_SENSITIVE_LAYERS=${FP8_SENSITIVE_LAYERS:-}
FP8_ROTATE=${FP8_ROTATE:-1}
WARMUP_MODE=${WARMUP_MODE:-}

export PYTHONPATH=$SGL/python:$REPO
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
export SGLANG_PER_STEP_TIMING=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ROT_NVFP4_REPO=$NVFP4_REPO
export SGLANG_ROT_NVFP4_SO_DIR=$NVFP4_SO_DIR
export SGLANG_ROT_NVFP4_MIX=$NVFP4_MIX
export SGLANG_ROT_NVFP4_MIX_BACKEND=$NVFP4_MIX_BACKEND
if [ -n "$NVFP4_OFFSETS" ]; then
  export SGLANG_ROT_NVFP4_OFFSETS=$NVFP4_OFFSETS
fi
export SGLANG_ROT_FP8_SM120A_ROTATE=$FP8_ROTATE
export SGLANG_ROT_FP8_SM120A_CALIB_MODE=$FP8_CALIB_MODE
if [ -n "$FP8_OFFSETS" ]; then
  export SGLANG_ROT_FP8_SM120A_OFFSETS=$FP8_OFFSETS
fi
if [ -n "$FP8_SENSITIVE_LAYERS" ]; then
  export SGLANG_ROT_FP8_SM120A_SENSITIVE_LAYERS=$FP8_SENSITIVE_LAYERS
fi
export SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS=${SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS:-0}

# The fp8 quantizer (.so in the research repo root, JIT fallback otherwise) is
# importable because $REPO is on PYTHONPATH above. rot_nvfp4 is component-scoped
# so the text encoder keeps its default backend.
exec "$PY" -u -c "
import sys
sys.path.insert(0, '$SGL/python')
sys.path.insert(0, '$REPO')
sys.argv = ['sglang', 'serve',
    '--model-path', '$MODEL',
    '--num-gpus', '$NG', '--tp-size', '$TP', '--ulysses-degree', '$ULY',
    '--host', '0.0.0.0', '--port', '$PORT',
    '--encoder-parallel', 'auto', '--model-variant', '$VARIANT',
    '--performance-mode', 'speed'] + (
    ['--component-attention-backends', 'transformer=rot_nvfp4']
    if '$BACKEND' == 'rot_nvfp4' else ['--attention-backend', '$BACKEND']) + (
    ['--warmup-mode', '$WARMUP_MODE'] if '$WARMUP_MODE' else []) + (
    ['--dit-layerwise-offload', 'true'] if '$DIT_OFFLOAD' == '1' else []) + (
    ['--layerwise-offload-components', '$LAYERWISE'] if '$LAYERWISE' else []) + (
    ['--component-residency'] + '$RESIDENCY'.split() if '$RESIDENCY' else [])
from sglang.cli.main import main
main()
"
