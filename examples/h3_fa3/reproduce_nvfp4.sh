#!/bin/bash
# NVFP4 kernel checks for MiniMax-H3 on SM120/SM121.
#
#   bash reproduce_nvfp4.sh
#   REAL_SHAPE=1 bash reproduce_nvfp4.sh
#
# The checks use the torch 2.14 venv and the prebuilt extensions under
# prebuilt/torch214.  The torch 2.13 build at the repo root is untouched.
set -e

PY=${PY:-/home/yiakwy/venvs/sglang-h3/bin/python}
SGL=${SGL:-/home/yiakwy/workspace/Github/sglang-fp8attn}
REPO=${REPO:-/home/yiakwy/workspace/Github/HKUST-Rotate_fp4_attn_qat-fork}
export SGLANG_ROT_NVFP4_REPO=$REPO
export SGLANG_ROT_NVFP4_SO_DIR=${SGLANG_ROT_NVFP4_SO_DIR:-$REPO/prebuilt/torch214}

echo "== kernel smoke =="
(cd "$REPO" && "$PY" smoke_test.py --seq 1024 --heads 56 --dtype bf16 --bench)

echo
echo "== kernel benchmark =="
(cd "$REPO" && "$PY" bench_blockscaled_fp4_attn.py \
    --seq 8192 --heads 56 --input-dtype bf16 --dtype bf16 \
    --warmup 3 --iters 10 --repeats 1)

echo
echo "== attention backends =="
"$PY" "$SGL/benchmark/kernels/attention/bench_h3_nvfp4.py" --seq 8192 16384 27216 --iters 5

if [ "${REAL_SHAPE:-0}" = "1" ]; then
  echo
  echo "== real ref2va shape =="
  "$PY" "$SGL/benchmark/kernels/attention/bench_h3_nvfp4.py" --seq 70339 --iters 3
fi

echo
echo "== unit tests =="
(cd "$SGL" && "$PY" -m pytest \
    python/sglang/multimodal_gen/test/unit/test_rot_nvfp4_attn.py -q)

echo
echo "== NVFP4 decomposition =="
(cd "$REPO" && "$PY" bench/nvfp4_bench/prefill/bench_nvfp4_kernels.py \
    --S 8192 --H 56 --iters 5)

if [ "${REAL_SHAPE:-0}" = "1" ]; then
  echo
  echo "== decomposition at real shape =="
  (cd "$REPO" && "$PY" bench/nvfp4_bench/prefill/bench_nvfp4_kernels.py \
      --S 70339 --H 56 --iters 3)
fi

echo
echo "All NVFP4 checks passed."
