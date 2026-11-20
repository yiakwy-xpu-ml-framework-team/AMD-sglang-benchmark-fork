#!/bin/bash
# MiniMax-H3 inference for the bullet-time rotating-platform prompt.
#
#   ./infer.sh
#   BACKEND=fa TAG=fp16 ./infer.sh
#   BACKEND=fp8_fa_sm120 TAG=fp8_sm120a ./infer.sh
#   BACKEND=rot_nvfp4 TAG=nvfp4 ./infer.sh
#   STEPS=24 PORT=30010 OUT=/tmp ./infer.sh
#   REUSE_SERVER=1 ./infer.sh
#
# The attention backend is the only A/B variable.  Model, prompt, seed, size
# and step count stay fixed, so PSNR between two outputs isolates attention
# precision.
set -e
# PY=/opt/sglang/bin/python3
PY=${PY:-/home/yiakwy/venvs/sglang-h3/bin/python}

HERE=$(cd "$(dirname "$0")" && pwd)

PORT=${PORT:-30110}

OUT=${OUT:-$(pwd)}
mkdir -p "$OUT"

STEPS=${STEPS:-30}
SECONDS_=${SECONDS_:-10}
SIZE=${SIZE:-768x1344}
ASPECT=${ASPECT:-9:16}
SHORT_EDGE=${SHORT_EDGE:-768}

SEED=${SEED:-42}
BACKEND=${BACKEND:-rot_fp8}

# MODEL=${MODEL:-/raid/yiakwy/MiniMax-H3}
MODEL=${MODEL:-/home/yiakwy/workspace/Github/models/MiniMax-H3}

# `REPO` is this SGLang tree: examples/h3_fa3 -> repo root.  The kernel
# research tree (attn_qat_infer + fp4attn/fp4preprocess) is `KERNEL_REPO`.
# launch_fp8_server.sh still calls its kernel tree `REPO`, so launch() below
# maps KERNEL_REPO -> REPO and passes SGL explicitly.
REPO=$(cd "$HERE/../.." && pwd)
SGL=${SGL:-$REPO}
KERNEL_REPO=${KERNEL_REPO:-/home/yiakwy/workspace/Github/HKUST-Rotate_fp4_attn_qat-fork}

# GB10 (SM121) defaults, matching launch_nvfp4_server.sh.  Override for other
# topologies, e.g. NG=4 TP=2 ULY=2 on a multi-GPU Hopper host.
NG=${NG:-1}
TP=${TP:-1}
ULY=${ULY:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DIT_OFFLOAD=${DIT_OFFLOAD:-0}
LAYERWISE=${LAYERWISE:-text_encoder,vae}
RESIDENCY=${RESIDENCY:-transformer=resident text_encoder=layerwise-offload vae=resident}
VARIANT=${VARIANT:-fl2va}

NVFP4_REPO=${NVFP4_REPO:-$KERNEL_REPO}
NVFP4_SO_DIR=${NVFP4_SO_DIR:-$NVFP4_REPO/prebuilt/torch214}
NVFP4_OFFSETS=${NVFP4_OFFSETS:-/tmp/opencode/nvfp4_offsets_h3}
NVFP4_MIX=${NVFP4_MIX:-all}
NVFP4_MIX_BACKEND=${NVFP4_MIX_BACKEND:-sdpa}
FP8_OFFSETS=${FP8_OFFSETS:-}
FP8_CALIB_MODE=${FP8_CALIB_MODE:-offline}
FP8_SENSITIVE_LAYERS=${FP8_SENSITIVE_LAYERS:-}
FP8_ROTATE=${FP8_ROTATE:-1}

case "$BACKEND" in
  fa)        TAG=${TAG:-fp16} ;;
  rot_fp8)   TAG=${TAG:-fp8}  ;;
  fp8_fa_sm120) TAG=${TAG:-fp8_sm120a} ;;
  rot_nvfp4) TAG=${TAG:-nvfp4} ;;
  *)         TAG=${TAG:-$BACKEND} ;;
esac

PROMPT='超真实时尚短片，电影感，超高清 2K 细节，在明亮高调白棚环境。
场景：人物站在旋转台上，周围有三件衣服环绕他一起旋转，英伦大衣，色彩鲜明的时尚夹克，时尚棒球帽和套装。
角色：一位酷似金城武的东方男士 约 20-24岁，短发利落，轮廓深邃，身形挺拔高挑，四肢修长，内搭白衬衫，脚踩漆皮牛津鞋。
核心机制：男士立于旋转台，每一次旋转进入子弹时间，然后人物迅速更换衣服，同时衣服旋转到视频前景正中，每一帧定格时衣物在风中微扬，人物摆出舒适自然的动作，然后快速旋转到下一个姿态，更换衣服，进入子弹时间。
分镜时间线：
  0-1秒：进入子弹时间，缓慢旋转，人物摆出轻松的动作，然后人物随转台迅速旋转1/3，更换为英伦大衣；
  1-2秒：再次进入子弹时间，人物摆出一个新造型缓慢旋转，然后人物随转台迅速再次旋转1/3，更换为色彩鲜明的时尚夹克；
  2-3秒：再次进入子弹时间，人物摆出一个新造型缓慢旋转，然后人物随转台迅速再次旋转1/3，更换为时尚棒球帽和套装；
  3-4秒：再次进入子弹时间，人物摆出夸张的棒球造型缓慢旋转，然后人物随转台迅速再次旋转1/3，恢复开始的造型。
细节关键词：皮肤质感、丝绸光泽、绒布纹理、金属扣件反光、灰尘粒子、8K 超清、锐利边缘、体积光、电影颗粒感。'

# ---- server management ----------------------------------------------------
server_backend() {
  # echo the attention backend of a live server on $PORT, or "" if none
  for p in $(pgrep -f "sglang.*serve.*--port $PORT" 2>/dev/null); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null \
      | sed -n -e 's/.*--attention-backend \([a-z0-9_]*\).*/\1/p' \
               -e 's/.*--component-attention-backends transformer=\([a-z0-9_]*\).*/\1/p'
    return 0
  done
  echo ""
}

launch() {
  echo "== launching server: backend=$BACKEND port=$PORT gpus=$CUDA_VISIBLE_DEVICES tp=$TP uly=$ULY =="
  tmux kill-session -t h3serve 2>/dev/null || true
  sleep 5
  # tmux inherits the *server* environment, not this shell's: pass everything
  # the launcher needs (paths, GPU set, topology, backend) explicitly.
  tmux new-session -d -s h3serve \
    "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' PORT='$PORT' BACKEND='$BACKEND' \
     PY='$PY' MODEL='$MODEL' VARIANT='$VARIANT' \
     REPO='$KERNEL_REPO' SGL='$SGL' \
     NG='$NG' TP='$TP' ULY='$ULY' DIT_OFFLOAD='$DIT_OFFLOAD' \
     LAYERWISE='$LAYERWISE' RESIDENCY='$RESIDENCY' \
     NVFP4_REPO='$NVFP4_REPO' NVFP4_SO_DIR='$NVFP4_SO_DIR' \
     NVFP4_OFFSETS='$NVFP4_OFFSETS' NVFP4_MIX='$NVFP4_MIX' \
     NVFP4_MIX_BACKEND='$NVFP4_MIX_BACKEND' \
     FP8_OFFSETS='$FP8_OFFSETS' FP8_CALIB_MODE='$FP8_CALIB_MODE' \
     FP8_SENSITIVE_LAYERS='$FP8_SENSITIVE_LAYERS' FP8_ROTATE='$FP8_ROTATE' \
     SGLANG_ROT_NVFP4_SAVE_OFFSETS='${SGLANG_ROT_NVFP4_SAVE_OFFSETS:-0}' \
     SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS='${SGLANG_ROT_FP8_SM120A_SAVE_OFFSETS:-0}' \
     $HERE/launch_fp8_server.sh 2>&1 | tee /tmp/h3_${TAG}_server.log"
  echo "waiting for server..."
  for i in $(seq 1 120); do
    sleep 5
    curl -sf http://localhost:$PORT/health >/dev/null 2>&1 && { echo "server up"; return 0; }
  done
  echo "SERVER FAILED — check /tmp/h3_${TAG}_server.log"; exit 1
}

if [ "${REUSE_SERVER:-0}" = "1" ]; then
  echo "reusing server on port $PORT (REUSE_SERVER=1)"
elif ! curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
  launch
elif [ "$(server_backend)" != "$BACKEND" ]; then
  echo "server running with backend='$(server_backend)', need '$BACKEND' — relaunching..."
  launch
else
  echo "server running with backend=$BACKEND ✓"
fi

# ---- generation -------------------------------------------------------------
OUT_FILE=$OUT/bullettime_${TAG}_${STEPS}st_seed${SEED}.mp4
echo "== generating ($STEPS steps, $SECONDS_ s, $SIZE, seed $SEED, $TAG = $BACKEND) =="

"$PY" "$HERE/gen_video.py" t2va \
  --tag "$TAG" --port "$PORT" --model "$MODEL" \
  --prompt "$PROMPT" --seconds "$SECONDS_" --steps "$STEPS" --seed "$SEED" \
  --size "$SIZE" --aspect "$ASPECT" --short-edge "$SHORT_EDGE" \
  --out "$OUT_FILE"

# ---- frames at the beat points ----------------------------------------------
N_FRAMES=$($PY -c "print(int(24 * $SECONDS_) + 3)")
F0=0
FQ=$((N_FRAMES / 4))
FH=$((N_FRAMES / 2))
FL=$((N_FRAMES - 1))
ffmpeg -y -loglevel error -i "$OUT_FILE" \
  -vf "select=eq(n\,$F0)+eq(n\,$FQ)+eq(n\,$FH)+eq(n\,$FL)" -vsync 0 \
  "$OUT/bullettime_${TAG}_${STEPS}st_seed${SEED}_f%d.png"
echo "frames: $OUT/bullettime_${TAG}_${STEPS}st_f{1..4}.png  (at $F0/$FQ/$FH/$FL)"
echo "video:  $OUT_FILE"
