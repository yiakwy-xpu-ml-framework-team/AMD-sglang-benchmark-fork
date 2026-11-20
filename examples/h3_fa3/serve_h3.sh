#!/bin/bash
# Minimal production-shape baseline: the same command used for the A/B, with
# no extra attention backend.  Override MODEL/PORT/VARIANT from the shell.
set -e

exec sglang serve \
  --model-path "${MODEL:-/home/yiakwy/workspace/Github/models/MiniMax-H3}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-30010}" \
  --model-variant "${VARIANT:-ref2va}"
