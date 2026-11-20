# Prompt calibration and layer-mix results (fl2va, GB10)

Using the same prompt in infer.sh as FP8, to produce 4s, 768x1344, 24 fps,
107 frames video.  

Using fp16/BF16 torch SDPA as baseline, 8 steps, 33.77 s/step on GB10.

Run the commands from examples/h3_fa3.  Prompt ids are v1 for the exact
infer.sh prompt, v1_resample_1 and v1_resample_2 for two rewrites.  Historical
task directories keep the old _orig/_para/_keywords suffixes; the tables below
use the new ids.

## 1. Offsets are only weakly prompt-dependent

### Reproduce

```bash
python calibration/calib.py run prompt-sweep --prefix 20261120_fl2va_prompt

python calibration/calib.py analyze offsets \
  --pattern '20261120_fl2va_prompt_*' \
  --layers blocks.0.attn blocks.25.attn blocks.49.attn

python calibration/calib.py analyze layer-groups \
  --pattern '20261120_fl2va_prompt_*'
```

The tables below are from the 20261116 run.  The commands write a new prefix
so a rerun does not mix with the historical task directories.  Each prompt
gets a fresh server with request warmup, so calibration sees that prompt
rather than the synthetic server warmup prompt.

### Data

The offset magnitude is stable across layers, while the prompt-to-prompt
change is a smaller part of it:

| layer | tensor | mean abs offset | std across prompts | relL2 to mean (v1 / resample_1 / resample_2) |
|---|---|---:|---:|---|
| blocks.0 | q | 0.439 | 0.103 | 0.436 / 0.224 / 0.268 |
| blocks.0 | k | 0.467 | 0.114 | 0.457 / 0.224 / 0.282 |
| blocks.25 | q | 0.693 | 0.152 | 0.294 / 0.167 / 0.205 |
| blocks.25 | k | 0.784 | 0.168 | 0.306 / 0.168 / 0.202 |
| blocks.49 | q | 1.503 | 0.272 | 0.441 / 0.203 / 0.282 |
| blocks.49 | k | 2.057 | 0.333 | 0.322 / 0.145 / 0.205 |

Pairwise q_offset relative L2 at the same layers:

| pair | block 0 | block 25 | block 49 |
|---|---:|---:|---:|
| v1–resample_1 | 0.608 | 0.420 | 0.589 |
| v1–resample_2 | 0.654 | 0.468 | 0.678 |
| resample_1–resample_2 | 0.219 | 0.224 | 0.204 |

Across all 52 layers:

| pair | q relL2 | q cosine | k relL2 | k cosine |
|---|---:|---:|---:|---:|
| v1–resample_1 | 0.391 | 0.915 | 0.384 | 0.918 |
| v1–resample_2 | 0.429 | 0.900 | 0.419 | 0.905 |
| resample_1–resample_2 | 0.190 | 0.981 | 0.179 | 0.983 |

### Conclusion

The shared component dominates.  Across all 104 q/k layer vectors, the
per-prompt standard deviation is 10–28% of the mean absolute offset, with a
median of 20%.  The mean prompt-to-prompt relative L2 across all 52 layers is
0.18–0.43, and cosine similarity is 0.90–0.98.  The two rewrites are much
closer to each other than either is to v1, so v1 is the outlier.  The offset is
therefore only weakly prompt-dependent, not prompt-specific.

This is enough to calibrate offline: one cache captures most of the vector,
and per-request online calibration is not required.  A cache based on a very
different prompt can still lose accuracy, so a prompt-specific cache remains
useful when the prompt is known.

## 2. Consensus offsets do not help

### Reproduce

```bash
python calibration/calib.py analyze consensus \
  --pattern '20261116_fl2va_prompt_*' \
  --out outputs/calibration/20261116_fl2va_prompt_consensus

python calibration/calib.py run calib-mode \
  --task-dir outputs/calibration/20261116_fl2va_consensus \
  --offsets outputs/calibration/20261116_fl2va_prompt_consensus/offsets \
  --mode offline --tag consensus --steps 8 --seed 42 \
  --reference outputs/tests/bullettime_fp16_8st_seed42.mp4
```

The baseline rows use the generic warmup cache and the v1 1-step cache from
sections 5 and 6.

### Data

| 8-step v1 config | PSNR mean | PSNR min | SSIM mean | per-step |
|---|---:|---:|---:|---:|
| fp16 SDPA reference | — | — | — | 33.77 s |
| all NVFP4, generic warmup cache | 11.35 dB | 8.09 | 0.7557 | 26.66 s |
| consensus cache | 11.16 dB | 8.85 | 0.7479 | 26.8 s |

### Conclusion

Consensus is 0.19 dB worse than the generic warmup cache and 2.58 dB worse
than the v1 1-step cache.  Averaging across prompts moves the cache toward the
rewrites and away from v1; it does not make the cache more robust.  Use the
generic cache or a prompt-specific cache, not a consensus.

## 3. Layer skip is not the quality lever

### Reproduce

```bash
python calibration/calib.py run layer-audit \
  --task-dir outputs/calibration/20261116_fl2va_prompt_audit \
  --offsets outputs/calibration/20261116_fl2va_prompt_orig/offsets

python calibration/calib.py run mix-test \
  --task-dir outputs/calibration/20261116_fl2va_mix_skip5 \
  --mix-file outputs/calibration/20261116_fl2va_prompt_audit/mix_skip_top5.txt \
  --tag skip5 --steps 8

python calibration/calib.py run mix-test \
  --task-dir outputs/calibration/20261116_fl2va_mix_skip5_fp8 \
  --mix-file outputs/calibration/20261116_fl2va_prompt_audit/mix_skip_top5.txt \
  --mix-backend fp8 --tag skip5fp8 --steps 8

python calibration/calib.py run mix-test \
  --task-dir outputs/calibration/20261116_fl2va_mix_skip10_fp8 \
  --mix-file outputs/calibration/20261116_fl2va_prompt_audit/mix_skip_top10.txt \
  --mix-backend fp8 --tag skip10fp8 --steps 8
```

### Data

fl2va audit at S=32997, one step, relL2 against SDPA.  Worst ten layers:

```text
45: 0.3108   49: 0.2275    1: 0.2125    3: 0.1387   46: 0.1321
43: 0.1308   42: 0.1297   41: 0.1242   39: 0.1233   19: 0.1198
```

All 8-step v1 videos against fp16:

| config | PSNR | SSIM | per-step | ΔPSNR vs all-NVFP4 |
|---|---:|---:|---:|---:|
| all NVFP4 | 11.35 | 0.7557 | 26.66 s | — |
| skip top5, SDPA | 11.54 | 0.7665 | 27.3 s | +0.19 |
| skip top5, FP8 | 11.63 | 0.7575 | 26.8 s | +0.28 |
| skip top10, FP8 | 11.35 | 0.7489 | 26.9 s | 0.00 |
| consensus offsets | 11.16 | 0.7479 | 26.8 s | −0.19 |

### Conclusion

The spread is under 0.3 dB and is not monotonic with the number of skipped
layers.  It is comparable to generative trajectory noise.  Five or ten layers
do not explain the quality gap; the 4-bit error is distributed over all 50
layers.

## 4. Calibration cost

### Reproduce

```bash
python calibration/calib.py analyze bench-offsets \
  --cache outputs/calibration/20261116_fl2va_prompt_orig/offsets \
  --seq 32997 --heads 56 --layers 52
```

The cache path is optional.  With it the command also times loading all 52
per-layer files.

### Data

```text
offline load: 23.1 ms total, 0.44 ms/layer, 52 layers
mean-first: 4.6 ms/tensor, 0.5 s for 52 layers, 1.03 GiB peak
rotate-then-mean: 45.6 ms/tensor, 4.7 s for 52 layers, 2.68 GiB peak
```

The 0.5 s and 4.7 s totals cover both Q and K for every layer.

### Conclusion

Mean-first calibration is about 10x faster than rotate-then-mean and uses
about 2.6x less peak memory.  Loading a complete offline cache costs about
23 ms, so a cached server pays almost nothing.  Only a first request without a
cache pays the 0.5 s online cost; the slower join order is not worth using.

An earlier ad-hoc run reported 263.9 ms and 8.4 ms per tensor.  The figures
above come from the command in this section and are the current reference;
the ratio and the memory reduction are the durable result.

## 5. Runtime calibration modes

### Reproduce

```bash
python calibration/calib.py run calib-mode \
  --task-dir outputs/calibration/20261116_fl2va_offline_orig \
  --offsets outputs/calibration/20261116_fl2va_prompt_orig/offsets \
  --mode offline --tag offlineorig --steps 8 --seed 42

python calibration/calib.py run calib-mode \
  --task-dir outputs/calibration/20261116_fl2va_online_all \
  --offsets /tmp/opencode/nvfp4_offsets_fl2va \
  --mode online --tag onlineall --steps 8 --seed 42

python calibration/calib.py run calib-mode \
  --task-dir outputs/calibration/20261116_fl2va_calib_hybrid_t40 \
  --offsets /tmp/opencode/nvfp4_offsets_fl2va \
  --mode hybrid \
  --sensitive-file outputs/calibration/20261116_fl2va_prompt_sensitivity/sensitive_layers_t0.40.txt \
  --tag hybrid --steps 8 --seed 42
```

### Data

The backend reads SGLANG_ROT_NVFP4_CALIB_MODE.  Offline loads an offset cache,
online recalibrates every layer per request, and hybrid recalibrates only the
layers in SGLANG_ROT_NVFP4_SENSITIVE_LAYERS.  At threshold 0.4 hybrid uses
layers 0-27, 47 and 49 online and 28-46 and 48 offline.  The debug run
confirmed the epoch hook: warmup epoch 1 with sequence 37712, real request
epoch 2 with sequence 32997, and all 50 DiT layers recalibrated on the request.

| 8-step v1 config | PSNR | SSIM | server e2e | steady step |
|---|---:|---:|---:|---:|
| fp16 SDPA | — | — | 297.93 s | 33.77 s |
| offline, warmup cache | 11.35 | 0.7557 | 241.07 s | 26.66 s |
| offline, v1 1-step cache | 13.74 | 0.7851 | 244.85 s | 26.6 s |
| hybrid, 30 online | 11.93 | 0.7648 | 246.18 s | 26.7 s |
| online, all 52 | 11.21 | 0.7471 | 247.07 s | 26.8 s |
| consensus cache | 11.16 | 0.7479 | 240.05 s | 26.8 s |

### Conclusion

Steady-state speed is unchanged at 26.6–26.8 s per step.  Online modes add
about 0.3–0.5 s to the first real step.  Relative to fp16, all NVFP4 modes are
about 1.21x end to end.

The 13.74 dB v1 cache came from a 1-step sample, while the online modes
calibrate on the 8-step request.  The single-seed PSNR ordering is therefore
not a clean comparison of calibration algorithms.  Section 6 repeats the
comparison on two seeds.

## 6. Two-seed confirmation

### Reproduce

```bash
python calibration/calib.py run seed-study --seed 42 --steps 8
python calibration/calib.py run seed-study --seed 43 --steps 8
```

The command runs the fp16 reference, generic all-load, hybrid and v1 1-step
cache arms in sequence, each with its own server.

### Data

| mode | seed42 PSNR | seed43 PSNR | mean PSNR | mean e2e | mean step |
|---|---:|---:|---:|---:|---:|
| fp16 SDPA | — | — | — | 299.7 s | 33.71 s |
| all-load generic | 11.35 | 13.43 | 12.39 | 242.5 s | 26.58 s |
| hybrid 30 online | 11.93 | 13.15 | 12.54 | 246.5 s | 26.77 s |
| v1 1-step cache | 13.74 | 13.73 | 13.74 | 245.2 s | 26.63 s |

### Conclusion

Hybrid is +0.15 dB on average, but the sign is not stable: +0.58 dB at seed42
and −0.28 dB at seed43.  It is not a reliable quality win.

The v1 cache is +1.35 dB over generic all-load and is stable across seeds
(13.74 and 13.73), even though the seed42 cache is used for the seed43 run.
This is further evidence that the offset is not strongly prompt- or
seed-specific once a good one-step sample exists.

All modes run at 26.5–26.8 s per step, about 1.26x faster per step and 1.22x
end to end versus fp16.  Hybrid adds about 0.5 s per request; the v1 cache adds
a one-time 1-step probe of about 27 s per new prompt, then no per-request cost.

## 7. Layer groups: sensitive vs stable

### Reproduce

```bash
python calibration/calib.py analyze layer-groups \
  --pattern '20261116_fl2va_prompt_*'

python calibration/calib.py run calib-mode \
  --task-dir outputs/calibration/20261116_fl2va_calib_hybrid_t40 \
  --offsets /tmp/opencode/nvfp4_offsets_fl2va \
  --mode hybrid \
  --sensitive-file outputs/calibration/20261116_fl2va_prompt_sensitivity/sensitive_layers_t0.40.txt \
  --tag hybrid --steps 8 --seed 42
```

### Data

Sensitive layers are blocks 0-27, 47 and 49, 30 layers in total.  Stable
layers are blocks 28-46 and 48, 20 layers.

| group | layers | q relL2 mean | k relL2 mean | q/k cosine |
|---|---:|---:|---:|---:|
| sensitive | 30 | 0.410 | 0.398 | 0.905 / 0.910 |
| stable | 20 | 0.235 | 0.230 | 0.970 / 0.970 |
| refiner | 2 | 0.252 | 0.237 | 0.966 / 0.969 |

Runtime comparison, 8-step v1, single seed:

| mode | PSNR | SSIM | per-step | calibration cost | server e2e |
|---|---:|---:|---:|---|---:|
| all load, generic cache | 11.35 | 0.7557 | 26.66 s | 23 ms once | 241.07 s |
| sensitive online + stable load | 11.93 | 0.7648 | 26.7 s | ~0.5 s/request | 246.18 s |
| all load, v1 1-step cache | 13.74 | 0.7851 | 26.6 s | precomputed | 244.85 s |
| all online | 11.21 | 0.7471 | 26.8 s | ~0.9 s/request | 247.07 s |

### Conclusion

Stable layers are about 1.7x more prompt-consistent than sensitive layers.
The early blocks 0-27 and the late blocks 47 and 49 carry most of the prompt
variation.  This supports an offline cache for the stable group and explains
why hybrid looked promising on one seed.

Hybrid is still only +0.58 dB on seed42; the two-seed mean is +0.15 dB and the
sign flips on seed43.  The single-seed table is superseded by section 6.  The
v1 1-step cache remains the accuracy winner, but it is loaded offline.

## 8. Conclusion

All tables above can be reproduced with the commands in their sections.  The
main result is that the offsets are less prompt-dependent than the raw
magnitudes suggest: the shared component dominates, with cosine similarity
0.90–0.98 and a mean prompt-to-prompt relative L2 of 0.18–0.43 across the 52
layers.  Offline calibration is therefore the right default, and a generic
cache is a reasonable fallback.

A prompt-specific 1-step cache is still worth precomputing when the prompt is
known: it is stable across seeds and +1.35 dB over the generic cache.  Load it
offline; do not recalibrate per request.  Hybrid is optional and not a
reliable win, online-all is not recommended, and a consensus cache does not
help.

Layer skipping gives at most +0.28 dB, within noise.  All-NVFP4 is a
speed-first mode; for texture close to fp16, use FP8 for a larger fraction of
the layers, or accept more steps or a different schedule.  The 4-step
"no bullet time" behavior is a sampling-schedule issue; 8 steps already
restores more of the intended beat structure.
