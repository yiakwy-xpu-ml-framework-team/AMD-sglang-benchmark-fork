# Prompt-dependent NVFP4 calibration

The calibration study asks how much the per-layer Q/K offsets move when the
prompt changes.  Each offset is a per-head, per-channel vector in the H64
rotated domain:

    o_q = H64(mean over batch and tokens of Q)
    o_k = H64(mean over batch and tokens of K)

The implementation reduces first and rotates the small mean, which is exact
because H64 is linear.  The vector is captured on the first forward of the
calibration request and then reused for the rest of that generation.

## Layout

    calib.py                 single entry point
    runner.py                server, request and comparison lifecycle
    tasks/                   experiment declarations
    analysis/                offset and layer analysis
    prompts/                 prompt rewrites used by the study
    results.md               measured results

The request client is ../gen_video.py.  It handles ref2va, fl2va and t2va and
is shared with infer.sh.

The former run_*.sh scripts and their helpers are replaced by task
declarations under tasks.  Each declaration builds a list of cases; the runner
starts one server per case, sends one request, stops the server, and optionally
compares the video with a reference.

Run the commands from examples/h3_fa3.  Output paths in the task defaults are
absolute; paths passed on the command line are resolved from the current
directory.

## Run

List the declarations:

    python calibration/calib.py list

Calibrate the three prompt rewrites.  A fresh server is started for every
prompt so calibration sees that prompt and not the server warmup prompt:

    python calibration/calib.py run prompt-sweep --prefix 20261120

Generate the fp16/SDPA reference for one seed:

    python calibration/calib.py run fp16-baseline --seed 43 --steps 8

Compare one calibration mode:

    python calibration/calib.py run calib-mode \
        --task-dir outputs/calibration/seed43_hybrid \
        --offsets /tmp/opencode/nvfp4_offsets_fl2va \
        --mode hybrid \
        --sensitive-file outputs/calibration/20261116_fl2va_prompt_sensitivity/sensitive_layers_t0.40.txt \
        --tag hybrid --seed 43 --steps 8 \
        --reference outputs/calibration/seed43_fp16/calib_fp16_8st_seed43.mp4

Audit every layer against the exact SDPA fallback, then rank the layers:

    python calibration/calib.py run layer-audit \
        --task-dir outputs/calibration/20261116_fl2va_prompt_audit \
        --offsets outputs/calibration/20261116_fl2va_prompt_orig/offsets

Generate a video with a layer mix produced by the audit:

    python calibration/calib.py run mix-test \
        --task-dir outputs/calibration/20261116_fl2va_mix_skip5 \
        --mix-file outputs/calibration/20261116_fl2va_prompt_audit/mix_skip_top5.txt \
        --tag skip5 --steps 8

Run the four-way comparison for a seed:

    python calibration/calib.py run seed-study --seed 43 --steps 8

Every run command accepts --dry-run, which prints the cases without starting a
server.  The process-wide NVFP4 environment and per-layer offsets are cached,
so each case must own a server process.  Stop any existing server on the
selected port before starting a run.

## Status

    python calibration/calib.py status

The command scans outputs/calibration and prints the number of offset files and
videos in each task directory.

## Analysis

Offsets across prompts:

    python calibration/calib.py analyze offsets \
        --pattern '20261116_fl2va_prompt_*' \
        --layers blocks.0.attn blocks.25.attn blocks.49.attn

Layer groups:

    python calibration/calib.py analyze layer-groups \
        --pattern '20261116_fl2va_prompt_*'

Prompt sensitivity and the sensitive/stable split:

    python calibration/calib.py analyze prompt-sensitivity \
        --pattern '20261116_fl2va_prompt_*' --threshold 0.4 --out <task_dir>

Consensus cache:

    python calibration/calib.py analyze consensus \
        --pattern '20261116_fl2va_prompt_*' \
        --out outputs/calibration/20261116_fl2va_prompt_consensus

Rank layers from an audit log:

    python calibration/calib.py analyze rank-layers \
        --log outputs/calibration/20261116_fl2va_prompt_audit/audit_raw.txt \
        --out outputs/calibration/20261116_fl2va_prompt_audit

Measure offline cache loading and online calibration cost:

    python calibration/calib.py analyze bench-offsets \
        --cache outputs/calibration/20261116_fl2va_prompt_orig/offsets \
        --seq 32997 --heads 56 --layers 52

The analysis writes CSV and JSON tables plus the two summary figures under
calibration/plots.  The generated 3D, heatmap and histogram PNG panels are
ignored by git and can be regenerated.

## Calibration modes

The backend reads SGLANG_ROT_NVFP4_CALIB_MODE:

- offline loads an offset cache, or calibrates once if the cache is missing;
- online recalibrates every layer on each request;
- hybrid recalibrates only the layers listed in
  SGLANG_ROT_NVFP4_SENSITIVE_LAYERS and loads the rest from the cache.

The prompt sweep uses request warmup so the first real request is the
calibration sample.  The mode comparison uses the normal server warmup, which
is why its generic cache comes from the warmup prompt.  Pass a prompt-specific
cache with --offsets to test the offline-cache deployment.

See results.md for the measured quality and timing tables.
