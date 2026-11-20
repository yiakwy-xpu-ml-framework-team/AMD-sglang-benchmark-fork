"""Per-frame SSIM / PSNR / MAD between two generated mp4 files.

    python compare_video.py --reference ref2va_fp8_all.mp4 \
        --candidate ref2va_nvfp4_all.mp4

The gate used for the H3 A/B is the mean over decoded frames; the min row makes
the worst frame visible. Both videos must have the same frame count.
"""

import argparse

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def frames(path):
    capture = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    reference = frames(args.reference)
    candidate = frames(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit(
            f"frame count mismatch: {len(reference)} vs {len(candidate)}"
        )

    rows = []
    for expected, actual in zip(reference, candidate):
        expected = expected.astype(np.float64)
        actual = actual.astype(np.float64)
        ssim = structural_similarity(
            expected, actual, channel_axis=2, data_range=255
        )
        mse = np.mean((expected - actual) ** 2)
        rows.append(
            (
                ssim,
                100.0 if mse == 0 else 10 * np.log10(255.0**2 / mse),
                np.mean(np.abs(expected - actual)),
            )
        )
    rows = np.asarray(rows)

    print(f"frames {len(rows)}")
    print(
        f"SSIM  mean {rows[:, 0].mean():.4f}  min {rows[:, 0].min():.4f}"
    )
    print(
        f"PSNR  mean {rows[:, 1].mean():.2f} dB  min {rows[:, 1].min():.2f}"
    )
    print(
        f"MAD   mean {rows[:, 2].mean():.2f}  max {rows[:, 2].max():.2f}"
    )


if __name__ == "__main__":
    main()
