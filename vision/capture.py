"""Grab a single frame from the laptop's overhead camera.

Standalone test: confirms the camera is reachable and saves a frame you can
look at before wiring up VLM detection on top of it.

Usage:
    python vision/capture.py                 # saves capture.jpg, default cam 0
    python vision/capture.py --index 1        # different camera index
    python vision/capture.py --out frame.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2


def capture_frame(index: int = 0, width: int = 1280, height: int = 720,
                   retries: int = 6, warmup_timeout_s: float = 10.0) -> "cv2.Mat":
    """Grab one frame, requesting a higher resolution than the camera default.

    Many USB webcams default to 640x480, which is too coarse for the VLM to
    reliably read small per-cell "c{col}r{row}" labels on a 7x8 grid. Request
    a larger frame if the camera supports it (silently falls back to the
    camera's max if it doesn't support the requested size).

    retries: on Windows, OpenCV's Media Foundation backend can transiently
    fail to grab a frame right after opening (especially if the camera was
    just opened/closed by a previous call) — retried automatically instead
    of failing the whole calibration/detection run over one flaky grab.

    warmup_timeout_s: keep re-reading frames until either the brightness
    check passes or this many real SECONDS have elapsed (not a fixed frame
    count). A real USB webcam confirmed needing ~30-45 quick reads before
    its auto-exposure kicks in, but a phone camera bridged in via something
    like "Link to Windows" can take several real seconds PER frame during
    its own stream startup -- a fixed read count silently under-warms it
    (each read returns near-instantly with a stale/black frame instead of
    actually waiting for a new one), which looked identical to the original
    slow-auto-exposure problem but has a different root cause and needs
    wall-clock time, not more reads, to fix.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            last_error = RuntimeError(f"Could not open camera index {index}")
            time.sleep(0.5)
            continue
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            deadline = time.monotonic() + warmup_timeout_s
            ok, frame = False, None
            reads = 0
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                reads += 1
                if ok and frame is not None and frame.mean() >= 5.0:
                    return frame
            if not ok or frame is None:
                last_error = RuntimeError(
                    f"Camera opened but returned no frame after {reads} reads "
                    f"over {warmup_timeout_s:.1f}s"
                )
                continue
            last_error = RuntimeError(
                f"Camera returned a near-black frame (mean brightness "
                f"{frame.mean():.2f}) even after {reads} reads over "
                f"{warmup_timeout_s:.1f}s -- may need more warm-up time "
                f"(try --warmup-timeout higher) or has a real exposure issue"
            )
        finally:
            cap.release()
        time.sleep(0.5 * (attempt + 1))  # backoff: 0.5s, 1.0s, 1.5s, ...

    raise last_error or RuntimeError(f"Failed to capture from camera index {index}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1, help="camera device index")
    ap.add_argument("--out", default="capture.jpg", help="output image path")
    ap.add_argument("--warmup-timeout", type=float, default=10.0,
                     help="seconds to keep retrying for a non-black frame "
                          "(raise this for slower camera bridges e.g. phone "
                          "cameras via Link to Windows)")
    args = ap.parse_args()

    frame = capture_frame(args.index, warmup_timeout_s=args.warmup_timeout)
    cv2.imwrite(args.out, frame)
    print(f"saved {args.out}  shape={frame.shape}")


if __name__ == "__main__":
    main()
