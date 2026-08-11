"""Non-GUI smoke test for a connected Intel RealSense D455.

Run this only after the camera is connected:

    python tests/d455_hardware_smoke.py --frames 300 --restart-check
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devices.d455 import D455Camera


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--restart-check", action="store_true")
    return parser.parse_args()


def capture_summary(camera, frame_count, voxel_size):
    if frame_count <= 0:
        raise ValueError("--frames must be positive")

    expected_shape = None
    valid_depth_ratios = []
    started_at = time.perf_counter()
    last_color = None
    last_depth_m = None

    for _ in range(frame_count):
        color, depth_raw = camera.get_data()
        if color.dtype != np.uint8:
            raise RuntimeError("D455 color frame is not uint8")
        if depth_raw.dtype != np.uint16:
            raise RuntimeError("D455 depth frame is not uint16")
        if color.shape[:2] != depth_raw.shape:
            raise RuntimeError("Aligned color and depth sizes do not match")
        if expected_shape is None:
            expected_shape = depth_raw.shape
        elif depth_raw.shape != expected_shape:
            raise RuntimeError("D455 frame size changed during capture")

        depth_m = camera.depth_to_meters(depth_raw, camera.depth_scale)
        if not np.isfinite(depth_m).all():
            raise RuntimeError("Depth frame contains NaN or Inf")
        valid_depth_ratios.append(float(np.count_nonzero(depth_raw) / depth_raw.size))
        last_color = color
        last_depth_m = depth_m

    elapsed = time.perf_counter() - started_at
    color_rgb = last_color[..., ::-1].astype(np.float32) / 255.0
    points, colors = camera.get_point_cloud(
        last_depth_m,
        camera.intrinsics,
        color_rgb,
    )
    valid_points = points[points[:, 2] > 0]
    valid_colors = colors[points[:, 2] > 0]
    if not np.isfinite(valid_points).all():
        raise RuntimeError("Generated point cloud contains NaN or Inf")

    voxel_points, _ = camera.voxelize(
        valid_points,
        valid_colors,
        voxel_size=voxel_size,
    )
    return {
        "device_name": camera.device_name,
        "serial": camera.serial,
        "frames": frame_count,
        "elapsed_seconds": elapsed,
        "measured_fps": frame_count / elapsed,
        "color_shape": list(last_color.shape),
        "depth_shape": list(last_depth_m.shape),
        "depth_scale": camera.depth_scale,
        "intrinsics": camera.intrinsics,
        "mean_valid_depth_ratio": float(np.mean(valid_depth_ratios)),
        "valid_point_count": int(valid_points.shape[0]),
        "voxel_point_count": int(voxel_points.shape[0]),
    }


def open_camera(args):
    return D455Camera(
        serial=args.serial,
        depth_size=(args.width, args.height),
        color_size=(args.width, args.height),
        fps=args.fps,
        timeout_ms=args.timeout_ms,
    )


def main():
    args = parse_args()
    with open_camera(args) as camera:
        summary = capture_summary(camera, args.frames, args.voxel_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.restart_check:
        with open_camera(args) as camera:
            color, depth = camera.get_data()
            print(
                "Restart check passed: color={}, depth={}".format(
                    color.shape,
                    depth.shape,
                )
            )


if __name__ == "__main__":
    main()
