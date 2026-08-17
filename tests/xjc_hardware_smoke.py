"""XJC六维力传感器只读冒烟测试。

该程序不会发送置零命令，也不会修改配置文件。流模式只在测试期间临时启用，
退出时由驱动关闭数据流并释放串口。

示例：
    python tests/xjc_hardware_smoke.py \
        --config configs/xjc_ft.json --samples 1000 --mode streaming
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devices.xjc_ft import XjcForceTorqueSensor, XjcProtocolError


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/xjc_ft.json")
    parser.add_argument("--port", default=None, help="覆盖配置中的串口设备")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--mode",
        choices=("polling", "streaming"),
        default="streaming",
    )
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def vector_stats(values):
    return {
        "mean": np.mean(values, axis=0).round(6).tolist(),
        "std": np.std(values, axis=0).round(6).tolist(),
        "min": np.min(values, axis=0).round(6).tolist(),
        "max": np.max(values, axis=0).round(6).tolist(),
    }


def capture(sensor, sample_count, mode, max_errors):
    if sample_count <= 0:
        raise ValueError("--samples must be positive")
    if max_errors < 0:
        raise ValueError("--max-errors must be non-negative")

    if mode == "streaming":
        sensor.enable_streaming()

    samples = []
    timestamps = []
    errors = 0
    started_at = time.perf_counter()
    while len(samples) < sample_count:
        try:
            wrench = sensor.read()
        except XjcProtocolError:
            errors += 1
            if errors > max_errors:
                raise RuntimeError(
                    "XJC protocol errors exceeded --max-errors ({})".format(
                        max_errors
                    )
                )
            continue
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise RuntimeError("XJC returned a non-finite or invalid wrench sample")
        samples.append(wrench)
        timestamps.append(time.perf_counter())

    elapsed = time.perf_counter() - started_at
    values = np.asarray(samples, dtype=np.float64)
    measured_rate = (
        (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]
        else 0.0
    )
    return {
        "port": sensor.port,
        "mode": mode,
        "configured_sample_rate_hz": sensor.sample_rate,
        "samples": len(samples),
        "protocol_errors": errors,
        "elapsed_seconds": round(elapsed, 6),
        "measured_sample_rate_hz": round(measured_rate, 3),
        "wrench_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "wrench_units": ["N", "N", "N", "Nm", "Nm", "Nm"],
        "statistics": vector_stats(values),
        "max_force_norm_n": round(
            float(np.max(np.linalg.norm(values[:, :3], axis=1))),
            6,
        ),
        "max_torque_norm_nm": round(
            float(np.max(np.linalg.norm(values[:, 3:], axis=1))),
            6,
        ),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    port = args.port if args.port is not None else config.get("port")
    with XjcForceTorqueSensor(
        port=port,
        baudrate=config.get("baudrate", 115200),
        timeout_s=config.get("timeout_s", 0.006),
        slave_address=config.get("slave_address", 1),
        sample_rate=config.get("sample_rate", 250),
        usb_selector=config.get("usb_selector"),
        transport=config.get("transport"),
    ) as sensor:
        summary = capture(
            sensor,
            sample_count=args.samples,
            mode=args.mode,
            max_errors=args.max_errors,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
