"""AUBO C5 RPC只读状态冒烟测试。

该程序只连接、登录并读取状态，不会上电、启动机器人、修改碰撞等级、设置TCP、
启用Servo模式、求逆解或发送运动命令。密码只从环境变量读取。

示例：
    export AUBO_PASSWORD='实际密码'
    python tests/aubo_readonly_smoke.py \
        --config configs/deploy_aubo.json --samples 20 --interval 0.1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devices.aubo_c5 import AuboC5Robot


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deploy_aubo.json")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.1)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _vector_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise RuntimeError("AUBO state history contains invalid values")
    return {
        "first": array[0].round(9).tolist(),
        "last": array[-1].round(9).tolist(),
        "min": np.min(array, axis=0).round(9).tolist(),
        "max": np.max(array, axis=0).round(9).tolist(),
    }


def collect_readonly_state(robot, sample_count, interval_s):
    if sample_count <= 0:
        raise ValueError("--samples must be positive")
    if not np.isfinite(interval_s) or interval_s < 0:
        raise ValueError("--interval must be non-negative")

    statuses = []
    joint_positions = []
    joint_velocities = []
    tcp_poses = []
    tcp_velocities = []
    started_at = time.perf_counter()
    for index in range(sample_count):
        statuses.append(robot.get_status())
        joint_positions.append(robot.get_joint_pos())
        joint_velocities.append(robot.get_joint_vel())
        tcp_poses.append(robot.get_tcp_pose_euler())
        tcp_velocities.append(robot.get_tcp_vel())
        if interval_s > 0 and index + 1 < sample_count:
            time.sleep(interval_s)
    elapsed = time.perf_counter() - started_at

    lower, upper = robot.get_joint_limits()
    joint_position_array = np.asarray(joint_positions, dtype=np.float64)
    if np.any(joint_position_array < lower) or np.any(joint_position_array > upper):
        raise RuntimeError("AUBO reported joint positions outside configured limits")

    joint_velocity_array = np.asarray(joint_velocities, dtype=np.float64)
    tcp_velocity_array = np.asarray(tcp_velocities, dtype=np.float64)
    return {
        "connection": {
            "robot_ip": robot.robot_ip,
            "rpc_port": robot.rpc_port,
            "robot_name": robot.robot_name,
        },
        "read_only": True,
        "motion_commands_sent": False,
        "samples": sample_count,
        "elapsed_seconds": round(elapsed, 6),
        "initial_status": statuses[0],
        "final_status": statuses[-1],
        "joint_position_rad": _vector_summary(joint_positions),
        "joint_position_deg_first": np.rad2deg(joint_position_array[0]).round(6).tolist(),
        "joint_velocity_rad_s": _vector_summary(joint_velocities),
        "max_abs_joint_velocity_rad_s": round(
            float(np.max(np.abs(joint_velocity_array))),
            9,
        ),
        "tcp_pose_xyz_m_rpy_rad": _vector_summary(tcp_poses),
        "tcp_velocity": _vector_summary(tcp_velocities),
        "max_abs_tcp_velocity": round(
            float(np.max(np.abs(tcp_velocity_array))),
            9,
        ),
        "joint_limits_rad": {
            "lower": lower.round(9).tolist(),
            "upper": upper.round(9).tolist(),
        },
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    robot_config = config["robot"]
    password_env = robot_config.get("password_env", "AUBO_PASSWORD")
    password = os.environ.get(password_env)
    if password is None:
        raise RuntimeError(
            "Set the {} environment variable for AUBO login".format(password_env)
        )

    with AuboC5Robot(
        robot_ip=robot_config["ip"],
        rpc_port=robot_config.get("rpc_port", 30004),
        username=robot_config.get("username", "aubo"),
        password=password,
        request_timeout_ms=robot_config.get("request_timeout_ms", 1000),
        max_joint_step_rad=robot_config.get(
            "max_joint_step_rad",
            np.deg2rad(0.1),
        ),
    ) as robot:
        summary = collect_readonly_state(
            robot,
            sample_count=args.samples,
            interval_s=args.interval,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
