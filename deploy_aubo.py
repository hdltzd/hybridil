"""Independent HybridIL deployment entry point for an AUBO C5.

The default configuration is replay-only and cannot move a robot. Real motion
requires both ``robot.allow_motion=true`` in the configuration and an explicit
``--execute`` command-line flag. This program never powers on the robot,
clears faults, changes safety settings, or zeros the force sensor.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from configs import pose as global_pose
from controllers.aubo_admit_hybrid import (
    AuboAdmitHybridController,
    WrenchCompensator,
)
from deployment.aubo_hybridil import (
    ActionChunkPlanner,
    CoordinateFrames,
    InferenceClient,
    PointCloudProcessor,
    ReplaySource,
    pose_matrix,
    transform_action_to_aubo_base,
    validate_transform,
)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def make_controller(path: str) -> AuboAdmitHybridController:
    config = load_json(path)
    return AuboAdmitHybridController(
        mass=config["mass"],
        stiffness=config["stiffness"],
        damping=config["damping"],
        max_acceleration=config["max_acceleration"],
        max_velocity=config["max_velocity"],
        max_pose_step=config["max_pose_step"],
        max_force=config["max_force"],
        max_torque=config["max_torque"],
        torque_admittance_threshold=config.get("torque_admittance_threshold", 0.1),
        contact_force_z=config.get("contact_force_z", -3.0),
        force_integral_gain=config.get("force_integral_gain", 0.5),
        force_integral_limit=config.get("force_integral_limit", 50.0),
        integral_window=config.get("integral_window", 250),
    )


def make_coordinate_frames(source_mode: str) -> CoordinateFrames:
    """完全从全局 pose.py 构造相机、公共 BASE 和 AUBO 基座坐标链。

    实时数据来自 D455，因此使用 ``D455_2_BASE``；回放文件由原 L515
    采集，因此使用 ``L515_2_BASE``。两者都先进入公共 ``BASE``，再进入
    AUBO 基座坐标系。新单臂系统约定 ``BASEr`` 就是 AUBO 基座，因此
    直接把 ``BASEr_2_BASE`` 作为 ``AUBO_BASE_2_BASE`` 使用。
    """

    camera_to_common_base = (
        global_pose.D455_2_BASE
        if source_mode == "live"
        else global_pose.L515_2_BASE
    )
    return CoordinateFrames(
        camera_to_common_base=camera_to_common_base,
        aubo_base_to_common_base=global_pose.BASEr_2_BASE,
    )


class LiveObservationSource:
    """Build live policy observations from D455, AUBO, and XJC state."""

    def __init__(self, config: Dict, frames, robot, sensor, camera, compensator) -> None:
        self.config = config
        self.robot = robot
        self.sensor = sensor
        self.camera = camera
        self.compensator = compensator
        self.frames = frames
        policy = config["policy"]
        self.processor = PointCloudProcessor(
            point_count=policy["point_count"],
            voxel_size=policy["voxel_size"],
            # 裁剪范围只读取全局 pose.py，避免部署配置产生第二套坐标定义。
            base_bounds=global_pose.BASE_SPACE,
            object_bounds=global_pose.OBJECT_SPACE,
            tool_bounds=global_pose.XJC_SPACE,
            seed=config.get("seed", 1),
        )
        horizon = int(policy["obs_horizon"])
        self.buffers = {
            "pointcloud": deque(maxlen=horizon),
            "robot0_eef_pos": deque(maxlen=horizon),
            "robot0_eef_quat": deque(maxlen=horizon),
            "robot0_eef_wrench": deque(maxlen=horizon),
        }

    def read_compensated_wrench(self, tcp_quat_xyzw_base) -> np.ndarray:
        raw = self.sensor.read()
        return self.compensator.compensate(raw, tcp_quat_xyzw_base)

    def next(self) -> Optional[Dict[str, np.ndarray]]:
        color_bgr, depth_raw = self.camera.get_data()
        depth_m = self.camera.depth_to_meters(depth_raw, self.camera.depth_scale)
        color_rgb = color_bgr[..., ::-1].astype(np.float32) / 255.0
        xyz_camera, rgb = self.camera.get_point_cloud(
            depth_m,
            self.camera.intrinsics,
            color_rgb,
        )

        tcp_in_base = self.robot.get_tcp_pose(matrix=True)
        tcp_in_camera = self.frames.tcp_in_camera(tcp_in_base)
        quat_base = Rot.from_matrix(tcp_in_base[:3, :3]).as_quat()
        wrench_tcp = self.read_compensated_wrench(quat_base)
        pointcloud = self.processor.process(
            xyz_camera,
            rgb,
            self.frames.camera_to_common_base,
            tcp_in_camera,
        )

        self.buffers["pointcloud"].append(pointcloud)
        self.buffers["robot0_eef_pos"].append(tcp_in_camera[:3, 3])
        self.buffers["robot0_eef_quat"].append(
            Rot.from_matrix(tcp_in_camera[:3, :3]).as_quat()
        )
        self.buffers["robot0_eef_wrench"].append(wrench_tcp)
        if len(self.buffers["pointcloud"]) < self.buffers["pointcloud"].maxlen:
            return None
        return {
            key: np.asarray(value, dtype=np.float32)
            for key, value in self.buffers.items()
        }


class AuboActionExecutor:
    """Execute validated position or admittance-hybrid action chunks."""

    def __init__(
        self,
        robot,
        wrench_reader,
        controller: AuboAdmitHybridController,
        camera_to_aubo_base,
        config: Dict,
    ) -> None:
        self.robot = robot
        self.wrench_reader = wrench_reader
        self.controller = controller
        self.camera_to_aubo_base = validate_transform(
            camera_to_aubo_base,
            "CAMERA_2_AUBO_BASE",
        )
        control_rate = float(config["control_rate_hz"])
        action_rate = float(config["action_rate_hz"])
        if not np.isfinite([control_rate, action_rate]).all() or min(
            control_rate,
            action_rate,
        ) <= 0:
            raise ValueError("control_rate_hz and action_rate_hz must be positive")
        self.control_period = 1.0 / control_rate
        self.action_period = 1.0 / action_rate
        self.control_timeout = float(config.get("control_timeout_s", 0.02))
        if (
            not np.isfinite(self.control_timeout)
            or self.control_timeout <= self.control_period
        ):
            raise ValueError("control_timeout_s must be greater than one control period")
        self.servo_duration = float(config.get("servo_duration_s", self.control_period))
        self.servo_lookahead = float(config.get("servo_lookahead_s", 0.1))
        self.servo_gain = float(config.get("servo_gain", 200.0))
        self.position_step = float(config.get("position_step_m", 0.002))
        self.rotation_step = float(config.get("rotation_step_rad", 0.01))
        self.workspace = np.asarray(
            config["workspace_bounds_aubo_base"],
            dtype=np.float64,
        )
        if self.workspace.shape != (3, 2) or np.any(self.workspace[:, 0] >= self.workspace[:, 1]):
            raise ValueError("workspace_bounds_aubo_base must have shape (3, 2)")
        self.mode = None

    def _check_target(self, position, wrench) -> None:
        if np.any(position < self.workspace[:, 0]) or np.any(position > self.workspace[:, 1]):
            raise RuntimeError("Predicted TCP target is outside the AUBO workspace")
        if np.linalg.norm(wrench[:3]) > self.controller.max_force:
            raise RuntimeError("Predicted force exceeds the configured safety limit")
        if np.linalg.norm(wrench[3:]) > self.controller.max_torque:
            raise RuntimeError("Predicted torque exceeds the configured safety limit")

    def _bounded_position_command(self, current, target_position, target_quat):
        delta_position = target_position - current[:3, 3]
        distance = float(np.linalg.norm(delta_position))
        if distance > self.position_step:
            delta_position *= self.position_step / distance
        command_position = current[:3, 3] + delta_position

        target_rotation = Rot.from_quat(target_quat).as_matrix()
        relative = Rot.from_matrix(current[:3, :3].T @ target_rotation).as_rotvec()
        angle = float(np.linalg.norm(relative))
        if angle > self.rotation_step:
            relative *= self.rotation_step / angle
        command_rotation = current[:3, :3] @ Rot.from_rotvec(relative).as_matrix()
        return command_position, Rot.from_matrix(command_rotation).as_quat()

    def _check_cycle_timeout(self, cycle_start: float, stage: str) -> None:
        elapsed = time.perf_counter() - cycle_start
        if elapsed > self.control_timeout:
            raise RuntimeError(
                "AUBO control cycle timeout after {} ({:.6f}s > {:.6f}s)".format(
                    stage,
                    elapsed,
                    self.control_timeout,
                )
            )

    def _send_pose(self, position, quat) -> None:
        joints = self.robot.inverse_kinematics(position, quat)
        self.robot.servo_joints(
            joints,
            duration_s=self.servo_duration,
            lookahead_s=self.servo_lookahead,
            gain=self.servo_gain,
        )

    def execute(self, mode: str, actions: np.ndarray) -> None:
        if mode not in ("position", "hybrid"):
            raise ValueError("mode must be position or hybrid")
        if mode != self.mode:
            self.controller.reset()
            self.mode = mode
        previous_cycle_start = None
        for action in actions:
            target_position, target_quat, wrench_tcp = transform_action_to_aubo_base(
                action,
                self.camera_to_aubo_base,
            )
            self._check_target(target_position, wrench_tcp)
            deadline = time.perf_counter() + self.action_period
            while True:
                cycle_start = time.perf_counter()
                if previous_cycle_start is None:
                    actual_dt = self.control_period
                else:
                    actual_dt = cycle_start - previous_cycle_start
                    if actual_dt > self.control_timeout:
                        raise RuntimeError(
                            "AUBO control cycle interval exceeded watchdog "
                            "({:.6f}s > {:.6f}s)".format(
                                actual_dt,
                                self.control_timeout,
                            )
                        )
                previous_cycle_start = cycle_start

                current = self.robot.get_tcp_pose(matrix=True)
                current_quat = Rot.from_matrix(current[:3, :3]).as_quat()
                measured_wrench = self.wrench_reader(current_quat)
                # 即使模型当前选择纯位置控制，也必须检查真实接触力。
                measured_wrench = self.controller.validate_measured_wrench(
                    measured_wrench
                )
                if mode == "position":
                    command_position, command_quat = self._bounded_position_command(
                        current,
                        target_position,
                        target_quat,
                    )
                else:
                    command_position, command_quat = self.controller.step(
                        current_position_base=current[:3, 3],
                        current_quat_xyzw_base=current_quat,
                        target_position_base=target_position,
                        target_quat_xyzw_base=target_quat,
                        measured_wrench_tcp=measured_wrench,
                        desired_wrench_tcp=wrench_tcp,
                        dt=actual_dt,
                    )
                # 传感器、状态读取、控制计算或 IK 前处理过慢时，不发送陈旧指令。
                self._check_cycle_timeout(cycle_start, "state/force/control")
                self._send_pose(command_position, command_quat)
                self._check_cycle_timeout(cycle_start, "servoJoint")
                remaining = cycle_start + self.control_period - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                if time.perf_counter() >= deadline:
                    break


def initialize_xjc_sensor(sensor, sensor_config: Dict) -> None:
    """在实时硬件初始化时清零XJC，再启动流式读取。

    清零是显式配置的硬件状态变更，仅由 ``source.mode=live`` 路径调用；
    回放和离线推理永远不会创建或清零传感器。
    """
    zero_on_startup = sensor_config.get("zero_on_startup", True)
    if not isinstance(zero_on_startup, bool):
        raise ValueError("XJC zero_on_startup must be a boolean")
    settle_seconds = float(sensor_config.get("zero_settle_seconds", 1.0))
    if not np.isfinite(settle_seconds) or settle_seconds < 0:
        raise ValueError("XJC zero_settle_seconds must be non-negative")

    if zero_on_startup:
        # 构造后尚未调用 enable_streaming，因此满足XJC清零的协议前提。
        print("XJC初始化：正在执行空载硬件清零")
        sensor.set_zero()
        if settle_seconds > 0:
            time.sleep(settle_seconds)

    if sensor_config.get("streaming", True):
        sensor.enable_streaming()


def build_live_hardware(config: Dict, frames: CoordinateFrames):
    from devices.aubo_c5 import AuboC5Robot
    from devices.d455 import D455Camera
    from devices.xjc_ft import XjcForceTorqueSensor

    robot_config = config["robot"]
    password_env = robot_config.get("password_env", "AUBO_PASSWORD")
    password = os.environ.get(password_env)
    if password is None:
        raise RuntimeError("Set the {} environment variable for AUBO login".format(password_env))
    robot = AuboC5Robot(
        robot_ip=robot_config["ip"],
        rpc_port=robot_config.get("rpc_port", 30004),
        username=robot_config.get("username", "aubo"),
        password=password,
        request_timeout_ms=robot_config.get("request_timeout_ms", 1000),
        max_joint_step_rad=robot_config.get("max_joint_step_rad", np.deg2rad(1.0)),
    )

    camera = None
    sensor = None
    try:
        camera_config = config["camera"]
        camera = D455Camera(
            serial=camera_config.get("serial"),
            depth_size=tuple(camera_config.get("depth_size", [1280, 720])),
            color_size=tuple(camera_config.get("color_size", [1280, 720])),
            fps=camera_config.get("fps", 30),
            timeout_ms=camera_config.get("timeout_ms", 3000),
            warmup_frames=camera_config.get("warmup_frames", 20),
        )
        sensor_config = load_json(config["force_sensor_config"])
        sensor = XjcForceTorqueSensor(
            port=sensor_config.get("port"),
            baudrate=sensor_config.get("baudrate", 115200),
            timeout_s=sensor_config.get("timeout_s", 0.1),
            slave_address=sensor_config.get("slave_address", 1),
            sample_rate=sensor_config.get("sample_rate", 250),
            usb_selector=sensor_config.get("usb_selector"),
        )
        initialize_xjc_sensor(sensor, sensor_config)
        compensator = WrenchCompensator(
            force_zero=sensor_config["force_zero"],
            torque_zero=sensor_config["torque_zero"],
            gravity_in_aubo_base=sensor_config["gravity_in_aubo_base"],
            center_of_mass_in_sensor=sensor_config["center_of_mass_in_sensor"],
            xjc_to_tcp=sensor_config["XJC_2_TCP"],
        )
        source = LiveObservationSource(config, frames, robot, sensor, camera, compensator)
        return robot, camera, sensor, source
    except Exception:
        if sensor is not None:
            sensor.close()
        if camera is not None:
            camera.close()
        robot.close()
        raise


def print_chunk_summary(call_index, source_index, chunk, camera_to_aubo_base) -> None:
    positions = []
    force_norms = []
    for action in chunk.actions:
        position, _, wrench = transform_action_to_aubo_base(
            action,
            camera_to_aubo_base,
        )
        positions.append(position)
        force_norms.append(float(np.linalg.norm(wrench[:3])))
    summary = {
        "policy_call": call_index,
        "source_index": source_index,
        "execution_mode": chunk.mode,
        "action_count": len(chunk.actions),
        "first_target_xyz_aubo_base": np.asarray(positions[0]).round(6).tolist(),
        "last_target_xyz_aubo_base": np.asarray(positions[-1]).round(6).tolist(),
        "max_predicted_force_n": round(max(force_norms), 6),
        "motion_sent": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deploy_aubo.json")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow real AUBO ServoJ commands; also requires allow_motion=true",
    )
    parser.add_argument(
        "--recorded-actions",
        action="store_true",
        help="Use HDF5 actions instead of contacting node.py (replay mode only)",
    )
    parser.add_argument("--max-policy-calls", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source_mode = config["source"]["mode"]
    if source_mode not in ("replay", "live"):
        raise ValueError("source.mode must be replay or live")
    if args.execute:
        if source_mode != "live":
            raise RuntimeError("--execute is prohibited with replay data")
        if not config["robot"].get("allow_motion", False):
            raise RuntimeError("Set robot.allow_motion=true as the second motion interlock")
        if not config["coordinates"].get("calibrated", False):
            raise RuntimeError("D455、公共 BASE 与 AUBO 基座的坐标链尚未完成标定")
        if args.recorded_actions:
            raise RuntimeError("--recorded-actions cannot be executed on hardware")

    max_calls = args.max_policy_calls
    if max_calls is None:
        max_calls = int(config["source"].get("max_policy_calls", 1))
    if max_calls <= 0:
        raise ValueError("max_policy_calls must be positive")

    policy_config = config["policy"]
    planner = ActionChunkPlanner(
        force_threshold=policy_config.get("force_threshold", 6.0),
        force_ratio=policy_config.get("force_ratio", 0.2),
    )
    frames = make_coordinate_frames(source_mode)
    camera_to_aubo_base = frames.camera_to_aubo_base
    if not config["coordinates"].get("calibrated", False):
        print(
            "警告：D455/公共BASE/AUBO基座坐标链尚未标定，转换结果仅供离线检查"
        )

    inference = None
    robot = camera = sensor = live_source = replay_source = executor = None
    try:
        if not args.recorded_actions:
            inference_config = config["inference"]
            inference = InferenceClient(
                endpoint=inference_config["endpoint"],
                timeout_ms=inference_config.get("timeout_ms", 120000),
            )

        if source_mode == "replay":
            source_config = config["source"]
            replay_source = ReplaySource(
                path=source_config["hdf5_path"],
                demo=source_config.get("demo", "demo_0"),
                obs_horizon=policy_config["obs_horizon"],
                action_horizon=policy_config["act_horizon"],
                start_index=source_config.get("start_index", 0),
            )
        else:
            robot, camera, sensor, live_source = build_live_hardware(config, frames)
            if args.execute:
                robot.require_motion_allowed()
                robot.enable_servo()
                executor = AuboActionExecutor(
                    robot=robot,
                    wrench_reader=live_source.read_compensated_wrench,
                    controller=make_controller(config["controller_config"]),
                    camera_to_aubo_base=camera_to_aubo_base,
                    config=config["execution"],
                )

        completed = 0
        while completed < max_calls:
            if replay_source is not None:
                observation, recorded_actions, source_index = replay_source.next()
            else:
                observation = live_source.next()
                if observation is None:
                    continue
                recorded_actions = None
                source_index = completed

            actions = (
                recorded_actions
                if args.recorded_actions
                else inference.predict(config["name"], observation)
            )
            chunk = planner.plan(actions[: policy_config["act_horizon"]])
            if executor is None:
                print_chunk_summary(completed, source_index, chunk, camera_to_aubo_base)
            else:
                executor.execute(chunk.mode, chunk.actions)
            completed += 1
    except KeyboardInterrupt:
        print("Deployment stopped by user")
    finally:
        if args.execute and robot is not None:
            robot.stop()
        if inference is not None:
            inference.close()
        if replay_source is not None:
            replay_source.close()
        if sensor is not None:
            sensor.close()
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
