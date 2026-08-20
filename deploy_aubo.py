"""Independent HybridIL deployment entry point for an AUBO C5.

Real motion requires both ``robot.allow_motion=true`` in the configuration and
an explicit ``--execute`` command-line flag. This program never powers on the
robot, clears faults, or changes controller safety settings. XJC startup zeroing
is controlled separately by ``zero_on_startup`` in its configuration.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import threading
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
    PlannedChunk,
    PointCloudProcessor,
    ReplaySource,
    pose_matrix,
    transform_action_to_aubo_base,
    validate_transform,
)
from deployment.aubo_runtime import (
    LatestWrenchCache,
    RuntimeStatistics,
    SingleServoControlLoop,
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
        max_force=config["max_force"],
        max_torque=config["max_torque"],
        position_max_acceleration=config.get("position_max_acceleration"),
        position_max_velocity=config.get("position_max_velocity"),
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

    def __init__(
        self,
        config: Dict,
        frames,
        robot,
        sensor,
        camera,
        compensator,
        robot_lock=None,
        wrench_cache=None,
    ) -> None:
        self.config = config
        self.robot = robot
        self.sensor = sensor
        self.camera = camera
        self.compensator = compensator
        self.wrench_cache = wrench_cache
        self.frames = frames
        # 推理保持线程和观测线程不能同时调用AUBO SDK。默认锁只用于
        # 离线单元测试；实机路径传入同一个共享锁。
        self.robot_lock = robot_lock or threading.RLock()
        # XJC只有一条串口字节流。保持线程、实时观测和动作控制若并发读取，
        # 会分别取走同一个29字节数据包的一部分，造成帧头超时或CRC错误。
        self.sensor_lock = threading.Lock()
        sensor_config = load_json(config["force_sensor_config"])
        self.sensor_read_retries = int(sensor_config.get("read_retries", 3))
        if self.sensor_read_retries <= 0:
            raise ValueError("XJC read_retries must be positive")
        self.safety_check = None
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
        if getattr(self, "wrench_cache", None) is not None:
            return self.wrench_cache.read_latest(tcp_quat_xyzw_base)
        # ser2net/socat链路可能偶发丢失一个250 Hz数据帧。只重试协议帧错误，
        # 连续失败仍抛出并触发机械臂停止；串口断开等其他异常不作掩盖。
        from devices.xjc_ft import XjcProtocolError

        with self.sensor_lock:
            last_error = None
            for _ in range(self.sensor_read_retries):
                try:
                    raw = self.sensor.read()
                    break
                except XjcProtocolError as error:
                    last_error = error
            else:
                raise XjcProtocolError(
                    "XJC streaming read failed {} consecutive times: {}".format(
                        self.sensor_read_retries,
                        last_error,
                    )
                ) from last_error
        return self.compensator.compensate(raw, tcp_quat_xyzw_base)

    def next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.safety_check is not None:
            self.safety_check()
        color_bgr, depth_raw = self.camera.get_data()
        depth_m = self.camera.depth_to_meters(depth_raw, self.camera.depth_scale)
        color_rgb = color_bgr[..., ::-1].astype(np.float32) / 255.0
        xyz_camera, rgb = self.camera.get_point_cloud(
            depth_m,
            self.camera.intrinsics,
            color_rgb,
        )

        with self.robot_lock:
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


class ServoHoldLoop:
    """在相机采集和模型推理期间持续保持一条安全的 AUBO ServoJ 流。

    AUBO 的 Servo 模式要求连续收到指令。扩散模型推理比一个 ServoJ 周期
    长得多，因此不能只在每个策略块内发送指令。该线程只重复最后一个已经
    通过安全检查的 TCP 目标；策略块执行期间由主线程独占发送控制指令。
    """

    def __init__(self, robot, wrench_reader, controller, config: Dict, robot_lock) -> None:
        self.robot = robot
        self.wrench_reader = wrench_reader
        self.controller = controller
        self.robot_lock = robot_lock
        self.period = 1.0 / float(config["control_rate_hz"])
        self.duration = self.period
        self.lookahead = float(config.get("servo_lookahead_s", 0.1))
        self.gain = float(config.get("servo_gain", 200.0))
        self._condition = threading.Condition()
        self._target = None
        self._paused = True
        self._sending = False
        self._closed = False
        self._failure = None
        self._thread = threading.Thread(
            target=self._run,
            name="aubo-servo-hold",
            daemon=True,
        )

    def start(self, position, quat) -> None:
        self.set_target(position, quat)
        with self._condition:
            self._paused = False
            self._condition.notify_all()
        self._thread.start()

    def set_target(self, position, quat) -> None:
        position = np.asarray(position, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if position.shape != (3,) or quat.shape != (4,):
            raise ValueError("Servo hold target must be xyz(3) and quaternion(4)")
        if not np.isfinite(position).all() or not np.isfinite(quat).all():
            raise ValueError("Servo hold target must be finite")
        if np.linalg.norm(quat) < 1e-12:
            raise ValueError("Servo hold quaternion must be non-zero")
        with self._condition:
            self._target = (position.copy(), quat / np.linalg.norm(quat))

    def pause(self) -> None:
        """停止保持发送，让策略块控制循环独占 AUBO ServoJ。"""
        with self._condition:
            self._paused = True
            # 如果刚好已有一条保持指令在发送，等待它完成；之后策略块期间
            # 不会再有保持线程调用ServoJ。
            while self._sending:
                self._condition.wait()

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def raise_if_failed(self) -> None:
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("AUBO Servo保持线程已停止：{}".format(failure)) from failure

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, 5.0 * self.period))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and (self._paused or self._target is None):
                    self._condition.wait()
                if self._closed:
                    return
                position, quat = self._target
                self._sending = True
            cycle_start = time.perf_counter()
            try:
                # 所有 AUBO SDK 调用均使用共享锁，防止与观测读取或策略块
                # 同时调用同一个 RPC 接口。
                with self.robot_lock:
                    current = self.robot.get_tcp_pose(matrix=True)
                    current_quat = Rot.from_matrix(current[:3, :3]).as_quat()
                    measured_wrench = self.wrench_reader(current_quat)
                    self.controller.validate_measured_wrench(measured_wrench)
                    self.robot.servo_tcp_pose(
                        position,
                        quat,
                        duration_s=self.duration,
                        lookahead_s=self.lookahead,
                        gain=self.gain,
                    )
            except Exception as error:
                try:
                    self.robot.stop()
                except Exception:
                    pass
                with self._condition:
                    self._failure = error
                    self._paused = True
                    self._sending = False
                    self._condition.notify_all()
                return
            with self._condition:
                self._sending = False
                self._condition.notify_all()
            remaining = self.period - (time.perf_counter() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)


class AuboActionExecutor:
    """Execute validated position or admittance-hybrid action chunks."""

    def __init__(
        self,
        robot,
        wrench_reader,
        controller: AuboAdmitHybridController,
        camera_to_aubo_base,
        config: Dict,
        robot_lock=None,
    ) -> None:
        self.robot = robot
        self.robot_lock = robot_lock or threading.RLock()
        self.wrench_reader = wrench_reader
        self.controller = controller
        self.camera_to_aubo_base = validate_transform(
            camera_to_aubo_base,
            "CAMERA_2_AUBO_BASE",
        )
        control_rate = float(config["control_rate_hz"])
        if not np.isfinite(control_rate) or control_rate <= 0:
            raise ValueError("control_rate_hz must be positive")
        self.control_period = 1.0 / control_rate
        self.trajectory_duration = float(config["trajectory_duration_s"])
        self.control_timeout = float(config.get("control_timeout_s", 0.02))
        if (
            not np.isfinite(self.control_timeout)
            or self.control_timeout <= self.control_period
        ):
            raise ValueError("control_timeout_s must be greater than one control period")
        self.servo_duration = self.control_period
        self.servo_lookahead = float(config.get("servo_lookahead_s", 0.1))
        self.servo_gain = float(config.get("servo_gain", 200.0))
        self.max_speed = float(config["max_tcp_speed_m_s"])
        self.max_angular_speed = float(config["max_tcp_angular_speed_rad_s"])
        self.workspace = np.asarray(
            config["workspace_bounds_aubo_base"],
            dtype=np.float64,
        )
        if self.workspace.shape != (3, 2) or np.any(self.workspace[:, 0] >= self.workspace[:, 1]):
            raise ValueError("workspace_bounds_aubo_base must have shape (3, 2)")
        self.mode = None
        self.last_command = None

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
        max_position_step = self.max_speed * self.control_period
        if distance > max_position_step:
            delta_position *= max_position_step / distance
        command_position = current[:3, 3] + delta_position

        target_rotation = Rot.from_quat(target_quat).as_matrix()
        relative = Rot.from_matrix(current[:3, :3].T @ target_rotation).as_rotvec()
        angle = float(np.linalg.norm(relative))
        max_rotation_step = self.max_angular_speed * self.control_period
        if angle > max_rotation_step:
            relative *= max_rotation_step / angle
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
        # 与参考项目的 process_command + send_command 一致：每一个笛卡尔
        # 目标都在同一机器人适配器内完成 IK 和 ServoJ，失败立即停止本进程
        # 持有的伺服，不能继续发送下一帧。
        self.robot.servo_tcp_pose(
            position,
            quat,
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
        action_period = self.trajectory_duration / len(actions)
        for action in actions:
            target_position, target_quat, wrench_tcp = transform_action_to_aubo_base(
                action,
                self.camera_to_aubo_base,
            )
            self._check_target(target_position, wrench_tcp)
            deadline = time.perf_counter() + action_period
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

                with self.robot_lock:
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
                with self.robot_lock:
                    self._send_pose(command_position, command_quat)
                self.last_command = (command_position.copy(), command_quat.copy())
                self._check_cycle_timeout(cycle_start, "servoJoint")
                remaining = cycle_start + self.control_period - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                if time.perf_counter() >= deadline:
                    break


def initialize_xjc_sensor(sensor, sensor_config: Dict) -> None:
    """按参考按摩项目的顺序检查、清零并启动XJC主动上传。"""
    zero_on_startup = sensor_config.get("zero_on_startup", True)
    if not isinstance(zero_on_startup, bool):
        raise ValueError("XJC zero_on_startup must be a boolean")

    prepare_seconds = float(sensor_config.get("zero_prepare_seconds", 0.5))
    response_timeout = float(sensor_config.get("zero_response_timeout_s", 0.1))
    retry_delay = float(sensor_config.get("zero_retry_delay_s", 0.5))
    settle_seconds = float(sensor_config.get("zero_settle_seconds", 1.0))
    zero_tolerance = float(sensor_config.get("zero_tolerance", 0.001))
    retries = int(sensor_config.get("zero_retries", 5))
    initial_disable_count = int(sensor_config.get("zero_initial_disable_count", 3))
    retry_disable_count = int(sensor_config.get("zero_retry_disable_count", 2))
    polling_count = int(sensor_config.get("zero_polling_count", 3))
    command_count = int(sensor_config.get("zero_command_count", 2))

    timing = [prepare_seconds, response_timeout, retry_delay, settle_seconds]
    if not np.isfinite(timing).all() or min(timing) < 0:
        raise ValueError("XJC zero timing values must be non-negative")
    counts = [
        retries,
        initial_disable_count,
        retry_disable_count,
        polling_count,
        command_count,
    ]
    if min(counts) <= 0:
        raise ValueError("XJC zero counts and retries must be positive")
    if not np.isfinite(zero_tolerance) or zero_tolerance <= 0:
        raise ValueError("XJC zero_tolerance must be positive")

    if zero_on_startup:
        print("XJC初始化：正在检查空载零点并执行必要的硬件清零")
        original_timeout = sensor.timeout_s
        sensor.timeout_s = max(original_timeout, response_timeout)
        verified_reading = None
        last_error = None
        try:
            # 参考程序先连续三次关闭主动上传，等待残留流数据停止。
            for _ in range(initial_disable_count):
                sensor.disable_streaming()
            if prepare_seconds > 0:
                time.sleep(prepare_seconds)

            for attempt in range(1, retries + 1):
                # 连续轮询三次，使用最后一次数据判断当前是否已处于零点。
                try:
                    for _ in range(polling_count):
                        verified_reading = np.asarray(
                            sensor.read_polling(),
                            dtype=np.float64,
                        )
                    if verified_reading.shape != (6,) or not np.isfinite(
                        verified_reading
                    ).all():
                        raise RuntimeError("XJC轮询返回无效六维数据")
                    if np.allclose(
                        verified_reading,
                        np.zeros(6),
                        atol=zero_tolerance,
                        rtol=0.0,
                    ):
                        last_error = None
                        print("XJC初始化：零点验证通过 {}".format(
                            np.round(verified_reading, 6).tolist()
                        ))
                        break
                    print("XJC初始化：检测到非零空载数据 {}".format(
                        np.round(verified_reading, 6).tolist()
                    ))
                except Exception as error:
                    verified_reading = None
                    last_error = error
                    print("XJC初始化：第{}/{}轮轮询失败：{}".format(
                        attempt,
                        retries,
                        error,
                    ))

                # 参考程序每轮再关闭两次、等待0.5秒、连续清零两次。
                for _ in range(retry_disable_count):
                    sensor.disable_streaming()
                if retry_delay > 0:
                    time.sleep(retry_delay)

                command_error = None
                for _ in range(command_count):
                    try:
                        sensor.set_zero()
                        command_error = None
                    except Exception as error:
                        command_error = error
                last_error = command_error

                if retry_delay > 0:
                    time.sleep(retry_delay)
                if attempt < retries:
                    print("XJC初始化：第{}/{}轮清零完成，重新轮询验证".format(
                        attempt,
                        retries,
                    ))
            else:
                verified_reading = None

            # 若最后一轮刚写完清零命令，额外执行三次轮询验证结果。
            if verified_reading is None or not np.allclose(
                verified_reading,
                np.zeros(6),
                atol=zero_tolerance,
                rtol=0.0,
            ):
                try:
                    for _ in range(polling_count):
                        verified_reading = np.asarray(
                            sensor.read_polling(),
                            dtype=np.float64,
                        )
                    last_error = None
                except Exception as error:
                    verified_reading = None
                    last_error = error

            if (
                verified_reading is None
                or verified_reading.shape != (6,)
                or not np.isfinite(verified_reading).all()
                or not np.allclose(
                    verified_reading,
                    np.zeros(6),
                    atol=zero_tolerance,
                    rtol=0.0,
                )
            ):
                detail = (
                    np.round(verified_reading, 6).tolist()
                    if verified_reading is not None
                    else str(last_error)
                )
                raise RuntimeError("XJC空载硬件清零后验证失败：{}".format(detail))
            print("XJC初始化：清零后验证通过 {}".format(
                np.round(verified_reading, 6).tolist()
            ))
        finally:
            sensor.timeout_s = original_timeout

        if settle_seconds > 0:
            time.sleep(settle_seconds)

    if sensor_config.get("streaming", True):
        sensor.enable_streaming()


def _robot_password(robot_config):
    password = robot_config.get("password")
    if not password:
        password_env = robot_config.get("password_env", "AUBO_PASSWORD")
        password = os.environ.get(password_env)
    if not isinstance(password, str) or not password:
        raise RuntimeError(
            "Set robot.password in the deployment config or set the {} environment variable "
            "for AUBO login".format(robot_config.get("password_env", "AUBO_PASSWORD"))
        )
    return password


def build_control_hardware(config: Dict, statistics, robot_lock=None):
    """只初始化AUBO与XJC，供保持测试和实时控制共用。"""
    from devices.aubo_c5 import AuboC5Robot
    from devices.xjc_ft import XjcForceTorqueSensor

    robot_config = config["robot"]
    robot = AuboC5Robot(
        robot_ip=robot_config["ip"],
        rpc_port=robot_config.get("rpc_port", 30004),
        username=robot_config.get("username", "aubo"),
        password=_robot_password(robot_config),
        robot_name=robot_config.get("robot_name"),
        request_timeout_ms=robot_config.get("request_timeout_ms", 1000),
    )

    sensor = None
    wrench_cache = None
    try:
        sensor_config = load_json(config["force_sensor_config"])
        sensor = XjcForceTorqueSensor(
            port=sensor_config.get("port"),
            baudrate=sensor_config.get("baudrate", 115200),
            timeout_s=sensor_config.get("timeout_s", 0.1),
            slave_address=sensor_config.get("slave_address", 1),
            sample_rate=sensor_config.get("sample_rate", 250),
            usb_selector=sensor_config.get("usb_selector"),
            transport=sensor_config.get("transport"),
        )
        initialize_xjc_sensor(sensor, sensor_config)
        compensator = WrenchCompensator(
            force_zero=sensor_config["force_zero"],
            torque_zero=sensor_config["torque_zero"],
            gravity_in_aubo_base=sensor_config["gravity_in_aubo_base"],
            center_of_mass_in_sensor=sensor_config["center_of_mass_in_sensor"],
            xjc_to_tcp=sensor_config["XJC_2_TCP"],
        )
        execution = config["execution"]
        wrench_cache = LatestWrenchCache(
            sensor=sensor,
            compensator=compensator,
            statistics=statistics,
            sample_rate_hz=sensor_config.get("sample_rate", 250),
            max_age_s=execution.get("xjc_max_sample_age_s", 0.02),
            read_retries=sensor_config.get("read_retries", 3),
        )
        wrench_cache.start()
        return robot, sensor, compensator, wrench_cache
    except Exception:
        if wrench_cache is not None:
            wrench_cache.close()
        if sensor is not None:
            sensor.close()
        robot.close()
        raise


def build_live_hardware(
    config: Dict,
    frames: CoordinateFrames,
    statistics,
    robot_lock=None,
):
    from devices.d455 import D455Camera

    robot = sensor = compensator = wrench_cache = camera = None
    try:
        robot, sensor, compensator, wrench_cache = build_control_hardware(
            config,
            statistics,
            robot_lock=robot_lock,
        )
        camera_config = config["camera"]
        camera = D455Camera(
            serial=camera_config.get("serial"),
            depth_size=tuple(camera_config.get("depth_size", [1280, 720])),
            color_size=tuple(camera_config.get("color_size", [1280, 720])),
            fps=camera_config.get("fps", 30),
            timeout_ms=camera_config.get("timeout_ms", 3000),
            warmup_frames=camera_config.get("warmup_frames", 20),
        )
        source = LiveObservationSource(
            config,
            frames,
            robot,
            sensor,
            camera,
            compensator,
            robot_lock=robot_lock,
            wrench_cache=wrench_cache,
        )
        return robot, camera, sensor, wrench_cache, source
    except Exception:
        if wrench_cache is not None:
            wrench_cache.close()
        if sensor is not None:
            sensor.close()
        if camera is not None:
            camera.close()
        robot.close()
        raise


def validate_deployment_config(config, args):
    """在连接、清零或运动任何硬件之前完成全部运行参数校验。"""
    source_mode = config["source"]["mode"]
    if source_mode not in ("replay", "live"):
        raise ValueError("source.mode must be replay or live")
    hold_seconds = float(args.hold_seconds or 0.0)
    if hold_seconds < 0 or not np.isfinite(hold_seconds):
        raise ValueError("--hold-seconds must be a finite non-negative number")
    if hold_seconds:
        if not args.execute:
            raise RuntimeError("--hold-seconds requires --execute")
        if source_mode != "live" or args.prepare or args.recorded_actions:
            raise RuntimeError(
                "--hold-seconds requires live mode and cannot be combined with --prepare or replay"
            )
    if args.execute:
        if source_mode != "live":
            raise RuntimeError("--execute is prohibited with replay data")
        if not config["robot"].get("allow_motion", False):
            raise RuntimeError("Set robot.allow_motion=true as the second motion interlock")
        if not config["coordinates"].get("calibrated", False):
            raise RuntimeError("D455、公共 BASE 与 AUBO 基座的坐标链尚未完成标定")
        if args.recorded_actions:
            raise RuntimeError("--recorded-actions cannot be executed on hardware")
    if args.prepare and not args.execute:
        raise RuntimeError("--prepare requires --execute because it moves the AUBO")

    execution = config["execution"]
    if execution.get("control_mode") != "position":
        raise ValueError(
            "当前单线程关节轨迹执行器仅支持 execution.control_mode=position"
        )
    position_command = execution.get("position_command", "servo_joint")
    if position_command != "send_command":
        raise ValueError("当前位置控制必须使用 execution.position_command=send_command")
    numeric_positive = (
        "control_rate_hz",
        "trajectory_duration_s",
        "control_timeout_s",
        "servo_lookahead_s",
        "servo_gain",
        "xjc_max_sample_age_s",
    )
    for key in numeric_positive:
        value = float(execution[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("execution.{} must be positive and finite".format(key))
    if int(execution.get("servo_queue_full_max_consecutive", 5)) <= 0:
        raise ValueError("servo_queue_full_max_consecutive must be positive")
    workspace = np.asarray(execution["workspace_bounds_aubo_base"], dtype=np.float64)
    if workspace.shape != (3, 2) or not np.isfinite(workspace).all() or np.any(
        workspace[:, 0] >= workspace[:, 1]
    ):
        raise ValueError("workspace_bounds_aubo_base must contain three valid ranges")
    protocol = config["inference"].get("protocol", "json")
    if protocol not in ("json", "numpy"):
        raise ValueError("inference.protocol must be json or numpy")
    sensor_config = load_json(config["force_sensor_config"])
    if float(sensor_config.get("sample_rate", 250)) <= 0:
        raise ValueError("XJC sample_rate must be positive")
    return hold_seconds


def plan_segments(planner, actions):
    """把完整预测转换为顺序控制段，同时保留位置/混合切换语义。"""
    remaining = np.asarray(actions, dtype=np.float64)
    segments = []
    while len(remaining):
        chunk = planner.plan(remaining)
        consumed = len(chunk.actions)
        if consumed <= 0:
            raise RuntimeError("Action planner consumed no actions")
        segments.append((chunk.mode, chunk.actions))
        remaining = remaining[consumed:]
    return segments


def move_to_point_times(trajectory_duration_s, point_count):
    """生成参考AUBO ``move_to_point(t, pose)`` 所需的累计到达时间序列。"""
    duration = float(trajectory_duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("trajectory_duration_s must be positive and finite")
    if int(point_count) <= 0:
        raise ValueError("point_count must be positive")
    return np.linspace(duration / int(point_count), duration, int(point_count))


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


def solve_continuous_joint_targets(
    robot,
    actions,
    camera_to_aubo_base,
    workspace_bounds,
    initial_seed=None,
):
    """连续求解整块IK；每一步都以上一步成功关节解作为下一步种子。"""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 13 or len(actions) == 0:
        raise ValueError("actions must have shape (N, 13), with N > 0")
    workspace = np.asarray(workspace_bounds, dtype=np.float64)
    if workspace.shape != (3, 2):
        raise ValueError("workspace_bounds must have shape (3, 2)")
    seed = (
        np.asarray(robot.get_joint_pos(), dtype=np.float64)
        if initial_seed is None
        else np.asarray(initial_seed, dtype=np.float64)
    )
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError("initial IK seed must contain six finite joints")
    targets = []
    for index, action in enumerate(actions):
        position, quat, _ = transform_action_to_aubo_base(
            action,
            camera_to_aubo_base,
        )
        if np.any(position < workspace[:, 0]) or np.any(position > workspace[:, 1]):
            raise RuntimeError(
                "Policy action {} is outside the AUBO workspace".format(index)
            )
        seed = np.asarray(
            robot.inverse_kinematics(position, quat, rest_joints=seed),
            dtype=np.float64,
        )
        targets.append(seed.copy())
    # inverse_kinematics遇到非零返回码会直接抛错，因此能进入列表的IK码均为0。
    return np.asarray(targets, dtype=np.float64), np.zeros(len(targets), dtype=np.int64)


def make_policy_ready_pose(
    first_action,
    camera_to_aubo_base,
    retreat_distance_m: float,
    tool_forward_axis: str,
):
    """由策略第一目标生成保持无接触距离的任务预备位姿。"""
    if not np.isfinite(retreat_distance_m) or retreat_distance_m <= 0:
        raise ValueError("preparation.retreat_distance_m must be positive")
    if tool_forward_axis not in ("+z", "-z"):
        raise ValueError("preparation.tool_forward_axis must be '+z' or '-z'")
    target_position, target_quat, _ = transform_action_to_aubo_base(
        first_action,
        camera_to_aubo_base,
    )
    target_rotation = Rot.from_quat(target_quat).as_matrix()
    forward = target_rotation[:, 2]
    if tool_forward_axis == "-z":
        forward = -forward
    ready_position = target_position - forward * float(retreat_distance_m)
    return ready_position, target_quat, target_position


def run_policy_preparation(
    config,
    robot,
    live_source,
    inference,
    camera_to_aubo_base,
    robot_lock,
):
    """先MoveJ到初始构型，再根据一次策略预测移动到无接触预备位姿。"""
    preparation = config.get("preparation", {})
    if not preparation.get("enabled", False):
        raise RuntimeError("Set preparation.enabled=true before using --prepare")

    waypoints = preparation.get("init_joint_waypoints_rad")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("preparation.init_joint_waypoints_rad must not be empty")
    move_velocity = float(preparation.get("movej_max_velocity_rad_s", 0.05))
    move_acceleration = float(preparation.get("movej_max_acceleration_rad_s2", 0.1))
    move_timeout = float(preparation.get("movej_timeout_s", 60.0))
    tolerance = float(preparation.get("movej_tolerance_rad", 0.005))

    print("准备阶段：正在低速移动到任务初始化关节构型")
    for index, waypoint in enumerate(waypoints):
        joints = np.asarray(waypoint, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError(
                "preparation.init_joint_waypoints_rad[{}] must contain 6 finite values".format(
                    index
                )
            )
        with robot_lock:
            robot.move_joints(
                joints,
                max_velocity=move_velocity,
                max_acceleration=move_acceleration,
                wait=True,
                timeout_s=move_timeout,
                tolerance_rad=tolerance,
            )
        print("准备阶段：初始化路点{}/{}已到达".format(index + 1, len(waypoints)))

    # 到达统一初始构型后才采集观测。第一次策略输出仅用于定位任务区域，
    # 不会被后面的实时控制继续使用。
    observation = None
    while observation is None:
        observation = live_source.next()
    actions = inference.predict(config["name"], observation)
    if np.asarray(actions).ndim != 2 or len(actions) == 0:
        raise RuntimeError("Policy returned no action for preparation")

    ready_position, ready_quat, predicted_position = make_policy_ready_pose(
        actions[0],
        camera_to_aubo_base,
        float(preparation.get("retreat_distance_m", 0.08)),
        preparation.get("tool_forward_axis", "+z"),
    )
    workspace = np.asarray(
        config["execution"]["workspace_bounds_aubo_base"],
        dtype=np.float64,
    )
    if workspace.shape != (3, 2):
        raise ValueError("workspace_bounds_aubo_base must have shape (3, 2)")
    if np.any(ready_position < workspace[:, 0]) or np.any(ready_position > workspace[:, 1]):
        raise RuntimeError("Policy-derived ready pose is outside the AUBO workspace")

    with robot_lock:
        ready_joints = robot.inverse_kinematics(ready_position, ready_quat)
    print(json.dumps({
        "preparation": "policy_ready_pose",
        "predicted_target_xyz_aubo_base": np.round(predicted_position, 6).tolist(),
        "ready_xyz_aubo_base": np.round(ready_position, 6).tolist(),
        "ready_quat_xyzw_aubo_base": np.round(ready_quat, 6).tolist(),
        "ready_joints_rad": np.round(ready_joints, 6).tolist(),
        "retreat_distance_m": float(preparation.get("retreat_distance_m", 0.08)),
    }, indent=2, ensure_ascii=False))

    if not preparation.get("move_to_policy_ready", False):
        print(
            "准备阶段预览完成：尚未移动到策略预备位姿；确认上面的位姿、关节角和"
            "MoveJ路径后，将preparation.move_to_policy_ready设为true再运行"
        )
        return False

    print("准备阶段：正在低速MoveJ到策略生成的无接触预备位姿")
    with robot_lock:
        robot.move_joints(
            ready_joints,
            max_velocity=move_velocity,
            max_acceleration=move_acceleration,
            wait=True,
            timeout_s=move_timeout,
            tolerance_rad=tolerance,
        )
    # obs_horizon大于1时也不能混入准备移动之前的旧观测。
    for buffer in live_source.buffers.values():
        buffer.clear()
    print("准备阶段完成：旧预测已丢弃，下一次观测将重新调用策略")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/deploy_aubo.json")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow real AUBO ServoJ commands; also requires allow_motion=true",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Move through configured init joints and a policy-derived ready pose before live control",
    )
    parser.add_argument(
        "--recorded-actions",
        action="store_true",
        help="Use HDF5 actions instead of contacting node.py (replay mode only)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep the current TCP pose with the single Servo loop; no D455 or node.py",
    )
    parser.add_argument("--max-policy-calls", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source_mode = config["source"]["mode"]
    hold_seconds = validate_deployment_config(config, args)

    max_calls = args.max_policy_calls
    if max_calls is None:
        max_calls = int(config["source"].get("max_policy_calls", 1))
    if max_calls <= 0:
        raise ValueError("max_policy_calls must be positive")

    policy_config = config["policy"]
    execution = config["execution"]
    frames = make_coordinate_frames(source_mode)
    camera_to_aubo_base = frames.camera_to_aubo_base
    if not config["coordinates"].get("calibrated", False):
        print(
            "警告：D455/公共BASE/AUBO基座坐标链尚未标定，转换结果仅供离线检查"
        )

    statistics = RuntimeStatistics()
    inference = None
    controller = None
    robot = camera = sensor = wrench_cache = live_source = replay_source = None
    servo_control = None
    robot_lock = threading.RLock()

    def create_inference_client():
        inference_config = config["inference"]
        return InferenceClient(
            endpoint=inference_config["endpoint"],
            timeout_ms=inference_config.get("timeout_ms", 120000),
            protocol=inference_config.get("protocol", "json"),
        )

    def get_live_observation():
        observation = None
        while observation is None:
            observation = live_source.next()
        return observation

    def predict_with_client(client, observation):
        started = time.perf_counter()
        actions = client.predict(config["name"], observation)
        latency = time.perf_counter() - started
        statistics.record_inference(latency, accepted=True)
        actions = np.asarray(actions, dtype=np.float64)[: int(policy_config["act_horizon"])]
        if actions.ndim != 2 or actions.shape[1] != 13 or len(actions) == 0:
            raise RuntimeError("Policy returned no valid 13-dimensional actions")
        return actions, latency

    def predict(observation):
        return predict_with_client(inference, observation)

    try:
        # 当前部署仅采用与原Flexiv非接触阶段一致的纯位置路径：
        # 模型TCP位姿 -> 启动Servo前连续IK -> 关节插值 -> AUBO ServoJ。

        # 保持模式和HDF5动作回放不需要连接模型服务器。
        if not args.recorded_actions and not hold_seconds:
            inference = create_inference_client()

        if hold_seconds:
            robot, sensor, compensator, wrench_cache = build_control_hardware(
                config,
                statistics,
                robot_lock=robot_lock,
            )
        elif source_mode == "replay":
            source_config = config["source"]
            replay_source = ReplaySource(
                path=source_config["hdf5_path"],
                demo=source_config.get("demo", "demo_0"),
                obs_horizon=policy_config["obs_horizon"],
                action_horizon=policy_config["act_horizon"],
                start_index=source_config.get("start_index", 0),
            )
        else:
            robot, camera, sensor, wrench_cache, live_source = build_live_hardware(
                config,
                frames,
                statistics,
                robot_lock=robot_lock,
            )
            if args.execute:
                robot.require_motion_allowed()
                if args.prepare:
                    preparation_complete = run_policy_preparation(
                        config=config,
                        robot=robot,
                        live_source=live_source,
                        inference=inference,
                        camera_to_aubo_base=camera_to_aubo_base,
                        robot_lock=robot_lock,
                    )
                    if not preparation_complete:
                        return

        # 保持测试不连接D455或node.py，但柔顺位置保持仍读取XJC闭环。
        if hold_seconds:
            robot.require_motion_allowed()
            robot.enable_servo()
            servo_control = SingleServoControlLoop(
                robot=robot,
                wrench_reader=wrench_cache.read,
                controller=controller,
                camera_to_aubo_base=camera_to_aubo_base,
                config=execution,
                robot_lock=robot_lock,
                statistics=statistics,
            )
            servo_control.start()
            print("当前位置保持测试开始：{:.3f}秒".format(hold_seconds))
            deadline = time.monotonic() + hold_seconds
            while time.monotonic() < deadline:
                servo_control.raise_if_failed()
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            servo_control.raise_if_failed()
            print(json.dumps({
                "hold_seconds": hold_seconds,
                "motion_mode": "hold_current_pose",
                "statistics": statistics.snapshot(),
            }, indent=2, ensure_ascii=False))
            return

        # 离线回放只打印位置目标，不启用Servo或混合力控。
        if not args.execute:
            completed = 0
            while completed < max_calls:
                if replay_source is not None:
                    observation, recorded_actions, source_index = replay_source.next()
                else:
                    observation = get_live_observation()
                    recorded_actions = None
                    source_index = completed
                if args.recorded_actions:
                    actions = np.asarray(recorded_actions, dtype=np.float64)[
                        : int(policy_config["act_horizon"])
                    ]
                else:
                    actions, _ = predict(observation)
                chunk = PlannedChunk("position", actions)
                print_chunk_summary(completed, source_index, chunk, camera_to_aubo_base)
                completed += 1
            return

        if execution["control_mode"] == "position":
            observation = get_live_observation()
            actions, inference_latency = predict(observation)
            workspace = np.asarray(
                execution["workspace_bounds_aubo_base"],
                dtype=np.float64,
            )
            # 启动Servo前连续求解完整10步IK。纯位置实时线程只插值这些
            # 关节路点并发送ServoJ，绝不在每个控制周期重新调用IK RPC。
            with robot_lock:
                nominal_joints, _ = solve_continuous_joint_targets(
                    robot,
                    actions,
                    camera_to_aubo_base,
                    workspace,
                )

            planner = ActionChunkPlanner(
                force_threshold=float(policy_config["force_threshold"]),
                force_ratio=float(policy_config["force_ratio"]),
                # 当前纯位置部署不让预测力改变执行方式。
                hybrid_force_enabled=False,
            )
            servo_control = SingleServoControlLoop(
                robot=robot,
                wrench_reader=wrench_cache.read,
                controller=controller,
                camera_to_aubo_base=camera_to_aubo_base,
                config=execution,
                robot_lock=robot_lock,
                statistics=statistics,
            )
            robot.require_motion_allowed()
            robot.enable_servo()
            servo_control.start()
            live_source.safety_check = servo_control.raise_if_failed

            accepted = 1
            current_latency = inference_latency
            current_source_index = 0
            timeout_s = max(
                5.0,
                float(execution["trajectory_duration_s"]) + 2.0,
            )
            revision = servo_control.move_to_point(
                move_to_point_times(execution["trajectory_duration_s"], len(actions)),
                plan_segments(planner, actions),
                joint_waypoints=nominal_joints,
            )
            while True:
                # 按参考move_to_point语义：当前模型完整动作块执行完毕，
                # 才采集新观测并请求下一次模型推理。
                if not servo_control.wait_executed_in_revision(
                    len(actions), timeout=timeout_s, revision=revision
                ):
                    raise RuntimeError("Timed out waiting for the complete move_to_point trajectory")
                servo_control.raise_if_failed()
                with robot_lock:
                    current_tcp = robot.get_tcp_pose(matrix=True)[:3, 3]
                print(json.dumps({
                    "policy_call": accepted - 1,
                    "source_index": current_source_index,
                    "execution_mode": "move_to_point_joint_position",
                    "action_count": len(actions),
                    "nominal_ik_count": len(nominal_joints),
                    "inference_latency_s": round(float(current_latency), 6),
                    "executed_actions": servo_control.executed_actions,
                    "tcp_position_aubo_base": np.round(current_tcp, 6).tolist(),
                    "motion_sent": True,
                    "statistics": statistics.snapshot(),
                }, indent=2, ensure_ascii=False))
                if accepted >= max_calls:
                    break

                # 当前10点完成后才开始下一轮：重新采集D455/AUBO/XJC观测，
                # 再推理、预求解完整10步IK、提交下一段move_to_point轨迹。
                observation = get_live_observation()
                actions, current_latency = predict(observation)
                with robot_lock:
                    nominal_joints, _ = solve_continuous_joint_targets(
                        robot,
                        actions,
                        camera_to_aubo_base,
                        workspace,
                    )
                servo_control.raise_if_failed()
                revision = servo_control.move_to_point(
                    move_to_point_times(execution["trajectory_duration_s"], len(actions)),
                    plan_segments(planner, actions),
                    joint_waypoints=nominal_joints,
                )
                current_source_index = accepted
                accepted += 1

            servo_control.raise_if_failed()
            return

    except KeyboardInterrupt:
        print("Deployment stopped by user")
    finally:
        if servo_control is not None:
            servo_control.close()
        if args.execute and robot is not None:
            robot.stop()
        if inference is not None:
            inference.close()
        if replay_source is not None:
            replay_source.close()
        if wrench_cache is not None:
            wrench_cache.close()
        if sensor is not None:
            sensor.close()
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()
        if args.execute:
            print(json.dumps({"final_statistics": statistics.snapshot()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
