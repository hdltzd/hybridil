"""AUBO实时部署的线程、背压、插值和统计基础设施。"""

from collections import deque
import json
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from deployment.aubo_hybridil import transform_action_to_aubo_base
from devices.aubo_c5 import AuboQueueFullError
from devices.xjc_ft import XjcProtocolError


class RuntimeStatistics:
    """线程安全的有界运行统计；不会随部署时长无限增长。"""

    def __init__(self, sample_window=10000):
        self._lock = threading.Lock()
        self._counts = {
            "servo_attempts": 0,
            "servo_success": 0,
            "servo_queue_full": 0,
            "servo_queue_full_max_consecutive": 0,
            "xjc_read_errors": 0,
            "policy_predictions_accepted": 0,
        }
        self._servo_rpc = deque(maxlen=sample_window)
        self._cycles = deque(maxlen=sample_window)
        self._xjc_ages = deque(maxlen=sample_window)
        self._inference = deque(maxlen=sample_window)
        self._success_times = deque(maxlen=sample_window)

    def record_servo(self, success, queue_full, rpc_seconds, consecutive):
        with self._lock:
            self._counts["servo_attempts"] += 1
            self._servo_rpc.append(float(rpc_seconds))
            if success:
                self._counts["servo_success"] += 1
                self._success_times.append(time.monotonic())
            if queue_full:
                self._counts["servo_queue_full"] += 1
            self._counts["servo_queue_full_max_consecutive"] = max(
                self._counts["servo_queue_full_max_consecutive"],
                int(consecutive),
            )

    def record_cycle(self, seconds):
        with self._lock:
            self._cycles.append(float(seconds))

    def record_xjc(self, age_seconds=None, error=False):
        with self._lock:
            if error:
                self._counts["xjc_read_errors"] += 1
            if age_seconds is not None:
                self._xjc_ages.append(float(age_seconds))

    def record_inference(self, seconds, accepted=True):
        with self._lock:
            self._inference.append(float(seconds))
            if accepted:
                self._counts["policy_predictions_accepted"] += 1

    @staticmethod
    def _timing(values):
        values = np.asarray(list(values), dtype=np.float64)
        if len(values) == 0:
            return {"mean_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}
        return {
            "mean_ms": round(float(values.mean() * 1000.0), 6),
            "max_ms": round(float(values.max() * 1000.0), 6),
            "p95_ms": round(float(np.percentile(values, 95) * 1000.0), 6),
        }

    def snapshot(self):
        with self._lock:
            counts = dict(self._counts)
            rpc = tuple(self._servo_rpc)
            cycles = tuple(self._cycles)
            ages = tuple(self._xjc_ages)
            inference = tuple(self._inference)
            success_times = tuple(self._success_times)
        attempts = counts["servo_attempts"]
        queue_ratio = counts["servo_queue_full"] / attempts if attempts else 0.0
        if len(success_times) >= 2 and success_times[-1] > success_times[0]:
            actual_hz = (len(success_times) - 1) / (success_times[-1] - success_times[0])
        else:
            actual_hz = 0.0
        return {
            **counts,
            "servo_queue_full_ratio": round(float(queue_ratio), 6),
            "actual_servo_send_hz": round(float(actual_hz), 3),
            "servo_rpc": self._timing(rpc),
            "control_cycle": self._timing(cycles),
            "xjc_sample_age": self._timing(ages),
            "inference_latency": self._timing(inference),
        }


class ServoCommandPacer:
    """为所有ServoJ尝试提供一个共享的最小发送间隔。"""

    def __init__(self, period_seconds, clock=time.monotonic, sleeper=time.sleep):
        if not math.isfinite(period_seconds) or period_seconds <= 0:
            raise ValueError("Servo period must be positive")
        self.period = float(period_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self._lock = threading.Lock()
        self._last_attempt = None

    def wait(self):
        with self._lock:
            now = self.clock()
            if self._last_attempt is not None:
                delay = self.period - (now - self._last_attempt)
                if delay > 0:
                    self.sleeper(delay)
                    now = self.clock()
            self._last_attempt = now
            return now


class ServoBackpressure:
    """处理AUBO code 2；单帧丢弃，连续超限才停止。"""

    def __init__(self, robot, pacer, statistics, max_consecutive=5):
        if int(max_consecutive) <= 0:
            raise ValueError("servo_queue_full_max_consecutive must be positive")
        self.robot = robot
        self.pacer = pacer
        self.statistics = statistics
        self.max_consecutive = int(max_consecutive)
        self.consecutive = 0

    def send(self, callback):
        self.pacer.wait()
        started = time.perf_counter()
        try:
            result = callback()
        except AuboQueueFullError:
            elapsed = time.perf_counter() - started
            self.consecutive += 1
            self.statistics.record_servo(False, True, elapsed, self.consecutive)
            if self.consecutive >= self.max_consecutive:
                try:
                    self.robot.stop()
                finally:
                    raise RuntimeError(
                        "AUBO ServoJ queue was full {} consecutive times".format(
                            self.consecutive
                        )
                    )
            return False, None
        except Exception:
            elapsed = time.perf_counter() - started
            self.statistics.record_servo(False, False, elapsed, self.consecutive)
            raise
        elapsed = time.perf_counter() - started
        self.consecutive = 0
        self.statistics.record_servo(True, False, elapsed, 0)
        return True, result


class LatestWrenchCache:
    """唯一读取XJC字节流的250 Hz采集线程。"""

    def __init__(
        self,
        sensor,
        compensator,
        statistics,
        sample_rate_hz=250.0,
        max_age_s=0.02,
        read_retries=3,
    ):
        if min(float(sample_rate_hz), float(max_age_s), int(read_retries)) <= 0:
            raise ValueError("XJC cache rate, age and retries must be positive")
        self.sensor = sensor
        self.compensator = compensator
        self.statistics = statistics
        self.period = 1.0 / float(sample_rate_hz)
        self.max_age = float(max_age_s)
        self.read_retries = int(read_retries)
        self._condition = threading.Condition()
        self._sample = None
        self._timestamp = None
        self._failure = None
        self._last_protocol_error = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="xjc-acquisition", daemon=True)

    def start(self, startup_timeout_s=2.0):
        self._thread.start()
        deadline = time.monotonic() + float(startup_timeout_s)
        with self._condition:
            while self._sample is None and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = (
                        ": {}".format(self._last_protocol_error)
                        if self._last_protocol_error is not None
                        else ""
                    )
                    raise RuntimeError(
                        "Timed out waiting for the first XJC sample{}".format(detail)
                    )
                self._condition.wait(remaining)
            if self._failure is not None:
                raise RuntimeError("XJC acquisition failed: {}".format(self._failure)) from self._failure

    def _run(self):
        consecutive = 0
        while True:
            with self._condition:
                if self._closed:
                    return
            started = time.monotonic()
            try:
                sample = np.asarray(self.sensor.read(), dtype=np.float64)
                if sample.shape != (6,) or not np.isfinite(sample).all():
                    raise RuntimeError("XJC returned an invalid wrench")
                consecutive = 0
                with self._condition:
                    self._sample = sample.copy()
                    self._timestamp = time.monotonic()
                    self._condition.notify_all()
            except XjcProtocolError as error:
                consecutive += 1
                self.statistics.record_xjc(error=True)
                # ser2net/socat链路在流模式刚启动或发生短暂抖动时，可能连续
                # 丢失若干个4 ms数据包。协议错误只计数并继续同步：首帧是否
                # 失败由start()的整体超时决定；取得首帧后则由read()的20 ms
                # 样本新鲜度看门狗决定是否停止机械臂。
                with self._condition:
                    self._last_protocol_error = error
            except Exception as error:
                self.statistics.record_xjc(error=True)
                with self._condition:
                    self._failure = error
                    self._condition.notify_all()
                return
            delay = self.period - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def read(self, tcp_quat_xyzw_base):
        return self._read(tcp_quat_xyzw_base, require_fresh=True)

    def read_latest(self, tcp_quat_xyzw_base):
        """返回最近一次有效样本，供策略观测使用。

        纯位置控制不应因为一次点云编码或模型推理使样本年龄超过实时阈值
        而停止Servo。采集线程永久失败或从未取得样本仍然会明确报错。
        """
        return self._read(tcp_quat_xyzw_base, require_fresh=False)

    def _read(self, tcp_quat_xyzw_base, require_fresh):
        with self._condition:
            failure = self._failure
            sample = None if self._sample is None else self._sample.copy()
            timestamp = self._timestamp
        if failure is not None:
            raise RuntimeError("XJC acquisition stopped: {}".format(failure)) from failure
        if sample is None or timestamp is None:
            raise RuntimeError("XJC cache has no sample")
        age = time.monotonic() - timestamp
        self.statistics.record_xjc(age_seconds=age)
        if require_fresh and age > self.max_age:
            raise RuntimeError(
                "XJC sample is stale ({:.6f}s > {:.6f}s)".format(age, self.max_age)
            )
        return self.compensator.compensate(sample, tcp_quat_xyzw_base)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def interpolate_pose(start_position, start_quat, end_position, end_quat, alpha):
    """线性位置插值和最短路径四元数SLERP。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start_position = np.asarray(start_position, dtype=np.float64)
    end_position = np.asarray(end_position, dtype=np.float64)
    start_quat = np.asarray(start_quat, dtype=np.float64)
    end_quat = np.asarray(end_quat, dtype=np.float64)
    start_quat /= np.linalg.norm(start_quat)
    end_quat /= np.linalg.norm(end_quat)
    if float(np.dot(start_quat, end_quat)) < 0.0:
        end_quat = -end_quat
    relative = Rot.from_quat(start_quat).inv() * Rot.from_quat(end_quat)
    quat = (Rot.from_quat(start_quat) * Rot.from_rotvec(relative.as_rotvec() * alpha)).as_quat()
    position = start_position + alpha * (end_position - start_position)
    return position, quat


def interpolate_wrench(start_wrench, end_wrench, alpha):
    """六维力/力矩的线性插值。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start = np.asarray(start_wrench, dtype=np.float64)
    end = np.asarray(end_wrench, dtype=np.float64)
    if start.shape != (6,) or end.shape != (6,):
        raise ValueError("wrenches must have shape (6,)")
    return start + alpha * (end - start)


class SingleServoControlLoop:
    """纯位置部署的单线程关节轨迹 ServoJ 执行器。

    每轮模型输出的10个TCP位姿会在启动Servo前连续求解为10个关节路点。
    本类的唯一实时线程只做关节插值和 ``servoJoint``：不在线求IK，也不
    运行第二个轨迹发布线程。这样避免Python线程竞争或IK RPC偶发延迟打断
    AUBO的Servo指令流。
    """

    def __init__(
        self,
        robot,
        wrench_reader,
        controller,
        camera_to_aubo_base,
        config,
        robot_lock,
        statistics,
    ):
        self.robot = robot
        self.wrench_reader = wrench_reader
        self.controller = controller
        self.camera_to_aubo_base = np.asarray(camera_to_aubo_base, dtype=np.float64)
        self.robot_lock = robot_lock
        self.statistics = statistics
        self.control_mode = config.get("control_mode", "position")
        if self.control_mode not in ("position", "force_position"):
            raise ValueError("control_mode must be position or force_position")
        self.control_rate = float(config["control_rate_hz"])
        self.period = 1.0 / self.control_rate
        self.trajectory_duration = float(config["trajectory_duration_s"])
        self.timeout = float(config["control_timeout_s"])
        # Servo命令持续时间始终等于控制周期，避免JSON中两处频率配置不一致。
        self.duration = self.period
        self.lookahead = float(config.get("servo_lookahead_s", 0.1))
        self.gain = float(config.get("servo_gain", 200.0))
        self.workspace = np.asarray(config["workspace_bounds_aubo_base"], dtype=np.float64)
        self.backpressure = ServoBackpressure(
            robot,
            ServoCommandPacer(self.period),
            statistics,
            config.get("servo_queue_full_max_consecutive", 5),
        )
        self._condition = threading.Condition()
        self._joint_trajectory = None
        self._desired = None
        self._last_command = None
        self._last_joint_command = None
        self._last_target_wrench = np.zeros(6, dtype=np.float64)
        self._accepted_actions = 0
        self._executed_actions = 0
        self._revision = 0
        self._active_revision = 0
        self._executed_in_revision = 0
        self._failure = None
        # 仅在控制线程异常时写入；必须在 robot.stop() 前采集，避免停止
        # Servo 后覆盖控制柜原始的安全状态。
        self._failure_snapshot = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="aubo-servo-control", daemon=True)

    @property
    def last_command(self):
        with self._condition:
            if self._last_command is None:
                return None
            return tuple(value.copy() for value in self._last_command)

    @property
    def executed_actions(self):
        with self._condition:
            return self._executed_actions

    @property
    def executed_in_revision(self):
        with self._condition:
            return self._executed_in_revision

    @property
    def failure_snapshot(self):
        """返回最近一次控制失败的只读诊断快照。"""
        with self._condition:
            if self._failure_snapshot is None:
                return None
            # JSON 往返可避免调用方意外修改内部嵌套字典。
            return json.loads(json.dumps(self._failure_snapshot))

    def start(self):
        with self.robot_lock:
            tcp = self.robot.get_tcp_pose(matrix=True)
            joints = self.robot.get_joint_pos()
        initial = (tcp[:3, 3].copy(), Rot.from_matrix(tcp[:3, :3]).as_quat())
        with self._condition:
            self._last_command = initial
            self._last_joint_command = np.asarray(joints, dtype=np.float64).copy()
            self._desired = ("position", initial[0].copy(), initial[1].copy(), np.zeros(6))
        self._thread.start()

    def _validate_action(self, mode, action):
        if self.control_mode == "position":
            mode = "position"
        if mode not in ("position", "hybrid"):
            raise ValueError("mode must be position or hybrid")
        action = np.asarray(action, dtype=np.float64)
        position, quat, wrench = transform_action_to_aubo_base(action, self.camera_to_aubo_base)
        if np.any(position < self.workspace[:, 0]) or np.any(position > self.workspace[:, 1]):
            raise RuntimeError("Predicted TCP target is outside the AUBO workspace")
        if self.control_mode == "force_position" and self.controller is None:
            raise RuntimeError("force_position mode requires an admittance controller")
        if mode == "hybrid":
            if self.controller is None:
                raise RuntimeError("Hybrid mode requires an admittance controller")
            if np.linalg.norm(wrench[:3]) > self.controller.max_force:
                raise RuntimeError("Predicted force exceeds the configured safety limit")
            if np.linalg.norm(wrench[3:]) > self.controller.max_torque:
                raise RuntimeError("Predicted torque exceeds the configured safety limit")
        return mode, action.copy()

    def _validated_items(self, segments):
        """验证模型动作段，并保留原始10个轨迹路点。"""
        actions = []
        for mode, segment_actions in segments:
            for action in np.asarray(segment_actions):
                actions.append(self._validate_action(mode, action))
        if not actions:
            raise ValueError("Servo trajectory must contain at least one action")

        return actions

    def move_to_point(self, t, segments, joint_waypoints):
        """提交一段完整的预求解关节轨迹。

        参数 ``t`` 是与参考 ``MassageRobot.move_to_point`` 相同的累计到达
        时间序列；``segments`` 仍保留原有的模型动作和模式检查。纯位置模式
        下，``joint_waypoints`` 是与模型点一一对应的连续IK结果。

        本方法不会替换仍在执行的动作，也不会自行发起下一次推理。调用方必须
        使用 ``wait_executed_in_revision`` 等待本修订的全部点执行完成后，才
        采集新观测并提交下一块。这与原Flexiv的“完整动作块→新观测→新推理”
        逻辑一致。
        """
        if self.control_mode != "position":
            raise RuntimeError(
                "force_position is not supported by the single-thread joint trajectory executor"
            )
        items = self._validated_items(segments)
        time_points = np.asarray(t, dtype=np.float64).reshape(-1)
        if len(time_points) != len(items):
            raise ValueError("move_to_point time points must match action count")
        if not np.isfinite(time_points).all() or time_points[0] <= 0.0:
            raise ValueError("move_to_point time points must be positive and finite")
        if len(time_points) > 1 and np.any(np.diff(time_points) <= 0.0):
            raise ValueError("move_to_point time points must be strictly increasing")
        joints = np.asarray(joint_waypoints, dtype=np.float64)
        expected_shape = (len(items), len(self._last_joint_command))
        if joints.shape != expected_shape or not np.isfinite(joints).all():
            raise ValueError(
                "joint_waypoints must have shape {} and contain finite values".format(
                    expected_shape
                )
            )
        with self._condition:
            self.raise_if_failed_locked()
            if self._joint_trajectory is not None:
                raise RuntimeError("AUBO move_to_point trajectory is already running")
            self._revision += 1
            revision = self._revision
            self._active_revision = revision
            self._executed_in_revision = 0
            self._accepted_actions += len(items)
            self._joint_trajectory = (
                revision,
                time.monotonic(),
                time_points.copy(),
                self._last_joint_command.copy(),
                joints.copy(),
            )
            self._condition.notify_all()
            return revision

    def wait_executed_in_revision(self, count, timeout=None, revision=None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while (
                (
                    (revision is not None and self._active_revision != revision)
                    or self._executed_in_revision < count
                )
                and self._failure is None
            ):
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
            self.raise_if_failed_locked()
            return (
                (revision is None or self._active_revision == revision)
                and self._executed_in_revision >= count
            )

    def wait_executed(self, count, timeout=None):
        """等待累计动作数，供连续轨迹预取使用。"""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._executed_actions < int(count) and self._failure is None:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
            self.raise_if_failed_locked()
            return self._executed_actions >= int(count)

    def raise_if_failed_locked(self):
        if self._failure is not None:
            raise RuntimeError("AUBO Servo控制线程已停止：{}".format(self._failure)) from self._failure

    def raise_if_failed(self):
        with self._condition:
            self.raise_if_failed_locked()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _compute_command(
        self,
        current,
        current_quat,
        mode,
        target_position,
        target_quat,
        target_wrench,
    ):
        """计算一个Servo目标；纯位置模式直接使用轨迹发布的期望位姿。"""
        compliant = self.control_mode == "force_position"
        if not compliant:
            return np.asarray(target_position, dtype=np.float64), np.asarray(target_quat, dtype=np.float64)
        if self.wrench_reader is None or self.controller is None:
            raise RuntimeError("Compliant position control requires XJC and a controller")
        measured = self.controller.validate_measured_wrench(
            self.wrench_reader(current_quat)
        )
        desired_wrench = (
            target_wrench if mode == "hybrid" else np.zeros(6, dtype=np.float64)
        )
        return self.controller.step(
            current_position_base=current[:3, 3],
            current_quat_xyzw_base=current_quat,
            target_position_base=target_position,
            target_quat_xyzw_base=target_quat,
            measured_wrench_tcp=measured,
            desired_wrench_tcp=desired_wrench,
            dt=self.period,
            limit_profile="hybrid" if mode == "hybrid" else "position",
        )

    def _desired_target(self):
        with self._condition:
            mode, position, quat, wrench = self._desired
            return mode, position.copy(), quat.copy(), wrench.copy()

    def _next_joint_target(self):
        """从稀疏关节路点计算当前Servo周期的一个插值关节目标。

        只执行数组运算，不访问AUBO SDK；这保证实时控制周期中不会出现
        inverseKinematics RPC，也不需要单独的轨迹发布线程。
        """
        with self._condition:
            if self._joint_trajectory is None:
                return self._last_joint_command.copy(), "position"
            revision, started, times, start_joints, waypoints = self._joint_trajectory
            elapsed = max(0.0, time.monotonic() - started)
            index = int(np.searchsorted(times, elapsed, side="left"))
            if index >= len(times):
                target = waypoints[-1].copy()
                completed = len(times)
            else:
                previous_time = 0.0 if index == 0 else float(times[index - 1])
                previous_joints = start_joints if index == 0 else waypoints[index - 1]
                alpha = (elapsed - previous_time) / max(
                    float(times[index]) - previous_time,
                    1e-12,
                )
                target = previous_joints + np.clip(alpha, 0.0, 1.0) * (
                    waypoints[index] - previous_joints
                )
                completed = int(np.count_nonzero(times <= elapsed + 1e-9))

            if completed > self._executed_in_revision:
                increment = completed - self._executed_in_revision
                self._executed_actions += increment
                self._executed_in_revision = completed
                self._condition.notify_all()
            if completed == len(times):
                self._joint_trajectory = None
            return np.asarray(target, dtype=np.float64).copy(), "position"

    def _capture_failure_snapshot(
        self,
        error,
        mode,
        target_position,
        target_quat,
        command_position=None,
        command_quat=None,
        joint_command=None,
    ):
        """记录异常发生时的AUBO状态，供区分保护停机与指令问题。

        此方法只能在 ``robot.stop()`` 之前调用。stop 会关闭Servo模式，
        若在其后读取状态，将无法判断Servo究竟是控制柜主动退出，还是本进程
        主动停止。
        """
        snapshot = {
            "event": "aubo_servo_control_failure",
            "timestamp_unix_s": round(time.time(), 6),
            "error_type": type(error).__name__,
            "error": str(error),
            "control_mode": self.control_mode,
            "trajectory_mode": mode,
            "desired_tcp_xyz_aubo_base": np.asarray(target_position).tolist(),
            "desired_tcp_quat_xyzw_aubo_base": np.asarray(target_quat).tolist(),
            "command_tcp_xyz_aubo_base": (
                None if command_position is None else np.asarray(command_position).tolist()
            ),
            "command_tcp_quat_xyzw_aubo_base": (
                None if command_quat is None else np.asarray(command_quat).tolist()
            ),
            "joint_command_rad": (
                None if joint_command is None else np.asarray(joint_command).tolist()
            ),
            "last_joint_command_rad": (
                None
                if self._last_joint_command is None
                else np.asarray(self._last_joint_command).tolist()
            ),
        }
        try:
            with self.robot_lock:
                snapshot["aubo_status_before_stop"] = self.robot.get_status()
                snapshot["joint_position_rad"] = self.robot.get_joint_pos().tolist()
                snapshot["joint_velocity_rad_s"] = self.robot.get_joint_vel().tolist()
                snapshot["tcp_pose_xyz_rpy_aubo_base"] = (
                    self.robot.get_tcp_pose_euler().tolist()
                )
        except Exception as status_error:
            snapshot["status_read_error"] = "{}: {}".format(
                type(status_error).__name__, status_error
            )

        with self._condition:
            self._failure_snapshot = snapshot
        print("AUBO失败瞬间状态（stop前）：")
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    def _run(self):
        previous_start = None
        overrun_count = 0
        current_mode = None
        while True:
            with self._condition:
                if self._closed:
                    return
            cycle_start = time.monotonic()
            target_position = None
            target_quat = None
            command_position = None
            command_quat = None
            joint_command = None
            mode = None
            try:
                if previous_start is not None and cycle_start - previous_start > self.timeout:
                    overrun_count += 1
                    if overrun_count == 1 or overrun_count % 100 == 0:
                        print(
                            "警告：AUBO控制周期延迟{:.6f}秒，累计{}次；位置控制继续运行".format(
                                cycle_start - previous_start,
                                overrun_count,
                            )
                        )
                previous_start = cycle_start
                if self.control_mode == "position":
                    joint_command, mode = self._next_joint_target()
                    target_wrench = np.zeros(6, dtype=np.float64)
                else:
                    mode, target_position, target_quat, target_wrench = self._desired_target()
                compliant = self.control_mode == "force_position"
                if compliant and current_mode != (self.control_mode, mode):
                    self.controller.reset()
                current_mode = (self.control_mode, mode)
                if compliant:
                    with self.robot_lock:
                        current = self.robot.get_tcp_pose(matrix=True)
                    current_quat = Rot.from_matrix(current[:3, :3]).as_quat()
                    command_position, command_quat = self._compute_command(
                        current,
                        current_quat,
                        mode,
                        target_position,
                        target_quat,
                        target_wrench,
                    )
                    # 插值和导纳计算后再次检查，防止控制器修正量把命令推出工作区。
                    if np.any(command_position < self.workspace[:, 0]) or np.any(
                        command_position > self.workspace[:, 1]
                    ):
                        raise RuntimeError("Interpolated TCP command is outside the AUBO workspace")
                    with self.robot_lock:
                        joint_command = self.robot.inverse_kinematics(
                            command_position,
                            command_quat,
                            rest_joints=self._last_joint_command,
                        )
                else:
                    current = None
                    current_quat = None
                with self.robot_lock:
                    accepted, _ = self.backpressure.send(
                        lambda: self.robot.send_command(
                            joint_command,
                            0,
                            duration_s=self.duration,
                            lookahead_s=self.lookahead,
                            gain=self.gain,
                        )
                    )
                if accepted:
                    self._last_joint_command = np.asarray(
                        joint_command,
                        dtype=np.float64,
                    ).copy()
                if accepted:
                    with self._condition:
                        self._last_command = (
                            np.asarray(command_position, dtype=np.float64).copy(),
                            np.asarray(command_quat, dtype=np.float64).copy(),
                        )
                self.statistics.record_cycle(time.monotonic() - cycle_start)
            except Exception as error:
                self._capture_failure_snapshot(
                    error=error,
                    mode=mode,
                    target_position=target_position,
                    target_quat=target_quat,
                    command_position=command_position,
                    command_quat=command_quat,
                    joint_command=joint_command,
                )
                try:
                    self.robot.stop()
                except Exception:
                    pass
                with self._condition:
                    self._failure = error
                    self._condition.notify_all()
                return


class AsyncInferenceWorker:
    """在工作线程内创建并独占ZeroMQ REQ socket。"""

    def __init__(self, client_factory, policy_name, statistics):
        self.client_factory = client_factory
        self.policy_name = policy_name
        self.statistics = statistics
        self._condition = threading.Condition()
        self._request = None
        self._results = deque()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="policy-inference", daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, token, observation):
        with self._condition:
            if self._request is not None:
                raise RuntimeError("An asynchronous inference request is already pending")
            self._request = (token, observation)
            self._condition.notify_all()

    def wait(self, token, timeout=None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while True:
                for result in self._results:
                    if result[0] == token:
                        self._results.remove(result)
                        _, actions, latency, error = result
                        if error is not None:
                            raise RuntimeError("Asynchronous policy inference failed: {}".format(error)) from error
                        return actions, latency
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for asynchronous policy inference")
                    self._condition.wait(remaining)

    def _run(self):
        client = self.client_factory()
        try:
            while True:
                with self._condition:
                    while not self._closed and self._request is None:
                        self._condition.wait()
                    if self._closed:
                        return
                    token, observation = self._request
                    self._request = None
                started = time.perf_counter()
                try:
                    actions = client.predict(self.policy_name, observation)
                    error = None
                except Exception as caught:
                    actions = None
                    error = caught
                latency = time.perf_counter() - started
                with self._condition:
                    self._results.append((token, actions, latency, error))
                    self._condition.notify_all()
        finally:
            client.close()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
