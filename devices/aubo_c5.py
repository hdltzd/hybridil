"""Safe, minimal AUBO C5 adapter for HybridIL deployment.

Constructing :class:`AuboC5Robot` only connects and logs in to the RPC
service. It never powers on the arm, unlocks a protective stop, changes the
collision level, starts the runtime machine, enables servo mode, or sends a
motion command. All state-changing operations are explicit method calls.

Pose conventions exposed by this module:

* positions are metres;
* quaternions accepted by public methods are scipy order ``[x, y, z, w]``;
* ``get_tcp_pose(matrix=False)`` mirrors the original Flexiv adapter and
  returns ``[x, y, z, qw, qx, qy, qz]``;
* raw AUBO TCP poses are ``[x, y, z, rx, ry, rz]`` with XYZ Euler angles.
"""

import time
from numbers import Real
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


class AuboError(RuntimeError):
    """Base error raised by the AUBO adapter."""


class AuboConnectionError(AuboError):
    """RPC connection or login failed."""


class AuboSafetyError(AuboError):
    """The robot state does not permit motion."""


class AuboCommandError(AuboError):
    """An AUBO SDK command returned a non-zero result."""


class AuboC5Robot:
    """AUBO C5 RPC state and motion adapter.

    The driver is intentionally independent of the massage-robot repository;
    only the vendor-provided ``pyaubo_sdk`` binary module is required.
    """

    DOF = 6
    RUNNING_MODE_VALUE = 8
    MOTION_SAFETY_MODE_VALUES = (1, 2)  # Normal and ReducedMode

    def __init__(
        self,
        robot_ip: str = "192.168.100.20",
        rpc_port: int = 30004,
        username: str = "aubo",
        password: str = "123456",
        request_timeout_ms: int = 1000,
        max_joint_step_rad: float = np.deg2rad(1.0),
    ) -> None:
        self._validate_connection_parameters(
            robot_ip,
            rpc_port,
            username,
            password,
            request_timeout_ms,
            max_joint_step_rad,
        )
        try:
            import pyaubo_sdk
        except ImportError as error:
            raise ImportError(
                "pyaubo_sdk is required to create an AuboC5Robot; use the "
                "aubo_massage environment or install a matching SDK build"
            ) from error

        self.sdk = pyaubo_sdk
        self.robot_ip = robot_ip
        self.rpc_port = int(rpc_port)
        self.username = username
        self.request_timeout_ms = int(request_timeout_ms)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.client = None
        self.robot_name = None
        self.robot = None
        self.robot_state = None
        self.robot_manage = None
        self.robot_config = None
        self.robot_algorithm = None
        self.motion_control = None
        self._closed = False
        # A read-only monitoring connection must not stop motion owned by a
        # different process when this adapter is closed.
        self._motion_owned = False

        try:
            self.client = self.sdk.RpcClient()
            self.client.setRequestTimeout(self.request_timeout_ms)
            self.client.connect(self.robot_ip, self.rpc_port)
            if not self.client.hasConnected():
                raise AuboConnectionError(
                    "Failed to connect to AUBO RPC at {}:{}".format(
                        self.robot_ip,
                        self.rpc_port,
                    )
                )

            self.client.login(username, password)
            if not self.client.hasLogined():
                raise AuboConnectionError("AUBO RPC login failed")

            robot_names = list(self.client.getRobotNames())
            if not robot_names:
                raise AuboConnectionError(
                    "AUBO RPC is connected but reported no robot interfaces"
                )
            self.robot_name = robot_names[0]
            self.robot = self.client.getRobotInterface(self.robot_name)
            self.robot_state = self.robot.getRobotState()
            self.robot_manage = self.robot.getRobotManage()
            self.robot_config = self.robot.getRobotConfig()
            self.robot_algorithm = self.robot.getRobotAlgorithm()
            self.motion_control = self.robot.getMotionControl()

            dof = int(self.robot_config.getDof())
            if dof != self.DOF:
                raise AuboConnectionError(
                    "Expected a 6-DOF AUBO C5, but {} reports {} DOF".format(
                        self.robot_name,
                        dof,
                    )
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_connection_parameters(
        robot_ip,
        rpc_port,
        username,
        password,
        request_timeout_ms,
        max_joint_step_rad,
    ) -> None:
        if not isinstance(robot_ip, str) or not robot_ip:
            raise ValueError("robot_ip must be a non-empty string")
        if not isinstance(rpc_port, int) or isinstance(rpc_port, bool):
            raise ValueError("rpc_port must be an integer")
        if rpc_port <= 0 or rpc_port > 65535:
            raise ValueError("rpc_port must be between 1 and 65535")
        if not isinstance(username, str) or not username:
            raise ValueError("username must be a non-empty string")
        if not isinstance(password, str):
            raise ValueError("password must be a string")
        if not isinstance(request_timeout_ms, int) or request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be a positive integer")
        if not isinstance(max_joint_step_rad, Real) or max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive")

    @staticmethod
    def _enum_description(value) -> Dict[str, object]:
        return {
            "name": getattr(value, "name", str(value)),
            "value": int(getattr(value, "value", value)),
        }

    @staticmethod
    def _as_vector(
        values: Sequence[float],
        size: int,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError("{} must have shape ({},)".format(name, size))
        if not np.isfinite(vector).all():
            raise ValueError("{} must contain only finite values".format(name))
        return vector

    def _require_open(self) -> None:
        if self._closed or self.client is None:
            raise AuboConnectionError("AUBO RPC connection is closed")
        if not self.client.hasConnected() or not self.client.hasLogined():
            raise AuboConnectionError("AUBO RPC connection or login was lost")

    def get_status(self) -> Dict[str, object]:
        """Return robot and safety state without modifying the controller."""
        self._require_open()
        robot_mode = self.robot_state.getRobotModeType()
        safety_mode = self.robot_state.getSafetyModeType()
        return {
            "robot_name": self.robot_name,
            "robot_mode": self._enum_description(robot_mode),
            "safety_mode": self._enum_description(safety_mode),
            "is_power_on": bool(self.robot_state.isPowerOn()),
            "is_steady": bool(self.robot_state.isSteady()),
            "is_collision_occurred": bool(
                self.robot_state.isCollisionOccurred()
            ),
            "servo_mode_enabled": bool(
                self.motion_control.isServoModeEnabled()
            ),
            "collision_level": int(self.robot_config.getCollisionLevel()),
            "dof": int(self.robot_config.getDof()),
        }

    def is_motion_allowed(self) -> bool:
        self._require_open()
        robot_mode = self.robot_state.getRobotModeType()
        safety_mode = self.robot_state.getSafetyModeType()
        robot_mode_value = int(getattr(robot_mode, "value", robot_mode))
        safety_mode_value = int(getattr(safety_mode, "value", safety_mode))
        return (
            robot_mode_value == self.RUNNING_MODE_VALUE
            and safety_mode_value in self.MOTION_SAFETY_MODE_VALUES
            and not self.robot_state.isCollisionOccurred()
        )

    def require_motion_allowed(self) -> None:
        if not self.is_motion_allowed():
            status = self.get_status()
            raise AuboSafetyError(
                "AUBO motion is not allowed: robot_mode={}, safety_mode={}, "
                "collision={}".format(
                    status["robot_mode"],
                    status["safety_mode"],
                    status["is_collision_occurred"],
                )
            )

    def get_joint_pos(self) -> np.ndarray:
        self._require_open()
        return self._as_vector(
            self.robot_state.getJointPositions(),
            self.DOF,
            "joint positions",
        )

    def get_joint_vel(self) -> np.ndarray:
        self._require_open()
        return self._as_vector(
            self.robot_state.getJointSpeeds(),
            self.DOF,
            "joint speeds",
        )

    def get_tcp_pose_euler(self) -> np.ndarray:
        self._require_open()
        return self._as_vector(
            self.robot_state.getTcpPose(),
            6,
            "TCP pose",
        )

    def get_tcp_pose(self, matrix: bool = False) -> np.ndarray:
        """Return the configured TCP pose in the AUBO base frame."""
        pose_euler = self.get_tcp_pose_euler()
        rotation = Rot.from_euler("xyz", pose_euler[3:], degrees=False)
        if matrix:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation.as_matrix()
            pose[:3, 3] = pose_euler[:3]
            return pose

        quat_xyzw = rotation.as_quat()
        return np.concatenate(
            [pose_euler[:3], quat_xyzw[3:], quat_xyzw[:3]]
        )

    def get_tcp_vel(self) -> np.ndarray:
        self._require_open()
        return self._as_vector(
            self.robot_state.getTcpSpeed(),
            6,
            "TCP speed",
        )

    def get_ext_wrench(self, base: bool = False) -> np.ndarray:
        """Read an AUBO force sensor when one is configured by the controller.

        The external XJC sensor used by the planned deployment will have its
        own adapter. This method only exposes force data already configured in
        the AUBO controller.
        """
        self._require_open()
        getter = (
            self.robot_state.getBaseForce
            if base
            else self.robot_state.getTcpForce
        )
        return self._as_vector(getter(), 6, "external wrench")

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        self._require_open()
        lower = self._as_vector(
            self.robot_config.getJointMinPositions(),
            self.DOF,
            "joint lower limits",
        )
        upper = self._as_vector(
            self.robot_config.getJointMaxPositions(),
            self.DOF,
            "joint upper limits",
        )
        return lower, upper

    def validate_joint_target(
        self,
        target_joints: Sequence[float],
        enforce_step_limit: bool = True,
    ) -> np.ndarray:
        target = self._as_vector(target_joints, self.DOF, "target joints")
        lower, upper = self.get_joint_limits()
        if np.any(target < lower) or np.any(target > upper):
            raise AuboSafetyError("Target joints exceed AUBO joint limits")
        if enforce_step_limit:
            current = self.get_joint_pos()
            max_step = float(np.max(np.abs(target - current)))
            if max_step > self.max_joint_step_rad:
                raise AuboSafetyError(
                    "Joint target step {:.6f} rad exceeds {:.6f} rad".format(
                        max_step,
                        self.max_joint_step_rad,
                    )
                )
        return target

    def inverse_kinematics(
        self,
        xyz_in_base: Sequence[float],
        quat_xyzw_in_base: Sequence[float],
        rest_joints: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Solve one AUBO IK target without moving the arm."""
        self._require_open()
        xyz = self._as_vector(xyz_in_base, 3, "TCP xyz")
        quat = self._as_vector(quat_xyzw_in_base, 4, "TCP quaternion")
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-12:
            raise ValueError("TCP quaternion norm must be non-zero")
        quat = quat / quat_norm
        euler = Rot.from_quat(quat).as_euler("xyz", degrees=False)
        pose_command = np.concatenate([xyz, euler]).tolist()
        seed = (
            self.get_joint_pos()
            if rest_joints is None
            else self._as_vector(rest_joints, self.DOF, "IK seed")
        )
        result, code = self.robot_algorithm.inverseKinematics(
            seed.tolist(),
            pose_command,
        )
        if int(code) != 0:
            raise AuboCommandError(
                "AUBO inverseKinematics failed with code {}".format(code)
            )
        return self.validate_joint_target(result, enforce_step_limit=False)

    def power_on_and_startup(
        self,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        """Explicitly power on and release brakes without changing safety settings."""
        self._require_open()
        if timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("startup timeouts must be positive")
        code = int(self.robot_manage.poweron())
        if code != 0:
            raise AuboCommandError("AUBO poweron failed with code {}".format(code))
        code = int(self.robot_manage.startup())
        if code != 0:
            raise AuboCommandError("AUBO startup failed with code {}".format(code))

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            mode = self.robot_state.getRobotModeType()
            if int(getattr(mode, "value", mode)) == self.RUNNING_MODE_VALUE:
                return
            time.sleep(poll_interval_s)
        raise AuboCommandError("Timed out waiting for AUBO Running mode")

    def power_off(self) -> None:
        self._require_open()
        self.stop()
        code = int(self.robot_manage.poweroff())
        if code != 0:
            raise AuboCommandError("AUBO poweroff failed with code {}".format(code))

    def set_tcp_offset(self, tcp_offset_xyzrpy: Sequence[float]) -> None:
        """Explicitly configure flange-to-TCP [x,y,z,rx,ry,rz]."""
        self._require_open()
        tcp_offset = self._as_vector(
            tcp_offset_xyzrpy,
            6,
            "TCP offset",
        )
        code = int(self.robot_config.setTcpOffset(tcp_offset.tolist()))
        if code != 0:
            raise AuboCommandError(
                "AUBO setTcpOffset failed with code {}".format(code)
            )

    def enable_servo(self, timeout_s: float = 1.0) -> None:
        self.require_motion_allowed()
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        code = int(self.motion_control.setServoMode(True))
        if code != 0:
            raise AuboCommandError(
                "AUBO setServoMode(True) failed with code {}".format(code)
            )
        # 从 SDK 接受启用命令开始，本进程就必须负责停止这段运动。
        # 即使后续状态轮询超时，也不能把机器人误判为“未持有运动”。
        self._motion_owned = True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.motion_control.isServoModeEnabled():
                return
            time.sleep(0.005)
        self.stop()
        raise AuboCommandError("Timed out enabling AUBO servo mode")

    def disable_servo(self) -> None:
        self._require_open()
        if self.motion_control.isServoModeEnabled():
            code = int(self.motion_control.setServoMode(False))
            if code != 0:
                raise AuboCommandError(
                    "AUBO setServoMode(False) failed with code {}".format(code)
                )
        self._motion_owned = False

    def servo_joints(
        self,
        target_joints: Sequence[float],
        duration_s: float = 0.008,
        lookahead_s: float = 0.1,
        gain: float = 200.0,
    ) -> None:
        """Send one bounded ServoJ sample; servo mode must already be enabled."""
        self.require_motion_allowed()
        if not self.motion_control.isServoModeEnabled():
            raise AuboSafetyError("AUBO servo mode is not enabled")
        if duration_s <= 0 or lookahead_s <= 0 or gain <= 0:
            raise ValueError("ServoJ timing and gain values must be positive")
        target = self.validate_joint_target(target_joints, enforce_step_limit=True)
        code = int(
            self.motion_control.servoJoint(
                target.tolist(),
                0,
                0,
                float(duration_s),
                float(lookahead_s),
                float(gain),
            )
        )
        if code != 0:
            self.stop()
            raise AuboCommandError(
                "AUBO servoJoint failed with code {}".format(code)
            )

    def move_joints(
        self,
        target_joints: Sequence[float],
        max_velocity: float = 0.1,
        max_acceleration: float = 0.2,
        wait: bool = True,
        timeout_s: float = 30.0,
        tolerance_rad: float = 0.005,
    ) -> None:
        """Execute one conservative discrete MoveJ target."""
        self.require_motion_allowed()
        if min(max_velocity, max_acceleration, timeout_s, tolerance_rad) <= 0:
            raise ValueError("MoveJ limits and timeout must be positive")
        if self.motion_control.isServoModeEnabled():
            raise AuboSafetyError("Disable servo mode before using move_joints")
        target = self.validate_joint_target(target_joints, enforce_step_limit=True)
        self._motion_owned = True
        code = int(
            self.motion_control.moveJoint(
                target.tolist(),
                float(max_acceleration),
                float(max_velocity),
                0,
                0,
            )
        )
        if code != 0:
            raise AuboCommandError(
                "AUBO moveJoint failed with code {}".format(code)
            )
        if not wait:
            return

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.require_motion_allowed()
            error = float(np.max(np.abs(self.get_joint_pos() - target)))
            if error <= tolerance_rad and self.robot_state.isSteady():
                return
            time.sleep(0.01)
        self.stop()
        raise AuboCommandError("Timed out waiting for AUBO MoveJ target")

    def stop(self) -> None:
        """Request motion cessation; never powers off or clears a safety event."""
        if self._closed or self.motion_control is None or not self._motion_owned:
            return
        stop_error = None
        try:
            # pyaubo_sdk 0.22.x: stopMove(quick, all_tasks)。安全停止采用
            # 快速停止，并停止机器人上的全部运动任务。
            code = int(self.motion_control.stopMove(True, True))
            if code != 0:
                stop_error = AuboCommandError(
                    "AUBO stopMove(True, True) failed with code {}".format(code)
                )
        except Exception as error:
            stop_error = error
        finally:
            try:
                if self.motion_control.isServoModeEnabled():
                    code = int(self.motion_control.setServoMode(False))
                    if code != 0 and stop_error is None:
                        stop_error = AuboCommandError(
                            "AUBO setServoMode(False) failed with code {}".format(code)
                        )
            except Exception as error:
                if stop_error is None:
                    stop_error = error
            finally:
                self._motion_owned = False
        if stop_error is not None:
            raise stop_error

    def close(self) -> None:
        """Stop commanded motion, log out, and disconnect; repeated calls are safe."""
        if self._closed:
            return
        try:
            if self.client is not None and self.client.hasConnected():
                try:
                    self.stop()
                except Exception:
                    pass
                if self.client.hasLogined():
                    self.client.logout()
                self.client.disconnect()
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
