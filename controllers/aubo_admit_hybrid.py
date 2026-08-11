"""Pure force/pose control utilities for the AUBO HybridIL deployment."""

from collections import deque
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("{} must be a finite ({},) vector".format(name, size))
    return result


class WrenchCompensator:
    """将 XJC 原始六维力转换为经过重力补偿的 TCP 坐标系六维力。"""

    def __init__(
        self,
        force_zero: Sequence[float],
        torque_zero: Sequence[float],
        gravity_in_aubo_base: Sequence[float],
        center_of_mass_in_sensor: Sequence[float],
        xjc_to_tcp: Sequence[Sequence[float]],
    ) -> None:
        self.force_zero = _vector(force_zero, 3, "force_zero")
        self.torque_zero = _vector(torque_zero, 3, "torque_zero")
        self.gravity_in_aubo_base = _vector(
            gravity_in_aubo_base,
            3,
            "gravity_in_aubo_base",
        )
        self.center_of_mass = _vector(
            center_of_mass_in_sensor,
            3,
            "center_of_mass_in_sensor",
        )
        self.xjc_to_tcp = np.asarray(xjc_to_tcp, dtype=np.float64)
        if self.xjc_to_tcp.shape != (4, 4):
            raise ValueError("XJC_2_TCP must have shape (4, 4)")
        if not np.isfinite(self.xjc_to_tcp).all():
            raise ValueError("XJC_2_TCP must contain finite values")

    def compensate(
        self,
        raw_wrench_sensor: Sequence[float],
        tcp_quat_xyzw_in_aubo_base: Sequence[float],
    ) -> np.ndarray:
        raw = _vector(raw_wrench_sensor, 6, "raw_wrench_sensor")
        quat = _vector(tcp_quat_xyzw_in_aubo_base, 4, "tcp quaternion")
        quat /= np.linalg.norm(quat)
        base_rotation_tcp = Rot.from_quat(quat).as_matrix()
        rotation_sensor_to_tcp = self.xjc_to_tcp[:3, :3]
        base_rotation_sensor = base_rotation_tcp @ rotation_sensor_to_tcp

        gravity_sensor = base_rotation_sensor.T @ self.gravity_in_aubo_base
        force_sensor = raw[:3] - self.force_zero - gravity_sensor
        torque_sensor = (
            raw[3:]
            - self.torque_zero
            - np.cross(self.center_of_mass, gravity_sensor)
        )

        sensor_origin_in_tcp = self.xjc_to_tcp[:3, 3]
        force_tcp = rotation_sensor_to_tcp @ force_sensor
        torque_tcp = (
            rotation_sensor_to_tcp @ torque_sensor
            + np.cross(sensor_origin_in_tcp, force_tcp)
        )
        return np.concatenate([force_tcp, torque_tcp])


class AuboAdmitHybridController:
    """Admittance controller adapted from the supplied AUBO reference."""

    def __init__(
        self,
        mass: Sequence[float],
        stiffness: Sequence[float],
        damping: Sequence[float],
        max_acceleration: Sequence[float],
        max_velocity: Sequence[float],
        max_pose_step: Sequence[float],
        max_force: float,
        max_torque: float,
        torque_admittance_threshold: float = 0.1,
        contact_force_z: float = -3.0,
        force_integral_gain: float = 0.5,
        force_integral_limit: float = 50.0,
        integral_window: int = 250,
    ) -> None:
        self.mass = _vector(mass, 6, "mass")
        self.stiffness = _vector(stiffness, 6, "stiffness")
        self.damping = _vector(damping, 6, "damping")
        if np.any(self.mass <= 0):
            raise ValueError("mass values must be positive")
        self.mass_inverse = 1.0 / self.mass
        self.max_acceleration = _vector(max_acceleration, 6, "max_acceleration")
        self.max_velocity = _vector(max_velocity, 6, "max_velocity")
        self.max_pose_step = _vector(max_pose_step, 6, "max_pose_step")
        if np.any(self.max_acceleration <= 0) or np.any(self.max_velocity <= 0):
            raise ValueError("acceleration and velocity limits must be positive")
        if np.any(self.max_pose_step <= 0):
            raise ValueError("pose step limits must be positive")
        if min(max_force, max_torque, torque_admittance_threshold) <= 0:
            raise ValueError("force, torque, and threshold limits must be positive")
        if integral_window <= 0:
            raise ValueError("integral_window must be positive")

        self.max_force = float(max_force)
        self.max_torque = float(max_torque)
        self.torque_admittance_threshold = float(torque_admittance_threshold)
        self.contact_force_z = float(contact_force_z)
        self.force_integral_gain = float(force_integral_gain)
        self.force_integral_limit = float(force_integral_limit)
        self.force_integral = deque([0.0] * integral_window, maxlen=integral_window)
        self.desired_twist_tcp = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self.desired_twist_tcp[:] = 0.0
        self.force_integral.extend([0.0] * self.force_integral.maxlen)

    @staticmethod
    def _clip(vector: np.ndarray, limits: np.ndarray) -> np.ndarray:
        return np.clip(vector, -limits, limits)

    def validate_measured_wrench(
        self,
        measured_wrench_tcp: Sequence[float],
    ) -> np.ndarray:
        """校验实测 TCP 六维力；位置与混合控制均必须调用。"""
        measured_wrench = _vector(
            measured_wrench_tcp,
            6,
            "measured_wrench",
        )
        if np.linalg.norm(measured_wrench[:3]) > self.max_force:
            raise RuntimeError("Measured force exceeds the configured safety limit")
        if np.linalg.norm(measured_wrench[3:]) > self.max_torque:
            raise RuntimeError("Measured torque exceeds the configured safety limit")
        return measured_wrench

    def step(
        self,
        current_position_base: Sequence[float],
        current_quat_xyzw_base: Sequence[float],
        target_position_base: Sequence[float],
        target_quat_xyzw_base: Sequence[float],
        measured_wrench_tcp: Sequence[float],
        desired_wrench_tcp: Sequence[float],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(dt) or dt <= 0 or dt > 0.1:
            raise ValueError("dt must be in (0, 0.1] seconds")
        current_position = _vector(current_position_base, 3, "current_position")
        target_position = _vector(target_position_base, 3, "target_position")
        current_quat = _vector(current_quat_xyzw_base, 4, "current quaternion")
        target_quat = _vector(target_quat_xyzw_base, 4, "target quaternion")
        measured_wrench = self.validate_measured_wrench(measured_wrench_tcp)
        desired_wrench = _vector(desired_wrench_tcp, 6, "desired_wrench")

        current_rotation = Rot.from_quat(current_quat).as_matrix()
        target_rotation = Rot.from_quat(target_quat).as_matrix()
        pose_error_tcp = np.zeros(6, dtype=np.float64)
        pose_error_tcp[:3] = current_rotation.T @ (
            target_position - current_position
        )
        pose_error_tcp[3:] = Rot.from_matrix(
            current_rotation.T @ target_rotation
        ).as_rotvec()

        wrench_error = measured_wrench - desired_wrench
        torque_enabled = (
            np.linalg.norm(wrench_error[3:]) > self.torque_admittance_threshold
        )
        if not torque_enabled:
            wrench_error[3:] = 0.0
            self.desired_twist_tcp[3:] = 0.0

        if measured_wrench[2] < self.contact_force_z:
            self.force_integral.append(wrench_error[2] * dt)
        else:
            self.force_integral.append(0.0)
        integral = np.clip(
            sum(self.force_integral),
            -self.force_integral_limit,
            self.force_integral_limit,
        )
        wrench_error[2] += self.force_integral_gain * integral

        acceleration = self.mass_inverse * (
            wrench_error
            - self.damping * self.desired_twist_tcp
            + self.stiffness * pose_error_tcp
        )
        acceleration = self._clip(acceleration, self.max_acceleration)
        self.desired_twist_tcp += acceleration * dt
        self.desired_twist_tcp = self._clip(
            self.desired_twist_tcp,
            self.max_velocity,
        )
        delta_pose_tcp = self._clip(
            self.desired_twist_tcp * dt,
            self.max_pose_step,
        )

        command_position = current_position + current_rotation @ delta_pose_tcp[:3]
        command_rotation = current_rotation @ Rot.from_rotvec(
            delta_pose_tcp[3:]
        ).as_matrix()
        command_quat = Rot.from_matrix(command_rotation).as_quat()
        return command_position, command_quat

    @staticmethod
    def orthogonalize_wrench_trajectory(actions: Sequence[Sequence[float]]) -> np.ndarray:
        """Apply the ForceMimic orthogonal force projection to 13-D actions."""
        result = np.asarray(actions, dtype=np.float64).copy()
        if result.ndim != 2 or result.shape[1] != 13:
            raise ValueError("actions must have shape (N, 13)")
        displacement = result[-1, :3] - result[0, :3]
        distance = float(np.linalg.norm(displacement))
        if distance < 0.005:
            raise ValueError("Action trajectory is too short to define motion direction")
        motion_direction = displacement / distance
        rotations = Rot.from_quat(result[:, 3:7]).as_matrix()
        force_in_camera = (rotations @ result[:, 7:10, None])[:, :, 0]
        force_in_camera -= (
            force_in_camera @ motion_direction
        )[:, None] * motion_direction
        result[:, 7:10] = (
            rotations.transpose(0, 2, 1) @ force_in_camera[:, :, None]
        )[:, :, 0]
        return result
