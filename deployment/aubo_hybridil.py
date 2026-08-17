"""Hardware-independent pieces of the HybridIL AUBO deployment pipeline.

坐标约定
--------
模型位姿为相机坐标系下的 ``[x, y, z, qx, qy, qz, qw]``，模型六维力为
TCP 坐标系下的 ``[Fx, Fy, Fz, Tx, Ty, Tz]``。公共 ``BASE`` 与机器人基座
不是同一个坐标系；相机到 AUBO 基座的变换必须通过公共 ``BASE`` 组合得到。
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from controllers.aubo_admit_hybrid import AuboAdmitHybridController


ACTION_DIM = 13


@dataclass(frozen=True)
class CoordinateFrames:
    """与 ``configs/pose.py`` 命名方向一致的坐标变换链。

    ``A_2_B`` 表示把 A 坐标系中的点变换到 B 坐标系。因而：

    新单臂部署将 ``pose.py`` 的 ``BASEr`` 定义为 AUBO 基座，因此：

    ``CAMERA_2_AUBO_BASE = inv(BASEr_2_BASE) @ CAMERA_2_BASE``。
    """

    camera_to_common_base: np.ndarray
    aubo_base_to_common_base: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_to_common_base",
            validate_transform(self.camera_to_common_base, "CAMERA_2_BASE"),
        )
        object.__setattr__(
            self,
            "aubo_base_to_common_base",
            validate_transform(
                self.aubo_base_to_common_base,
                "BASEr_2_BASE（AUBO_BASE_2_BASE）",
            ),
        )

    @property
    def camera_to_aubo_base(self) -> np.ndarray:
        """返回 ``CAMERA_2_AUBO_BASE``。"""

        return np.linalg.inv(self.aubo_base_to_common_base) @ self.camera_to_common_base

    def tcp_in_camera(self, tcp_in_aubo_base: Sequence[Sequence[float]]) -> np.ndarray:
        """把 AUBO SDK 返回的 ``TCP_2_AUBO_BASE`` 转成 ``TCP_2_CAMERA``。"""

        tcp_to_aubo = validate_transform(tcp_in_aubo_base, "TCP_2_AUBO_BASE")
        return np.linalg.inv(self.camera_to_aubo_base) @ tcp_to_aubo


def _finite_array(values, shape, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError("{} must have shape {}".format(name, shape))
    if not np.isfinite(result).all():
        raise ValueError("{} must contain only finite values".format(name))
    return result


def validate_transform(transform, name: str = "transform") -> np.ndarray:
    matrix = _finite_array(transform, (4, 4), name)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("{} must be a homogeneous transform".format(name))
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("{} rotation must be orthonormal".format(name))
    if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
        raise ValueError("{} rotation determinant must be +1".format(name))
    return matrix


def pose_matrix(position: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    xyz = _finite_array(position, (3,), "position")
    quat = _finite_array(quat_xyzw, (4,), "quaternion")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rot.from_quat(quat / norm).as_matrix()
    result[:3, 3] = xyz
    return result


def transform_action_to_aubo_base(
    action: Sequence[float],
    camera_to_aubo_base: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(xyz_aubo_base, quat_aubo_base, wrench_tcp)``。

    六维力保持在 TCP 坐标系，因为 AUBO 导纳控制器需要将它与 TCP 坐标系下的
    实测六维力直接比较。
    """
    action_array = _finite_array(action, (ACTION_DIM,), "action")
    camera_to_aubo_array = validate_transform(
        camera_to_aubo_base,
        "CAMERA_2_AUBO_BASE",
    )
    target_in_camera = pose_matrix(action_array[:3], action_array[3:7])
    target_in_aubo_base = camera_to_aubo_array @ target_in_camera
    target_quat = Rot.from_matrix(target_in_aubo_base[:3, :3]).as_quat()
    return target_in_aubo_base[:3, 3], target_quat, action_array[7:].copy()


@dataclass(frozen=True)
class PlannedChunk:
    """One executable prefix selected from a policy prediction."""

    mode: str
    actions: np.ndarray


class ActionChunkPlanner:
    """Reproduce the ForceMimic force-threshold execution state machine."""

    def __init__(self, force_threshold: float = 6.0, force_ratio: float = 0.2):
        if not np.isfinite(force_threshold) or force_threshold <= 0:
            raise ValueError("force_threshold must be positive")
        if not np.isfinite(force_ratio) or not 0.0 <= force_ratio <= 1.0:
            raise ValueError("force_ratio must be between 0 and 1")
        self.force_threshold = float(force_threshold)
        self.force_ratio = float(force_ratio)
        self.initial_force_detection = True

    @staticmethod
    def _actions(values) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != ACTION_DIM or len(result) == 0:
            raise ValueError("actions must have shape (N, 13), with N > 0")
        if not np.isfinite(result).all():
            raise ValueError("actions must contain only finite values")
        quat_norms = np.linalg.norm(result[:, 3:7], axis=1)
        if np.any(quat_norms < 1e-12):
            raise ValueError("action quaternions must be non-zero")
        result = result.copy()
        result[:, 3:7] /= quat_norms[:, None]
        return result

    def plan(self, actions: Sequence[Sequence[float]]) -> PlannedChunk:
        actions_array = self._actions(actions)
        forces = np.linalg.norm(actions_array[:, 7:10], axis=1)
        high = forces > self.force_threshold
        enough_high_force = int(np.count_nonzero(high)) > len(high) * self.force_ratio

        if self.initial_force_detection:
            if not enough_high_force:
                return PlannedChunk("position", actions_array)
            self.initial_force_detection = False
            first_high = int(np.argmax(high))
            if first_high > 0:
                return PlannedChunk("position", actions_array[:first_high])
            return PlannedChunk("hybrid", self._orthogonalize(actions_array))

        if np.all(high):
            return PlannedChunk("hybrid", self._orthogonalize(actions_array))
        if enough_high_force:
            first_low = int(np.argmax(~high))
            if first_low > 0:
                prefix = actions_array[:first_low]
                return PlannedChunk("hybrid", self._orthogonalize(prefix))
            first_high = int(np.argmax(high))
            if first_high > 0:
                return PlannedChunk("position", actions_array[:first_high])
        return PlannedChunk("position", actions_array)

    @staticmethod
    def _orthogonalize(actions: np.ndarray) -> np.ndarray:
        # A one-sample prefix cannot define a trajectory direction. Retaining
        # its wrench is safer than inventing an arbitrary projection plane.
        if len(actions) < 2:
            return actions.copy()
        try:
            return AuboAdmitHybridController.orthogonalize_wrench_trajectory(actions)
        except ValueError as error:
            if "too short" not in str(error):
                raise
            return actions.copy()


class PointCloudProcessor:
    """Construct the point-cloud observation expected by HybridIL."""

    def __init__(
        self,
        point_count: int,
        voxel_size: float,
        base_bounds: Sequence[Sequence[float]],
        object_bounds: Sequence[Sequence[float]],
        tool_bounds: Sequence[Sequence[float]],
        seed: int = 1,
    ) -> None:
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        if voxel_size < 0:
            raise ValueError("voxel_size cannot be negative")
        self.point_count = int(point_count)
        self.voxel_size = float(voxel_size)
        self.base_bounds = self._bounds(base_bounds, "base_bounds")
        self.object_bounds = self._bounds(object_bounds, "object_bounds")
        self.tool_bounds = self._bounds(tool_bounds, "tool_bounds")
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _bounds(bounds, name: str) -> np.ndarray:
        result = _finite_array(bounds, (3, 2), name)
        if np.any(result[:, 0] >= result[:, 1]):
            raise ValueError("{} lower bounds must be below upper bounds".format(name))
        return result

    @staticmethod
    def _inside(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        return np.all(
            (points > bounds[:, 0]) & (points < bounds[:, 1]),
            axis=1,
        )

    def process(
        self,
        xyz_camera: np.ndarray,
        rgb: np.ndarray,
        camera_to_common_base: Sequence[Sequence[float]],
        tcp_in_camera: Sequence[Sequence[float]],
    ) -> np.ndarray:
        xyz = np.asarray(xyz_camera, dtype=np.float64)
        colors = np.asarray(rgb, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_camera must have shape (N, 3)")
        if colors.shape != xyz.shape:
            raise ValueError("rgb must have the same shape as xyz_camera")
        camera_to_common_array = validate_transform(
            camera_to_common_base,
            "CAMERA_2_BASE",
        )
        tcp_in_camera_array = validate_transform(tcp_in_camera, "tcp_in_camera")

        finite = np.isfinite(xyz).all(axis=1) & np.isfinite(colors).all(axis=1)
        finite &= xyz[:, 2] > 0.0
        xyz = xyz[finite]
        colors = colors[finite]
        if len(xyz) == 0:
            raise RuntimeError("D455 frame contains no finite positive-depth points")

        xyz_common_base = self._transform_points(xyz, camera_to_common_array)
        xyz_tcp = self._transform_points(xyz, np.linalg.inv(tcp_in_camera_array))
        keep = self._inside(xyz_common_base, self.base_bounds)
        keep &= self._inside(xyz_common_base, self.object_bounds) | self._inside(
            xyz_tcp,
            self.tool_bounds,
        )
        # Same hand-removal convention as eval.py: discard points behind TCP.
        keep &= xyz_tcp[:, 2] > 0.0
        xyz = xyz[keep]
        colors = colors[keep]
        if len(xyz) == 0:
            raise RuntimeError("All D455 points were removed by deployment clipping")

        if self.voxel_size > 0:
            voxel_index = np.floor(xyz / self.voxel_size).astype(np.int64)
            _, selected = np.unique(voxel_index, axis=0, return_index=True)
            xyz = xyz[selected]
            colors = colors[selected]

        replace = len(xyz) < self.point_count
        selected = self.rng.choice(len(xyz), self.point_count, replace=replace)
        return np.concatenate([xyz[selected], colors[selected]], axis=1).astype(np.float32)

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return points @ transform[:3, :3].T + transform[:3, 3]


class ReplaySource:
    """Read policy observations and recorded actions from one robomimic demo."""

    OBSERVATION_KEYS = (
        "pointcloud",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_eef_wrench",
    )

    def __init__(
        self,
        path: str,
        demo: str,
        obs_horizon: int,
        action_horizon: int,
        start_index: int = 0,
    ) -> None:
        if obs_horizon <= 0 or action_horizon <= 0:
            raise ValueError("observation and action horizons must be positive")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        try:
            import h5py
        except ImportError as error:
            raise ImportError(
                "h5py is required only when source.mode is replay"
            ) from error
        self.file = h5py.File(str(self.path), "r")
        data = self.file.get("data")
        if data is None or demo not in data:
            self.close()
            raise KeyError("demo {!r} was not found in {}".format(demo, path))
        self.demo_name = demo
        self.demo = data[demo]
        self.obs = self.demo["obs"]
        missing = [key for key in self.OBSERVATION_KEYS if key not in self.obs]
        if missing:
            self.close()
            raise KeyError("replay observation is missing {}".format(missing))
        if "action_with_wrench" not in self.demo:
            self.close()
            raise KeyError("replay demo has no action_with_wrench dataset")
        self.length = len(self.demo["action_with_wrench"])
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.index = max(int(start_index), self.obs_horizon - 1)
        if self.index >= self.length:
            self.close()
            raise ValueError("start_index is outside the replay demo")

    def next(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
        if self.file is None or self.index >= self.length:
            raise StopIteration
        index = self.index
        obs_slice = slice(index - self.obs_horizon + 1, index + 1)
        observation = {
            key: np.asarray(self.obs[key][obs_slice])
            for key in self.OBSERVATION_KEYS
        }
        action_stop = min(index + self.action_horizon, self.length)
        recorded_actions = np.asarray(
            self.demo["action_with_wrench"][index:action_stop],
            dtype=np.float64,
        )
        self.index += max(1, len(recorded_actions))
        return observation, recorded_actions, index

    def close(self) -> None:
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class InferenceClient:
    """Validated ZeroMQ client for the existing ``node.py`` policy server."""

    def __init__(self, endpoint: str, timeout_ms: int = 120000, protocol="json") -> None:
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise ValueError("endpoint must be a tcp:// ZeroMQ endpoint")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if protocol not in ("json", "numpy"):
            raise ValueError("inference protocol must be json or numpy")
        self.protocol = protocol
        try:
            import zmq
        except ImportError as error:
            raise ImportError("pyzmq is required for policy inference") from error
        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)

    def predict(self, name: str, observation: Dict[str, np.ndarray]) -> np.ndarray:
        validated = {}
        for key in ReplaySource.OBSERVATION_KEYS:
            if key not in observation:
                raise KeyError("observation is missing {!r}".format(key))
            value = np.asarray(observation[key])
            if not np.isfinite(value).all():
                raise ValueError("observation {!r} contains NaN or Inf".format(key))
            validated[key] = value
        try:
            if self.protocol == "numpy":
                self.socket.send_multipart(encode_numpy_request(name, validated))
                actions = decode_numpy_response(self.socket.recv_multipart())
            else:
                payload = {"name": name}
                payload.update({key: value.tolist() for key, value in validated.items()})
                self.socket.send_string(json.dumps(payload))
                response = json.loads(self.socket.recv_string())
                actions = np.asarray(response.get("action"), dtype=np.float64)
        except self.zmq.Again as error:
            raise TimeoutError("Timed out communicating with the policy server") from error
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) == 0:
            raise RuntimeError("policy server returned an invalid action shape")
        if not np.isfinite(actions).all():
            raise RuntimeError("policy server returned NaN or Inf")
        return actions

    def close(self) -> None:
        if getattr(self, "socket", None) is not None:
            self.socket.close(linger=0)
            self.socket = None
        if getattr(self, "context", None) is not None:
            self.context.term()
            self.context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


NUMPY_PROTOCOL = "hybridil-numpy-v1"


def _array_from_frame(description, frame):
    dtype = np.dtype(description["dtype"])
    shape = tuple(int(value) for value in description["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(frame) != expected:
        raise ValueError("NumPy protocol frame size does not match its metadata")
    return np.frombuffer(frame, dtype=dtype).reshape(shape).copy()


def encode_numpy_request(name, observation):
    """将策略观测编码为小型JSON头和连续NumPy字节帧。"""
    descriptions = []
    frames = []
    for key in ReplaySource.OBSERVATION_KEYS:
        value = np.ascontiguousarray(observation[key])
        descriptions.append({"name": key, "dtype": value.dtype.str, "shape": value.shape})
        frames.append(value.tobytes(order="C"))
    header = {
        "protocol": NUMPY_PROTOCOL,
        "name": name,
        "arrays": descriptions,
    }
    return [json.dumps(header).encode("utf-8"), *frames]


def decode_numpy_request(parts):
    if len(parts) < 2:
        raise ValueError("NumPy policy request must contain metadata and array frames")
    header = json.loads(parts[0].decode("utf-8"))
    if header.get("protocol") != NUMPY_PROTOCOL:
        raise ValueError("Unsupported NumPy policy protocol")
    descriptions = header.get("arrays", [])
    if len(descriptions) != len(parts) - 1:
        raise ValueError("NumPy policy request frame count mismatch")
    observation = {
        description["name"]: _array_from_frame(description, frame)
        for description, frame in zip(descriptions, parts[1:])
    }
    missing = [key for key in ReplaySource.OBSERVATION_KEYS if key not in observation]
    if missing:
        raise KeyError("NumPy policy request is missing {}".format(missing))
    return header["name"], observation


def encode_numpy_response(name, actions):
    actions = np.ascontiguousarray(actions)
    header = {
        "protocol": NUMPY_PROTOCOL,
        "name": name,
        "action": {"dtype": actions.dtype.str, "shape": actions.shape},
    }
    return [json.dumps(header).encode("utf-8"), actions.tobytes(order="C")]


def decode_numpy_response(parts):
    if len(parts) != 2:
        raise ValueError("NumPy policy response must contain metadata and one action frame")
    header = json.loads(parts[0].decode("utf-8"))
    if header.get("protocol") != NUMPY_PROTOCOL:
        raise ValueError("Unsupported NumPy policy protocol")
    return np.asarray(_array_from_frame(header["action"], parts[1]), dtype=np.float64)
