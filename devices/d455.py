from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy as np


class D455Camera:
    """Minimal Intel RealSense D455 driver for policy deployment.

    Depth frames are aligned to the color stream. ``get_data`` returns the
    raw uint16 depth image; multiply it by ``depth_scale`` to obtain metres.
    Camera-to-reference-frame extrinsics intentionally do not belong here.
    """

    DEFAULT_DEPTH_SIZE = (1280, 720)
    DEFAULT_COLOR_SIZE = (1280, 720)
    DEFAULT_FPS = 30

    def __init__(
        self,
        serial: Optional[str] = None,
        depth_size: Tuple[int, int] = DEFAULT_DEPTH_SIZE,
        color_size: Tuple[int, int] = DEFAULT_COLOR_SIZE,
        fps: int = DEFAULT_FPS,
        timeout_ms: int = 3000,
        warmup_frames: int = 20,
        init_retries: int = 5,
    ) -> None:
        depth_size = self._validate_stream_size("depth_size", depth_size)
        color_size = self._validate_stream_size("color_size", color_size)
        if not isinstance(fps, Integral) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if not isinstance(timeout_ms, Integral) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if not isinstance(warmup_frames, Integral) or isinstance(warmup_frames, bool) or warmup_frames < 0:
            raise ValueError("warmup_frames must be a non-negative integer")
        if not isinstance(init_retries, Integral) or isinstance(init_retries, bool) or init_retries <= 0:
            raise ValueError("init_retries must be a positive integer")
        if serial is not None and not isinstance(serial, str):
            raise ValueError("serial must be a string or None")

        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise ImportError(
                "pyrealsense2 is required to create a D455Camera"
            ) from error

        self.serial = None
        self.timeout_ms = timeout_ms
        self.init_retries = init_retries
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._pipeline_started = False

        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("No Intel RealSense device was detected")

        available_devices = []
        for device in devices:
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            available_devices.append((device_name, device_serial))

        if serial:
            selected = next(
                (item for item in available_devices if item[1] == serial),
                None,
            )
            if selected is None:
                available_serials = ", ".join(item[1] for item in available_devices)
                raise RuntimeError(
                    "Requested RealSense serial {!r} was not found; available "
                    "serials: {}".format(serial, available_serials)
                )
        else:
            selected = next(
                (item for item in available_devices if "D455" in item[0].upper()),
                None,
            )
            if selected is None:
                available_names = ", ".join(item[0] for item in available_devices)
                raise RuntimeError(
                    "No Intel RealSense D455 was detected; available devices: "
                    + available_names
                )

        self.device_name, self.serial = selected
        if "D455" not in self.device_name.upper():
            raise RuntimeError(
                "RealSense serial {!r} is {}, not a D455".format(
                    self.serial,
                    self.device_name,
                )
            )
        self.config.enable_device(self.serial)

        depth_width, depth_height = depth_size
        color_width, color_height = color_size
        self.config.enable_stream(
            rs.stream.depth,
            depth_width,
            depth_height,
            rs.format.z16,
            fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            color_width,
            color_height,
            rs.format.bgr8,
            fps,
        )

        self.align = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter()

        try:
            pipeline_profile = self.pipeline.start(self.config)
            self._pipeline_started = True
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            # RealSense reports metres represented by one raw depth unit.
            self.depth_scale = float(depth_sensor.get_depth_scale())

            aligned_frames = None
            for _ in range(max(1, warmup_frames)):
                aligned_frames = self._wait_for_aligned_frames(timeout_ms)

            color_frame = aligned_frames.get_color_frame()
            color_intrinsics = (
                color_frame.get_profile()
                .as_video_stream_profile()
                .get_intrinsics()
            )
            self.intrinsics = [
                color_intrinsics.ppx,
                color_intrinsics.ppy,
                color_intrinsics.fx,
                color_intrinsics.fy,
            ]
            # Alias retained for compatibility with the data-collection driver.
            self.color_intrinsics = self.intrinsics
            self.mtx = np.array(
                [
                    [self.intrinsics[2], 0.0, self.intrinsics[0]],
                    [0.0, self.intrinsics[3], self.intrinsics[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_stream_size(name: str, size: Sequence[int]) -> Tuple[int, int]:
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError("{} must be a (width, height) pair".format(name))
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
            for value in size
        ):
            raise ValueError("{} values must be positive integers".format(name))
        return int(size[0]), int(size[1])

    def _wait_for_aligned_frames(self, timeout_ms: int):
        last_error = None
        for _ in range(self.init_retries):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                aligned_frames = self.align.process(frames)
                if (
                    aligned_frames.get_depth_frame()
                    and aligned_frames.get_color_frame()
                ):
                    return aligned_frames
            except RuntimeError as error:
                last_error = error

        message = "Failed to get aligned color and depth frames from D455"
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def get_data(
        self,
        hole_filling: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return one aligned ``(color_bgr, depth_raw)`` frame pair."""
        if not self._pipeline_started:
            raise RuntimeError("D455 pipeline is not running")

        aligned_frames = self._wait_for_aligned_frames(
            self.timeout_ms if timeout_ms is None else timeout_ms
        )
        depth_frame = aligned_frames.get_depth_frame()
        if hole_filling:
            depth_frame = self.hole_filling.process(depth_frame)
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        return color_image, depth_image

    def get(self, timeout_ms: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ForceCapture-compatible shorthand for ``get_data``."""
        return self.get_data(timeout_ms=timeout_ms)

    @staticmethod
    def depth_to_meters(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
        return np.asarray(depth_raw, dtype=np.float32) * depth_scale

    @staticmethod
    def get_point_cloud(
        depth: np.ndarray,
        intrinsics: Sequence[float],
        color: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Back-project a depth image in metres into the color camera frame."""
        if depth.ndim != 2:
            raise ValueError("depth must have shape (height, width)")
        if len(intrinsics) != 4:
            raise ValueError("intrinsics must be [ppx, ppy, fx, fy]")
        if color is not None and color.shape[:2] != depth.shape:
            raise ValueError("color and depth image sizes must match")

        height, width = depth.shape
        pix_x, pix_y = np.meshgrid(np.arange(width), np.arange(height))
        x = (pix_x - intrinsics[0]) * depth / intrinsics[2]
        y = (pix_y - intrinsics[1]) * depth / intrinsics[3]
        xyz = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
        rgb = None if color is None else color.reshape(-1, color.shape[-1])
        return xyz, rgb

    # Compatibility alias used by the ForceCapture reference driver.
    img2pc = get_point_cloud

    @staticmethod
    def transform_pc(pc_camera: np.ndarray, camera_to_target: np.ndarray) -> np.ndarray:
        if pc_camera.ndim != 2 or pc_camera.shape[1] != 3:
            raise ValueError("pc_camera must have shape (N, 3)")
        if camera_to_target.shape != (4, 4):
            raise ValueError("camera_to_target must have shape (4, 4)")

        homogeneous = np.concatenate(
            [pc_camera, np.ones((pc_camera.shape[0], 1), dtype=pc_camera.dtype)],
            axis=-1,
        )
        return (homogeneous @ camera_to_target.T)[:, :3]

    @staticmethod
    def voxelize(
        pc_xyz: np.ndarray,
        pc_rgb: np.ndarray,
        voxel_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if pc_xyz.ndim != 2 or pc_xyz.shape[1] != 3:
            raise ValueError("pc_xyz must have shape (N, 3)")
        if pc_rgb.ndim != 2 or pc_rgb.shape[1] != 3:
            raise ValueError("pc_rgb must have shape (N, 3)")
        if pc_xyz.shape[0] != pc_rgb.shape[0]:
            raise ValueError("pc_xyz and pc_rgb must have the same point count")

        finite = np.isfinite(pc_xyz).all(axis=1) & np.isfinite(pc_rgb).all(axis=1)
        xyz = np.asarray(pc_xyz[finite])
        rgb = np.asarray(pc_rgb[finite])
        if len(xyz) == 0:
            return xyz.reshape(0, 3), rgb.reshape(0, 3)

        # 部署环境只依赖 NumPy；每个体素稳定保留输入中的第一个点。
        voxel_index = np.floor(xyz / voxel_size).astype(np.int64)
        _, selected = np.unique(voxel_index, axis=0, return_index=True)
        selected.sort()
        return xyz[selected], rgb[selected]

    def close(self) -> None:
        """Stop the RealSense pipeline; repeated calls are safe."""
        if getattr(self, "_pipeline_started", False):
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline_started = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
