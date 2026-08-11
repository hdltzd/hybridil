import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from devices.d455 import D455Camera


class TestD455CameraMath(unittest.TestCase):
    def test_depth_to_meters(self):
        depth_raw = np.array([[0, 1000], [1500, 2000]], dtype=np.uint16)

        depth_m = D455Camera.depth_to_meters(depth_raw, 0.001)

        np.testing.assert_allclose(
            depth_m,
            np.array([[0.0, 1.0], [1.5, 2.0]], dtype=np.float32),
        )

    def test_get_point_cloud(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        color = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
             [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
        )

        xyz, rgb = D455Camera.get_point_cloud(
            depth,
            intrinsics=[0.0, 0.0, 1.0, 1.0],
            color=color,
        )

        np.testing.assert_allclose(
            xyz,
            np.array(
                [[0.0, 0.0, 1.0],
                 [2.0, 0.0, 2.0],
                 [0.0, 3.0, 3.0],
                 [4.0, 4.0, 4.0]]
            ),
        )
        np.testing.assert_array_equal(rgb, color.reshape(-1, 3))

    def test_transform_pc(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        camera_to_target = np.eye(4)
        camera_to_target[:3, 3] = [0.5, -1.0, 2.0]

        transformed = D455Camera.transform_pc(points, camera_to_target)

        np.testing.assert_allclose(
            transformed,
            np.array([[1.5, 1.0, 5.0], [-0.5, -1.0, 4.0]]),
        )

    def test_rejects_mismatched_color_size(self):
        with self.assertRaisesRegex(ValueError, "sizes must match"):
            D455Camera.get_point_cloud(
                np.ones((2, 2)),
                [0.0, 0.0, 1.0, 1.0],
                np.ones((3, 2, 3)),
            )

    def test_voxelize(self):
        points = np.array(
            [[0.0, 0.0, 1.0], [0.0001, 0.0001, 1.0], [0.1, 0.0, 1.0]]
        )
        colors = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

        downsampled_points, downsampled_colors = D455Camera.voxelize(
            points,
            colors,
            voxel_size=0.01,
        )

        self.assertEqual(downsampled_points.shape, (2, 3))
        self.assertEqual(downsampled_colors.shape, (2, 3))

    def test_rejects_invalid_voxel_input(self):
        with self.assertRaisesRegex(ValueError, "same point count"):
            D455Camera.voxelize(
                np.ones((2, 3)),
                np.ones((1, 3)),
                voxel_size=0.01,
            )

    def test_voxelize_drops_nonfinite_points_without_open3d(self):
        points = np.array([[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        downsampled_points, downsampled_colors = D455Camera.voxelize(
            points,
            colors,
            voxel_size=0.01,
        )

        np.testing.assert_allclose(downsampled_points, [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(downsampled_colors, [[1.0, 0.0, 0.0]])


class TestD455CameraLifecycle(unittest.TestCase):
    @staticmethod
    def _fake_realsense(devices):
        camera_info = SimpleNamespace(name="name", serial_number="serial")

        class FakeDevice:
            def __init__(self, name, serial):
                self.info = {"name": name, "serial": serial}

            def get_info(self, key):
                return self.info[key]

        fake_devices = [FakeDevice(name, serial) for name, serial in devices]
        return SimpleNamespace(
            pipeline=lambda: object(),
            config=lambda: object(),
            context=lambda: SimpleNamespace(query_devices=lambda: fake_devices),
            camera_info=camera_info,
        )

    def test_reports_when_no_realsense_is_connected(self):
        fake_rs = self._fake_realsense([])
        with patch.dict("sys.modules", {"pyrealsense2": fake_rs}):
            with self.assertRaisesRegex(RuntimeError, "No Intel RealSense"):
                D455Camera(warmup_frames=0)

    def test_reports_requested_serial_mismatch(self):
        fake_rs = self._fake_realsense(
            [("Intel RealSense D455", "available-serial")]
        )
        with patch.dict("sys.modules", {"pyrealsense2": fake_rs}):
            with self.assertRaisesRegex(RuntimeError, "requested-serial"):
                D455Camera(serial="requested-serial", warmup_frames=0)

    def test_rejects_a_non_d455_device(self):
        fake_rs = self._fake_realsense(
            [("Intel RealSense L515", "l515-serial")]
        )
        with patch.dict("sys.modules", {"pyrealsense2": fake_rs}):
            with self.assertRaisesRegex(RuntimeError, "No Intel RealSense D455"):
                D455Camera(warmup_frames=0)

    def test_parameters_are_validated_before_loading_sdk(self):
        invalid_arguments = [
            {"depth_size": (0, 720)},
            {"color_size": (1280, -1)},
            {"fps": 0},
            {"timeout_ms": 0},
            {"warmup_frames": -1},
            {"init_retries": 0},
            {"serial": 123},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    D455Camera(**arguments)

    def test_get_forwards_timeout(self):
        camera = D455Camera.__new__(D455Camera)
        received = {}

        def fake_get_data(hole_filling=False, timeout_ms=None):
            received["hole_filling"] = hole_filling
            received["timeout_ms"] = timeout_ms
            return "color", "depth"

        camera.get_data = fake_get_data

        frames = camera.get(timeout_ms=1234)

        self.assertEqual(frames, ("color", "depth"))
        self.assertEqual(
            received,
            {"hole_filling": False, "timeout_ms": 1234},
        )

    def test_close_is_idempotent(self):
        class FakePipeline:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1

        camera = D455Camera.__new__(D455Camera)
        camera.pipeline = FakePipeline()
        camera._pipeline_started = True

        camera.close()
        camera.close()

        self.assertEqual(camera.pipeline.stop_calls, 1)
        self.assertFalse(camera._pipeline_started)


if __name__ == "__main__":
    unittest.main()
