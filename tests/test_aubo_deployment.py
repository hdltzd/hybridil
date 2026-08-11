import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from configs import pose as global_pose
from controllers.aubo_admit_hybrid import AuboAdmitHybridController
from deploy_aubo import (
    AuboActionExecutor,
    initialize_xjc_sensor,
    make_coordinate_frames,
)
from deployment.aubo_hybridil import (
    ActionChunkPlanner,
    CoordinateFrames,
    PointCloudProcessor,
    ReplaySource,
    transform_action_to_aubo_base,
)
from devices.aubo_c5 import AuboC5Robot, AuboCommandError


def action_trajectory(count, force):
    actions = np.zeros((count, 13), dtype=np.float64)
    actions[:, 0] = np.linspace(0.0, 0.02, count)
    actions[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    actions[:, 7:10] = force
    return actions


class TestCoordinateConversion(unittest.TestCase):
    def test_replay_and_live_use_the_matching_pose_camera_transform(self):
        replay_frames = make_coordinate_frames("replay")
        live_frames = make_coordinate_frames("live")

        np.testing.assert_allclose(
            replay_frames.camera_to_common_base,
            global_pose.L515_2_BASE,
        )
        np.testing.assert_allclose(
            live_frames.camera_to_common_base,
            global_pose.D455_2_BASE,
        )
        np.testing.assert_allclose(
            live_frames.aubo_base_to_common_base,
            global_pose.BASEr_2_BASE,
        )

    def test_camera_to_aubo_base_is_composed_through_common_base(self):
        camera_to_common = np.eye(4)
        camera_to_common[:3, 3] = [1.0, 2.0, 3.0]
        aubo_to_common = np.eye(4)
        aubo_to_common[:3, 3] = [0.5, 1.0, 1.5]
        frames = CoordinateFrames(camera_to_common, aubo_to_common)

        np.testing.assert_allclose(
            frames.camera_to_aubo_base,
            np.linalg.inv(aubo_to_common) @ camera_to_common,
        )

    def test_tcp_pose_is_converted_from_aubo_base_to_camera(self):
        camera_to_common = np.eye(4)
        camera_to_common[:3, 3] = [1.0, 0.0, 0.0]
        aubo_to_common = np.eye(4)
        tcp_to_aubo = np.eye(4)
        tcp_to_aubo[:3, 3] = [2.0, 0.0, 0.0]
        frames = CoordinateFrames(camera_to_common, aubo_to_common)

        tcp_to_camera = frames.tcp_in_camera(tcp_to_aubo)

        np.testing.assert_allclose(tcp_to_camera[:3, 3], [1.0, 0.0, 0.0])

    def test_action_pose_is_transformed_but_tcp_wrench_is_not(self):
        action = action_trajectory(1, [1.0, 2.0, 3.0])[0]
        action[:3] = [0.1, 0.2, 0.3]
        action[10:] = [0.4, 0.5, 0.6]
        camera_to_aubo_base = np.eye(4)
        camera_to_aubo_base[:3, :3] = Rot.from_euler("z", 90, degrees=True).as_matrix()
        camera_to_aubo_base[:3, 3] = [1.0, 0.0, 0.0]

        position, quaternion, wrench = transform_action_to_aubo_base(
            action,
            camera_to_aubo_base,
        )

        np.testing.assert_allclose(position, [0.8, 0.1, 0.3], atol=1e-8)
        np.testing.assert_allclose(
            Rot.from_quat(quaternion).as_matrix(),
            camera_to_aubo_base[:3, :3],
            atol=1e-8,
        )
        np.testing.assert_allclose(wrench, action[7:])


class TestActionChunkPlanner(unittest.TestCase):
    def test_low_force_selects_position(self):
        planner = ActionChunkPlanner(force_threshold=6.0, force_ratio=0.2)
        chunk = planner.plan(action_trajectory(10, [0.0, 0.0, 2.0]))
        self.assertEqual(chunk.mode, "position")
        self.assertEqual(len(chunk.actions), 10)
        self.assertTrue(planner.initial_force_detection)

    def test_contact_entry_executes_position_prefix(self):
        planner = ActionChunkPlanner(force_threshold=6.0, force_ratio=0.2)
        actions = action_trajectory(10, [0.0, 0.0, 7.0])
        actions[:3, 7:10] = [0.0, 0.0, 1.0]

        chunk = planner.plan(actions)

        self.assertEqual(chunk.mode, "position")
        self.assertEqual(len(chunk.actions), 3)
        self.assertFalse(planner.initial_force_detection)

    def test_hybrid_force_is_projected_orthogonal_to_motion(self):
        planner = ActionChunkPlanner(force_threshold=6.0, force_ratio=0.2)
        actions = action_trajectory(10, [7.0, 2.0, 0.0])

        chunk = planner.plan(actions)

        self.assertEqual(chunk.mode, "hybrid")
        np.testing.assert_allclose(chunk.actions[:, 7], 0.0, atol=1e-8)
        np.testing.assert_allclose(chunk.actions[:, 8], 2.0, atol=1e-8)


class TestPointCloudProcessor(unittest.TestCase):
    def test_filters_and_pads_to_policy_point_count(self):
        processor = PointCloudProcessor(
            point_count=5,
            voxel_size=0.0,
            base_bounds=[[-1, 1], [-1, 1], [0, 2]],
            object_bounds=[[-1, 1], [-1, 1], [0, 2]],
            tool_bounds=[[-1, 1], [-1, 1], [0, 2]],
            seed=1,
        )
        xyz = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0]])
        rgb = np.ones((3, 3))

        result = processor.process(xyz, rgb, np.eye(4), np.eye(4))

        self.assertEqual(result.shape, (5, 6))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result[:, 2] > 0))


class TestReplaySource(unittest.TestCase):
    def test_observation_horizon_and_action_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.hdf5"
            with h5py.File(str(path), "w") as file:
                demo = file.create_group("data/demo_0")
                obs = demo.create_group("obs")
                obs.create_dataset("pointcloud", data=np.zeros((5, 2, 6)))
                obs.create_dataset("robot0_eef_pos", data=np.zeros((5, 3)))
                obs.create_dataset("robot0_eef_quat", data=np.tile([0, 0, 0, 1], (5, 1)))
                obs.create_dataset("robot0_eef_wrench", data=np.zeros((5, 6)))
                demo.create_dataset("action_with_wrench", data=action_trajectory(5, [0, 0, 1]))

            with ReplaySource(str(path), "demo_0", 2, 3, start_index=0) as source:
                observation, actions, index = source.next()

            self.assertEqual(index, 1)
            self.assertEqual(observation["pointcloud"].shape, (2, 2, 6))
            self.assertEqual(actions.shape, (3, 13))


class TestAuboMotionOwnership(unittest.TestCase):
    class FakeMotionControl:
        def __init__(self):
            self.stop_calls = 0
            self.stop_arguments = None
            self.stop_return_code = 0
            self.servo_enabled = False

        def stopMove(self, quick, all_tasks):
            self.stop_calls += 1
            self.stop_arguments = (quick, all_tasks)
            return self.stop_return_code

        def isServoModeEnabled(self):
            return self.servo_enabled

        def setServoMode(self, enabled):
            self.servo_enabled = enabled
            return 0

    def make_robot(self, owns_motion):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        robot._closed = False
        robot._motion_owned = owns_motion
        robot.motion_control = self.FakeMotionControl()
        return robot

    def test_read_only_adapter_does_not_send_stop(self):
        robot = self.make_robot(owns_motion=False)

        robot.stop()

        self.assertEqual(robot.motion_control.stop_calls, 0)

    def test_owned_motion_is_stopped_and_released(self):
        robot = self.make_robot(owns_motion=True)
        robot.motion_control.servo_enabled = True

        robot.stop()

        self.assertEqual(robot.motion_control.stop_calls, 1)
        self.assertEqual(robot.motion_control.stop_arguments, (True, True))
        self.assertFalse(robot.motion_control.servo_enabled)
        self.assertFalse(robot._motion_owned)

    def test_stop_error_code_is_reported_after_servo_is_disabled(self):
        robot = self.make_robot(owns_motion=True)
        robot.motion_control.servo_enabled = True
        robot.motion_control.stop_return_code = -1

        with self.assertRaises(AuboCommandError):
            robot.stop()

        self.assertFalse(robot.motion_control.servo_enabled)
        self.assertFalse(robot._motion_owned)


class TestXjcInitialization(unittest.TestCase):
    class FakeSensor:
        def __init__(self):
            self.calls = []

        def set_zero(self):
            self.calls.append("set_zero")

        def enable_streaming(self):
            self.calls.append("enable_streaming")

    def test_live_initialization_zeroes_before_starting_stream(self):
        sensor = self.FakeSensor()

        initialize_xjc_sensor(
            sensor,
            {
                "zero_on_startup": True,
                "zero_settle_seconds": 0.0,
                "streaming": True,
            },
        )

        self.assertEqual(sensor.calls, ["set_zero", "enable_streaming"])

    def test_zero_can_be_explicitly_disabled_for_read_only_diagnosis(self):
        sensor = self.FakeSensor()

        initialize_xjc_sensor(
            sensor,
            {
                "zero_on_startup": False,
                "zero_settle_seconds": 0.0,
                "streaming": True,
            },
        )

        self.assertEqual(sensor.calls, ["enable_streaming"])


class TestAuboActionExecutorSafety(unittest.TestCase):
    class FakeRobot:
        def __init__(self):
            self.servo_calls = 0

        def get_tcp_pose(self, matrix=False):
            self.assert_matrix = matrix
            return np.eye(4)

        def inverse_kinematics(self, position, quat):
            return np.zeros(6)

        def servo_joints(self, joints, **kwargs):
            self.servo_calls += 1

    @staticmethod
    def make_controller():
        return AuboAdmitHybridController(
            mass=np.ones(6),
            stiffness=np.ones(6),
            damping=np.ones(6),
            max_acceleration=np.ones(6),
            max_velocity=np.ones(6),
            max_pose_step=np.ones(6) * 0.01,
            max_force=20.0,
            max_torque=3.0,
        )

    @staticmethod
    def make_config(control_rate=125, timeout=0.02):
        return {
            "control_rate_hz": control_rate,
            "action_rate_hz": 1000,
            "control_timeout_s": timeout,
            "position_step_m": 0.0001,
            "rotation_step_rad": 0.001,
            "workspace_bounds_aubo_base": [
                [-0.8, 0.8],
                [-0.8, 0.8],
                [-0.2, 1.2],
            ],
        }

    @staticmethod
    def one_action():
        action = np.zeros((1, 13), dtype=np.float64)
        action[0, 2] = 0.1
        action[0, 6] = 1.0
        return action

    def test_position_mode_reads_force_and_rejects_overload_before_motion(self):
        robot = self.FakeRobot()
        wrench_calls = []

        def overloaded_wrench(quat):
            wrench_calls.append(quat)
            return np.array([21.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        executor = AuboActionExecutor(
            robot,
            overloaded_wrench,
            self.make_controller(),
            np.eye(4),
            self.make_config(),
        )

        with self.assertRaisesRegex(RuntimeError, "Measured force"):
            executor.execute("position", self.one_action())

        self.assertEqual(len(wrench_calls), 1)
        self.assertEqual(robot.servo_calls, 0)

    def test_watchdog_prevents_a_stale_position_command(self):
        robot = self.FakeRobot()

        def slow_wrench(quat):
            time.sleep(0.004)
            return np.zeros(6)

        executor = AuboActionExecutor(
            robot,
            slow_wrench,
            self.make_controller(),
            np.eye(4),
            self.make_config(control_rate=1000, timeout=0.002),
        )

        with self.assertRaisesRegex(RuntimeError, "control cycle timeout"):
            executor.execute("position", self.one_action())

        self.assertEqual(robot.servo_calls, 0)


if __name__ == "__main__":
    unittest.main()
