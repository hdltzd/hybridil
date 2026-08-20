import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from configs import pose as global_pose
from controllers.aubo_admit_hybrid import AuboAdmitHybridController
from deploy_aubo import (
    AuboActionExecutor,
    LiveObservationSource,
    initialize_xjc_sensor,
    make_coordinate_frames,
    make_controller,
    make_policy_ready_pose,
    move_to_point_times,
    solve_continuous_joint_targets,
    load_json,
    validate_deployment_config,
)
from deployment.aubo_hybridil import (
    ActionChunkPlanner,
    CoordinateFrames,
    PointCloudProcessor,
    ReplaySource,
    transform_action_to_aubo_base,
)
from deployment.aubo_runtime import (
    LatestWrenchCache,
    RuntimeStatistics,
    ServoBackpressure,
    ServoCommandPacer,
    SingleServoControlLoop,
    interpolate_pose,
    interpolate_wrench,
)
from devices.aubo_c5 import (
    AuboC5Robot,
    AuboCommandError,
    AuboQueueFullError,
)
from devices.xjc_ft import XjcProtocolError


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


class TestContinuousIK(unittest.TestCase):
    class Robot:
        def __init__(self):
            self.seeds = []

        def get_joint_pos(self):
            return np.zeros(6)

        def inverse_kinematics(self, position, quaternion, rest_joints=None):
            seed = np.asarray(rest_joints, dtype=np.float64)
            self.seeds.append(seed.copy())
            return seed + 0.1

    def test_each_ik_uses_previous_solution_as_seed(self):
        robot = self.Robot()
        actions = action_trajectory(3, [0.0, 0.0, 0.0])

        targets, codes = solve_continuous_joint_targets(
            robot,
            actions,
            np.eye(4),
            [[-1, 1], [-1, 1], [-1, 1]],
        )

        np.testing.assert_allclose(robot.seeds[0], np.zeros(6))
        np.testing.assert_allclose(robot.seeds[1], np.full(6, 0.1))
        np.testing.assert_allclose(robot.seeds[2], np.full(6, 0.2))
        np.testing.assert_allclose(targets[-1], np.full(6, 0.3))
        np.testing.assert_array_equal(codes, np.zeros(3, dtype=np.int64))


class TestLiveWrenchReading(unittest.TestCase):
    class IdentityCompensator:
        def compensate(self, raw, quaternion):
            return np.asarray(raw, dtype=np.float64)

    class IntermittentSensor:
        def __init__(self, failures):
            self.failures = failures
            self.calls = 0

        def read(self):
            self.calls += 1
            if self.calls <= self.failures:
                raise XjcProtocolError("missing streaming header")
            return np.arange(6, dtype=np.float64)

    def make_source(self, failures, retries=3):
        source = LiveObservationSource.__new__(LiveObservationSource)
        source.sensor = self.IntermittentSensor(failures)
        source.compensator = self.IdentityCompensator()
        source.sensor_lock = threading.Lock()
        source.sensor_read_retries = retries
        return source

    def test_transient_streaming_errors_are_retried(self):
        source = self.make_source(failures=2, retries=3)

        wrench = source.read_compensated_wrench([0, 0, 0, 1])

        np.testing.assert_array_equal(wrench, np.arange(6, dtype=np.float64))
        self.assertEqual(source.sensor.calls, 3)

    def test_consecutive_streaming_errors_still_stop_deployment(self):
        source = self.make_source(failures=3, retries=3)

        with self.assertRaisesRegex(XjcProtocolError, "3 consecutive times"):
            source.read_compensated_wrench([0, 0, 0, 1])


class TestPolicyReadyPose(unittest.TestCase):
    def test_positive_tool_z_retreats_along_negative_tool_z(self):
        action = np.zeros(13, dtype=np.float64)
        action[:3] = [0.1, 0.2, 0.3]
        action[3:7] = [0.0, 0.0, 0.0, 1.0]

        ready, quat, target = make_policy_ready_pose(
            action,
            np.eye(4),
            retreat_distance_m=0.08,
            tool_forward_axis="+z",
        )

        np.testing.assert_allclose(target, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(ready, [0.1, 0.2, 0.22])
        np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])

    def test_negative_tool_z_retreats_along_positive_tool_z(self):
        action = np.zeros(13, dtype=np.float64)
        action[:3] = [0.1, 0.2, 0.3]
        action[3:7] = [0.0, 0.0, 0.0, 1.0]

        ready, _, _ = make_policy_ready_pose(
            action,
            np.eye(4),
            retreat_distance_m=0.08,
            tool_forward_axis="-z",
        )

        np.testing.assert_allclose(ready, [0.1, 0.2, 0.38])


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
    def test_reference_send_command_delegates_to_servoj(self):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        calls = []
        robot.servo_joints = lambda joints, **kwargs: calls.append(
            (np.asarray(joints).copy(), kwargs)
        )

        robot.send_command(np.arange(6), 0, duration_s=0.01)

        np.testing.assert_array_equal(calls[0][0], np.arange(6))
        self.assertEqual(calls[0][1]["duration_s"], 0.01)

    def test_reference_send_command_rejects_failed_ik(self):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        robot.servo_joints = lambda *args, **kwargs: self.fail("must not send")

        with self.assertRaisesRegex(AuboCommandError, "code -23"):
            robot.send_command(np.zeros(6), -23)

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

    def test_cartesian_servo_pipeline_stops_after_ik_failure(self):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        calls = []

        def failed_ik(position, quaternion):
            calls.append("ik")
            raise AuboCommandError("IK failed")

        def stopped():
            calls.append("stop")

        robot.inverse_kinematics = failed_ik
        robot.servo_joints = lambda *args, **kwargs: calls.append("servoj")
        robot.stop = stopped

        with self.assertRaisesRegex(AuboCommandError, "IK failed"):
            robot.servo_tcp_pose([0, 0, 0], [0, 0, 0, 1])

        self.assertEqual(calls, ["ik", "stop"])

    def test_cartesian_servo_pipeline_calls_ik_then_servoj(self):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        calls = []
        joints = np.arange(6, dtype=np.float64)
        robot.inverse_kinematics = lambda position, quaternion: calls.append("ik") or joints
        robot.servo_joints = lambda value, **kwargs: calls.append(("servoj", value))
        robot.stop = lambda: calls.append("stop")

        result = robot.servo_tcp_pose([0, 0, 0], [0, 0, 0, 1])

        np.testing.assert_array_equal(result, joints)
        self.assertEqual(calls[0], "ik")
        self.assertEqual(calls[1][0], "servoj")
        self.assertNotIn("stop", calls)

    def test_cartesian_queue_full_does_not_stop_servo(self):
        robot = AuboC5Robot.__new__(AuboC5Robot)
        calls = []
        robot.inverse_kinematics = lambda position, quaternion: np.zeros(6)
        robot.servo_joints = lambda *args, **kwargs: (_ for _ in ()).throw(
            AuboQueueFullError("queue full")
        )
        robot.stop = lambda: calls.append("stop")

        with self.assertRaises(AuboQueueFullError):
            robot.servo_tcp_pose([0, 0, 0], [0, 0, 0, 1])

        self.assertEqual(calls, [])

    def test_servoj_code_two_maps_to_queue_full_without_stop(self):
        class Motion:
            def isServoModeEnabled(self):
                return True

            def servoJoint(self, *args):
                return 2

        robot = AuboC5Robot.__new__(AuboC5Robot)
        robot.motion_control = Motion()
        robot.require_motion_allowed = lambda: None
        robot.validate_joint_target = lambda target: np.asarray(target)
        stops = []
        robot.stop = lambda: stops.append(True)

        with self.assertRaises(AuboQueueFullError):
            robot.servo_joints(np.zeros(6))

        self.assertEqual(stops, [])

    def test_other_servoj_error_stops_immediately(self):
        class Motion:
            def isServoModeEnabled(self):
                return True

            def servoJoint(self, *args):
                return -5

        robot = AuboC5Robot.__new__(AuboC5Robot)
        robot.motion_control = Motion()
        robot.require_motion_allowed = lambda: None
        robot.validate_joint_target = lambda target: np.asarray(target)
        stops = []
        robot.stop = lambda: stops.append(True)

        with self.assertRaisesRegex(AuboCommandError, "code -5"):
            robot.servo_joints(np.zeros(6))

        self.assertEqual(stops, [True])


class TestServoRuntime(unittest.TestCase):
    class FakeRobot:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    def test_queue_full_success_resets_consecutive_count(self):
        robot = self.FakeRobot()
        stats = RuntimeStatistics()
        backpressure = ServoBackpressure(
            robot,
            ServoCommandPacer(0.000001),
            stats,
            max_consecutive=5,
        )

        for _ in range(2):
            accepted, _ = backpressure.send(
                lambda: (_ for _ in ()).throw(AuboQueueFullError("full"))
            )
            self.assertFalse(accepted)
        accepted, _ = backpressure.send(lambda: "ok")

        self.assertTrue(accepted)
        self.assertEqual(backpressure.consecutive, 0)
        self.assertEqual(stats.snapshot()["servo_queue_full"], 2)
        self.assertEqual(robot.stop_calls, 0)

    def test_five_consecutive_queue_full_stops(self):
        robot = self.FakeRobot()
        backpressure = ServoBackpressure(
            robot,
            ServoCommandPacer(0.000001),
            RuntimeStatistics(),
            max_consecutive=5,
        )
        callback = lambda: (_ for _ in ()).throw(AuboQueueFullError("full"))

        for _ in range(4):
            self.assertFalse(backpressure.send(callback)[0])
        with self.assertRaisesRegex(RuntimeError, "5 consecutive"):
            backpressure.send(callback)

        self.assertEqual(robot.stop_calls, 1)

    def test_pacer_prevents_back_to_back_attempts(self):
        class Clock:
            def __init__(self):
                self.now = 0.0
                self.sleeps = []

            def read(self):
                return self.now

            def sleep(self, seconds):
                self.sleeps.append(seconds)
                self.now += seconds

        clock = Clock()
        pacer = ServoCommandPacer(0.01, clock=clock.read, sleeper=clock.sleep)
        pacer.wait()
        pacer.wait()

        self.assertEqual(len(clock.sleeps), 1)
        self.assertAlmostEqual(clock.sleeps[0], 0.01)

    def test_pose_interpolation_uses_slerp(self):
        end_quat = Rot.from_euler("z", 90, degrees=True).as_quat()
        position, quat = interpolate_pose(
            [0, 0, 0],
            [0, 0, 0, 1],
            [2, 0, 0],
            end_quat,
            0.5,
        )
        np.testing.assert_allclose(position, [1, 0, 0])
        np.testing.assert_allclose(
            Rot.from_quat(quat).as_euler("xyz", degrees=True)[2],
            45.0,
            atol=1e-8,
        )

    def test_wrench_interpolation_is_linear(self):
        np.testing.assert_allclose(
            interpolate_wrench(np.zeros(6), np.arange(6), 0.5),
            np.arange(6) * 0.5,
        )

    def test_admittance_position_uses_xjc_with_zero_desired_wrench(self):
        class Controller:
            max_force = 20.0
            max_torque = 3.0

            def __init__(self):
                self.desired = None

            def validate_measured_wrench(self, wrench):
                return np.asarray(wrench, dtype=np.float64)

            def step(self, **kwargs):
                self.desired = kwargs["desired_wrench_tcp"].copy()
                return (
                    np.asarray(kwargs["target_position_base"]),
                    np.asarray(kwargs["target_quat_xyzw_base"]),
                )

        controller = Controller()
        config = load_json("configs/deploy_aubo.json")["execution"]
        config["control_mode"] = "force_position"
        loop = SingleServoControlLoop(
            robot=object(),
            wrench_reader=lambda quaternion: np.ones(6),
            controller=controller,
            camera_to_aubo_base=np.eye(4),
            config=config,
            robot_lock=threading.RLock(),
            statistics=RuntimeStatistics(),
        )

        position, quaternion = loop._compute_command(
            np.eye(4),
            np.array([0.0, 0.0, 0.0, 1.0]),
            "position",
            np.array([0.1, 0.2, 0.3]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.ones(6) * 9.0,
        )

        np.testing.assert_array_equal(controller.desired, np.zeros(6))
        np.testing.assert_array_equal(position, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_force_hybrid_uses_predicted_wrench(self):
        class Controller:
            max_force = 20.0
            max_torque = 3.0

            def __init__(self):
                self.desired = None

            def validate_measured_wrench(self, wrench):
                return np.asarray(wrench, dtype=np.float64)

            def step(self, **kwargs):
                self.desired = kwargs["desired_wrench_tcp"].copy()
                return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])

        controller = Controller()
        config = load_json("configs/deploy_aubo.json")["execution"]
        config["control_mode"] = "force_position"
        loop = SingleServoControlLoop(
            robot=object(),
            wrench_reader=lambda quaternion: np.zeros(6),
            controller=controller,
            camera_to_aubo_base=np.eye(4),
            config=config,
            robot_lock=threading.RLock(),
            statistics=RuntimeStatistics(),
        )
        predicted = np.arange(6, dtype=np.float64)

        loop._compute_command(
            np.eye(4),
            np.array([0.0, 0.0, 0.0, 1.0]),
            "hybrid",
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
            predicted,
        )

        np.testing.assert_array_equal(controller.desired, predicted)

    def test_move_to_point_submits_sparse_joint_waypoints(self):
        """纯位置move_to_point只保存10个IK关节路点，不预生成稠密TCP点。"""
        loop = SingleServoControlLoop(
            robot=object(),
            wrench_reader=None,
            controller=None,
            camera_to_aubo_base=np.eye(4),
            config=load_json("configs/deploy_aubo.json")["execution"],
            robot_lock=threading.RLock(),
            statistics=RuntimeStatistics(),
        )
        loop._last_joint_command = np.zeros(6)
        segments = [("position", action_trajectory(10, [0, 0, 0]))]
        t = move_to_point_times(0.5, 10)
        joint_waypoints = np.tile(np.arange(6, dtype=np.float64), (10, 1))

        self.assertEqual(loop.move_to_point(t, segments, joint_waypoints), 1)
        revision, _, stored_times, _, stored_joints = loop._joint_trajectory
        self.assertEqual(revision, 1)
        self.assertEqual(len(stored_times), len(t))
        self.assertEqual(len(stored_joints), len(t))

    def test_move_to_point_times_match_reference_cumulative_t_semantics(self):
        np.testing.assert_allclose(
            move_to_point_times(0.5, 10),
            np.linspace(0.05, 0.5, 10),
        )

class TestLatestWrenchCache(unittest.TestCase):
    class Sensor:
        def read(self):
            return np.arange(6, dtype=np.float64)

    class Compensator:
        def compensate(self, wrench, quaternion):
            return np.asarray(wrench) + 1.0

    def test_cache_returns_latest_compensated_sample(self):
        cache = LatestWrenchCache(
            self.Sensor(),
            self.Compensator(),
            RuntimeStatistics(),
            sample_rate_hz=250,
            max_age_s=0.02,
        )
        cache.start()
        try:
            np.testing.assert_array_equal(
                cache.read([0, 0, 0, 1]),
                np.arange(6, dtype=np.float64) + 1.0,
            )
        finally:
            cache.close()

    def test_stale_cache_is_rejected(self):
        cache = LatestWrenchCache(
            self.Sensor(),
            self.Compensator(),
            RuntimeStatistics(),
            sample_rate_hz=250,
            max_age_s=0.001,
        )
        cache._sample = np.zeros(6)
        cache._timestamp = time.monotonic() - 1.0

        with self.assertRaisesRegex(RuntimeError, "stale"):
            cache.read([0, 0, 0, 1])

    def test_startup_recovers_after_more_than_three_protocol_timeouts(self):
        class StartupSensor:
            def __init__(self):
                self.calls = 0

            def read(self):
                self.calls += 1
                if self.calls <= 5:
                    raise XjcProtocolError("waiting for streaming header")
                return np.arange(6, dtype=np.float64)

        sensor = StartupSensor()
        statistics = RuntimeStatistics()
        cache = LatestWrenchCache(
            sensor,
            self.Compensator(),
            statistics,
            sample_rate_hz=250,
            max_age_s=0.02,
            read_retries=3,
        )
        cache.start(startup_timeout_s=0.2)
        try:
            self.assertGreater(sensor.calls, 3)
            self.assertEqual(statistics.snapshot()["xjc_read_errors"], 5)
            np.testing.assert_array_equal(
                cache.read([0, 0, 0, 1]),
                np.arange(6, dtype=np.float64) + 1.0,
            )
        finally:
            cache.close()

    def test_missing_stream_fails_on_overall_startup_timeout(self):
        class MissingStreamSensor:
            def read(self):
                raise XjcProtocolError("waiting for streaming header")

        cache = LatestWrenchCache(
            MissingStreamSensor(),
            self.Compensator(),
            RuntimeStatistics(),
            sample_rate_hz=250,
            max_age_s=0.02,
        )
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "Timed out waiting for the first XJC sample",
            ):
                cache.start(startup_timeout_s=0.02)
        finally:
            cache.close()


class TestDeploymentConfiguration(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "hold_seconds": 0.0,
            "execute": False,
            "prepare": False,
            "recorded_actions": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_approved_one_hundred_hz_baseline_is_valid(self):
        config = load_json("configs/deploy_aubo.json")
        self.assertEqual(validate_deployment_config(config, self.make_args()), 0.0)
        self.assertEqual(config["execution"]["control_rate_hz"], 100)
        self.assertEqual(config["execution"]["control_mode"], "position")
        self.assertEqual(config["execution"]["position_command"], "send_command")
        self.assertEqual(config["controller_config"], "configs/aubo_admit_hybrid.json")
        self.assertNotIn("servo_duration_s", config["execution"])
        self.assertNotIn("position_step_m", config["execution"])
        self.assertNotIn("rotation_step_rad", config["execution"])
        self.assertNotIn("prefetch_after_actions", config["execution"])
        self.assertNotIn("command_rate_hz", config["execution"])
        self.assertNotIn("max_tcp_speed_m_s", config["execution"])
        self.assertNotIn("max_tcp_angular_speed_rad_s", config["execution"])
        loop = SingleServoControlLoop(
            robot=object(),
            wrench_reader=None,
            controller=None,
            camera_to_aubo_base=np.eye(4),
            config=config["execution"],
            robot_lock=threading.RLock(),
            statistics=RuntimeStatistics(),
        )
        self.assertAlmostEqual(loop.period, 0.01)
        self.assertAlmostEqual(loop.duration, loop.period)

    def test_hold_requires_execute_and_rejects_prepare(self):
        config = load_json("configs/deploy_aubo.json")
        with self.assertRaisesRegex(RuntimeError, "requires --execute"):
            validate_deployment_config(
                config,
                self.make_args(hold_seconds=30.0),
            )
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            validate_deployment_config(
                config,
                self.make_args(hold_seconds=30.0, execute=True, prepare=True),
            )

    def test_non_contact_profile_is_faster_than_contact_profile(self):
        controller = make_controller("configs/aubo_admit_hybrid.json")
        kwargs = {
            "current_position_base": np.zeros(3),
            "current_quat_xyzw_base": [0.0, 0.0, 0.0, 1.0],
            "target_position_base": [0.1, 0.0, 0.0],
            "target_quat_xyzw_base": [0.0, 0.0, 0.0, 1.0],
            "measured_wrench_tcp": np.zeros(6),
            "desired_wrench_tcp": np.zeros(6),
            "dt": 0.01,
        }
        position_command, _ = controller.step(
            **kwargs,
            limit_profile="position",
        )
        controller.reset()
        hybrid_command, _ = controller.step(
            **kwargs,
            limit_profile="hybrid",
        )
        self.assertGreater(position_command[0], hybrid_command[0])


class TestXjcInitialization(unittest.TestCase):
    class FakeSensor:
        def __init__(self):
            self.calls = []
            self.timeout_s = 0.006

        def disable_streaming(self):
            self.calls.append("disable_streaming")

        def set_zero(self):
            self.calls.append("set_zero")

        def read_polling(self):
            self.calls.append("read_polling")
            return np.zeros(6)

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

        self.assertEqual(
            sensor.calls,
            [
                "disable_streaming",
                "disable_streaming",
                "disable_streaming",
                "read_polling",
                "read_polling",
                "read_polling",
                "enable_streaming",
            ],
        )

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

    def test_zero_retries_and_restores_realtime_timeout(self):
        class RetrySensor(self.FakeSensor):
            def __init__(self):
                super().__init__()
                self.zeroed = False

            def set_zero(self):
                self.calls.append("set_zero")
                self.zeroed = True

            def read_polling(self):
                self.calls.append("read_polling")
                return np.zeros(6) if self.zeroed else np.ones(6)

        sensor = RetrySensor()
        initialize_xjc_sensor(
            sensor,
            {
                "zero_on_startup": True,
                "zero_prepare_seconds": 0.0,
                "zero_response_timeout_s": 0.1,
                "zero_retries": 2,
                "zero_retry_delay_s": 0.0,
                "zero_initial_disable_count": 1,
                "zero_retry_disable_count": 1,
                "zero_polling_count": 1,
                "zero_command_count": 2,
                "zero_settle_seconds": 0.0,
                "streaming": False,
            },
        )

        self.assertEqual(
            sensor.calls,
            [
                "disable_streaming",
                "read_polling",
                "disable_streaming",
                "set_zero",
                "set_zero",
                "read_polling",
            ],
        )
        self.assertEqual(sensor.timeout_s, 0.006)


class TestAuboActionExecutorSafety(unittest.TestCase):
    class FakeRobot:
        def __init__(self):
            self.servo_calls = 0

        def get_tcp_pose(self, matrix=False):
            self.assert_matrix = matrix
            return np.eye(4)

        def servo_tcp_pose(self, position, quat, **kwargs):
            self.servo_calls += 1
            return np.zeros(6)

    @staticmethod
    def make_controller():
        return AuboAdmitHybridController(
            mass=np.ones(6),
            stiffness=np.ones(6),
            damping=np.ones(6),
            max_acceleration=np.ones(6),
            max_velocity=np.ones(6),
            max_force=20.0,
            max_torque=3.0,
        )

    @staticmethod
    def make_config(control_rate=125, timeout=0.02):
        return {
            "control_rate_hz": control_rate,
            "trajectory_duration_s": 0.001,
            "control_timeout_s": timeout,
            "max_tcp_speed_m_s": 0.1,
            "max_tcp_angular_speed_rad_s": 0.2,
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
