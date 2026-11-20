"""Launch test: simulator and NMPC nodes fly the figure-8 (10 m/s peak) in real time for
15 s. Checks tracking error, solver status and latency against the 10 ms control period.
"""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from quadrotor_nmpc_msgs.msg import ControllerStats, QuadState
from rclpy.qos import QoSProfile, ReliabilityPolicy

RUN_S = 15.0


@pytest.mark.launch_test
def generate_test_description():
    sim = launch_ros.actions.Node(package="quadrotor_nmpc_ros", executable="sim_node",
                                  parameters=[{"v_peak": 10.0}])
    nmpc = launch_ros.actions.Node(package="quadrotor_nmpc_ros", executable="nmpc_node",
                                   parameters=[{"v_peak": 10.0}])
    return launch.LaunchDescription([sim, nmpc, launch_testing.actions.ReadyToTest()])


class TestClosedLoop(unittest.TestCase):
    def test_drives_in_real_time(self):
        rclpy.init()
        node = rclpy.create_node("closed_loop_probe")
        states, stats = [], []
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(QuadState, "/quad/state", states.append, qos)
        node.create_subscription(ControllerStats, "/nmpc/stats", stats.append, 10)
        t_end = time.time() + RUN_S
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        rclpy.shutdown()

        self.assertGreater(len(states), 0.7 * RUN_S * 100)
        self.assertGreater(len(stats), 0.7 * RUN_S * 100)
        self.assertTrue(all(m.status == 0 for m in stats[1:]))
        self.assertLess(max(m.position_error for m in stats), 0.2)
        lat = sorted(m.latency_ms for m in stats[1:])
        self.assertLess(lat[len(lat) // 2], 5.0)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
