#!/usr/bin/env python3
"""Record the running ROS 2 closed loop for a while and print a summary (JSON).

    ros2 launch quadrotor_nmpc_ros figure8.launch.py &
    python3 flight_probe.py --seconds 9 [--out flight.json]
"""
import argparse
import json
import time

import numpy as np
import rclpy
from quadrotor_nmpc_msgs.msg import ControllerStats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node("flight_probe")
    S = []
    node.create_subscription(
        ControllerStats, "/nmpc/stats",
        lambda m: S.append((m.solve_time_ms, m.step_time_ms, m.latency_ms, m.status,
                            m.position_error)), 100)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    rclpy.shutdown()
    S = np.array(S[1:])
    e = S[:, 4]

    def q(v, p):
        return round(float(np.percentile(v, p)), 3)

    out = dict(
        seconds=a.seconds, steps=int(len(S)), failed_solves=int((S[:, 3] != 0).sum()),
        rms_error_m=round(float(np.sqrt(np.mean(e ** 2))), 4), max_error_m=round(float(e.max()), 4),
        solve_ms_median=q(S[:, 0], 50), solve_ms_p99=q(S[:, 0], 99),
        step_ms_median=q(S[:, 1], 50), latency_ms_median=q(S[:, 2], 50),
        latency_ms_p99=q(S[:, 2], 99), latency_ms_max=round(float(S[:, 2].max()), 3))
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
