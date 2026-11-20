"""Quadrotor simulator + NMPC flying the figure-8.

    ros2 launch quadrotor_nmpc_ros figure8.launch.py v_peak:=10.0
    ros2 launch quadrotor_nmpc_ros figure8.launch.py rotor3_max:=3.5 limits:="[6.56,5.53,3.48,6.27]"

``rotor3_max`` caps rotor 3 in the simulator (damaged propeller); ``limits`` are the
per-rotor thrusts the NMPC may use (e.g. the ones ddma certified).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    v = float(LaunchConfiguration("v_peak").perform(context))
    cap = float(LaunchConfiguration("rotor3_max").perform(context))
    lim = [float(s) for s in LaunchConfiguration("limits").perform(context).strip("[]").split(",")]
    return [
        Node(package="quadrotor_nmpc_ros", executable="sim_node", name="quad_sim",
             parameters=[{"v_peak": v, "state_rate_hz": 100.0,
                          "rotor_max_thrust": [1e9, 1e9, cap, 1e9]}], output="screen"),
        Node(package="quadrotor_nmpc_ros", executable="nmpc_node", name="nmpc",
             parameters=[{"v_peak": v, "thrust_limits": lim}], output="screen"),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("v_peak", default_value="10.0"),
        DeclareLaunchArgument("rotor3_max", default_value="1e9"),
        DeclareLaunchArgument("limits", default_value="[7.0,7.0,7.0,7.0]"),
        OpaqueFunction(function=_nodes),
    ])
