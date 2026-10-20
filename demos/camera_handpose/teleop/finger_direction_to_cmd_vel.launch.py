#!/usr/bin/env python3

# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bridge /handpose/finger_direction (Vector3) -> key_vel (Twist) for twist_mux.

The bridge script lives in this demo (not in the cix_ros_noe_handpose
package): the inference packages stay generic, demo-level robot control
glue stays with the demo.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "finger_direction_to_cmd_vel.py"
)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "input_topic", default_value="/handpose/finger_direction"
            ),
            DeclareLaunchArgument("output_topic", default_value="key_vel"),
            DeclareLaunchArgument("linear_speed", default_value="0.25"),
            DeclareLaunchArgument("angular_speed", default_value="0.8"),
            DeclareLaunchArgument("cmd_vel_timeout_sec", default_value="0.35"),
            ExecuteProcess(
                cmd=[
                    "python3", SCRIPT, "--ros-args",
                    "-p", "use_sim_time:=", LaunchConfiguration("use_sim_time"),
                    "-p", "input_topic:=", LaunchConfiguration("input_topic"),
                    "-p", "output_topic:=", LaunchConfiguration("output_topic"),
                    "-p", "linear_speed:=", LaunchConfiguration("linear_speed"),
                    "-p", "angular_speed:=", LaunchConfiguration("angular_speed"),
                    "-p", "cmd_vel_timeout_sec:=",
                    LaunchConfiguration("cmd_vel_timeout_sec"),
                ],
                name="finger_direction_to_cmd_vel",
                output="screen",
            ),
        ]
    )
