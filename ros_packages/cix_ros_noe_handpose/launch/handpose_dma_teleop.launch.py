# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Launch the zero-copy NPU handpose node (C++ ``handpose_dmabuf_node``) only.

This launch file is camera-agnostic: it starts just the
``handpose_dmabuf_node`` node and points it at an image topic and a dmabuf
broker socket. The camera node and the broker are expected to be provided
separately (e.g. by a demo launch file) -- this package deliberately does
not depend on any specific camera package.

Zero-copy works when:
  - the camera publishes ``sensor_msgs/Image`` with an encoding carrying a
    ``;dmabuf_handle=N`` suffix (fd registered with the broker), AND
  - ``broker_socket_path`` matches the path the broker listens on.

Otherwise the node transparently falls back to memcpy via cv_bridge.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument("image_topic", default_value="/cam0_stream1"))
    ld.add_action(DeclareLaunchArgument(
        "broker_socket_path", default_value="",
        description="Broker UDS path; empty = built-in default from cix_ros_camera.",
    ))
    ld.add_action(DeclareLaunchArgument(
        "npu_model_path", default_value="",
        description="NPU model .cix path; empty = share/cix_ros_noe_handpose/models/handpose_resnet50_size256.cix",
    ))
    ld.add_action(DeclareLaunchArgument("output_topic", default_value="/handpose/image_out"))
    ld.add_action(DeclareLaunchArgument("debug_topic", default_value="/handpose/debug_image"))
    ld.add_action(DeclareLaunchArgument("direction_topic", default_value="/handpose/finger_direction"))
    ld.add_action(DeclareLaunchArgument("publish_output_image", default_value="True"))
    ld.add_action(DeclareLaunchArgument("publish_debug_image", default_value="True"))
    ld.add_action(DeclareLaunchArgument("debug_image_scale", default_value="0.5"))

    handpose = Node(
        package="cix_ros_noe_handpose",
        executable="handpose_dmabuf_node",
        name="handpose_dmabuf_node",
        output="screen",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "broker_socket_path": LaunchConfiguration("broker_socket_path"),
            "npu_model_path": LaunchConfiguration("npu_model_path"),
            "output_topic": LaunchConfiguration("output_topic"),
            "debug_topic": LaunchConfiguration("debug_topic"),
            "direction_topic": LaunchConfiguration("direction_topic"),
            "publish_output_image": LaunchConfiguration("publish_output_image"),
            "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            "debug_image_scale": LaunchConfiguration("debug_image_scale"),
        }],
    )
    ld.add_action(handpose)

    return ld
