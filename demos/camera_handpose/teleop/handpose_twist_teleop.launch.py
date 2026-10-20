#!/usr/bin/env python3

# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""USB-camera NPU handpose -> /handpose/finger_direction (Vector3).

The teleop script lives in this demo (not in the cix_ros_noe_handpose
package): the inference packages stay generic, demo-level robot control
glue stays with the demo. Base control: finger_direction_to_cmd_vel.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "handpose_twist_teleop.py"
)


def generate_launch_description():
    pkg_share = get_package_share_directory("cix_ros_noe_handpose")
    default_cix = os.path.join(pkg_share, "models", "handpose_resnet50_size256.cix")
    fastdds_xml = os.path.join(pkg_share, "config", "fastdds_shm_only.xml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock (/clock)",
    )
    npu_model_path_arg = DeclareLaunchArgument(
        "npu_model_path",
        default_value=default_cix,
        description="Path to handpose_resnet50_size256.cix",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/color/image_raw",
        description="Input subscription when use_usb_camera:=false (not published here).",
    )
    use_usb_camera_arg = DeclareLaunchArgument(
        "use_usb_camera",
        default_value="true",
        description="Read frames directly from a USB camera with OpenCV instead of ROS image topic.",
    )
    video_device_arg = DeclareLaunchArgument(
        "video_device",
        default_value="auto",
        description=(
            "V4L device e.g. /dev/video0, or 'auto' to scan /dev/video* and prefer "
            "udev ID_BUS=usb with a successful frame read."
        ),
    )
    camera_width_arg = DeclareLaunchArgument("camera_width", default_value="640")
    camera_height_arg = DeclareLaunchArgument("camera_height", default_value="480")
    camera_fps_arg = DeclareLaunchArgument("camera_fps", default_value="30.0")
    camera_frame_id_arg = DeclareLaunchArgument("camera_frame_id", default_value="usb_camera")
    publish_output_image_arg = DeclareLaunchArgument(
        "publish_output_image",
        default_value="true",
        description="Publish each inference frame as bgr8 on output_image_topic.",
    )
    output_image_topic_arg = DeclareLaunchArgument(
        "output_image_topic",
        default_value="/handpose/image_out",
        description="Raw camera frame used for NPU (USB or subscribed).",
    )
    publish_debug_image_arg = DeclareLaunchArgument("publish_debug_image", default_value="true")
    debug_image_topic_arg = DeclareLaunchArgument(
        "debug_image_topic",
        default_value="/handpose/debug_image",
    )
    index_vs_other_ratio_arg = DeclareLaunchArgument(
        "index_vs_other_ratio",
        default_value="1.15",
        description="When require_index_dominant:=true, index must beat other fingers by this ratio.",
    )
    min_finger_len_arg = DeclareLaunchArgument(
        "min_finger_len_px",
        default_value="6.0",
        description="Min index length (px) for a direction (coarse).",
    )
    direction_axis_ratio_arg = DeclareLaunchArgument(
        "direction_axis_ratio",
        default_value="1.4",
        description="4-way split: horizontal if |dx|>=ratio*|dy|. Lower=wider L/R zone.",
    )
    publish_finger_direction_arg = DeclareLaunchArgument(
        "publish_finger_direction",
        default_value="true",
        description="Publish geometry_msgs/Vector3 compass (-1/0/1 on x,y; NONE=zeros).",
    )
    finger_direction_topic_arg = DeclareLaunchArgument(
        "finger_direction_topic",
        default_value="/handpose/finger_direction",
    )
    require_index_arg = DeclareLaunchArgument(
        "require_index_dominant",
        default_value="false",
    )
    match_sim_dds_arg = DeclareLaunchArgument(
        "match_simulated_robot_dds",
        default_value="false",
        description=(
            "If true: match simulated_robot.launch.py DDS (e.g. ROS_DOMAIN_ID=61). "
            "Use when running in a second terminal alongside that sim."
        ),
    )

    dds_env_group = GroupAction(
        condition=IfCondition(
            PythonExpression(
                ["'", LaunchConfiguration("match_simulated_robot_dds"), "' == 'true'"]
            )
        ),
        actions=[
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", "61"),
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
            SetEnvironmentVariable("IGN_IP", "127.0.0.1"),
            SetEnvironmentVariable("ROS_AUTOMATIC_DISCOVERY_RANGE", "OFF"),
            SetEnvironmentVariable("ROS_STATIC_PEERS", "127.0.0.1"),
            SetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE", fastdds_xml),
            SetEnvironmentVariable("FASTRTPS_LOG_VERBOSITY", "ERROR"),
        ],
    )

    node = ExecuteProcess(
        cmd=[
            "python3", SCRIPT, "--ros-args",
            "-p", "use_sim_time:=", LaunchConfiguration("use_sim_time"),
            "-p", "npu_model_path:=", LaunchConfiguration("npu_model_path"),
            "-p", "image_topic:=", LaunchConfiguration("image_topic"),
            "-p", "use_usb_camera:=", LaunchConfiguration("use_usb_camera"),
            "-p", "video_device:=", LaunchConfiguration("video_device"),
            "-p", "camera_width:=", LaunchConfiguration("camera_width"),
            "-p", "camera_height:=", LaunchConfiguration("camera_height"),
            "-p", "camera_fps:=", LaunchConfiguration("camera_fps"),
            "-p", "camera_frame_id:=", LaunchConfiguration("camera_frame_id"),
            "-p", "publish_output_image:=", LaunchConfiguration("publish_output_image"),
            "-p", "output_image_topic:=", LaunchConfiguration("output_image_topic"),
            "-p", "publish_debug_image:=", LaunchConfiguration("publish_debug_image"),
            "-p", "debug_image_topic:=", LaunchConfiguration("debug_image_topic"),
            "-p", "min_finger_len_px:=", LaunchConfiguration("min_finger_len_px"),
            "-p", "direction_axis_ratio:=", LaunchConfiguration("direction_axis_ratio"),
            "-p", "publish_finger_direction:=",
            LaunchConfiguration("publish_finger_direction"),
            "-p", "finger_direction_topic:=",
            LaunchConfiguration("finger_direction_topic"),
            "-p", "require_index_dominant:=", LaunchConfiguration("require_index_dominant"),
            "-p", "index_vs_other_ratio:=", LaunchConfiguration("index_vs_other_ratio"),
        ],
        name="handpose_twist_teleop",
        output="screen",
    )

    return LaunchDescription(
        [
            match_sim_dds_arg,
            dds_env_group,
            use_sim_time_arg,
            npu_model_path_arg,
            image_topic_arg,
            use_usb_camera_arg,
            video_device_arg,
            camera_width_arg,
            camera_height_arg,
            camera_fps_arg,
            camera_frame_id_arg,
            publish_output_image_arg,
            output_image_topic_arg,
            publish_debug_image_arg,
            debug_image_topic_arg,
            min_finger_len_arg,
            direction_axis_ratio_arg,
            publish_finger_direction_arg,
            finger_direction_topic_arg,
            require_index_arg,
            index_vs_other_ratio_arg,
            node,
        ]
    )
