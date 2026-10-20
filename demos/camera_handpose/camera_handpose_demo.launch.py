# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""camera_handpose demo launch: broker + V4L2 camera + handpose (zero-copy).

Brings up the three processes that make the zero-copy handpose pipeline:
  1. cix_ros_dmabuf_broker  -- UDS fd relay (cix_ros_dmabuf package)
  2. v4l2_dmabuf_camera     -- V4L2 EXPBUF -> dmabuf fd publisher (cix_ros_camera)
  3. handpose_dmabuf_node   -- C++ NPU inference, zero-copy subscriber (cix_ros_noe_handpose)

Optionally also starts the demo's teleop bridge (finger_direction_to_cmd_vel,
in this demo's teleop/ directory), which maps the published finger direction
to geometry_msgs/Twist on key_vel (e.g. for twist_mux) -- off by default
because it drives a robot base.

The camera publisher side of cix_ros_camera connects to the broker at its
built-in default socket path /tmp/cix_ros_dmabuf_broker.sock, so the broker is
started with that same default (no arg) and handpose is pointed at it via
broker_socket_path. rqt_image_view is NOT started here -- launch it separately
on a graphical session (see run_demo.sh / README).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

TELEOP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "teleop", "finger_direction_to_cmd_vel.py"
)


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument("device", default_value="/dev/video9"))
    ld.add_action(DeclareLaunchArgument("width", default_value="1280"))
    ld.add_action(DeclareLaunchArgument("height", default_value="720"))
    ld.add_action(DeclareLaunchArgument("fps", default_value="30.0"))
    ld.add_action(DeclareLaunchArgument("pixel_format", default_value="RGB3"))
    ld.add_action(DeclareLaunchArgument("image_topic", default_value="/cam0_stream1"))
    ld.add_action(DeclareLaunchArgument("broker_socket_path",
                                        default_value="/tmp/cix_ros_dmabuf_broker.sock"))
    ld.add_action(DeclareLaunchArgument("use_broker", default_value="True"))
    ld.add_action(DeclareLaunchArgument("debug_image_scale", default_value="0.5"))
    ld.add_action(DeclareLaunchArgument("teleop", default_value="False",
                                        description="Also start the demo's "
                                        "finger_direction -> Twist bridge (key_vel)."))

    device = LaunchConfiguration("device")
    width = LaunchConfiguration("width")
    height = LaunchConfiguration("height")
    fps = LaunchConfiguration("fps")
    pixel_format = LaunchConfiguration("pixel_format")
    image_topic = LaunchConfiguration("image_topic")
    broker_socket_path = LaunchConfiguration("broker_socket_path")
    use_broker = LaunchConfiguration("use_broker")
    debug_image_scale = LaunchConfiguration("debug_image_scale")
    teleop = LaunchConfiguration("teleop")

    # 1. dmabuf broker (listens on the default socket path unless overridden).
    broker = Node(
        package="cix_ros_dmabuf",
        executable="cix_ros_dmabuf_broker",
        name="dmabuf_broker",
        output="screen",
        arguments=[broker_socket_path],
        condition=IfCondition(PythonExpression([use_broker])),
    )
    ld.add_action(broker)

    # 2. V4L2 dmabuf camera publisher.
    camera = Node(
        package="cix_ros_camera",
        executable="v4l2_dmabuf_camera",
        name="v4l2_dmabuf_camera",
        output="screen",
        parameters=[{
            "device": device,
            "pixel_format": pixel_format,
            "width": ParameterValue(width, value_type=int),
            "height": ParameterValue(height, value_type=int),
            "fps": ParameterValue(fps, value_type=float),
            "topic": image_topic,
        }],
    )
    ld.add_action(camera)

    # 3. Handpose C++ NPU inference node (zero-copy subscriber via the broker).
    handpose = Node(
        package="cix_ros_noe_handpose",
        executable="handpose_dmabuf_node",
        name="handpose_dmabuf_node",
        output="screen",
        parameters=[{
            "image_topic": image_topic,
            "broker_socket_path": broker_socket_path,
            "publish_debug_image": True,
            "publish_output_image": True,
            "debug_image_scale": ParameterValue(debug_image_scale, value_type=float),
        }],
    )
    ld.add_action(handpose)

    # 4. Optional: teleop bridge (Vector3 finger direction -> Twist on key_vel).
    # A plain demo script, so it is executed directly rather than via ros2 run.
    teleop_bridge = ExecuteProcess(
        cmd=["python3", TELEOP_SCRIPT, "--ros-args",
             "-p", "use_sim_time:=false"],
        name="finger_direction_to_cmd_vel",
        output="screen",
        condition=IfCondition(PythonExpression([teleop])),
    )
    ld.add_action(teleop_bridge)

    return ld
