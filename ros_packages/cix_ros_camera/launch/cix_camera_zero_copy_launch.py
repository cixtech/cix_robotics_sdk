# Copyright 2026 Cix Technology Group Co., Ltd
# SPDX-License-Identifier: BSD-3-Clause
#
# Cross-process zero-copy demo launch (V4L2 dmabuf path)
# ===============================================
# Three independent processes:
#
#   1. broker     - cix_ros_dmabuf_broker      (UDS fd relay)
#   2. camera     - v4l2_dmabuf_camera          (V4L2 EXPBUF -> dmabuf fd)
#   3. subscriber - SubTestNode                 (fetches fd via broker, zero-copy)
#
# After startup:
#   - broker listens on /tmp/cix_ros_dmabuf_broker.sock
#   - camera connects to broker and registers each buffer's fd
#     (publisher fd->handle cache)
#   - subscriber connects to broker and obtains a dup'd fd via the handle
#     (same physical memory)
#   - the whole pipeline is cross-process zero-copy
#
# Verification:
#   ros2 topic hz /cam0_stream1     # ~20 fps
#   ros2 topic bw /cam0_stream1     # bandwidth far below the frame size
#                                   # (only header + encoding)
# ===============================================

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # 1. broker process
    broker = ExecuteProcess(
        cmd=['ros2', 'run', 'cix_ros_dmabuf', 'cix_ros_dmabuf_broker'],
        name='dmabuf_broker',
        output='screen',
    )

    # 2. V4L2 dmabuf capture node (publisher)
    camera = Node(
        package='cix_ros_camera',
        executable='v4l2_dmabuf_camera',
        name='v4l2_dmabuf_camera',
        output='screen',
        parameters=[{
            'device': '/dev/video9',
            'width': 1280,
            'height': 720,
            'fps': 30.0,
            'pixel_format': 'RGB3',
            'topic': '/cam0_stream1',
            'frame_id': 'camera',
        }],
    )

    # 3. Subscriber container (SubTestNode component, zero-copy fd via broker)
    subscriber_container = ComposableNodeContainer(
        name='cix_camera_subscriber',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='cix_ros_camera',
                plugin='cix_ros_camera::SubTestNode',
                name='sub_test',
                remappings=[
                    ('/image', '/cam0_stream1'),
                ],
                parameters=[{
                    'dump': False,
                }],
            ),
        ],
        output='screen',
    )

    return LaunchDescription([broker, camera, subscriber_container])
