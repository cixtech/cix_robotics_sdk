#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""NPU bolt OBB (yolo_obb.cix): yolov8_msgs / vision_msgs output + optional debug Image."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock (/clock)',
    )
    npu_model_path_arg = DeclareLaunchArgument(
        'npu_model_path',
        default_value='',
        description='Absolute path to yolo_obb.cix; empty uses share/cix_ros_noe_yolo/model/yolo_obb.cix',
    )
    npu_model_name_arg = DeclareLaunchArgument(
        'npu_model_name',
        default_value='yolo_obb',
        description='Basename under share/.../model/ when npu_model_path is empty',
    )
    onnx_yolov8_root_arg = DeclareLaunchArgument(
        'onnx_yolov8_root',
        default_value='',
        description='Override root containing yolo_postprocess/; empty uses package share',
    )
    class_names_arg = DeclareLaunchArgument(
        'class_names',
        default_value='bolt',
        description='OBB class names, comma-separated',
    )
    num_classes_arg = DeclareLaunchArgument(
        'num_classes',
        default_value='1',
        description='OBB head class count (yolo_obb.cix: 1)',
    )
    conf_thr_arg = DeclareLaunchArgument('conf_thr', default_value='0.90')
    nms_thr_arg = DeclareLaunchArgument('nms_thr', default_value='0.40')
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image_raw',
    )
    detections_topic_arg = DeclareLaunchArgument(
        'detections_topic',
        default_value='/vision/detections_2d',
        description='vision_msgs/Detection2DArray (when output_format includes vision_msgs)',
    )
    output_format_arg = DeclareLaunchArgument(
        'output_format',
        default_value='yolov8_obb',
        description='vision_msgs | yolov8_obb | both',
    )
    yolov8_inference_topic_arg = DeclareLaunchArgument(
        'yolov8_inference_topic',
        default_value='/Yolov8_Inference',
        description='yolov8_msgs/Yolov8Inference (when output_format includes yolov8_obb)',
    )
    class_filter_arg = DeclareLaunchArgument(
        'class_filter',
        default_value='bolt',
        description='Comma-separated class names; empty = no filter',
    )
    npu_active_topic_arg = DeclareLaunchArgument(
        'npu_active_topic',
        default_value='/npu_vision_active',
    )
    npu_active_heartbeat_hz_arg = DeclareLaunchArgument(
        'npu_active_heartbeat_hz',
        default_value='2.0',
    )
    publish_npu_active_heartbeat_arg = DeclareLaunchArgument(
        'publish_npu_active_heartbeat',
        default_value='true',
        description='If true, periodically republish true on npu_active_topic; if false, only initial true',
    )
    reenable_ultralytics_arg = DeclareLaunchArgument(
        'reenable_ultralytics_on_exit',
        default_value='true',
        description='Publish false on npu_active_topic on exit',
    )
    publish_debug_image_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='false',
        description='Publish sensor_msgs/Image with OBB boxes on debug_image_topic',
    )
    debug_image_topic_arg = DeclareLaunchArgument(
        'debug_image_topic',
        default_value='/inference_result',
        description='BGR Image topic with drawn detections',
    )
    profile_timing_arg = DeclareLaunchArgument(
        'profile_timing',
        default_value='false',
        description='Log preprocess/npu/postprocess ms (throttled)',
    )

    node = Node(
        package='cix_ros_noe_yolo',
        executable='npu_yolo_obb_inference',
        name='cix_ros_noe_yolo',
        output='screen',
        parameters=[
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'npu_model_path': LaunchConfiguration('npu_model_path'),
                'npu_model_name': LaunchConfiguration('npu_model_name'),
                'onnx_yolov8_root': LaunchConfiguration('onnx_yolov8_root'),
                'class_names': LaunchConfiguration('class_names'),
                'num_classes': LaunchConfiguration('num_classes'),
                'conf_thr': LaunchConfiguration('conf_thr'),
                'nms_thr': LaunchConfiguration('nms_thr'),
                'image_topic': LaunchConfiguration('image_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'output_format': LaunchConfiguration('output_format'),
                'yolov8_inference_topic': LaunchConfiguration('yolov8_inference_topic'),
                'class_filter': LaunchConfiguration('class_filter'),
                'npu_active_topic': LaunchConfiguration('npu_active_topic'),
                'publish_npu_active_heartbeat': LaunchConfiguration(
                    'publish_npu_active_heartbeat'
                ),
                'npu_active_heartbeat_hz': LaunchConfiguration('npu_active_heartbeat_hz'),
                'reenable_ultralytics_on_exit': LaunchConfiguration(
                    'reenable_ultralytics_on_exit'
                ),
                'publish_debug_image': LaunchConfiguration('publish_debug_image'),
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                'profile_timing': LaunchConfiguration('profile_timing'),
            }
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            npu_model_path_arg,
            npu_model_name_arg,
            onnx_yolov8_root_arg,
            class_names_arg,
            num_classes_arg,
            conf_thr_arg,
            nms_thr_arg,
            image_topic_arg,
            detections_topic_arg,
            output_format_arg,
            yolov8_inference_topic_arg,
            class_filter_arg,
            npu_active_topic_arg,
            publish_npu_active_heartbeat_arg,
            npu_active_heartbeat_hz_arg,
            reenable_ultralytics_arg,
            publish_debug_image_arg,
            debug_image_topic_arg,
            profile_timing_arg,
            node,
        ]
    )
