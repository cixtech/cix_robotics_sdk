#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""NPU YOLOv8-seg: Detection2DArray + optional overlay debug Image (``seg_inference_npu.py``-aligned)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock (/clock)",
    )
    npu_model_path_arg = DeclareLaunchArgument(
        "npu_model_path",
        default_value="",
        description="Absolute path to seg .cix; if empty, npu_model_name under package share/model/",
    )
    npu_model_name_arg = DeclareLaunchArgument(
        "npu_model_name",
        default_value="yolov8l_seg",
        description="Basename under share/.../model/ when npu_model_path is empty",
    )
    onnx_yolov8_root_arg = DeclareLaunchArgument(
        "onnx_yolov8_root",
        default_value="",
        description="Override root containing yolo_postprocess/; empty uses package share",
    )
    conf_thr_arg = DeclareLaunchArgument("conf_thr", default_value="0.35")
    iou_thr_arg = DeclareLaunchArgument("iou_thr", default_value="0.45")
    seg_nm_arg = DeclareLaunchArgument(
        "seg_nm",
        default_value="32",
        description="Mask coefficient channels (proto dim 0)",
    )
    seg_proto_hw_arg = DeclareLaunchArgument(
        "seg_proto_hw",
        default_value="160",
        description="Proto mask spatial size (HxW); yolov8l_seg default 160",
    )
    seg_num_classes_arg = DeclareLaunchArgument(
        "seg_num_classes",
        default_value="80",
        description="COCO class count; det head C = 4 + seg_num_classes + seg_nm",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/image_raw",
    )
    detections_topic_arg = DeclareLaunchArgument(
        "detections_topic",
        default_value="/vision/seg_detections_2d",
    )
    npu_active_topic_arg = DeclareLaunchArgument(
        "npu_active_topic",
        default_value="/npu_seg_active",
    )
    npu_active_heartbeat_hz_arg = DeclareLaunchArgument(
        "npu_active_heartbeat_hz",
        default_value="2.0",
    )
    publish_npu_active_heartbeat_arg = DeclareLaunchArgument(
        "publish_npu_active_heartbeat",
        default_value="true",
    )
    reenable_ultralytics_arg = DeclareLaunchArgument(
        "reenable_ultralytics_on_exit",
        default_value="true",
    )
    publish_debug_image_arg = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="true",
    )
    debug_image_topic_arg = DeclareLaunchArgument(
        "debug_image_topic",
        default_value="/vision/yolo_seg_debug_image",
    )

    node = Node(
        package="cix_ros_noe_yolo",
        executable="npu_yolo_seg_inference",
        name="npu_yolo_seg_inference",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "npu_model_path": LaunchConfiguration("npu_model_path"),
                "npu_model_name": LaunchConfiguration("npu_model_name"),
                "onnx_yolov8_root": LaunchConfiguration("onnx_yolov8_root"),
                "conf_thr": LaunchConfiguration("conf_thr"),
                "iou_thr": LaunchConfiguration("iou_thr"),
                "seg_nm": LaunchConfiguration("seg_nm"),
                "seg_proto_hw": LaunchConfiguration("seg_proto_hw"),
                "seg_num_classes": LaunchConfiguration("seg_num_classes"),
                "image_topic": LaunchConfiguration("image_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "npu_active_topic": LaunchConfiguration("npu_active_topic"),
                "publish_npu_active_heartbeat": LaunchConfiguration(
                    "publish_npu_active_heartbeat"
                ),
                "npu_active_heartbeat_hz": LaunchConfiguration("npu_active_heartbeat_hz"),
                "reenable_ultralytics_on_exit": LaunchConfiguration(
                    "reenable_ultralytics_on_exit"
                ),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
                "debug_image_topic": LaunchConfiguration("debug_image_topic"),
            }
        ],
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            npu_model_path_arg,
            npu_model_name_arg,
            onnx_yolov8_root_arg,
            conf_thr_arg,
            iou_thr_arg,
            seg_nm_arg,
            seg_proto_hw_arg,
            seg_num_classes_arg,
            image_topic_arg,
            detections_topic_arg,
            npu_active_topic_arg,
            publish_npu_active_heartbeat_arg,
            npu_active_heartbeat_hz_arg,
            reenable_ultralytics_arg,
            publish_debug_image_arg,
            debug_image_topic_arg,
            node,
        ]
    )
