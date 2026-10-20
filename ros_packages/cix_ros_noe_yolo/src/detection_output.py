# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
Convert NPU YOLO-OBB detection rows into ROS messages.

Pure helpers for ``npu_yolo_obb_inference`` (no ROS node here):

  image -> NpuYoloObbEngine.detect() -> List[DetectionRow]
                                        |
                          detection_output (this module)
                                        |
                    Detection2DArray / Yolov8Inference -> publish

``output_format`` selects publishers:
  - ``vision_msgs``  -> ``rows_to_detection2d_array``  -> detections_topic
  - ``yolov8_obb``   -> ``rows_to_yolov8_inference``  -> /Yolov8_Inference
  - ``both``         -> both

Each DetectionRow is (corners 4x2, class_name, confidence) from the OBB engine.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import PoseWithCovariance
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

# Matches NpuYoloObbEngine.detect() return type
DetectionRow = Tuple[np.ndarray, str, float]

# ROS parameter output_format (case-insensitive)
OUTPUT_VISION_MSGS = 'vision_msgs'
OUTPUT_YOLOV8_OBB = 'yolov8_obb'
OUTPUT_BOTH = 'both'
VALID_OUTPUT_FORMATS = frozenset({OUTPUT_VISION_MSGS, OUTPUT_YOLOV8_OBB, OUTPUT_BOTH})


# --- output_format / class_filter parsing -------------------------------------


def parse_output_format(value: str) -> Tuple[bool, bool]:
    """
    Parse ``output_format`` parameter.

    Returns:
        (publish_vision_msgs, publish_yolov8_obb): whether to create each publisher.
    Unknown values publish vision_msgs only.
    """
    key = (value or '').strip().lower()
    if key == OUTPUT_YOLOV8_OBB:
        return False, True
    if key == OUTPUT_BOTH:
        return True, True
    # Unknown / vision_msgs / empty -> vision_msgs only (safe default for generic tools)
    return True, False


def parse_class_filter(value) -> List[str]:
    """
    Parse ``class_filter`` parameter.

    Accepts comma-separated string (e.g. ``bolt`` or ``bolt,nut``) or a list.
    Empty list means no filtering (keep all classes).
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [part.strip() for part in text.split(',') if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def filter_rows(rows: Sequence[DetectionRow], class_filter: Sequence[str]) -> List[DetectionRow]:
    """Filter detection rows by class name; no-op when class_filter is empty."""
    if not class_filter:
        return list(rows)
    allowed = set(class_filter)
    return [row for row in rows if row[1] in allowed]


# --- geometry: OBB corners <-> message-friendly forms -------------------------


def bbox_xyxy_to_corners(bbox_xyxy: np.ndarray) -> np.ndarray:
    """
    Axis-aligned box (x1,y1,x2,y2) -> 4x2 corners, order matches Moveit2YoloObb UI polygon.

    Corner order: top-left -> top-right -> bottom-right -> bottom-left
    (flattened to 8 floats for InferenceResult.coordinates).
    """
    x1, y1, x2, y2 = (float(bbox_xyxy[i]) for i in range(4))
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def row_bbox_to_corners(bbox: np.ndarray) -> np.ndarray:
    """Normalize engine output to 4x2 float32 corners for both message types."""
    arr = np.asarray(bbox, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (4, 2):
        return arr
    if arr.size == 8:
        return arr.reshape(4, 2)
    # Fallback: axis-aligned box from first four scalars
    if arr.size >= 4:
        return bbox_xyxy_to_corners(arr.reshape(-1)[:4])
    raise ValueError(f"unsupported bbox shape for message conversion: {arr.shape}")


# --- ROS message builders -----------------------------------------------------


def rows_to_detection2d_array(header: Header, rows: Sequence[DetectionRow]) -> Detection2DArray:
    """
    Build ``vision_msgs/Detection2DArray`` (rotated box as center + size + theta).

    OBB corners are fitted with minAreaRect; for generic nodes on detections_topic.
    """
    out = Detection2DArray()
    out.header = header
    for bbox, class_name, conf in rows:
        det = Detection2D()
        det.header = header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis = ObjectHypothesis()
        hyp.hypothesis.class_id = str(class_name)
        hyp.hypothesis.score = float(conf)
        hyp.pose = PoseWithCovariance()
        det.results.append(hyp)
        corners = row_bbox_to_corners(bbox)
        # vision_msgs Detection2D uses center + size + theta, not raw vertices
        rect = cv2.minAreaRect(corners.astype(np.float32))
        (cx, cy), (bw, bh), angle_deg = rect
        det.bbox = BoundingBox2D()
        det.bbox.center.position.x = float(cx)
        det.bbox.center.position.y = float(cy)
        det.bbox.center.theta = math.radians(float(angle_deg))
        det.bbox.size_x = max(0.0, float(bw))
        det.bbox.size_y = max(0.0, float(bh))
        out.detections.append(det)
    return out


def rows_to_yolov8_inference(header: Header, rows: Sequence[DetectionRow]):
    """
    Build ``yolov8_msgs/Yolov8Inference`` (Moveit2YoloObb custom OBB message).

    Requires ``yolov8_msgs`` in the workspace (source tree: ``yolo_msgs/``).
    Each ``InferenceResult``: class_name + coordinates[8] (four vertex x,y pairs).
    """
    # Lazy import: yolov8_msgs is only required when this converter runs
    from yolov8_msgs.msg import InferenceResult, Yolov8Inference

    out = Yolov8Inference()
    out.header = header
    for bbox, class_name, conf in rows:
        item = InferenceResult()
        item.class_name = str(class_name)
        # float64[8]: vertex order must match GraphExecuter OBB overlay / click-to-grasp
        item.coordinates = row_bbox_to_corners(bbox).reshape(-1).astype(np.float64).tolist()
        item.confidence = float(conf)
        out.yolov8_inference.append(item)
    return out
