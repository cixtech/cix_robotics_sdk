#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""NPU YOLOv8-pose: image -> Detection2DArray (person boxes) + Float32MultiArray keypoints + optional debug Image."""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovariance
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)


def _as_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _as_bool(val, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return default


def _pose_rows_to_detections(header, rows, class_names):
    """rows: list of (bbox_xyxy, score, cls_id, kpts (17,3))."""
    out = Detection2DArray()
    out.header = header
    for bbox, score, cls_id, _kpts in rows:
        d = Detection2D()
        d.header = header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis = ObjectHypothesis()
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        hyp.hypothesis.class_id = str(name)
        hyp.hypothesis.score = float(score)
        hyp.pose = PoseWithCovariance()
        d.results.append(hyp)
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        d.bbox = BoundingBox2D()
        d.bbox.center.position.x = (x1 + x2) / 2.0
        d.bbox.center.position.y = (y1 + y2) / 2.0
        d.bbox.center.theta = 0.0
        d.bbox.size_x = max(0.0, x2 - x1)
        d.bbox.size_y = max(0.0, y2 - y1)
        out.detections.append(d)
    return out


def _pose_rows_to_float32_layout(rows):
    """Flatten keypoints as [person, 17, 3] row-major."""
    msg = Float32MultiArray()
    p = len(rows)
    if p == 0:
        msg.layout.dim = [
            MultiArrayDimension(label="person", size=0, stride=0),
            MultiArrayDimension(label="kpt", size=17, stride=3),
            MultiArrayDimension(label="xyv", size=3, stride=1),
        ]
        return msg
    flat: list[float] = []
    for _bb, _sc, _ci, kpts in rows:
        flat.extend(kpts.reshape(-1).astype(np.float32).tolist())
    msg.data = [float(x) for x in flat]
    msg.layout.dim = [
        MultiArrayDimension(label="person", size=p, stride=17 * 3),
        MultiArrayDimension(label="kpt", size=17, stride=3),
        MultiArrayDimension(label="xyv", size=3, stride=1),
    ]
    return msg


def _draw_pose_debug(bgr, rows, kpt_vis_thr: float):
    from yolo_postprocess.yolov8_pose_decode import COCO17_SKELETON

    out = bgr.copy()
    for bbox, _score, _cid, kpts in rows:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pts = []
        for i in range(17):
            x, y, v = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
            if v >= kpt_vis_thr:
                pts.append((int(round(x)), int(round(y))))
            else:
                pts.append(None)
        for a, b in COCO17_SKELETON:
            pa, pb = pts[a], pts[b]
            if pa is not None and pb is not None:
                cv2.line(out, pa, pb, (255, 128, 0), 2, cv2.LINE_AA)
        for p in pts:
            if p is not None:
                cv2.circle(out, p, 4, (0, 128, 255), -1, cv2.LINE_AA)
    return out


def _overlay_fps_top_left(bgr, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    x, y = 8, 24
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 4
    cv2.rectangle(
        bgr,
        (x - pad, y - th - pad),
        (x + tw + pad, y + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


class NpuYoloPoseInference(Node):
    def __init__(self):
        super().__init__("npu_yolo_pose_inference")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("npu_model_path", "")
        self.declare_parameter("npu_model_name", "yolov8s-pose")
        self.declare_parameter("onnx_yolov8_root", "")
        self.declare_parameter("conf_thr", 0.25)
        self.declare_parameter("nms_thr", 0.45)
        self.declare_parameter("kpt_vis_thr", 0.25)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/vision/pose_detections_2d")
        self.declare_parameter("pose_keypoints_topic", "/vision/pose_keypoints_flat")
        self.declare_parameter("npu_active_topic", "/npu_pose_active")
        self.declare_parameter("publish_npu_active_heartbeat", True)
        self.declare_parameter("npu_active_heartbeat_hz", 2.0)
        self.declare_parameter("reenable_ultralytics_on_exit", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "/vision/yolo_pose_debug_image")

        if self.get_parameter("use_sim_time").get_parameter_value().bool_value:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        conf = _as_float(self.get_parameter("conf_thr").value, 0.25)
        nms = _as_float(self.get_parameter("nms_thr").value, 0.45)

        try:
            from cix_ros_noe_yolo.npu_yolo_pose_engine import NpuYoloPoseV8Engine
        except ImportError as e:
            self.get_logger().error(f"cix_ros_noe_yolo pose: {e}")
            raise

        import os

        path = self.get_parameter("npu_model_path").get_parameter_value().string_value.strip()
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory

                share = get_package_share_directory("cix_ros_noe_yolo")
            except Exception:
                share = ""
            if not share:
                raise RuntimeError("npu_model_path empty and cix_ros_noe_yolo share not found")

            name = (
                self.get_parameter("npu_model_name").get_parameter_value().string_value.strip()
                or "yolov8s-pose"
            )
            if not name.endswith(".cix"):
                name = f"{name}.cix"
            path = os.path.join(share, "model", name)
            if not os.path.isfile(path):
                legacy = os.path.join(share, "onnx_yolov8_n", name)
                if os.path.isfile(legacy):
                    path = legacy

        if not os.path.isfile(path):
            from cix_ros_noe_yolo.model_assets import model_missing_hint

            raise RuntimeError(model_missing_hint(path))

        onnx_root = self.get_parameter("onnx_yolov8_root").get_parameter_value().string_value or ""
        self._engine = NpuYoloPoseV8Engine(
            path, onnx_yolov8_root=onnx_root, conf_thr=conf, nms_thr=nms
        )
        from yolo_postprocess.label.coco_classes import COCO_CLASSES

        self._coco = list(COCO_CLASSES)
        self._kpt_vis_thr = _as_float(self.get_parameter("kpt_vis_thr").value, 0.25)
        self.get_logger().info(f"NPU pose inference loaded: {path}")

        self.bridge = CvBridge()
        img_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value.strip()
            or "/camera/image_raw"
        )
        det_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value.strip()
            or "/vision/pose_detections_2d"
        )
        kpt_topic = (
            self.get_parameter("pose_keypoints_topic").get_parameter_value().string_value.strip()
            or "/vision/pose_keypoints_flat"
        )

        self._det_pub = self.create_publisher(Detection2DArray, det_topic, qos_profile_sensor_data)
        self._kpt_pub = self.create_publisher(Float32MultiArray, kpt_topic, qos_profile_sensor_data)

        self._viz_pub = None
        if _as_bool(self.get_parameter("publish_debug_image").value, True):
            dbg_topic = (
                self.get_parameter("debug_image_topic").get_parameter_value().string_value.strip()
                or "/vision/yolo_pose_debug_image"
            )
            self._viz_pub = self.create_publisher(Image, dbg_topic, qos_profile_sensor_data)
            self.get_logger().info(f"Publishing pose debug image on {dbg_topic}")

        npu_topic = (
            self.get_parameter("npu_active_topic").get_parameter_value().string_value.strip()
            or "/npu_pose_active"
        )
        hz = _as_float(self.get_parameter("npu_active_heartbeat_hz").value, 2.0)
        hz = max(0.2, min(hz, 50.0))
        npu_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._npu_active_pub = self.create_publisher(Bool, npu_topic, npu_qos)
        self._npu_active_pub.publish(Bool(data=True))
        self._npu_active_timer = None
        if _as_bool(self.get_parameter("publish_npu_active_heartbeat").value, True):
            self._npu_active_timer = self.create_timer(1.0 / hz, self._npu_active_heartbeat)

        self.create_subscription(Image, img_topic, self._image_cb, qos_profile_sensor_data)

        self._fps_last_mono = None
        self._fps_smoothed = 0.0

    def _npu_active_heartbeat(self):
        self._npu_active_pub.publish(Bool(data=True))

    def _image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rows = self._engine.detect_pose(cv_image)
            self._det_pub.publish(_pose_rows_to_detections(msg.header, rows, self._coco))
            self._kpt_pub.publish(_pose_rows_to_float32_layout(rows))
            if self._viz_pub is not None:
                vis = _draw_pose_debug(cv_image, rows, self._kpt_vis_thr)
                now = time.monotonic()
                if self._fps_last_mono is not None:
                    dt = now - self._fps_last_mono
                    if dt > 1e-6:
                        inst = 1.0 / dt
                        if self._fps_smoothed <= 0.0:
                            self._fps_smoothed = inst
                        else:
                            self._fps_smoothed = 0.85 * self._fps_smoothed + 0.15 * inst
                self._fps_last_mono = now
                _overlay_fps_top_left(vis, self._fps_smoothed)
                img_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
                img_msg.header = msg.header
                self._viz_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"NPU pose inference error: {e}")

    def destroy_node(self):
        if self.get_parameter("reenable_ultralytics_on_exit").get_parameter_value().bool_value:
            try:
                self._npu_active_pub.publish(Bool(data=False))
            except Exception:
                pass
        if self._npu_active_timer is not None:
            try:
                self._npu_active_timer.cancel()
            except Exception:
                pass
        try:
            self._engine.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NpuYoloPoseInference()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        # Ctrl+C / launch teardown often shuts the context already; a second
        # rclpy.shutdown() raises RCLError and triggers Ubuntu's Apport dialog.
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
