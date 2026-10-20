#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""Standalone NPU YOLO COCO inference: image -> /vision/detections_2d + optional debug Image + /npu_vision_active."""

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovariance
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
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


def _draw_rows_bgr(bgr, rows):
    """rows: list of (bbox_xyxy, class_name str, conf float). Returns annotated BGR image."""
    out = bgr.copy()
    for bbox, class_name, conf in rows:
        x1 = int(float(bbox[0]))
        y1 = int(float(bbox[1]))
        x2 = int(float(bbox[2]))
        y2 = int(float(bbox[3]))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {float(conf):.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 4, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _overlay_fps_top_left(bgr, fps: float) -> None:
    """Draw smoothed FPS at top-left (BGR, in-place)."""
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


def _rows_to_detection2d_array(header, rows):
    """rows: list of (bbox_xyxy ndarray, class_name str, conf float)."""
    out = Detection2DArray()
    out.header = header
    for bbox, class_name, conf in rows:
        d = Detection2D()
        d.header = header
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis = ObjectHypothesis()
        hyp.hypothesis.class_id = str(class_name)
        hyp.hypothesis.score = float(conf)
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


class NpuYoloCocoInference(Node):
    def __init__(self):
        super().__init__("npu_yolo_coco_inference")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("npu_model_path", "")
        self.declare_parameter("npu_model_name", "yolov8n")
        self.declare_parameter("onnx_yolov8_root", "")
        self.declare_parameter("conf_thr", 0.25)
        self.declare_parameter("nms_thr", 0.45)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/vision/detections_2d")
        self.declare_parameter("npu_active_topic", "/npu_vision_active")
        self.declare_parameter("publish_npu_active_heartbeat", True)
        self.declare_parameter("npu_active_heartbeat_hz", 2.0)
        self.declare_parameter("reenable_ultralytics_on_exit", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "/vision/yolo_debug_image")

        conf = _as_float(self.get_parameter("conf_thr").value, 0.25)
        nms = _as_float(self.get_parameter("nms_thr").value, 0.45)

        try:
            from cix_ros_noe_yolo.npu_yolo_coco_engine import NpuYoloCocoEngine
        except ImportError as e:
            self.get_logger().error(f"cix_ros_noe_yolo: {e}")
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
                self.get_logger().error(
                    "npu_model_path is empty and package share could not be resolved"
                )
                raise RuntimeError("npu_model_path required when cix_ros_noe_yolo is not installed")

            name = (
                self.get_parameter("npu_model_name").get_parameter_value().string_value.strip()
                or "yolov8n"
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

            hint = model_missing_hint(path)
            self.get_logger().error(hint)
            raise RuntimeError(hint)

        onnx_root = self.get_parameter("onnx_yolov8_root").get_parameter_value().string_value or ""
        self._engine = NpuYoloCocoEngine(
            path, onnx_yolov8_root=onnx_root, conf_thr=conf, nms_thr=nms
        )
        self.get_logger().info(f"NPU COCO inference node loaded: {path}")

        self.bridge = CvBridge()
        img_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value.strip()
            or "/camera/image_raw"
        )
        det_topic = (
            self.get_parameter("detections_topic").get_parameter_value().string_value.strip()
            or "/vision/detections_2d"
        )

        self._det_pub = self.create_publisher(Detection2DArray, det_topic, qos_profile_sensor_data)

        self._viz_pub = None
        if _as_bool(self.get_parameter("publish_debug_image").value, True):
            dbg_topic = (
                self.get_parameter("debug_image_topic").get_parameter_value().string_value.strip()
                or "/vision/yolo_debug_image"
            )
            self._viz_pub = self.create_publisher(Image, dbg_topic, qos_profile_sensor_data)
            self.get_logger().info(f"Publishing debug image (BGR boxes) on {dbg_topic}")

        npu_topic = (
            self.get_parameter("npu_active_topic").get_parameter_value().string_value.strip()
            or "/npu_vision_active"
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
        else:
            self.get_logger().info(
                "NPU active Bool: initial true published; periodic heartbeat disabled"
            )

        self.create_subscription(Image, img_topic, self._image_cb, qos_profile_sensor_data)

        self._fps_last_mono = None
        self._fps_smoothed = 0.0

    def _npu_active_heartbeat(self):
        self._npu_active_pub.publish(Bool(data=True))

    def _image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rows = self._engine.detect(cv_image)
            out = _rows_to_detection2d_array(msg.header, rows)
            self._det_pub.publish(out)
            if self._viz_pub is not None:
                vis = _draw_rows_bgr(cv_image, rows)
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
            self.get_logger().error(f"NPU inference error: {e}")

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
        node = NpuYoloCocoInference()
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
