# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NPU YOLO-OBB inference (yolo_obb.cix bolt) with yolov8_msgs / vision_msgs output."""

from __future__ import annotations

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from vision_msgs.msg import Detection2DArray

from cix_ros_noe_yolo.detection_output import (
    OUTPUT_VISION_MSGS,
    OUTPUT_YOLOV8_OBB,
    VALID_OUTPUT_FORMATS,
    filter_rows,
    parse_class_filter,
    parse_output_format,
    rows_to_detection2d_array,
    rows_to_yolov8_inference,
)
from cix_ros_noe_yolo.npu_yolo_obb_engine import NpuYoloObbEngine


def _as_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _as_bool(val, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('1', 'true', 'yes', 'on')
    return default


def _parse_class_names(value) -> list[str]:
    if value is None:
        return ['bolt']
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ['bolt']
        return [part.strip() for part in text.split(',') if part.strip()]
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names if names else ['bolt']
    text = str(value).strip()
    return [text] if text else ['bolt']


def _draw_obb_rows_bgr(bgr: np.ndarray, rows) -> np.ndarray:
    """rows: (corners 4x2, class_name, conf)."""
    out = bgr.copy()
    for corners, class_name, conf in rows:
        pts = np.asarray(corners, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        label = f'{class_name} {float(conf):.2f}'
        cv2.putText(
            out,
            label,
            (int(pts[0][0]), max(int(pts[0][1]) - 4, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def _overlay_fps_top_left(bgr, fps: float, label: str = 'NPU Inference FPS') -> None:
    text = f'{label}: {fps:.1f}'
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
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


class NpuYoloObbInference(Node):
    def __init__(self):
        super().__init__('cix_ros_noe_yolo')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.declare_parameter('npu_model_path', '')
        self.declare_parameter('npu_model_name', 'yolo_obb')
        self.declare_parameter('onnx_yolov8_root', '')
        self.declare_parameter('class_names', 'bolt')
        self.declare_parameter('num_classes', 1)
        self.declare_parameter('conf_thr', 0.90)
        self.declare_parameter('nms_thr', 0.40)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('detections_topic', '/vision/detections_2d')
        self.declare_parameter('output_format', OUTPUT_YOLOV8_OBB)
        self.declare_parameter('yolov8_inference_topic', '/Yolov8_Inference')
        self.declare_parameter('class_filter', 'bolt')
        self.declare_parameter('npu_active_topic', '/npu_vision_active')
        self.declare_parameter('publish_npu_active_heartbeat', True)
        self.declare_parameter('npu_active_heartbeat_hz', 2.0)
        self.declare_parameter('reenable_ultralytics_on_exit', True)
        self.declare_parameter('publish_debug_image', False)
        self.declare_parameter('debug_image_topic', '/inference_result')
        self.declare_parameter('profile_timing', False)

        output_format_raw = (
            self.get_parameter('output_format').get_parameter_value().string_value.strip()
        )
        if output_format_raw.lower() not in VALID_OUTPUT_FORMATS:
            self.get_logger().warning(
                f'Unknown output_format "{output_format_raw}", using {OUTPUT_YOLOV8_OBB}'
            )
            output_format_raw = OUTPUT_YOLOV8_OBB
        self._publish_vision, self._publish_yolov8 = parse_output_format(output_format_raw)
        self._class_filter = parse_class_filter(self.get_parameter('class_filter').value)

        conf = _as_float(self.get_parameter('conf_thr').value, 0.90)
        nms = _as_float(self.get_parameter('nms_thr').value, 0.40)

        path = self._resolve_model_path()
        if not os.path.isfile(path):
            from cix_ros_noe_yolo.model_assets import model_missing_hint

            hint = model_missing_hint(path)
            self.get_logger().error(hint)
            raise RuntimeError(hint)

        class_names = _parse_class_names(self.get_parameter('class_names').value)
        try:
            num_classes = int(self.get_parameter('num_classes').value)
        except (TypeError, ValueError):
            num_classes = len(class_names)

        self._profile_timing = _as_bool(
            self.get_parameter('profile_timing').value, False
        )
        onnx_root = (
            self.get_parameter('onnx_yolov8_root').get_parameter_value().string_value.strip()
        )

        try:
            self._engine = NpuYoloObbEngine(
                path,
                class_names=class_names,
                conf_thr=conf,
                nms_thr=nms,
                num_classes=num_classes,
                profile_timing=self._profile_timing,
                onnx_yolov8_root=onnx_root,
            )
        except ImportError as e:
            self.get_logger().error(f'cix_ros_noe_yolo NPU OBB: {e}')
            raise

        self.get_logger().info(
            f'NPU OBB inference loaded: {path} (classes={class_names})'
        )

        self.bridge = CvBridge()
        img_topic = (
            self.get_parameter('image_topic').get_parameter_value().string_value.strip()
            or '/image_raw'
        )

        self._det_pub = None
        if self._publish_vision:
            det_topic = (
                self.get_parameter('detections_topic').get_parameter_value().string_value.strip()
                or '/vision/detections_2d'
            )
            self._det_pub = self.create_publisher(
                Detection2DArray, det_topic, qos_profile_sensor_data
            )
            self.get_logger().info(f'Publishing {OUTPUT_VISION_MSGS} on {det_topic}')

        self._yolo_pub = None
        if self._publish_yolov8:
            from yolov8_msgs.msg import Yolov8Inference

            yolo_topic = (
                self.get_parameter('yolov8_inference_topic')
                .get_parameter_value()
                .string_value.strip()
                or '/Yolov8_Inference'
            )
            self._yolo_pub = self.create_publisher(Yolov8Inference, yolo_topic, 10)
            self.get_logger().info(f'Publishing {OUTPUT_YOLOV8_OBB} on {yolo_topic}')
            if self._class_filter:
                self.get_logger().info(f'class_filter={self._class_filter}')

        self._viz_pub = None
        if _as_bool(self.get_parameter('publish_debug_image').value, False):
            dbg_topic = (
                self.get_parameter('debug_image_topic').get_parameter_value().string_value.strip()
                or '/inference_result'
            )
            self._viz_pub = self.create_publisher(Image, dbg_topic, qos_profile_sensor_data)
            self.get_logger().info(f'Publishing debug image (BGR OBB) on {dbg_topic}')

        self._setup_npu_active_publisher()
        self.create_subscription(Image, img_topic, self._image_cb, qos_profile_sensor_data)
        self._npu_fps_pub = self.create_publisher(Float32, '/npu_inference_fps', 10)

        self._timing_log_counter = 0

        self.get_logger().info(f'output_format={output_format_raw}')
        if self._profile_timing:
            self.get_logger().info('profile_timing enabled (inference ms in log)')

    def _resolve_model_path(self) -> str:
        path = self.get_parameter('npu_model_path').get_parameter_value().string_value.strip()
        if path:
            return path

        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory('cix_ros_noe_yolo')
        name = (
            self.get_parameter('npu_model_name').get_parameter_value().string_value.strip()
            or 'yolo_obb'
        )
        if not name.endswith('.cix'):
            name = f'{name}.cix'
        return os.path.join(share, 'model', name)

    def _setup_npu_active_publisher(self) -> None:
        npu_topic = (
            self.get_parameter('npu_active_topic').get_parameter_value().string_value.strip()
            or '/npu_vision_active'
        )
        hz = _as_float(self.get_parameter('npu_active_heartbeat_hz').value, 2.0)
        hz = max(0.2, min(hz, 50.0))
        npu_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._npu_active_pub = self.create_publisher(Bool, npu_topic, npu_qos)
        self._npu_active_pub.publish(Bool(data=True))
        self._npu_active_timer = None
        if _as_bool(self.get_parameter('publish_npu_active_heartbeat').value, True):
            self._npu_active_timer = self.create_timer(1.0 / hz, self._npu_active_heartbeat)
        else:
            self.get_logger().info(
                'NPU active Bool: initial true published; periodic heartbeat disabled'
            )

    def _npu_active_heartbeat(self):
        self._npu_active_pub.publish(Bool(data=True))

    def _publish_detections(self, header, rows):
        filtered = filter_rows(rows, self._class_filter)
        if self._det_pub is not None:
            self._det_pub.publish(rows_to_detection2d_array(header, filtered))
        if self._yolo_pub is not None:
            self._yolo_pub.publish(rows_to_yolov8_inference(header, filtered))

    def _image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rows = self._engine.detect(cv_image)
            self._publish_detections(msg.header, rows)

            npu_infer_fps = self._engine.npu_infer_fps
            if npu_infer_fps > 0.0:
                self._npu_fps_pub.publish(Float32(data=npu_infer_fps))

            if self._profile_timing:
                self._timing_log_counter += 1
                if self._timing_log_counter >= 30:
                    self._timing_log_counter = 0
                    t = self._engine.last_timing_ms
                    npu_infer = t.get('npu_infer')
                    if npu_infer is not None:
                        self.get_logger().info(
                            f'inference ms: preprocess={t.get("preprocess", 0):.1f} '
                            f'npu={t.get("npu", 0):.1f} '
                            f'(load={t.get("npu_load", 0):.1f} '
                            f'infer={npu_infer:.1f} '
                            f'get={t.get("npu_get", 0):.1f}) '
                            f'postprocess={t.get("postprocess", 0):.1f} '
                            f'total={t.get("total", 0):.1f}'
                        )
                    else:
                        self.get_logger().info(
                            f'inference ms: preprocess={t.get("preprocess", 0):.1f} '
                            f'npu={t.get("npu", 0):.1f} postprocess={t.get("postprocess", 0):.1f} '
                            f'total={t.get("total", 0):.1f}'
                        )

            if self._viz_pub is not None:
                vis = _draw_obb_rows_bgr(
                    cv_image, filter_rows(rows, self._class_filter)
                )
                if npu_infer_fps > 0.0:
                    _overlay_fps_top_left(vis, npu_infer_fps)
                img_msg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
                img_msg.header = msg.header
                self._viz_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'NPU inference error: {e}')

    def destroy_node(self):
        if self.get_parameter('reenable_ultralytics_on_exit').get_parameter_value().bool_value:
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
        node = NpuYoloObbInference()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
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


if __name__ == '__main__':
    main()
