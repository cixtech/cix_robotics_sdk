#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NPU handpose: finger direction (Vector3), raw frames on output_image_topic, optional debug; base via finger_direction_to_cmd_vel."""

from __future__ import annotations

import glob
import os
import re
import subprocess

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cix_ros_noe_handpose.handpose_gesture import (
    HandPoseNpuGesture,
    render_handpose_debug,
    robot_command_to_discrete_xyz,
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
        return val.lower() in ("1", "true", "yes")
    return default


def _udev_properties(device_path: str) -> dict:
    props: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["udevadm", "info", "-q", "property", "-n", device_path],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return props
        for line in out.stdout.splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip()
    except Exception:
        pass
    return props


def _list_video_devices() -> list[str]:
    paths = sorted(
        glob.glob("/dev/video[0-9]*"),
        key=lambda p: int(re.search(r"(\d+)$", p).group(1)) if re.search(r"(\d+)$", p) else 0,
    )
    return [p for p in paths if os.path.exists(p)]


def _try_open_v4l(path: str) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def pick_usb_video_device(logger) -> str:
    """Pick first readable /dev/video* that udev marks as USB; else first readable device."""
    devices = _list_video_devices()
    if not devices:
        raise RuntimeError("no /dev/video* devices found")

    usb_readable: list[str] = []
    any_readable: list[str] = []

    for dev in devices:
        cap = _try_open_v4l(dev)
        if cap is None:
            continue
        cap.release()
        any_readable.append(dev)
        props = _udev_properties(dev)
        if props.get("ID_BUS", "").lower() == "usb":
            usb_readable.append(dev)

    if usb_readable:
        chosen = usb_readable[0]
        logger.info(f"handpose teleop: auto-selected USB V4L device {chosen}")
        return chosen
    if any_readable:
        chosen = any_readable[0]
        logger.warning(
            f"handpose teleop: no udev USB-tagged readable device; using {chosen}"
        )
        return chosen
    raise RuntimeError("could not read frames from any /dev/video* device")


class HandposeTwistTeleop(Node):
    def __init__(self) -> None:
        super().__init__("handpose_twist_teleop")

        # Launch may inject use_sim_time before declare; avoid duplicate declare_parameter.
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter(
            "image_topic",
            "/camera/color/image_raw",
            ParameterDescriptor(
                description="Input when use_usb_camera:=false only. Not used for publishing."
            ),
        )
        self.declare_parameter("npu_model_path", "")
        self.declare_parameter("use_usb_camera", True)
        self.declare_parameter(
            "video_device",
            "auto",
            ParameterDescriptor(
                description=(
                    "V4L path e.g. /dev/video0, or 'auto' to pick first readable "
                    "device with udev ID_BUS=usb."
                )
            ),
        )
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("camera_frame_id", "usb_camera")
        self.declare_parameter("publish_output_image", True)
        self.declare_parameter(
            "output_image_topic",
            "/handpose/image_out",
            ParameterDescriptor(
                description="bgr8 image of each frame used for NPU (USB and subscribed input)."
            ),
        )
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_topic", "/handpose/debug_image")
        self.declare_parameter(
            "min_finger_len_px",
            6.0,
            ParameterDescriptor(
                description="Min index segment length (px) to accept a direction (coarse)."
            ),
        )
        self.declare_parameter("require_index_dominant", False)
        self.declare_parameter("index_vs_other_ratio", 1.15)
        self.declare_parameter(
            "direction_axis_ratio",
            1.4,
            ParameterDescriptor(
                description=(
                    "Coarse 4-way split: horizontal if |dx| >= ratio*|dy|. "
                    "Lower = wider left/right vs up/down (less angle precision needed)."
                )
            ),
        )
        self.declare_parameter("publish_finger_direction", True)
        self.declare_parameter("finger_direction_topic", "/handpose/finger_direction")

        use_sim = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        if use_sim:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        path = self.get_parameter("npu_model_path").get_parameter_value().string_value.strip()
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory

                share = get_package_share_directory("cix_ros_noe_handpose")
                path = os.path.join(share, "models", "handpose_resnet50_size256.cix")
            except Exception:
                path = ""
        if not path or not os.path.isfile(path):
            self.get_logger().error(
                "npu_model_path missing or not a file; set param or install package with models/"
            )
            raise RuntimeError("handpose model path required")

        self._dir_pub = None
        if _as_bool(self.get_parameter("publish_finger_direction").value, True):
            fd_topic = (
                self.get_parameter("finger_direction_topic").get_parameter_value().string_value.strip()
                or "/handpose/finger_direction"
            )
            self._dir_pub = self.create_publisher(Vector3, fd_topic, 10)
            self.get_logger().info(
                f"handpose teleop: coarse compass on {fd_topic} (x=L/R, y=U/D image, z=0)"
            )
        self._out_img_pub = None
        self._debug_img_pub = None

        self._bridge = CvBridge()
        self._usb_timer = None
        self._cap = None
        self._camera_frame_id = (
            self.get_parameter("camera_frame_id").get_parameter_value().string_value.strip()
            or "usb_camera"
        )
        self._publish_debug_image_enabled = _as_bool(
            self.get_parameter("publish_debug_image").value,
            True,
        )
        self._debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value.strip()
            or "/handpose/debug_image"
        )
        self._publish_output_image_enabled = _as_bool(
            self.get_parameter("publish_output_image").value,
            True,
        )
        self._output_image_topic = (
            self.get_parameter("output_image_topic").get_parameter_value().string_value.strip()
            or "/handpose/image_out"
        )
        if self._publish_output_image_enabled:
            self._out_img_pub = self.create_publisher(
                Image, self._output_image_topic, qos_profile_sensor_data
            )
            self.get_logger().info(f"handpose: raw bgr8 -> {self._output_image_topic}")

        self._use_usb_camera = _as_bool(
            self.get_parameter("use_usb_camera").value,
            True,
        )
        width = int(_as_float(self.get_parameter("camera_width").value, 640))
        height = int(_as_float(self.get_parameter("camera_height").value, 480))
        fps = max(1.0, _as_float(self.get_parameter("camera_fps").value, 30.0))
        img_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value.strip()
            or "/camera/color/image_raw"
        )

        if self._use_usb_camera:
            raw_video = self.get_parameter("video_device").get_parameter_value().string_value.strip()
            if not raw_video or raw_video.lower() in ("auto", "detect", "scan"):
                video_device = pick_usb_video_device(self.get_logger())
            else:
                video_device = raw_video
            self._cap = cv2.VideoCapture(video_device)
            if not self._cap.isOpened():
                raise RuntimeError(f"could not open USB camera: {video_device}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            if self._publish_debug_image_enabled:
                self._debug_img_pub = self.create_publisher(
                    Image, self._debug_image_topic, qos_profile_sensor_data
                )
            self.get_logger().info(
                f"handpose: USB {video_device}; raw out {self._output_image_topic}"
            )
        else:
            self.create_subscription(Image, img_topic, self._image_cb, qos_profile_sensor_data)
            if self._publish_debug_image_enabled:
                self._debug_img_pub = self.create_publisher(
                    Image, self._debug_image_topic, qos_profile_sensor_data
                )
            self.get_logger().info(
                f"handpose: subscribe in {img_topic} -> raw out {self._output_image_topic}"
            )

        from NOE_Engine import EngineInfer

        self._engine = EngineInfer(path)
        self._gesture = HandPoseNpuGesture(
            self._engine,
            min_finger_len_px=_as_float(self.get_parameter("min_finger_len_px").value, 6.0),
            require_index_dominant=_as_bool(
                self.get_parameter("require_index_dominant").value,
                False,
            ),
            index_vs_other_ratio=_as_float(self.get_parameter("index_vs_other_ratio").value, 1.15),
            direction_axis_ratio=_as_float(
                self.get_parameter("direction_axis_ratio").value,
                1.4,
            ),
        )

        if self._use_usb_camera:
            self._usb_timer = self.create_timer(1.0 / fps, self._usb_camera_cb)

        self.get_logger().info(
            f"handpose: model={path}; direction + output_image + (opt) debug; base: finger_direction_to_cmd_vel"
        )

    def _image_cb(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process_bgr(bgr)
        except Exception as e:
            self.get_logger().error(f"handpose teleop: {e}")

    def _usb_camera_cb(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.get_logger().warning("handpose teleop: USB camera read failed", throttle_duration_sec=2.0)
            return
        try:
            self._process_bgr(frame)
        except Exception as e:
            self.get_logger().error(f"handpose teleop: {e}")

    def _process_bgr(self, bgr) -> None:
        if self._out_img_pub is not None:
            self._publish_output_image(bgr)
        res = self._gesture.infer_frame(bgr)
        if self._dir_pub is not None:
            x, y, z = robot_command_to_discrete_xyz(res.command)
            self._dir_pub.publish(Vector3(x=float(x), y=float(y), z=float(z)))
        if self._debug_img_pub is not None:
            debug = render_handpose_debug(bgr, res.keypoints_px, res.command)
            self._publish_debug_image(debug)

    def _publish_output_image(self, bgr) -> None:
        if self._out_img_pub is None:
            return
        msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._camera_frame_id
        self._out_img_pub.publish(msg)

    def _publish_debug_image(self, bgr) -> None:
        if self._debug_img_pub is None:
            return
        msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._camera_frame_id
        self._debug_img_pub.publish(msg)

    def destroy_node(self) -> None:
        timer = getattr(self, "_usb_timer", None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
            self._usb_timer = None
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
        eng = getattr(self, "_engine", None)
        if eng is not None:
            try:
                eng.clean()
            except Exception:
                pass
            self._engine = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = HandposeTwistTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"handpose_twist_teleop: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
