#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Zero-copy NPU handpose teleop using ISP dmabuf.

HandposeDmaTeleop directly consumes dmabuf images from ISP ROS node via
the broker service. Uses the DmaBufferImage wrapper to achieve true zero
copy: no memcpy between ISP driver -> ROS transport -> NPU inference.

Performance:
    Same process composable: shared_ptr passthrough, absolute zero copy
    Cross process with broker: only fd passed over Unix socket (SCM_RIGHTS)
    Cross process without broker: falls back to standard Image.data memcpy
"""

from __future__ import annotations

import os
import re
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from cix_ros_noe_handpose.handpose_gesture import (
    HandPoseNpuGesture,
    render_handpose_debug,
    robot_command_to_discrete_xyz,
)

# dmabuf zero-copy transport (cix_ros_dmabuf package)
from cix_ros_dmabuf.dma_image import BrokerClient, get_broker_client

DMA_SUPPORT = True


class HandposeDmaTeleop(Node):
    """Zero-copy NPU handpose teleop using dmabuf fd transport."""

    def __init__(self) -> None:
        super().__init__("handpose_dma_teleop")

        self.declare_parameter(
            "image_topic",
            "/cam0_stream1",
            ParameterDescriptor(description="Input image from ISP ROS node"),
        )
        self.declare_parameter("npu_model_path", "")
        self.declare_parameter("publish_output_image", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter(
            "debug_image_topic", "/handpose/debug_image",
            ParameterDescriptor(description="Debug overlay image output")
        )
        self.declare_parameter(
            "debug_image_scale", 0.5,
            ParameterDescriptor(
                description=(
                    "Downscale factor for the published debug image (0-1]. The "
                    "overlay is rendered at full resolution then resized before "
                    "publishing, so a full-res raw frame (e.g. 1280x720 BGR = "
                    "2.76 MB at 20 fps = ~55 MB/s) does not flood the DDS link "
                    "and stall rqt_image_view. 1.0 = no downscale."
                )
            ),
        )
        self.declare_parameter(
            "min_finger_len_px", 6.0,
            ParameterDescriptor(description="Minimum index-finger segment length in px"),
        )
        self.declare_parameter("require_index_dominant", False)
        self.declare_parameter("index_vs_other_ratio", 1.15)
        self.declare_parameter(
            "direction_axis_ratio", 1.4,
            ParameterDescriptor(description="dx threshold ratio for left/right"),
        )
        self.declare_parameter(
            "use_broker", DMA_SUPPORT,
            ParameterDescriptor(description="Try zero-copy via dmabuf broker when available"),
        )
        self.declare_parameter(
            "broker_socket_path", "",
            ParameterDescriptor(
                description=(
                    "Unix socket path of the dmabuf broker. Empty = use "
                    "DMABUF_BROKER_SOCK env var or the built-in default. Must "
                    "match the path the camera's broker listens on."
                )
            ),
        )

        path = self.get_parameter("npu_model_path").get_parameter_value().string_value.strip()
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                share = get_package_share_directory("cix_ros_noe_handpose")
                path = os.path.join(share, "models", "handpose_resnet50_size256.cix")
            except Exception:
                path = ""
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"handpose model not found: '{path}'")

        img_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value.strip()
            or "/cam0_stream1"
        )

        # Initialize broker client for zero-copy (will fail silently if not running)
        self._broker_ok = False
        if self.get_parameter("use_broker").get_parameter_value().bool_value and DMA_SUPPORT:
            sock_path = (
                self.get_parameter("broker_socket_path")
                .get_parameter_value().string_value.strip()
            ) or None
            client = get_broker_client(sock_path)
            self._broker_ok = client.is_connected
            if self._broker_ok:
                self.get_logger().info(
                    f"✅ dmabuf broker connected at {client._socket_path}; "
                    f"will use zero-copy from topic '{img_topic}'"
                )
            else:
                self.get_logger().warning(
                    "⚠ dmabuf broker not running; falling back to memcpy"
                )

        self._publish_output_image = (
            self.get_parameter("publish_output_image").get_parameter_value().bool_value
        )
        self._publish_debug_image = (
            self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        )
        self._debug_image_scale = float(
            self.get_parameter("debug_image_scale").get_parameter_value().double_value
        )
        if not (0.0 < self._debug_image_scale <= 1.0):
            self.get_logger().warning(
                f"debug_image_scale={self._debug_image_scale} out of (0,1]; using 1.0"
            )
            self._debug_image_scale = 1.0

        if self._publish_output_image:
            self._out_img_pub = self.create_publisher(
                Image, "/handpose/image_out", qos_profile_sensor_data
            )

        if self._publish_debug_image:
            debug_topic = (
                self.get_parameter("debug_image_topic")
                .get_parameter_value().string_value.strip()
                or "/handpose/debug_image"
            )
            self._debug_img_pub = self.create_publisher(
                Image, debug_topic, qos_profile_sensor_data
            )

        self._dir_pub = self.create_publisher(Vector3, "/handpose/finger_direction", 10)

        self._bridge = CvBridge()

        # Initialize NPU engine
        from NOE_Engine import EngineInfer
        self._engine = EngineInfer(path)
        self._gesture = HandPoseNpuGesture(
            self._engine,
            min_finger_len_px=float(
                self.get_parameter("min_finger_len_px").get_parameter_value().double_value
            ),
            require_index_dominant=(
                self.get_parameter("require_index_dominant").get_parameter_value().bool_value
            ),
            index_vs_other_ratio=float(
                self.get_parameter("index_vs_other_ratio").get_parameter_value().double_value
            ),
            direction_axis_ratio=float(
                self.get_parameter("direction_axis_ratio").get_parameter_value().double_value
            ),
        )

        self._frame_count = 0
        self._zero_copy_count = 0

        # Rolling FPS estimate (inference cadence), shown on the debug overlay.
        self._last_t = 0.0
        self._fps = 0.0

        # Subscribe image (standard ROS Image)
        self.create_subscription(Image, img_topic, self._image_cb, qos_profile_sensor_data)
        self.get_logger().info(f"✅ handpose_dma_teleop ready; listening to {img_topic}")

    def _parse_handle_from_encoding(self, encoding: str) -> tuple[str, int]:
        """Extract dmabuf handle from encoding if present.

        Format: "nv12;dmabuf_handle=12345" -> ("nv12", 12345)
        """
        match = re.search(r"dmabuf_handle=(\d+)", encoding)
        if match:
            fmt = encoding.split(";", 1)[0].strip()
            return fmt, int(match.group(1))
        return encoding, 0

    def _image_cb(self, msg: Image) -> None:
        try:
            self._frame_count += 1

            # Rolling FPS (exponential smoothing of inter-frame interval).
            now = time.monotonic()
            if self._last_t:
                dt = now - self._last_t
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst
            self._last_t = now

            # Check if this message carries a dmabuf handle via broker
            encoding, handle = self._parse_handle_from_encoding(msg.encoding)
            bgr = None

            if handle != 0 and self._broker_ok:
                # Zero-copy path
                fd, _ = get_broker_client().fetch_fd(handle)
                if fd >= 0:
                    try:
                        self._zero_copy_count += 1
                        # Direct dmabuf inference
                        result = self._gesture.infer_dmabuf(
                            fd, msg.step * msg.height,
                            msg.width, msg.height,
                            stride=msg.step,
                            encoding=encoding,
                        )

                        # For debug/display we still need BGR (memcpy needed)
                        if self._publish_output_image or self._publish_debug_image:
                            bgr = self._dmabuf_to_bgr(fd, msg.width, msg.height, msg.step, encoding)
                    finally:
                        os.close(fd)  # close our dup of the fd
                else:
                    # Fetch failed, fall back to memcpy
                    bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                    result = self._gesture.infer_frame(bgr)
            else:
                # Standard memcpy path
                bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                result = self._gesture.infer_frame(bgr)

            # Publish direction
            x, y, z = robot_command_to_discrete_xyz(result.command)
            self._dir_pub.publish(Vector3(x=float(x), y=float(y), z=float(z)))

            # Publish outputs
            if bgr is None and (self._publish_output_image or self._publish_debug_image):
                # No BGR yet (zero-copy didn't render it), but we still need it
                if handle != 0 and self._broker_ok:
                    fd2, _ = get_broker_client().fetch_fd(handle)
                    if fd2 >= 0:
                        bgr = self._dmabuf_to_bgr(fd2, msg.width, msg.height, msg.step, encoding)
                        os.close(fd2)

            if bgr is not None:
                if self._publish_output_image:
                    out_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
                    out_msg.header = msg.header
                    self._out_img_pub.publish(out_msg)

                if self._publish_debug_image:
                    debug = render_handpose_debug(bgr, result.keypoints_px, result.command)
                    # Downscale before publishing so the raw frame does not flood
                    # the DDS link and stall rqt_image_view (see debug_image_scale).
                    if self._debug_image_scale < 1.0:
                        debug = cv2.resize(
                            debug,
                            (int(debug.shape[1] * self._debug_image_scale),
                             int(debug.shape[0] * self._debug_image_scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    # FPS overlay, top-right corner.
                    fps_text = f"{self._fps:4.1f} fps"
                    (tw, th), _ = cv2.getTextSize(
                        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    h_img, w_img = debug.shape[:2]
                    org = (w_img - tw - 10, 30)
                    cv2.rectangle(
                        debug,
                        (org[0] - 4, org[1] - th - 4),
                        (w_img - 6, org[1] + 6),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        debug,
                        fps_text,
                        org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                    debug_msg.header = msg.header
                    self._debug_img_pub.publish(debug_msg)

            # Log stats
            if self._frame_count % 100 == 0:
                zero_copy_pct = 100.0 * self._zero_copy_count / self._frame_count
                self.get_logger().info(
                    f"Frames: {self._frame_count}, Zero-copy: {self._zero_copy_count} "
                    f"({zero_copy_pct:.1f}%)"
                )

        except Exception as e:
            self.get_logger().error(f"handpose dma teleop: {e}")

    def _dmabuf_to_bgr(self, fd: int, width: int, height: int, stride: int,
                       encoding: str = "nv12"):
        """Helper: render a dmabuf frame to BGR for display.

        Supports NV12 (stride = Y plane stride) and packed rgb8/bgr8
        (stride = row bytes, i.e. width*3 when unpadded).
        """
        import mmap
        enc = encoding.lower()

        # All mmap-backed numpy views must be consumed (and deleted) inside the
        # `with` block: np.frombuffer(mm, ...) keeps the buffer exported, so
        # mm.close() raises "cannot close exported pointers exist" if a view
        # outlives the block. The cv2/np results below own their memory and are
        # safe to return after the mmap is closed.

        if enc in ("rgb8", "bgr8"):
            row_bytes = width * 3
            map_size = stride * height
            with mmap.mmap(fd, map_size, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
                buf = np.frombuffer(mm, dtype=np.uint8, count=map_size, offset=0)
                buf = buf.reshape(height, stride)[:, :row_bytes].reshape(height, width, 3)
                # cv2 works in BGR; flip channels if the buffer is RGB-ordered.
                # ascontiguousarray forces a copy independent of the mmap.
                bgr = np.ascontiguousarray(buf[:, :, ::-1]) if enc == "rgb8" \
                    else np.ascontiguousarray(buf)
                del buf
            return bgr

        # Default: NV12 (stride = Y plane stride, total size = stride*height*3/2)
        with mmap.mmap(fd, stride * height * 3 // 2, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
            y_plane = np.frombuffer(mm, dtype=np.uint8, count=stride * height, offset=0)
            y_plane = y_plane.reshape(height, stride)[:, :width]

            uv_plane = np.frombuffer(mm, dtype=np.uint8, count=stride * height // 2,
                                     offset=stride * height)
            uv_plane = uv_plane.reshape(height // 2, stride // 2, 2)[:, :width // 2, :]

            # NV12 -> I420 -> BGR
            yuv_i420 = np.empty((height * 3 // 2, width), dtype=np.uint8)
            yuv_i420[:height] = y_plane
            yuv_i420[height:height + height // 4] = uv_plane[:, :, 0].reshape(height // 2, width // 2)
            yuv_i420[height + height // 4:] = uv_plane[:, :, 1].reshape(height // 2, width // 2)
            bgr = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
            del y_plane, uv_plane, yuv_i420
        return bgr

    def destroy_node(self) -> None:
        # Release the NPU engine explicitly (graph/job + NOE fds); without
        # this the engine is only freed when the process exits.
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
        node = HandposeDmaTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"handpose_dma_teleop: {e}")
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
