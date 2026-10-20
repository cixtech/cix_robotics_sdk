# SPDX-License-Identifier: Apache-2.0
"""Hand pose keypoints -> coarse 4-way index direction (no ROS). Engine injected (NOE EngineInfer).

Zero-copy extension for ISP dmabuf:
    preprocess_nv12_dmabuf(fd, size, width, height) -> tensor
    HandPoseNpuGesture.infer_dmabuf(fd, size, w, h, stride, fmt) -> FrameResult
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

import cv2
import numpy as np

IMG_SIZE = 256
IDX_MCP, IDX_TIP = 5, 8
MID_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
MID_MCP, RING_MCP, PINKY_MCP = 9, 13, 17


class RobotCommand(str, Enum):
    NONE = "none"
    FORWARD = "forward"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


def preprocess_bgr(bgr: np.ndarray, size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    image = cv2.resize(bgr, dsize=size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)


def preprocess_nv12_dmabuf(fd: int, size: int, width: int, height: int,
                            stride: int | None = None,
                            out_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """Zero-copy preprocess: NV12 dmabuf -> NPU input tensor.

    Args:
        fd: dmabuf file descriptor (NV12 format, Y plane then UV plane)
        size: total buffer bytes
        width: image width
        height: image height
        stride: line stride (padded width), same as width if no padding
        out_size: (out_w, out_h) for NPU input

    Returns:
        NCHW float32 tensor (1, 3, out_h, out_w) normalized [0,1]
    """
    if stride is None:
        stride = width

    # Mmap dmabuf into process memory (zero copy!)
    import mmap
    y_size = stride * height
    uv_size = (stride * height) // 2  # (w/2 * h/2) * 2 planes = w*h/2

    # Keep all mmap-backed views inside the `with` block and delete them before
    # close (see preprocess_rgb_dmabuf for the rationale). cv2.cvtColor returns
    # an independent array that is safe to use after the mmap is closed.
    with mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
        # Y plane (uint8, h x stride_y)
        y_plane = np.frombuffer(mm, dtype=np.uint8, count=y_size, offset=0)
        y_plane = y_plane.reshape(height, stride)[:, :width]

        # UV plane (uint8 interleaved)
        uv_plane = np.frombuffer(mm, dtype=np.uint8, count=uv_size, offset=y_size)
        uv_plane = uv_plane.reshape(height // 2, stride // 2, 2)[:, :width // 2, :]

        # NV12 -> I420 -> BGR (yuv_i420 is a fresh np.empty array; assignments copy)
        yuv_i420 = np.empty((height * 3 // 2, width), dtype=np.uint8)
        yuv_i420[:height] = y_plane
        yuv_i420[height:height + height // 4] = uv_plane[:, :, 0].reshape(height // 2, width // 2)
        yuv_i420[height + height // 4:] = uv_plane[:, :, 1].reshape(height // 2, width // 2)
        bgr = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
        del y_plane, uv_plane, yuv_i420

    # Resize and normalize for NPU
    image = cv2.resize(bgr, dsize=out_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)


def preprocess_rgb_dmabuf(fd: int, size: int, width: int, height: int,
                           stride: int | None = None,
                           out_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                           rgb: bool = True) -> np.ndarray:
    """Zero-copy preprocess: packed RGB/BGR dmabuf -> NPU input tensor.

    Args:
        fd: dmabuf file descriptor (packed 3-byte-per-pixel, e.g. RGB3)
        size: total buffer bytes
        width: image width
        height: image height
        stride: line stride in bytes (padded width*3); defaults to width*3
        out_size: (out_w, out_h) for NPU input
        rgb: True if the buffer is RGB ordered (will be converted to BGR to match
            the NPU model's expected input); False if already BGR

    Returns:
        NCHW float32 tensor (1, 3, out_h, out_w) normalized [0,1]
    """
    if stride is None:
        stride = width * 3

    import mmap
    row_bytes = width * 3
    # Do every operation that touches the mmap inside the `with` block and delete
    # the numpy views before it closes. np.frombuffer(mm, ...) creates views that
    # keep the mmap buffer exported; if they outlive the block, mm.close() raises
    # "cannot close exported pointers exist". cv2.resize returns an independent
    # array, so it is safe to keep after the mmap is gone.
    with mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
        buf = np.frombuffer(mm, dtype=np.uint8, count=stride * height, offset=0)
        # Crop each row to the unpadded width, then drop the trailing channel axis.
        buf = buf.reshape(height, stride)[:, :row_bytes].reshape(height, width, 3)
        # The NPU model was trained on BGR (see preprocess_bgr); convert if needed.
        bgr = buf[:, :, ::-1] if rgb else buf
        image = cv2.resize(bgr, dsize=out_size)
        del buf, bgr

    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)


def raw_output_to_keypoints_pixels(raw: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    flat = np.asarray(raw).reshape(-1)
    if flat.size != 42:
        raise ValueError(f"expected 42 values, got {flat.size}")
    xy = flat.reshape(21, 2).astype(np.float32)
    xy[:, 0] *= float(img_w)
    xy[:, 1] *= float(img_h)
    return xy


def classify_index_finger_direction(
    keypoints_px: np.ndarray,
    min_finger_len_px: float = 6.0,
    axis_ratio: float = 1.4,
) -> RobotCommand:
    """Coarse up/down/left/right: use dominant image axis with a loose ratio (not sub-degree)."""
    p5 = keypoints_px[IDX_MCP]
    p8 = keypoints_px[IDX_TIP]
    dx = float(p8[0] - p5[0])
    dy = float(p8[1] - p5[1])
    sq = dx * dx + dy * dy
    if sq < min_finger_len_px * min_finger_len_px:
        return RobotCommand.NONE
    adx = abs(dx)
    ady = abs(dy)
    # Horizontal if |dx| is "large enough" vs |dy|; ratio<1 widens left/right
    # sectors vs 45 deg split, ratio>1 widens up/down (1.4 => ~54 deg boundary).
    if adx >= axis_ratio * ady:
        return RobotCommand.LEFT if dx < 0 else RobotCommand.RIGHT
    return RobotCommand.FORWARD if dy < 0 else RobotCommand.BACK


def is_index_mostly_extended(
    keypoints_px: np.ndarray,
    index_vs_other_ratio: float = 1.15,
) -> bool:
    def dist(a: int, b: int) -> float:
        d = keypoints_px[a] - keypoints_px[b]
        return float(np.linalg.norm(d))

    index_len = dist(IDX_MCP, IDX_TIP)
    others = [
        dist(MID_MCP, MID_TIP),
        dist(RING_MCP, RING_TIP),
        dist(PINKY_MCP, PINKY_TIP),
    ]
    if index_len < 1e-3:
        return False
    return index_len >= max(others) * index_vs_other_ratio


# 21-pt hand skeleton edges.
HAND_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def render_handpose_debug(
    bgr: np.ndarray,
    keypoints_px: np.ndarray,
    command: RobotCommand,
) -> np.ndarray:
    """Debug overlay: bbox, skeleton, index finger line, cmd text."""
    out = bgr.copy()
    h, w = out.shape[:2]
    pts = keypoints_px.astype(np.int32)

    xs, ys = pts[:, 0], pts[:, 1]
    margin = 8
    x1 = max(0, int(np.min(xs)) - margin)
    y1 = max(0, int(np.min(ys)) - margin)
    x2 = min(w - 1, int(np.max(xs)) + margin)
    y2 = min(h - 1, int(np.max(ys)) + margin)
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 2)

    for a, b in HAND_SKELETON_EDGES:
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 1, cv2.LINE_AA)

    i0, i1 = IDX_MCP, IDX_TIP
    cv2.line(out, tuple(pts[i0]), tuple(pts[i1]), (0, 220, 255), 3, cv2.LINE_AA)
    cv2.circle(out, tuple(pts[i0]), 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(out, tuple(pts[i1]), 6, (0, 128, 255), -1, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if i not in (IDX_MCP, IDX_TIP):
            cv2.circle(out, tuple(p), 3, (180, 180, 180), -1, cv2.LINE_AA)

    cv2.putText(
        out,
        f"cmd: {command.value}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def robot_command_to_discrete_xyz(command: RobotCommand) -> tuple[float, float, float]:
    """Discrete compass in image plane for messaging (no sub-degree precision)."""
    if command == RobotCommand.NONE:
        return 0.0, 0.0, 0.0
    if command == RobotCommand.FORWARD:
        return 0.0, -1.0, 0.0
    if command == RobotCommand.BACK:
        return 0.0, 1.0, 0.0
    if command == RobotCommand.LEFT:
        return -1.0, 0.0, 0.0
    if command == RobotCommand.RIGHT:
        return 1.0, 0.0, 0.0
    return 0.0, 0.0, 0.0


@dataclass
class FrameResult:
    command: RobotCommand
    raw_output: np.ndarray
    keypoints_px: np.ndarray
    index_extended_ok: bool


class HandPoseNpuGesture:
    """Runs NPU handpose model and maps index finger to coarse RobotCommand."""

    def __init__(
        self,
        engine: Any,
        min_finger_len_px: float = 6.0,
        require_index_dominant: bool = False,
        index_vs_other_ratio: float = 1.15,
        direction_axis_ratio: float = 1.4,
    ) -> None:
        self._engine = engine
        self._min_finger_len_px = min_finger_len_px
        self._require_index_dominant = require_index_dominant
        self._index_vs_other_ratio = index_vs_other_ratio
        self._direction_axis_ratio = direction_axis_ratio

    def infer_frame(self, bgr: np.ndarray) -> FrameResult:
        h, w = bgr.shape[:2]
        tensor = preprocess_bgr(bgr)
        raw = self._engine.forward([tensor])[0]
        raw = np.asarray(raw).reshape(-1)
        kpts = raw_output_to_keypoints_pixels(raw, w, h)
        ext_ok = (
            is_index_mostly_extended(kpts, self._index_vs_other_ratio)
            if self._require_index_dominant
            else True
        )
        if not ext_ok:
            cmd = RobotCommand.NONE
        else:
            cmd = classify_index_finger_direction(
                kpts,
                self._min_finger_len_px,
                self._direction_axis_ratio,
            )
        return FrameResult(command=cmd, raw_output=raw, keypoints_px=kpts, index_extended_ok=ext_ok)

    def infer_dmabuf(self, fd: int, size: int, width: int, height: int,
                     stride: int | None = None, encoding: str = "nv12") -> FrameResult:
        """Zero-copy inference directly from dmabuf.

        Args:
            fd: dmabuf file descriptor
            size: total buffer size in bytes
            width: image width
            height: image height
            stride: line stride (bytes), same as width if not padded
            encoding: pixel format -- "nv12"/"yuv420sp", "rgb8", or "bgr8"

        Returns:
            FrameResult with hand pose keypoints and command

        Raises:
            ValueError: if encoding is not supported
        """
        enc = encoding.lower()
        if enc in ("nv12", "yuv420sp"):
            tensor = preprocess_nv12_dmabuf(fd, size, width, height, stride=stride)
        elif enc == "rgb8":
            tensor = preprocess_rgb_dmabuf(fd, size, width, height, stride=stride, rgb=True)
        elif enc == "bgr8":
            tensor = preprocess_rgb_dmabuf(fd, size, width, height, stride=stride, rgb=False)
        else:
            raise ValueError(
                f"Unsupported dmabuf encoding '{encoding}'; "
                "supported: nv12/yuv420sp, rgb8, bgr8"
            )

        raw = self._engine.forward([tensor])[0]
        raw = np.asarray(raw).reshape(-1)
        kpts = raw_output_to_keypoints_pixels(raw, width, height)
        ext_ok = (
            is_index_mostly_extended(kpts, self._index_vs_other_ratio)
            if self._require_index_dominant
            else True
        )
        if not ext_ok:
            cmd = RobotCommand.NONE
        else:
            cmd = classify_index_finger_direction(
                kpts,
                self._min_finger_len_px,
                self._direction_axis_ratio,
            )
        return FrameResult(command=cmd, raw_output=raw, keypoints_px=kpts, index_extended_ok=ext_ok)
