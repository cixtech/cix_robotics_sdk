# SPDX-License-Identifier: Apache-2.0
"""YOLOv8-pose NPU decode aligned with CIX ``inference_npu.py`` / ``preprocess_yolov8seg`` + ``scale_boxes_kpts``.

Letterbox to 640×640, BGR→RGB, /255; raw head ``(C, 8400)`` with C in {56, 135}; NMS and map back
using ``pad_w``, ``pad_h``, and ``ratio`` (same as onnx_yolov8_n utils).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .object_detect_postprocess import xywh2xyxy

# (bbox_xyxy, score, class_id, keypoints (17,3) x,y in original image, v visibility)
PoseRow = Tuple[np.ndarray, float, int, np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def preprocess_yolov8_pose_from_bgr(
    img_bgr: np.ndarray,
    dtype=np.float32,
    model_height: int = 640,
    model_width: int = 640,
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    """
    Same geometry and normalization as ``utils.image_process.preprocess_yolov8seg``,
    but accepts a BGR ``uint8`` image array (ROS / OpenCV).

    Returns:
        img_nchw: (1, 3, H, W) float32 RGB, /255
        ratio: (r, r) scale applied to (H, W) before padding
        pad_w, pad_h: letterbox padding (floats, half each side convention matches SDK)
    """
    shape = img_bgr.shape[:2]
    new_shape = (model_height, model_width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    pad_w = (new_shape[1] - new_unpad[0]) / 2.0
    pad_h = (new_shape[0] - new_unpad[1]) / 2.0
    img = img_bgr
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    # HWC BGR -> CHW RGB, /255 (matches preprocess_yolov8seg)
    img_chw = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=dtype) / 255.0
    img_nchw = np.expand_dims(img_chw, axis=0).astype(np.float32)
    return img_nchw, ratio, pad_w, pad_h


def nms_keypoints_indices(dets: np.ndarray, iou_thresh: float) -> np.ndarray:
    """NMS on xyxy boxes in first four columns; scores in column 4. Returns row indices into ``dets``."""
    if dets.size == 0:
        return np.zeros((0,), dtype=np.int32)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return np.asarray(keep, dtype=np.int32)


def scale_boxes_kpts(
    boxes: np.ndarray,
    ratio: Tuple[float, float],
    shape: tuple,
    pad_w: float,
    pad_h: float,
) -> np.ndarray:
    """Same as ``utils.pose_postprocess.scale_boxes_kpts`` (expects xywh in cols 0–3, 56 columns total)."""
    gain = ratio[0]
    pad = (pad_w, pad_h)
    out = boxes.astype(np.float32, copy=True)
    out[:, 0] -= pad[0]
    out[:, 1] -= pad[1]
    out[:, :4] /= gain
    num_kpts = out.shape[1] // 3
    for kid in range(2, num_kpts + 1):
        out[:, kid * 3 - 1] = (out[:, kid * 3 - 1] - pad[0]) / gain
        out[:, kid * 3] = (out[:, kid * 3] - pad[1]) / gain
    h_lim, w_lim = int(shape[0]), int(shape[1])
    top_left_x = out[:, 0].clip(0, w_lim)
    top_left_y = out[:, 1].clip(0, h_lim)
    bottom_right_x = (out[:, 0] + out[:, 2]).clip(0, w_lim)
    bottom_right_y = (out[:, 1] + out[:, 3]).clip(0, h_lim)
    out[:, 0] = top_left_x
    out[:, 1] = top_left_y
    out[:, 2] = bottom_right_x
    out[:, 3] = bottom_right_y
    return out


def decode_pose_sdk(
    pred_hw: np.ndarray,
    conf_thr: float,
    nms_thr: float,
    image_shape: tuple,
    ratio: Tuple[float, float],
    pad_w: float,
    pad_h: float,
) -> List[PoseRow]:
    """
    Decode raw pose head using the same steps as ``inference_npu.py``.

    Args:
        pred_hw: (C, 8400) with C in {56, 135}.
        image_shape: ``(H, W)`` or ``(H, W, C)`` (uses first two dims for clipping).
    """
    c, a = pred_hw.shape[0], pred_hw.shape[1]
    if a != 8400:
        raise ValueError(f"expected 8400 anchors, got {a}")
    if c not in (56, 135):
        raise ValueError(f"unsupported pose channel count C={c}; expected 56 or 135")

    p = pred_hw.T.astype(np.float32)  # (8400, C)

    if c == 56:
        m = p[:, 4] > conf_thr
    else:
        cls_logits = p[:, 4:84]
        cls_probs = _sigmoid(cls_logits)
        conf_all = np.max(cls_probs, axis=1)
        m = conf_all > conf_thr

    if not np.any(m):
        return []

    pred_full = p[m].copy()
    cls_kept = (
        np.zeros(pred_full.shape[0], dtype=np.int32)
        if c == 56
        else np.argmax(_sigmoid(pred_full[:, 4:84]), axis=1).astype(np.int32)
    )

    # ``scale_boxes_kpts`` only supports the 56-wide layout (see onnx_yolov8_n pose_postprocess).
    work = np.zeros((pred_full.shape[0], 56), dtype=np.float32)
    work[:, :4] = pred_full[:, :4]
    if c == 56:
        work[:, 4] = pred_full[:, 4]
        work[:, 5:56] = pred_full[:, 5:56]
    else:
        cls_probs = _sigmoid(pred_full[:, 4:84])
        work[:, 4] = np.max(cls_probs, axis=1).astype(np.float32)
        work[:, 5:56] = pred_full[:, 84:135]

    bboxs_xyxy = xywh2xyxy(work)
    keep_idx = nms_keypoints_indices(bboxs_xyxy, nms_thr)
    if keep_idx.size == 0:
        return []

    sel = bboxs_xyxy[keep_idx]
    cls_out = cls_kept[keep_idx]

    sel[:, 2] = sel[:, 2] - sel[:, 0]
    sel[:, 3] = sel[:, 3] - sel[:, 1]
    scaled = scale_boxes_kpts(sel, ratio, image_shape, pad_w, pad_h)

    out_rows: List[PoseRow] = []
    for i in range(scaled.shape[0]):
        row = scaled[i]
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        score = float(row[4])
        kflat = row[5:56].astype(np.float32)
        kpt = kflat.reshape(17, 3)
        kpt[:, 2] = _sigmoid(kpt[:, 2])
        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
        out_rows.append((bb, score, int(cls_out[i]), kpt))
    return out_rows
