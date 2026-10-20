# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""YOLO-OBB NPU postprocess (optimized, aligned with ``inference_npu 1.py``)."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

_INV255_F32 = np.float32(1.0 / 255.0)
_PI_2 = np.float32(0.5 * math.pi)
_PI_34 = np.float32(0.75 * math.pi)
_RAD2DEG_F32 = np.float32(180.0 / math.pi)
_MAX_NMS = 300
_EMPTY_BOXES = np.zeros((0, 7), dtype=np.float32)


def letterbox_preprocess(
    image_bgr: np.ndarray,
    model_height: int = 640,
    model_width: int = 640,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Returns (NCHW blob, letterbox gain, (pad_w, pad_h))."""
    shape = image_bgr.shape[:2]
    new_shape = (model_height, model_width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w = (new_shape[1] - new_unpad[0]) / 2
    pad_h = (new_shape[0] - new_unpad[1]) / 2

    img = image_bgr
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    blob = np.empty((1, 3, img.shape[0], img.shape[1]), dtype=dtype)
    if dtype == np.float32:
        blob[0, 0] = img[:, :, 2].astype(np.float32, copy=False) * _INV255_F32
        blob[0, 1] = img[:, :, 1].astype(np.float32, copy=False) * _INV255_F32
        blob[0, 2] = img[:, :, 0].astype(np.float32, copy=False) * _INV255_F32
    else:
        blob[0, 0] = (img[:, :, 2] * _INV255_F32).astype(dtype)
        blob[0, 1] = (img[:, :, 1] * _INV255_F32).astype(dtype)
        blob[0, 2] = (img[:, :, 0] * _INV255_F32).astype(dtype)
    return blob, r, (pad_w, pad_h)


def _nms_xywh_indices(xywh: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray | None:
    n = xywh.shape[0]
    if n == 0:
        return None
    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    tlwh = np.empty((n, 4), dtype=np.float32)
    tlwh[:, 0] = cx - bw * 0.5
    tlwh[:, 1] = cy - bh * 0.5
    tlwh[:, 2] = bw
    tlwh[:, 3] = bh
    idx = cv2.dnn.NMSBoxes(tlwh, scores, score_threshold=0.0, nms_threshold=iou_thr)
    if idx is None or len(idx) == 0:
        return None
    return np.asarray(idx, dtype=np.intp).reshape(-1)


def _scale_boxes_inplace(
    boxes: np.ndarray, pad: tuple[float, float], gain: float
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    pad_w, pad_h = pad
    boxes[:, 0] -= pad_w
    boxes[:, 1] -= pad_h
    boxes[:, :4] /= gain
    return boxes


def _postprocess_obb_single_class(
    pred: np.ndarray,
    conf_thr: float,
    iou_thr: float,
    letterbox_pad: tuple[float, float] | None,
    letterbox_gain: float | None,
) -> np.ndarray:
    scores = pred[4]
    idx = np.flatnonzero(scores > conf_thr)
    if idx.size == 0:
        return _EMPTY_BOXES

    if idx.size > _MAX_NMS:
        sub = scores[idx]
        idx = idx[np.argpartition(sub, -_MAX_NMS)[-_MAX_NMS:]]

    m = idx.size
    boxes = np.empty((m, 7), dtype=np.float32)
    boxes[:, 0] = pred[0, idx]
    boxes[:, 1] = pred[1, idx]
    boxes[:, 2] = pred[2, idx]
    boxes[:, 3] = pred[3, idx]
    boxes[:, 4] = scores[idx]
    boxes[:, 5] = 0.0
    angles = pred[5, idx]
    mask = (angles >= _PI_2) & (angles <= _PI_34)
    boxes[:, 6] = np.where(mask, angles - np.pi, angles) * _RAD2DEG_F32

    keep = _nms_xywh_indices(boxes[:, :4], boxes[:, 4], iou_thr)
    if keep is None:
        return _EMPTY_BOXES
    boxes = boxes[keep]

    if letterbox_pad is not None and letterbox_gain is not None:
        _scale_boxes_inplace(boxes, letterbox_pad, letterbox_gain)
    return boxes


def _pred_as_channels_first(outputs: np.ndarray, num_ch: int) -> np.ndarray:
    pred = np.squeeze(outputs)
    if pred.ndim == 3:
        pred = pred.reshape(num_ch, -1)
    elif pred.ndim != 2:
        raise ValueError(f"expected 2D output (C, A), got shape={outputs.shape}")
    if pred.shape[0] == num_ch:
        return pred
    if pred.shape[1] == num_ch:
        return pred.T
    raise ValueError(
        f"output channels {pred.shape} != 4+nc+1={num_ch}; check num_classes or export"
    )


def postprocess_obb(
    outputs: np.ndarray,
    conf_thr: float,
    iou_thr: float,
    num_classes: int = 1,
    letterbox_pad: tuple[float, float] | None = None,
    letterbox_gain: float | None = None,
) -> np.ndarray:
    """
    YOLO OBB: filter → angle norm → NMS → optional map to original image coords.

    ``outputs`` may be (C, A) or (1, C, A). With ``letterbox_pad`` / ``letterbox_gain``
    from ``letterbox_preprocess``, scaling is done in one pass.
    """
    num_ch = 4 + num_classes + 1
    pred = _pred_as_channels_first(outputs, num_ch)

    if num_classes == 1:
        return _postprocess_obb_single_class(
            pred, conf_thr, iou_thr, letterbox_pad, letterbox_gain
        )

    cls_scores = pred[4 : 4 + num_classes]
    class_ids = cls_scores.argmax(axis=0)
    scores = cls_scores[class_ids, np.arange(pred.shape[1])]
    idx = np.flatnonzero(scores > conf_thr)
    if idx.size == 0:
        return _EMPTY_BOXES
    if idx.size > _MAX_NMS:
        sub = scores[idx]
        idx = idx[np.argpartition(sub, -_MAX_NMS)[-_MAX_NMS:]]

    m = idx.size
    boxes = np.empty((m, 7), dtype=np.float32)
    boxes[:, 0] = pred[0, idx]
    boxes[:, 1] = pred[1, idx]
    boxes[:, 2] = pred[2, idx]
    boxes[:, 3] = pred[3, idx]
    boxes[:, 4] = scores[idx]
    boxes[:, 5] = class_ids[idx].astype(np.float32)
    angles = pred[-1, idx]
    mask = (angles >= _PI_2) & (angles <= _PI_34)
    boxes[:, 6] = np.where(mask, angles - np.pi, angles) * _RAD2DEG_F32

    keep = _nms_xywh_indices(boxes[:, :4], boxes[:, 4], iou_thr)
    if keep is None:
        return _EMPTY_BOXES
    boxes = boxes[keep]

    if letterbox_pad is not None and letterbox_gain is not None:
        _scale_boxes_inplace(boxes, letterbox_pad, letterbox_gain)
    return boxes


def scale_obb_to_original(
    boxes: np.ndarray,
    img_shape: tuple[int, int],
    input_shape: tuple[int, int] = (640, 640),
) -> np.ndarray:
    """Legacy helper when pad/gain were not passed into ``postprocess_obb``."""
    if boxes.size == 0:
        return boxes

    h, w = img_shape[:2]
    gain = min(input_shape[0] / h, input_shape[1] / w)
    pad_w = (input_shape[1] - w * gain) / 2
    pad_h = (input_shape[0] - h * gain) / 2

    out = boxes.copy()
    return _scale_boxes_inplace(out, (pad_w, pad_h), gain)


def boxes7_to_corners(boxes_7: np.ndarray) -> list[np.ndarray]:
    """(N,7) OBB rows → list of (4,2) corner arrays in image pixels."""
    corners_list: list[np.ndarray] = []
    for box in boxes_7:
        pts = cv2.boxPoints(
            (
                (float(box[0]), float(box[1])),
                (float(box[2]), float(box[3])),
                float(box[6]),
            )
        )
        corners_list.append(np.asarray(pts, dtype=np.float32))
    return corners_list


def draw_obb_boxes(
    image_bgr: np.ndarray,
    boxes_7: np.ndarray,
    class_names: Sequence[str],
) -> np.ndarray:
    vis = image_bgr.copy()
    for box in boxes_7:
        cx, cy, bw, bh, score, cls_id, angle_deg = box
        points = cv2.boxPoints(((cx, cy), (bw, bh), angle_deg)).astype(np.int32)
        cv2.polylines(vis, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
        cv2.putText(
            vis,
            f"{name} {score:.2f}",
            (int(points[0][0]), int(points[0][1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis
