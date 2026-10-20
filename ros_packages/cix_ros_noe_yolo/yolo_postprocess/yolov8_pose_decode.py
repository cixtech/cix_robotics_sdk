# SPDX-License-Identifier: Apache-2.0
"""Decode YOLOv8-pose style NPU heads (56 or 135 channels × 8400 anchors) to boxes + 17 COCO keypoints.

**Note:** ``npu_yolo_pose_engine`` uses :mod:`yolov8_pose_npu_decode` (letterbox + RGB + ``scale_boxes_kpts``)
to match ``inference_npu.py``. This module’s ``decode_pose_predictions`` assumes stretch-resize 640×640
(legacy); prefer the NPU decode path for ``yolov8s-pose.cix``.

Layouts (common Ultralytics-style raw heads):
  - 56 = 4 (xywh) + 1 (person score) + 17*3 (kpt x,y,v raw)
  - 135 = 4 (xywh) + 80 (class logits) + 17*3 (kpt x,y,v raw)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .object_detect_postprocess import xywh2xyxy

# COCO-17 skeleton pairs (index into 17 keypoints) for drawing lines
COCO17_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def decode_pose_predictions(
    pred_hw: np.ndarray,
    conf_thr: float,
    nms_thr: float,
    src_wh: Tuple[int, int],
    model_wh: Tuple[int, int] = (640, 640),
) -> List[Tuple[np.ndarray, float, int, np.ndarray]]:
    """
    Args:
        pred_hw: array shaped (C, 8400) after forward (C in {56, 135}).
        conf_thr, nms_thr: thresholds.
        src_wh: (H, W) original image.
        model_wh: (H, W) model input (default 640).

    Returns:
        List of (bbox_xyxy_orig, score, class_id, keypoints_xyv_orig)
        keypoints_xyv_orig shape (17, 3) float32 — x,y in original pixels, v in [0,1].
    """
    c, a = pred_hw.shape[0], pred_hw.shape[1]
    if a != 8400:
        raise ValueError(f"expected 8400 anchors, got {a}")
    if c not in (56, 135):
        raise ValueError(f"unsupported pose channel count C={c}; expected 56 or 135")

    p = pred_hw.T.astype(np.float32)  # (8400, C)

    xywh = p[:, :4]
    if c == 56:
        cls_scores = p[:, 4:5]
        kraw = p[:, 5:56].reshape(-1, 17, 3)
        conf = _sigmoid(cls_scores[:, 0])
        cls_id = np.zeros(len(conf), dtype=np.int32)
    else:
        cls_logits = p[:, 4:84]
        kraw = p[:, 84:135].reshape(-1, 17, 3)
        cls_probs = _sigmoid(cls_logits)
        conf = np.max(cls_probs, axis=1)
        cls_id = np.argmax(cls_probs, axis=1).astype(np.int32)

    m = conf > conf_thr
    if not np.any(m):
        return []

    xywh_f = xywh[m]
    conf_f = conf[m]
    cls_f = cls_id[m]
    kraw_f = kraw[m]

    xyxy = xywh2xyxy(xywh_f)
    boxes = np.asarray(xyxy, dtype=np.float32)

    h0, w0 = float(model_wh[0]), float(model_wh[1])
    h1, w1 = float(src_wh[0]), float(src_wh[1])
    sx = w1 / w0
    sy = h1 / h0

    wh_list = []
    for i in range(len(xywh_f)):
        x, y, ww, hh = [float(v) for v in xywh_f[i]]
        x1 = x - ww / 2.0
        y1 = y - hh / 2.0
        wh_list.append([x1, y1, ww, hh])
    idx = cv2.dnn.NMSBoxes(wh_list, conf_f.tolist(), conf_thr, nms_thr)
    if idx is None or len(idx) == 0:
        return []
    keep = np.asarray(idx).flatten()

    out: List[Tuple[np.ndarray, float, int, np.ndarray]] = []
    for j in keep:
        j = int(j)
        bb = boxes[j].copy()
        bb[0] *= sx
        bb[1] *= sy
        bb[2] *= sx
        bb[3] *= sy
        kpt = kraw_f[j].copy()
        kpt[:, 0] *= sx
        kpt[:, 1] *= sy
        kpt[:, 2] = _sigmoid(kpt[:, 2])
        out.append((bb.astype(np.float32), float(conf_f[j]), int(cls_f[j]), kpt.astype(np.float32)))
    return out
