# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""YOLOv8 instance-seg postprocess (subset of onnx_yolov8_n ``segment_postprocess`` for NPU two-head output)."""

from __future__ import annotations

import cv2
import numpy as np


def masks2segments(masks: np.ndarray) -> list:
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype("float32"))
    return segments


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_mask(masks: np.ndarray, im0_shape: tuple, ratio_pad: tuple | None = None) -> np.ndarray:
    im1_shape = masks.shape[:2]
    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        pad = ratio_pad[1]
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def process_mask(
    protos: np.ndarray,
    masks_in: np.ndarray,
    bboxes: np.ndarray,
    im0_shape: tuple,
) -> np.ndarray:
    c, mh, mw = protos.shape
    masks = (
        np.matmul(masks_in, protos.reshape((c, -1)))
        .reshape((-1, mh, mw))
        .transpose(1, 2, 0)
    )
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(masks, im0_shape)
    masks = np.einsum("HWN -> NHW", masks)
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)


def postprocess_yolov8seg(
    preds: list,
    im0: np.ndarray,
    ratio: float | tuple,
    pad_w: float,
    pad_h: float,
    conf_threshold: float,
    iou_threshold: float,
    nm: int = 32,
):
    """Same logic as onnx_yolov8_n ``segment_postprocess.postprocess_yolov8seg`` (``ratio`` may be ``(r, r)``)."""
    x, protos = preds[0], preds[1]
    x = np.einsum("bcn->bnc", x)
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
    if len(x) == 0:
        return [], [], []
    x = np.c_[
        x[..., :4],
        np.amax(x[..., 4:-nm], axis=-1),
        np.argmax(x[..., 4:-nm], axis=-1),
        x[..., -nm:],
    ]
    idx = cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)
    if idx is None or len(idx) == 0:
        return [], [], []
    x = x[np.asarray(idx).reshape(-1)]
    if len(x) == 0:
        return [], [], []
    x[..., [0, 1]] -= x[..., [2, 3]] / 2
    x[..., [2, 3]] += x[..., [0, 1]]
    x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
    rmin = min(ratio) if isinstance(ratio, (tuple, list)) else float(ratio)
    x[..., :4] /= rmin
    x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
    x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
    masks = process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
    segments = masks2segments(masks)
    return x[..., :6], segments, masks
