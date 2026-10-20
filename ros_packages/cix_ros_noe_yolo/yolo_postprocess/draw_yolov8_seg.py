# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""Minimal YOLOv8-seg overlay (from onnx_yolov8_n ``draw.draw_yolov8seg``)."""

from __future__ import annotations

import cv2
import numpy as np

from .label.coco_classes import COCO_CLASSES


class Colors:
    def __init__(self):
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


def draw_yolov8seg(
    im: np.ndarray,
    bboxes: list,
    segments: list,
    color_palette=None,
) -> np.ndarray:
    if color_palette is None:
        color_palette = Colors()
    im_canvas = im.copy()
    for (*box, conf, cls_), segment in zip(bboxes, segments):
        if segment is None or len(segment) < 3:
            continue
        cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)
        cv2.fillPoly(im_canvas, np.int32([segment]), color_palette(int(cls_), bgr=True))
        cv2.rectangle(
            im,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color_palette(int(cls_), bgr=True),
            1,
            cv2.LINE_AA,
        )
        cn = COCO_CLASSES[int(cls_)] if 0 <= int(cls_) < len(COCO_CLASSES) else str(int(cls_))
        cv2.putText(
            im,
            f"{cn}: {conf:.3f}",
            (int(box[0]), int(max(box[1] - 9, 12))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_palette(int(cls_), bgr=True),
            2,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
