# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""CIX NOE YOLOv8-seg NPU inference (no ROS). Two-head output like ``seg_inference_npu.py``."""

from __future__ import annotations

import sys
from typing import Any, List, Tuple

import numpy as np

from cix_ros_noe_yolo.npu_yolo_coco_engine import find_onnx_yolov8_root

SegResult = Tuple[np.ndarray, list, np.ndarray]


class NpuYoloSegV8Engine:
    """Loads a YOLOv8-seg ``.cix`` (e.g. ``yolov8l_seg``): letterbox RGB input, two outputs, SDK postprocess."""

    def __init__(
        self,
        model_path: str,
        onnx_yolov8_root: str = "",
        conf_thr: float = 0.35,
        iou_thr: float = 0.45,
        nm: int = 32,
        proto_hw: int = 160,
        num_classes: int = 80,
    ) -> None:
        path = (model_path or "").strip()
        if not path:
            raise ValueError("model_path must be set to a .cix file.")
        onnx_root = find_onnx_yolov8_root(onnx_yolov8_root)
        if not onnx_root:
            raise RuntimeError(
                "Could not locate yolo_postprocess (set onnx_yolov8_root or install cix_ros_noe_yolo)."
            )
        if onnx_root not in sys.path:
            sys.path.insert(0, onnx_root)
        try:
            from NOE_Engine import EngineInfer
        except ImportError as e:
            raise RuntimeError(f"NPU import failed (NOE_Engine): {e}") from e
        try:
            self._engine = EngineInfer(path)
        except Exception as e:
            raise RuntimeError(f"EngineInfer({path}) failed: {e}") from e
        self._onnx_root = onnx_root
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)
        self._nm = int(nm)
        self._proto_hw = int(proto_hw)
        self._feat_c = 4 + int(num_classes) + self._nm

    def close(self) -> None:
        eng = getattr(self, "_engine", None)
        if eng is not None:
            try:
                eng.clean()
            except Exception:
                pass
            self._engine = None

    def detect_seg(self, cv_image: np.ndarray) -> SegResult:
        """Returns ``(boxes (N,6) xyxy+score+cls, segments, masks bool (N,H,W))`` in original image space."""
        from yolo_postprocess.segment_postprocess_yolov8 import postprocess_yolov8seg
        from yolo_postprocess.yolov8_pose_npu_decode import preprocess_yolov8_pose_from_bgr

        im, ratio, pad_w, pad_h = preprocess_yolov8_pose_from_bgr(cv_image)
        preds: Any = self._engine.forward([im])
        if not isinstance(preds, (list, tuple)) or len(preds) < 2:
            raise RuntimeError("seg model forward must return at least two tensors [det, proto]")
        p0 = np.asarray(preds[0], dtype=np.float32)
        p1 = np.asarray(preds[1], dtype=np.float32)
        p0 = np.reshape(p0, (1, self._feat_c, 8400))
        p1 = np.reshape(p1, (1, self._nm, self._proto_hw, self._proto_hw))
        boxes, segments, masks = postprocess_yolov8seg(
            [p0, p1],
            cv_image,
            ratio,
            pad_w,
            pad_h,
            self._conf_thr,
            self._iou_thr,
            nm=self._nm,
        )
        if isinstance(boxes, list) and len(boxes) == 0:
            h, w = cv_image.shape[:2]
            return (
                np.zeros((0, 6), dtype=np.float32),
                [],
                np.zeros((0, h, w), dtype=bool),
            )
        b = np.asarray(boxes, dtype=np.float32)
        m = np.asarray(masks, dtype=bool)
        return b, segments, m

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
