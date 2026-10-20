# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""CIX NOE YOLOv8-pose NPU inference (no ROS). Raw head C×8400 with C in {56, 135}."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

from cix_ros_noe_yolo.npu_yolo_coco_engine import find_onnx_yolov8_root

# (bbox_xyxy, score, class_id, keypoints (17,3) xyv in original image pixels)
PoseRow = Tuple[np.ndarray, float, int, np.ndarray]


class NpuYoloPoseV8Engine:
    """Loads a pose ``.cix`` and decodes 56- or 135-channel YOLOv8-pose heads."""

    def __init__(
        self,
        model_path: str,
        onnx_yolov8_root: str = "",
        conf_thr: float = 0.25,
        nms_thr: float = 0.45,
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
        self._nms_thr = float(nms_thr)

    def close(self) -> None:
        eng = getattr(self, "_engine", None)
        if eng is not None:
            try:
                eng.clean()
            except Exception:
                pass
            self._engine = None

    def detect_pose(self, cv_image: np.ndarray) -> List[PoseRow]:
        """Returns list of (bbox_xyxy, score, class_id, keypoints_xyv) in original image coordinates."""
        from yolo_postprocess.yolov8_pose_npu_decode import decode_pose_sdk, preprocess_yolov8_pose_from_bgr

        data, ratio, pad_w, pad_h = preprocess_yolov8_pose_from_bgr(cv_image)
        raw = np.asarray(self._engine.forward(data.astype(np.float32))[0], dtype=np.float32)
        raw = np.squeeze(raw)
        if raw.ndim == 1:
            if raw.size % 8400 != 0:
                raise RuntimeError(
                    f"pose forward output size {raw.size} not divisible by 8400; check .cix export"
                )
            c = raw.size // 8400
            pred_hw = raw.reshape(c, 8400)
        elif raw.ndim == 2:
            if raw.shape[1] == 8400:
                pred_hw = raw
            elif raw.shape[0] == 8400:
                pred_hw = raw.T
            else:
                raise RuntimeError(
                    f"pose forward shape {raw.shape}: expected (*, 8400) or (8400, *)"
                )
        else:
            raise RuntimeError(f"pose forward unexpected ndim={raw.ndim}")

        return decode_pose_sdk(
            pred_hw,
            self._conf_thr,
            self._nms_thr,
            cv_image.shape,
            ratio,
            pad_w,
            pad_h,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
