# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""CIX NOE YOLOv8 COCO NPU inference (no ROS)."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# Detection row: (bbox_xyxy ndarray shape (4,), class_name str, confidence float)
DetectionRow = Tuple[np.ndarray, str, float]


POSTPROCESS_SUBDIR = "yolo_postprocess"
_LEGACY_POSTPROCESS_SUBDIR = "utils"


def _has_postprocess_dir(root: str) -> bool:
    return os.path.isdir(os.path.join(root, POSTPROCESS_SUBDIR)) or os.path.isdir(
        os.path.join(root, _LEGACY_POSTPROCESS_SUBDIR)
    )


def find_onnx_yolov8_root(explicit: str) -> str:
    """Directory that contains ``yolo_postprocess/`` (postprocess + labels). May be package share or legacy onnx_yolov8_n."""
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("cix_ros_noe_yolo")
        if _has_postprocess_dir(share):
            return share
        legacy = os.path.join(share, "onnx_yolov8_n")
        if os.path.isdir(legacy) and _has_postprocess_dir(legacy):
            return legacy
    except Exception:
        pass
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if _has_postprocess_dir(str(p)):
            return str(p)
        legacy = p / "onnx_yolov8_n"
        if legacy.is_dir() and _has_postprocess_dir(str(legacy)):
            return str(legacy)
    return ""


class NpuYoloCocoEngine:
    """
    Loads a COCO detection ``.cix`` via NOE EngineInfer and runs postprocess from ``yolo_postprocess/``.
    """

    def __init__(
        self,
        model_path: str,
        onnx_yolov8_root: str = "",
        conf_thr: float = 0.25,
        nms_thr: float = 0.45,
    ) -> None:
        path = (model_path or "").strip()
        if not path:
            raise ValueError("model_path must be set to yolov8n.cix (absolute path recommended).")
        onnx_root = find_onnx_yolov8_root(onnx_yolov8_root)
        if not onnx_root:
            raise RuntimeError(
                "Could not locate yolo_postprocess (set onnx_yolov8_root to a directory containing yolo_postprocess/, or install cix_ros_noe_yolo)."
            )
        if onnx_root not in sys.path:
            sys.path.insert(0, onnx_root)
        try:
            from NOE_Engine import EngineInfer
            from yolo_postprocess.object_detect_postprocess import clip_boxes, postprocess_yolo, xywh2xyxy
            from yolo_postprocess.label.coco_classes import COCO_CLASSES
        except ImportError as e:
            raise RuntimeError(f"NPU import failed (NOE_Engine / yolo_postprocess): {e}") from e
        try:
            self._engine = EngineInfer(path)
        except Exception as e:
            raise RuntimeError(f"EngineInfer({path}) failed: {e}") from e
        self._postprocess_yolo = postprocess_yolo
        self._xywh2xyxy = xywh2xyxy
        self._clip_boxes = clip_boxes
        self._coco_classes: Sequence[str] = COCO_CLASSES
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

    @staticmethod
    def preprocess_bgr640(cv_image: np.ndarray):
        """Letterbox to 640x640 (constant 114 padding), BGR NCHW float32 blob.

        Returns (src_shape, data, gain, (pad_w, pad_h)); ``gain``/``pad`` map
        letterbox coordinates back to source pixels (see :meth:`detect`).
        """
        src_shape = cv_image.shape[:2]  # (h, w)
        new_shape = (640, 640)  # (h, w)
        gain = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])
        new_unpad = int(round(src_shape[1] * gain)), int(round(src_shape[0] * gain))
        pad_w = (new_shape[1] - new_unpad[0]) / 2
        pad_h = (new_shape[0] - new_unpad[1]) / 2

        img = cv_image
        if src_shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        tensor = img.astype(np.float32) / 255.0
        data = np.expand_dims(tensor.transpose(2, 0, 1), axis=0).astype(np.float32)
        return src_shape, data, gain, (pad_w, pad_h)

    def detect(self, cv_image: np.ndarray) -> List[DetectionRow]:
        """Returns rows (bbox_xyxy, class_name, conf) in original image coordinates."""
        src_shape, data, gain, (pad_w, pad_h) = self.preprocess_bgr640(cv_image)
        pred = self._engine.forward(data.astype(np.float32))[0]
        pred = np.reshape(pred, (84, 8400))
        pred = np.transpose(pred, (1, 0))
        raw = self._postprocess_yolo(pred, self._conf_thr, self._nms_thr)
        if raw is None or len(raw) == 0:
            return []
        if raw.ndim == 1:
            raw = np.expand_dims(raw, 0)
        bbox_xywh = raw[:, :4]
        bbox_xyxy = self._xywh2xyxy(bbox_xywh)
        # Undo the letterbox: remove padding, scale back, clip to source frame
        bbox_xyxy = (bbox_xyxy - np.array([pad_w, pad_h, pad_w, pad_h], dtype=np.float32)) / gain
        self._clip_boxes(bbox_xyxy, src_shape)

        rows: List[DetectionRow] = []
        nclass = len(self._coco_classes)
        for i in range(len(bbox_xyxy)):
            conf = float(raw[i, 4])
            cls_id = int(raw[i, 5])
            if cls_id < 0 or cls_id >= nclass:
                continue
            class_name = self._coco_classes[cls_id]
            rows.append((bbox_xyxy[i], class_name, conf))
        return rows

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
