# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""NPU YOLO-OBB inference (``yolo_obb.cix``), optimized per ``inference_npu 1.py``."""

from __future__ import annotations

import sys
import time
from typing import List, Sequence, Tuple

import numpy as np

from cix_ros_noe_yolo.npu_yolo_coco_engine import find_onnx_yolov8_root

try:
    from libnoe import NOE_TENSOR_TYPE_OUTPUT
except ImportError:
    NOE_TENSOR_TYPE_OUTPUT = 1  # type: ignore[misc, assignment]

# (corners 4x2 float32, class_name, confidence)
ObbDetectionRow = Tuple[np.ndarray, str, float]


class NpuYoloObbEngine:
    """
    Loads a YOLO-OBB ``.cix`` (e.g. ``yolo_obb.cix``) via NOE ``EngineInfer``.

    Output head: ``(4+nc+1, num_anchors)`` — bolt single-class: ``(6, 8400)``.

    Reuses a single NPU job across frames (no per-frame create/clean).
    """

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str] = ("bolt",),
        conf_thr: float = 0.90,
        nms_thr: float = 0.40,
        num_classes: int | None = None,
        input_size: tuple[int, int] = (640, 640),
        profile_timing: bool = False,
        onnx_yolov8_root: str = "",
    ) -> None:
        path = (model_path or "").strip()
        if not path:
            raise ValueError("model_path must point to a .cix OBB model (e.g. yolo_obb.cix).")

        onnx_root = find_onnx_yolov8_root(onnx_yolov8_root)
        if not onnx_root:
            raise RuntimeError(
                "Could not locate yolo_postprocess (set onnx_yolov8_root or install cix_ros_noe_yolo)."
            )
        if onnx_root not in sys.path:
            sys.path.insert(0, onnx_root)
        try:
            from yolo_postprocess.npu_yolo_obb_postprocess import (
                boxes7_to_corners,
                letterbox_preprocess,
                postprocess_obb,
            )
        except ImportError as e:
            raise RuntimeError(f"OBB postprocess import failed: {e}") from e

        self._boxes7_to_corners = boxes7_to_corners
        self._letterbox_preprocess = letterbox_preprocess
        self._postprocess_obb = postprocess_obb

        self._class_names = tuple(class_names)
        if not self._class_names:
            raise ValueError("class_names must not be empty for OBB models.")
        self._num_classes = (
            num_classes if num_classes is not None else len(self._class_names)
        )
        self._conf_thr = float(conf_thr)
        self._nms_thr = float(nms_thr)
        self._input_size = input_size
        self._dtype = np.float32
        self._profile_timing = profile_timing
        self._last_timing_ms: dict[str, float] = {}
        self._job_id: int | None = None

        try:
            from NOE_Engine import EngineInfer
        except ImportError as e:
            raise RuntimeError(f"NOE_Engine import failed: {e}") from e

        try:
            self._engine = EngineInfer(path)
        except Exception as e:
            raise RuntimeError(f"EngineInfer({path}) failed: {e}") from e

        self._output_channels = 4 + self._num_classes + 1
        out_shape = (
            self._engine.get_output_shape(0)
            if hasattr(self._engine, "get_output_shape")
            else None
        )
        self._output_anchors = int(out_shape[-1]) if out_shape is not None else 8400
        self._create_reused_job()

    @property
    def last_timing_ms(self) -> dict[str, float]:
        """Last ``detect()`` breakdown (ms): preprocess, npu*, postprocess, total."""
        return dict(self._last_timing_ms)

    @property
    def npu_infer_fps(self) -> float:
        """Instant NPU sync-infer FPS from ``EngineInfer.get_cur_fps()`` (last forward)."""
        eng = getattr(self, "_engine", None)
        if eng is None or not hasattr(eng, "get_cur_fps"):
            return 0.0
        try:
            dur = getattr(eng, "_dur_time", 0.0)
            if dur <= 0.0:
                return 0.0
            return float(eng.get_cur_fps())
        except Exception:
            return 0.0

    def _create_reused_job(self) -> None:
        eng = self._engine
        ret, job_id = eng.npu.noe_create_job(eng.graph_id, eng.job_cfg)
        if ret != 0:
            raise RuntimeError("npu: noe_create_job failed")
        self._job_id = job_id
        eng.job_id = job_id

    def _destroy_reused_job(self) -> None:
        if self._job_id is None:
            return
        eng = getattr(self, "_engine", None)
        if eng is not None:
            try:
                eng.npu.noe_clean_job(self._job_id)
            except Exception:
                pass
        self._job_id = None

    def close(self) -> None:
        self._destroy_reused_job()
        eng = getattr(self, "_engine", None)
        if eng is not None:
            try:
                eng.clean()
            except Exception:
                pass
            self._engine = None

    def _quantize_input(self, eng, index: int, input_data: np.ndarray) -> np.ndarray:
        quantized = np.round(
            input_data.astype(np.float32) * eng.in_tensor_desc[index].scale
            - eng.in_tensor_desc[index].zero_point
        )
        return np.clip(
            quantized, eng.input_dtype_min[index], eng.input_dtype_max[index]
        ).astype(eng.input_type[index])

    def _forward_reuse(self, blob: np.ndarray) -> np.ndarray:
        """Run NPU infer on a persistent job (load → sync infer → get output)."""
        eng = self._engine
        job_id = self._job_id
        if job_id is None:
            raise RuntimeError("NPU job not initialized")

        t_load0 = time.perf_counter()
        input_data = self._quantize_input(eng, 0, blob)
        if len(input_data.tobytes()) != eng.in_tensor_desc[0].size:
            raise RuntimeError(
                f"input byte size {len(input_data.tobytes())} != "
                f"expected {eng.in_tensor_desc[0].size}"
            )
        eng.npu.noe_load_tensor(job_id, 0, input_data.tobytes())
        t_load1 = time.perf_counter()

        eng.npu.noe_job_infer_sync(job_id, -1)
        t_infer1 = time.perf_counter()
        eng._dur_time = t_infer1 - t_load1
        eng._dur_time_list.append(eng._dur_time)
        eng._acc_time += eng._dur_time
        eng._cnt_time += 1

        retmap, data = eng.npu.noe_get_tensor(job_id, NOE_TENSOR_TYPE_OUTPUT, 0)
        if retmap != 0:
            raise RuntimeError("npu: noe_get_tensor failed")
        output_data = np.frombuffer(data, dtype=eng.output_type[0])
        raw = (
            output_data.astype(np.float32) + eng.out_tensor_desc[0].zero_point
        ) / eng.out_tensor_desc[0].scale
        t_get1 = time.perf_counter()

        if self._profile_timing:
            load_ms = (t_load1 - t_load0) * 1000.0
            infer_ms = (t_infer1 - t_load1) * 1000.0
            get_ms = (t_get1 - t_infer1) * 1000.0
            self._last_timing_ms["npu_load"] = load_ms
            self._last_timing_ms["npu_infer"] = infer_ms
            self._last_timing_ms["npu_get"] = get_ms
            self._last_timing_ms["npu"] = load_ms + infer_ms + get_ms

        return raw

    def _forward_boxes7(self, cv_image: np.ndarray) -> np.ndarray:
        h, w = self._input_size
        t0 = time.perf_counter()
        blob, gain, pad = self._letterbox_preprocess(
            cv_image, model_height=h, model_width=w, dtype=self._dtype
        )
        t1 = time.perf_counter()
        raw = self._forward_reuse(blob)
        pred = raw.reshape(self._output_channels, self._output_anchors)
        t2 = time.perf_counter()
        boxes = self._postprocess_obb(
            pred,
            self._conf_thr,
            self._nms_thr,
            num_classes=self._num_classes,
            letterbox_pad=pad,
            letterbox_gain=gain,
        )
        t3 = time.perf_counter()
        if self._profile_timing:
            self._last_timing_ms["preprocess"] = (t1 - t0) * 1000.0
            self._last_timing_ms.setdefault("npu", (t2 - t1) * 1000.0)
            self._last_timing_ms["postprocess"] = (t3 - t2) * 1000.0
            self._last_timing_ms["total"] = (t3 - t0) * 1000.0
        return boxes

    def detect(self, cv_image: np.ndarray) -> List[ObbDetectionRow]:
        """Returns rows with true OBB corners (4,2), class name, confidence."""
        boxes_orig = self._forward_boxes7(cv_image)

        rows: List[ObbDetectionRow] = []
        for corners, box in zip(self._boxes7_to_corners(boxes_orig), boxes_orig):
            cls_id = int(box[5])
            if cls_id < 0 or cls_id >= len(self._class_names):
                continue
            rows.append((corners, self._class_names[cls_id], float(box[4])))
        return rows

    def detect_boxes7(self, cv_image: np.ndarray) -> np.ndarray:
        """Raw (N,7) boxes in original image coords — for debug drawing."""
        return self._forward_boxes7(cv_image)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
