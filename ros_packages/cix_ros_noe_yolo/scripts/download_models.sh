#!/bin/bash
# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
# Download the precompiled YOLO `.cix` NPU models for cix_ros_noe_yolo
# from the CIX AI model hub on ModelScope.
#
# The models are NOT distributed with this repository or the SDK deb
# (they are derived from Ultralytics checkpoints, license AGPL-3.0).
# Downloading them accepts that license; see the repo NOTICE.md.
# (The single-class bolt OBB model yolo_obb.cix is the exception: it ships
# with the downstream standalone demo repository that uses it instead.)
#
# Works from two locations:
#   - a robotics_sdk checkout:      ros_packages/cix_ros_noe_yolo/scripts/
#   - an installed SDK package:     <prefix>/share/cix_ros_noe_yolo/scripts/
#     (may need sudo: the default DEST is the share dir, root-owned)
#
#   bash download_models.sh                       # all models
#   MODELS="yolov8n yolo_obb" bash download_models.sh
#   DEST=/some/dir bash download_models.sh        # override target dir
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${DEST:-$SCRIPT_DIR/../model}"   # <pkg>/model/ (repo layout == share layout)
mkdir -p "$DEST"
HUB="https://modelscope.cn/models/cix/ai_model_hub/resolve/master"

# name | path in cix/ai_model_hub | expected size in bytes
MODELS_ALL=(
  "yolov8n|models/ComputeVision/Object_Detection/onnx_yolov8_n/yolov8n.cix|4702156"
  "yolov8s|models/ComputeVision/Object_Detection/onnx_yolov8_s/yolov8s.cix|13424980"
  "yolov8s-pose|models/ComputeVision/Pose_Estimation/onnx_yolov8s_pose/yolov8s-pose.cix|13372348"
  "yolov8l_seg|models/ComputeVision/Semantic_Segmentation/onnx_yolov8_segmentation/yolov8l_seg.cix|58021536"
  "yolov11_n|models/ComputeVision/Object_Detection/onnx_yolov11_n/yolov11_n.cix|4433560"
  "yolov12_n|models/ComputeVision/Object_Detection/onnx_yolov12_n/yolov12_n.cix|5680100"
  "yolov13_n|models/ComputeVision/Object_Detection/onnx_yolov13_n/yolov13_n.cix|8365348"
)

rc=0
for entry in "${MODELS_ALL[@]}"; do
  name="${entry%%|*}"; rest="${entry#*|}"
  path="${rest%%|*}"; size="${rest#*|}"

  if [[ -n "${MODELS:-}" ]] && ! grep -qw "$name" <<<"$MODELS"; then
    continue
  fi

  out="$DEST/$name.cix"
  if [[ -s "$out" ]]; then
    echo "SKIP  $name.cix (already present)"
    continue
  fi

  echo "GET   $name.cix <- $HUB/$path"
  if ! curl -fL --retry 3 -o "$out" "$HUB/$path"; then
    echo "FAIL  $name.cix download failed" >&2
    rm -f "$out"
    rc=1
    continue
  fi

  actual=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [[ "$size" != "0" && "$actual" != "$size" ]]; then
    echo "FAIL  $name.cix size mismatch: expected $size, got $actual" >&2
    rm -f "$out"
    rc=1
  else
    echo "OK    $name.cix ($actual bytes)"
  fi
done

if [[ "$rc" != 0 ]]; then
  echo >&2
  echo "Some models failed to download. Check the ModelScope repo:" >&2
  echo "  https://modelscope.cn/models/cix/ai_model_hub" >&2
  exit "$rc"
fi
