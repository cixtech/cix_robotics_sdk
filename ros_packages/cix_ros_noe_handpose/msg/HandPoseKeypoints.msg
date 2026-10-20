# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 21 hand keypoints from the handpose NPU model.
#
# keypoints are in input-image pixel coordinates (x right, y down, origin at
# the top-left corner of the frame the model actually saw). Index semantics
# follow the standard 21-point hand skeleton:
#   0 wrist, 1-4 thumb (MCP->TIP), 5-8 index, 9-12 middle,
#   13-16 ring, 17-20 pinky.
# Each point is a geometry_msgs/Point with z unused (0.0).

std_msgs/Header header
geometry_msgs/Point[] keypoints
