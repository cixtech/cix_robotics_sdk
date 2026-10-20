#!/bin/bash
# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
# Source this in any terminal that runs ros2 CLI tools against the running
# camera_handpose demo, so the CLI sees the demo's topics (matches run_demo.sh:
# Cyclone DDS localhost-only, ROS_DOMAIN_ID=0).  Usage:
#   source ~/robotics_sdk/demos/camera_handpose/env.sh
_DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SDK_DIR="$(cd "$_DEMO_DIR/../.." && pwd)"
# ROS 2 distro: Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04); humble preferred
# when both are installed. Override with ROS_DISTRO=<distro>.
if [ -z "${ROS_DISTRO:-}" ] || [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  for _d in humble jazzy; do
    if [ -f "/opt/ros/$_d/setup.bash" ]; then ROS_DISTRO="$_d"; break; fi
  done
fi
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
[ -f "$_SDK_DIR/install/setup.bash" ] && source "$_SDK_DIR/install/setup.bash"
set -u
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$_DEMO_DIR/config/cyclonedds_localhost.xml"
export LD_LIBRARY_PATH="/usr/share/cix/lib:${LD_LIBRARY_PATH:-}"
