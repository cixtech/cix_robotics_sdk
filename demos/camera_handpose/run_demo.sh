#!/bin/bash
# =============================================================================
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# run_demo.sh — camera_handpose zero-copy demo (broker + camera + handpose)
#
# Brings up the full zero-copy pipeline:
#   cix_ros_dmabuf_broker  (UDS fd relay)
#   v4l2_dmabuf_camera     (V4L2 EXPBUF -> dmabuf fd, publishes /cam0_stream1)
#   handpose_dmabuf_node   (C++ NPU inference, zero-copy subscriber via broker)
#   rqt_image_view         (optional, previews /handpose/debug_image)
#
# Usage (on the board):
#   bash ~/robotics_sdk/demos/camera_handpose/run_demo.sh [start [rqt_topic]]
#   bash ~/robotics_sdk/demos/camera_handpose/run_demo.sh stop
#
# Prereqs (checked below):
#   - cix_ros_camera and cix_ros_noe_handpose already colcon-built and on the
#     sourced setup.bash (this script only orchestrates; it does not build).
#   - ISP driver armcb_isp_v4l2 loaded, /dev/video9 present, isp_app running.
#
# Logs: /tmp/{broker,cam,handpose,rqt}.log
# =============================================================================
set -uo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "$DEMO_DIR/../.." && pwd)"

CAMERA_PKG=cix_ros_camera
DMABUF_PKG=cix_ros_dmabuf
# Handpose node: the C++ zero-copy node from cix_ros_noe_handpose (single-frame
# ~12 ms vs the Python node's ~60 ms). The Python node (handpose_dma_teleop)
# still ships in the package as a slower alternative and can be run manually.
HANDPOSE_PKG=cix_ros_noe_handpose
HANDPOSE_EXE=handpose_dmabuf_node
BROKER_EXE=cix_ros_dmabuf_broker
CAMERA_EXE=v4l2_dmabuf_camera
SOCK=/tmp/cix_ros_dmabuf_broker.sock          # cix_ros_camera publisher side hardcodes this

CAM_DEVICE=/dev/video9
CAM_WIDTH=1280
CAM_HEIGHT=720
CAM_FPS=30.0
CAM_FMT=RGB3
CAM_TOPIC=/cam0_stream1
DEFAULT_RQT_TOPIC=/handpose/debug_image
# Four-way direction split for the handpose node: horizontal only when
# |dx| >= DIRECTION_AXIS_RATIO*|dy| (1.4 => ~54 deg from vertical).
DIRECTION_AXIS_RATIO="${DIRECTION_AXIS_RATIO:-1.4}"

# Where the handpose model lives (installed by either handpose package).
MODEL_HINT="${MODEL_HINT:-}"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[0;33m'; B='\033[0;34m'; N='\033[0m'
ok()   { printf "${G}✓${N} %s\n" "$1"; }
fail() { printf "${R}✗${N} %s\n" "$1"; }
warn() { printf "${Y}!${N} %s\n" "$1"; }
hdr()  { printf "\n${B}== %s ==${N}\n" "$1"; }

# ROS 2 distro: Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04). Auto-detect
# from /opt/ros (humble preferred when both are installed); override with
# ROS_DISTRO=<distro>.
detect_ros_distro() {
  if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    return 0
  fi
  local d
  for d in humble jazzy; do
    if [ -f "/opt/ros/$d/setup.bash" ]; then ROS_DISTRO="$d"; return 0; fi
  done
  for d in /opt/ros/*/setup.bash; do
    [ -f "$d" ] && { ROS_DISTRO="$(basename "$(dirname "$d")")"; return 0; }
  done
  return 1
}

# -----------------------------------------------------------------------------
# ROS + DDS environment. Cyclone DDS localhost-only so the image stream stays
# on-loopback (rqt blank-screen root cause is a missing CYCLONEDDS_URI).
source_ros_env() {
  set +u
  if ! detect_ros_distro; then
    fail "no ROS 2 distro found under /opt/ros (set ROS_DISTRO=<distro>)"; exit 1
  fi
  source "/opt/ros/$ROS_DISTRO/setup.bash"
  # Source the SDK workspace so both packages are on the path (built per README).
  for ws_install in "$SDK_DIR"/install/setup.bash; do
    [ -f "$ws_install" ] && source "$ws_install"
  done
  set -u
  export ROS_DOMAIN_ID=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI="file://$DEMO_DIR/config/cyclonedds_localhost.xml"
  export LD_LIBRARY_PATH="/usr/share/cix/lib:${LD_LIBRARY_PATH:-}"
}

# -----------------------------------------------------------------------------
cleanup_procs() {
  pkill -9 -f "[c]ix_ros_dmabuf_broker" 2>/dev/null
  pkill -9 -f "[v]4l2_dmabuf_camera"    2>/dev/null
  pkill -9 -f "[h]andpose_dma_teleop"   2>/dev/null
  pkill -9 -f "[h]andpose_dmabuf_node"  2>/dev/null
  pkill -9 -f "[r]qt_image_view"        2>/dev/null
  # broker comm may be truncated to 15 chars; kill by socket-listener pid too.
  for p in $(ss -xlp 2>/dev/null | grep "$(basename "$SOCK")" | grep -oP 'pid=\K[0-9]+' | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  # Iceoryx RouDi daemon (only if we started it for SHM transport)
  if [ "${SHM_STARTED:-0}" = "1" ]; then
    pkill -9 -f "[i]ox-roudi" 2>/dev/null
  fi
  sleep 1
  rm -f "$SOCK"
}

# Fall back to a Cyclone config with SharedMemory disabled. Cyclone+iceoryx
# aborts the whole process (SIGABRT → Ubuntu "internal error" dialog) when
# <SharedMemory><Enable>true</Enable> is set but iox-roudi is not reachable.
disable_shm() {
  local noshm="/tmp/cyclonedds_localhost_noshm.$$.xml"
  sed 's|<Enable>true</Enable>|<Enable>false</Enable>|' \
    "$DEMO_DIR/config/cyclonedds_localhost.xml" > "$noshm"
  export CYCLONEDDS_URI="file://$noshm"
  warn "SHM disabled (RouDi unavailable) — debug_image may drop frames"
}

# Start the Iceoryx RouDi daemon if Cyclone shared-memory transport is enabled
# in the config. SHM lets same-host processes exchange the 0.69 MB/frame
# debug_image via a shared-memory pointer instead of serializing it over a
# loopback socket -- without it the debug_image subscription rate is capped at
# ~17 fps by per-frame deserialization; with it, it matches the 20 fps publish
# rate. RouDi must be running before any ROS node starts.
ensure_roudi() {
  SHM_STARTED=0
  grep -q "<Enable>true</Enable>" "$DEMO_DIR/config/cyclonedds_localhost.xml" 2>/dev/null \
    || return 0   # SHM not enabled in config -- nothing to do
  if pgrep -x iox-roudi >/dev/null 2>&1; then
    ok "RouDi already running"
    return 0
  fi
  if [ ! -x "/opt/ros/$ROS_DISTRO/bin/iox-roudi" ]; then
    disable_shm
    return 0
  fi
  # Truncate log first so a stale "ready" line cannot fool the wait loop.
  : > /tmp/roudi.log 2>/dev/null || rm -f /tmp/roudi.log
  nohup "/opt/ros/$ROS_DISTRO/bin/iox-roudi" \
    -c "/opt/ros/$ROS_DISTRO/etc/roudi_config_example.toml" \
    > /tmp/roudi.log 2>&1 < /dev/null &
  local ready=0
  for i in $(seq 1 30); do
    # roudi died right after start (e.g. its log redirect hit a stale
    # root/other-user-owned /tmp/roudi.log from a docker run -- fs.protected_regular
    # blocks even root from O_CREAT-opening such files). Never trust a "ready"
    # line from a stale log: require the process to be alive too.
    pgrep -x iox-roudi >/dev/null 2>&1 || break
    grep -q "ready for clients" /tmp/roudi.log 2>/dev/null && { ready=1; break; }
    sleep 0.2
  done
  if [ "$ready" -eq 1 ]; then ok "RouDi ready (SHM transport)"; SHM_STARTED=1
  else
    warn "RouDi did not become ready (see /tmp/roudi.log)"
    disable_shm
  fi
}

do_stop() {
  hdr "Stopping demo"
  cleanup_procs
  sleep 1
  printf "Remaining: "
  pgrep -af "cix_ros_dmabuf_broker|v4l2_dmabuf_camera|handpose_dma_teleop|handpose_dmabuf_node|rqt_image_view" 2>/dev/null \
    | grep -v pgrep | grep -v "run_demo" || echo "NONE"
}

# -----------------------------------------------------------------------------
preflight() {
  hdr "Pre-flight checks"
  local bad=0

  lsmod | grep -q armcb_isp_v4l2 && ok "ISP driver loaded" || { fail "ISP driver not loaded"; bad=1; }
  [ -e "$CAM_DEVICE" ] && ok "$CAM_DEVICE present" || { fail "$CAM_DEVICE missing"; bad=1; }
  if pgrep -f "isp_app" >/dev/null 2>&1; then ok "isp_app running"; else
    fail "isp_app not running (ISP not initialized)"
    printf "    fix: ${Y}sudo systemctl restart isp-daemon${N}\n"; bad=1
  fi

  local missing=""
  [ -x "$(command -v ros2)" ] || missing="$missing ros2"
  ros2 pkg prefix "$DMABUF_PKG"   >/dev/null 2>&1 || missing="$missing $DMABUF_PKG"
  ros2 pkg prefix "$CAMERA_PKG"   >/dev/null 2>&1 || missing="$missing $CAMERA_PKG"
  ros2 pkg prefix "$HANDPOSE_PKG" >/dev/null 2>&1 || missing="$missing $HANDPOSE_PKG"
  if [ -n "$missing" ]; then
    fail "Missing on ROS path:$missing"
    printf "    fix: colcon build --packages-select %s %s %s, then source the workspace setup.bash\n" \
      "$DMABUF_PKG" "$CAMERA_PKG" "$HANDPOSE_PKG"
    bad=1
  else ok "Packages on ROS path ($DMABUF_PKG, $CAMERA_PKG, $HANDPOSE_PKG)"; fi

  if [ "$bad" -ne 0 ]; then echo; fail "Pre-flight failed. Fix the above and re-run."; exit 1; fi

  # Resolve the NPU model file. The C++ node needs an explicit path; the Python
  # node can find its own, but we resolve it here for both so it is verifiable.
  local pkg_share
  pkg_share="$(ros2 pkg prefix "$HANDPOSE_PKG" 2>/dev/null)/share/$HANDPOSE_PKG"
  MODEL_FILE="$MODEL_HINT"
  if [ -z "$MODEL_FILE" ] && [ -f "$pkg_share/models/handpose_resnet50_size256.cix" ]; then
    MODEL_FILE="$pkg_share/models/handpose_resnet50_size256.cix"
  fi
  if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
    fail "handpose model not found (looked in $pkg_share/models; set MODEL_HINT)"
    bad=1
  else ok "model: $MODEL_FILE"; fi
}

# -----------------------------------------------------------------------------
do_start() {
  local rqt_topic="${1:-$DEFAULT_RQT_TOPIC}"
  source_ros_env
  preflight

  hdr "Cleaning stale processes"
  cleanup_procs

  # Iceoryx RouDi must be up before any ROS node if SHM transport is enabled
  # in the Cyclone config (see ensure_roudi).
  ensure_roudi

  # broker first, wait until its socket is actually listening (otherwise camera
  # and handpose silently fall back to memcpy).
  hdr "Start broker"
  setsid nohup ros2 run "$DMABUF_PKG" "$BROKER_EXE" > /tmp/broker.log 2>&1 < /dev/null &
  local ready=0
  for i in $(seq 1 50); do
    if [ -S "$SOCK" ] && ss -xl 2>/dev/null | grep -q "$(basename "$SOCK")"; then ready=1; break; fi
    sleep 0.1
  done
  [ "$ready" -eq 1 ] && ok "broker ready ($SOCK)" || { fail "broker socket not ready"; tail -5 /tmp/broker.log; exit 1; }

  hdr "Start camera"
  setsid nohup ros2 run "$CAMERA_PKG" "$CAMERA_EXE" --ros-args \
    -p device:="$CAM_DEVICE" -p pixel_format:="$CAM_FMT" \
    -p width:="$CAM_WIDTH" -p height:="$CAM_HEIGHT" -p fps:="$CAM_FPS" \
    -p topic:="$CAM_TOPIC" > /tmp/cam.log 2>&1 < /dev/null &
  local cam_ok=0
  for i in $(seq 1 40); do
    grep -q "started successfully" /tmp/cam.log 2>/dev/null && { cam_ok=1; break; }
    sleep 0.2
  done
  [ "$cam_ok" -eq 1 ] && ok "camera started" || { warn "camera did not report started"; tail -6 /tmp/cam.log; }

  hdr "Start handpose ($HANDPOSE_PKG/$HANDPOSE_EXE)"
  # C++ zero-copy node: broker_socket_path is passed explicitly so it matches
  # the broker this script started (defaults to the same path, but be explicit).
  setsid nohup ros2 run "$HANDPOSE_PKG" "$HANDPOSE_EXE" --ros-args \
    -p image_topic:="$CAM_TOPIC" \
    -p broker_socket_path:="$SOCK" \
    -p npu_model_path:="$MODEL_FILE" \
    -p publish_debug_image:=true \
    -p publish_output_image:=true \
    -p debug_image_scale:=0.5 \
    -p direction_axis_ratio:="$DIRECTION_AXIS_RATIO" > /tmp/handpose.log 2>&1 < /dev/null &
  sleep 4

  hdr "Verify zero-copy"
  local zc=0
  for i in $(seq 1 20); do
    grep -q "dmabuf broker connected" /tmp/handpose.log 2>/dev/null && { zc=1; break; }
    sleep 0.5
  done
  if [ "$zc" -eq 1 ]; then
    ok "zero-copy link active"
    grep -E "broker connected|Zero-copy" /tmp/handpose.log 2>/dev/null | tail -1 | sed 's/^/    /'
  else
    warn "zero-copy not confirmed (memcpy fallback?)"
    tail -6 /tmp/handpose.log
  fi

  # rqt preview (optional; needs a local graphical session).
  hdr "rqt preview ($rqt_topic)"
  cat > /tmp/launch_rqt.sh <<EOF
#!/bin/bash
source /opt/ros/$ROS_DISTRO/setup.bash
for ws in "$SDK_DIR"/install/setup.bash "$HOME"/ros2_ws/install/setup.bash; do
  [ -f "\$ws" ] && source "\$ws"
done
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Inherit the URI chosen by ensure_roudi (may be the no-SHM fallback).
export CYCLONEDDS_URI="$CYCLONEDDS_URI"
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
export XAUTHORITY=\$(ls /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
exec ros2 run rqt_image_view rqt_image_view "$rqt_topic"
EOF
  chmod +x /tmp/launch_rqt.sh
  setsid nohup /tmp/launch_rqt.sh > /tmp/rqt.log 2>&1 < /dev/null &
  sleep 5
  local sub=0
  for i in $(seq 1 10); do
    sub=$(ros2 topic info "$rqt_topic" 2>/dev/null | grep -ci "subscription" || true)
    [ "$sub" -ge 1 ] && break
    sleep 0.5
  done
  [ "$sub" -ge 1 ] && ok "rqt subscribed to $rqt_topic (count $sub)" \
    || warn "rqt subscription not seen (no graphical session? see /tmp/rqt.log)"

  hdr "Status"
  printf "  %-10s %s\n" "broker"   "$(pgrep -f "$BROKER_EXE"   >/dev/null && echo RUNNING || echo DOWN)"
  printf "  %-10s %s\n" "camera"   "$(pgrep -f "$CAMERA_EXE"   >/dev/null && echo RUNNING || echo DOWN)"
  printf "  %-10s %s\n" "handpose" "$(pgrep -f "$HANDPOSE_EXE" >/dev/null && echo RUNNING || echo DOWN)"
  printf "  %-10s %s\n" "rqt"      "$(pgrep -f rqt_image_view  >/dev/null && echo "RUNNING ($rqt_topic)" || echo DOWN)"
  printf "  %-10s %s\n" "logs"     "/tmp/{broker,cam,handpose,rqt}.log"
  echo
  printf "Stop: ${Y}bash $DEMO_DIR/run_demo.sh stop${N}  (or bash $DEMO_DIR/stop_demo.sh)\n"
}

case "${1:-start}" in
  start) do_start "${2:-$DEFAULT_RQT_TOPIC}" ;;
  stop)  do_stop ;;
  *) echo "Usage: bash $0 [start [rqt_topic] | stop]"; echo "  default: start, rqt topic = $DEFAULT_RQT_TOPIC"; exit 1 ;;
esac
