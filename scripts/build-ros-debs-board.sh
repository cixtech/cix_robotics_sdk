#!/bin/bash
# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
# Board-native (aarch64) build of robotics_sdk ros_packages into arm64 debs.
#
# Self-contained: this script sources no other script and depends only on the
# board's ROS 2 install, colcon, and dpkg-deb. Run it ON the board, under
# either supported distro: Ubuntu 22.04 jammy / ROS 2 Humble, or Ubuntu 24.04
# noble / ROS 2 Jazzy.
#
# Scope: ONLY packages under ros_packages/ are built into debs (one deb per
# package). Demos ship source-only from the repo and are NOT built into debs
# (see docs/RELEASE.md). External fork demos (cix-panda-vision-grasp, cix-autonombot)
# live in their own repos and are never built here either.
#
# The distro is taken from ROS_DISTRO (or derived from ROS_PREFIX) and drives
# everything distro-specific: deb names (cix-ros-<distro>-*), control Depends
# (ros-<distro>-*), and the install prefix.
#
#   bash scripts/build-ros-debs-board.sh [unit ...]    # no args -> build all ros_packages
#   bash scripts/build-ros-debs-board.sh cix-ros-humble-noe-yolo cix-ros-humble-camera
#   ROS_DISTRO=jazzy bash scripts/build-ros-debs-board.sh          # cix-ros-jazzy-* debs
#   REPACK=1 bash scripts/build-ros-debs-board.sh [unit ...]   # skip colcon, repack existing install/
#   OUT=/tmp/debs ROS_PREFIX=/opt/ros/jazzy bash scripts/build-ros-debs-board.sh [unit ...]
set -uo pipefail

# --- paths (defaults match the P1 board layout; all overridable via env) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="$(cd "$SCRIPT_DIR/.." && pwd)"            # repo root = parent of scripts/
OUT="${OUT:-$SDK/out}"                             # default: <repo>/out (board must be able to write here; override with OUT=<dir>)
# ROS_DISTRO/ROS_PREFIX: if only one is set, derive the other. Defaults: humble.
if [ -n "${ROS_PREFIX:-}" ] && [ -z "${ROS_DISTRO:-}" ]; then
  ROS_DISTRO="$(basename "$ROS_PREFIX")"
else
  ROS_DISTRO="${ROS_DISTRO:-humble}"
fi
case "$ROS_DISTRO" in
  humble) UBUNTU_CODENAME="jammy" ;;
  jazzy)  UBUNTU_CODENAME="noble" ;;
  *) UBUNTU_CODENAME="unknown" ;;
esac
ROS_PREFIX="${ROS_PREFIX:-/opt/ros/$ROS_DISTRO}"
SHIPPED="$OUT/shipped.txt"                      # pkg names already in a deb, to dedup across units

[ -f "$ROS_PREFIX/setup.bash" ] || { echo "ROS2 not found at $ROS_PREFIX (set ROS_DISTRO or ROS_PREFIX=<your ROS 2 install>)"; exit 1; }
echo "ROS distro: $ROS_DISTRO ($ROS_PREFIX, Ubuntu $UBUNTU_CODENAME)"
set +u; # shellcheck disable=SC1091
source "$ROS_PREFIX/setup.bash"; set -u
mkdir -p "$OUT"
: > "$SHIPPED"

# python site-packages dir under each unit's install/ (colcon layout).
# Depending on how the board's Python was configured (Debian/Ubuntu patched
# sysconfig vs vanilla), ament installs to either
# lib/pythonX.Y/site-packages or local/lib/pythonX.Y/dist-packages — detect
# the real layout from the install tree instead of guessing.
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.10)"

py_site_of() {  # $1=inst -> python dir relative to that install prefix
  if [ -d "$1/local/lib/python${PY_VER}/dist-packages" ]; then
    echo "local/lib/python${PY_VER}/dist-packages"
  else
    echo "lib/python${PY_VER}/site-packages"
  fi
}

# unit_debname | src_path relative to SDK | extra_srcs (comma-sep siblings for cross-pkg refs, optional)
# Build order matters for dedup: cix_ros_dmabuf ships first, so the later
# units that build it as a sibling skip it via the [dedup] path.
UNITS=(
  "cix-ros-${ROS_DISTRO}-dmabuf|ros_packages/cix_ros_dmabuf"
  "cix-ros-${ROS_DISTRO}-noe-yolo|ros_packages/cix_ros_noe_yolo"
  "cix-ros-${ROS_DISTRO}-camera|ros_packages/cix_ros_camera|ros_packages/cix_ros_dmabuf"
  "cix-ros-${ROS_DISTRO}-noe-handpose|ros_packages/cix_ros_noe_handpose|ros_packages/cix_ros_dmabuf"
)

version_of() { sed -n 's/.*<version>\(.*\)<\/version>.*/\1/p' "$1" | head -1; }

collect_pkg_into_stage() {  # $1=pkg $2=inst $3=stage
  local pkg=$1 inst=$2 stage=$3
  local PY_SITE; PY_SITE=$(py_site_of "$inst")
  mkdir -p "$stage$ROS_PREFIX/share"
  cp -a "$inst/share/$pkg" "$stage$ROS_PREFIX/share/" 2>/dev/null
  for idx in "$inst"/share/ament_index/resource_index/*/; do
    [ -e "$idx" ] || continue
    local idxtype; idxtype=$(basename "$idx")
    [ -e "$idx/$pkg" ] || continue
    mkdir -p "$stage$ROS_PREFIX/share/ament_index/resource_index/$idxtype"
    cp -a "$idx/$pkg" "$stage$ROS_PREFIX/share/ament_index/resource_index/$idxtype/"
  done
  [ -d "$inst/lib/$pkg" ] && { mkdir -p "$stage$ROS_PREFIX/lib"; cp -a "$inst/lib/$pkg" "$stage$ROS_PREFIX/lib/"; }
  # ament_python packages may use a Python module name != ROS package name
  # (e.g. cix_ros_noe_sfa3d ships module `sfa`); read the real top-level
  # module(s) from the package's egg-info instead of assuming dir == $pkg.
  local egginfo mod
  for egginfo in "$inst/$PY_SITE/$pkg"-*.egg-info; do
    [ -d "$egginfo" ] || continue
    mkdir -p "$stage$ROS_PREFIX/$PY_SITE"
    cp -a "$egginfo" "$stage$ROS_PREFIX/$PY_SITE/"
    [ -f "$egginfo/top_level.txt" ] || continue
    while IFS= read -r mod; do
      [ -n "$mod" ] || continue
      [ -d "$inst/$PY_SITE/$mod" ] && cp -a "$inst/$PY_SITE/$mod" "$stage$ROS_PREFIX/$PY_SITE/"
    done < "$egginfo/top_level.txt"
  done
  # fallback: module dir literally named $pkg (no egg-info present)
  if [ -d "$inst/$PY_SITE/$pkg" ] && [ ! -e "$stage$ROS_PREFIX/$PY_SITE/$pkg" ]; then
    mkdir -p "$stage$ROS_PREFIX/$PY_SITE"
    cp -a "$inst/$PY_SITE/$pkg" "$stage$ROS_PREFIX/$PY_SITE/"
  fi
  [ -d "$inst/include/$pkg" ] && { mkdir -p "$stage$ROS_PREFIX/include"; cp -a "$inst/include/$pkg" "$stage$ROS_PREFIX/include/"; }
  local so
  for so in "$inst"/lib/lib${pkg}*.so*; do
    [ -e "$so" ] || continue
    mkdir -p "$stage$ROS_PREFIX/lib"; cp -a "$so" "$stage$ROS_PREFIX/lib/"
  done
}

# Map a package.xml runtime dependency (<depend>/<exec_depend>) to the apt
# package providing it at runtime. Emits a note and returns empty for deps
# that are not installable via apt (excluded from Depends).
deb_for_dep() {  # $1 = dependency name from package.xml
  case "$1" in
    # SDK sibling packages built by this same script
    cix_ros_dmabuf)       echo "cix-ros-${ROS_DISTRO}-dmabuf" ;;
    cix_ros_camera)       echo "cix-ros-${ROS_DISTRO}-camera" ;;
    cix_ros_noe_handpose) echo "cix-ros-${ROS_DISTRO}-noe-handpose" ;;
    cix_ros_noe_yolo)     echo "cix-ros-${ROS_DISTRO}-noe-yolo" ;;
    # rosdep keys that are already Ubuntu/Debian package names
    lib*|python3-*)       echo "$1" ;;
    # community package, not published to the ROS 2 apt repo (no
    # ros-<distro>-yolov8-msgs exists); users colcon-build it from source
    yolov8_msgs)          echo "  [dep-skip] yolov8_msgs (no ros-${ROS_DISTRO}-yolov8-msgs deb; build from source)" >&2; echo "" ;;
    # everything else is a ROS package -> ros-<distro>-<pkg>; ROS deb names
    # use dashes where package.xml names use underscores
    # (e.g. cv_bridge -> ros-<distro>-cv-bridge)
    *)                    echo "ros-${ROS_DISTRO}-${1//_/-}" ;;
  esac
}

# Runtime dependencies of a package: <depend> + <exec_depend> entries
# (buildtool_depend/test_depend are build-time only and excluded).
runtime_deps_of() {  # $1=package.xml -> one dep name per line
  sed -n -e 's/^[[:space:]]*<depend>\(.*\)<\/depend>.*/\1/p' \
         -e 's/^[[:space:]]*<exec_depend>\(.*\)<\/exec_depend>.*/\1/p' "$1"
}

make_deb() {  # $1=debname $2=version $3=stage $4=deps $5=description
  local debname=$1 ver=$2 stage=$3 deps=$4 desc=$5
  mkdir -p "$stage/DEBIAN"
  {
    echo "Package: $debname"
    echo "Version: $ver"
    echo "Architecture: arm64"
    # Section/Priority are required by archive tools (reprepro/aptly/PPA
    # ingestion) even though dpkg-deb itself does not enforce them.
    echo "Section: science"
    echo "Priority: optional"
    echo "Maintainer: CIX Robotics Team <noreply@cixtech.com>"
    [ -n "$deps" ] && echo "Depends: $deps"
    echo "Description: $desc"
  } > "$stage/DEBIAN/control"
  if dpkg-deb -b --root-owner-group "$stage" "$OUT/${debname}_${ver}_arm64.deb" 2>/dev/null; then
    echo "  [ok] ${debname}_${ver}_arm64.deb"
  else
    echo "  [FAIL] $debname"
  fi
  rm -rf "$stage"
}

build_ros_unit() {  # $1=name $2=src $3=extra_srcs(comma-sep, kept as siblings for cross-pkg refs)
  local name=$1 src=$2 extra=${3:-}
  local ws=$OUT/ws/$name inst="$OUT/ws/$name/install"
  if [ "${REPACK:-0}" = "1" ] && [ -d "$inst/share" ]; then
    echo "=== repack: $name ==="
  else
    rm -rf "$ws"; mkdir -p "$ws/src"
    ln -sf "$SDK/$src" "$ws/src/src0"
    local e
    local IFS=','
    for e in $extra; do [ -n "$e" ] && ln -sf "$SDK/$e" "$ws/src/$(basename "$e")"; done
    unset IFS
    echo "=== build: $name ($SDK/$src) ==="
    ( cd "$ws" && colcon build \
        --merge-install --continue-on-error \
        --install-base "$inst" \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        > "$ws/build.log" 2>&1 )
    echo "  colcon rc=$? (log $ws/build.log)"
  fi
  local stage=$OUT/stage/$name ver="" first=1 shipped=0 skipped=0
  local dep_list="" d
  rm -rf "$stage"; mkdir -p "$stage"
  for pxml in "$inst"/share/*/package.xml; do
    [ -f "$pxml" ] || continue
    local pkg; pkg=$(basename "$(dirname "$pxml")")
    if grep -qx "$pkg" "$SHIPPED" 2>/dev/null; then
      echo "  [dedup] $pkg (already in an earlier unit's deb)"; skipped=$((skipped+1)); continue
    fi
    collect_pkg_into_stage "$pkg" "$inst" "$stage"
    echo "$pkg" >> "$SHIPPED"; shipped=$((shipped+1))
    [ -n "$first" ] && { ver=$(version_of "$pxml"); first=""; }
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      dep_list="$dep_list $(deb_for_dep "$d")"
    done < <(runtime_deps_of "$pxml")
    echo "  + $pkg"
  done
  # dedup + deterministic order -> comma-separated Depends value
  local deps; deps=$(printf '%s\n' $dep_list | sed '/^$/d' | sort -u | paste -sd, | sed 's/,/, /g')
  # YOLO .cix models are Ultralytics-derived (AGPL-3.0) and must not ship in
  # the deb; users fetch them from ModelScope after install:
  #   sudo bash <prefix>/share/cix_ros_noe_yolo/scripts/download_models.sh
  # (the handpose model is Apache-2.0 and does ship.)
  if [ "$name" = "cix-ros-${ROS_DISTRO}-noe-yolo" ]; then
    local f
    for f in "$stage$ROS_PREFIX/share/cix_ros_noe_yolo/model/"*.cix; do
      [ -e "$f" ] || continue
      rm -f "$f"
      echo "  [strip] $(basename "$f") (AGPL model, not bundled; run download_models.sh after install)"
    done
  fi
  ver=${ver:-0.1.0}
  make_deb "$name" "$ver" "$stage" "$deps" "$name - $shipped ROS2 package(s) from robotics_sdk, built natively on board (aarch64/$UBUNTU_CODENAME, ROS 2 $ROS_DISTRO). Skipped $skipped dup(s)."
  grep -E 'Failed|Could NOT find' "$ws/build.log" 2>/dev/null | head -8
}

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then TARGETS=(); for u in "${UNITS[@]}"; do TARGETS+=("${u%%|*}"); done; fi
for t in "${TARGETS[@]}"; do
  entry=""; for u in "${UNITS[@]}"; do [ "${u%%|*}" = "$t" ] && entry="$u"; done
  [ -z "$entry" ] && { echo "unknown unit: $t"; continue; }
  IFS='|' read -r name src extra <<< "$entry"
  build_ros_unit "$name" "$src" "$extra"
done

echo; echo "=== debs ==="; ls -1 "$OUT"/*.deb 2>/dev/null
echo "count: $(ls "$OUT"/*.deb 2>/dev/null | wc -l)"
