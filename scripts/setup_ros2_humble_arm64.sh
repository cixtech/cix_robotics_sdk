#!/usr/bin/env bash
# =============================================================================
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# setup_ros2_humble_arm64.sh
# Platform : Ubuntu 22.04 (Jammy) ARM64/aarch64
# ROS 2    : Humble Hawksbill
# Includes : ROS 2 desktop-full, Gazebo Fortress, MoveIt 2, Nav2/SLAM,
#            robot tools, DDS middleware, Python libs, ARM and dev tools,
#            colcon workspace, shell block, colcon mixin, VSCode config.
# Usage    : sudo bash setup_ros2_humble_arm64.sh
#
# PREREQUISITE - configure the CIX PPA first:
#     curl -fsSL https://oa-ppa.cixtech.com/cix-ppa-daily/cix-repo.sh | sudo sh
#
# This script installs only what it needs: it never runs a system-wide
# `apt upgrade` / `full-upgrade` and never runs `apt autoremove`. With the CIX
# PPA pinned at priority 1001 those would downgrade linux-libc-dev and replace
# xwayland with an older CIX build. Please keep it that way.
# =============================================================================

set -Eeuo pipefail

export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
INSTALL_DESKTOP_FULL="${INSTALL_DESKTOP_FULL:-1}"
INSTALL_GAZEBO="${INSTALL_GAZEBO:-1}"
INSTALL_MOVEIT="${INSTALL_MOVEIT:-1}"
INSTALL_NAV2="${INSTALL_NAV2:-1}"
INSTALL_HARDWARE_TOOLS="${INSTALL_HARDWARE_TOOLS:-1}"
INSTALL_PYTHON_EXTRAS="${INSTALL_PYTHON_EXTRAS:-1}"
CREATE_WORKSPACE="${CREATE_WORKSPACE:-1}"
APT_RETRIES="${APT_RETRIES:-3}"
FORCE_OSRF_GAZEBO="${FORCE_OSRF_GAZEBO:-1}"
SNAPSHOT_FALLBACK="${SNAPSHOT_FALLBACK:-1}"
ROS_SNAPSHOT_DATE="${ROS_SNAPSHOT_DATE:-2026-07-02}"
ROS_SNAPSHOT_MIRROR="${ROS_SNAPSHOT_MIRROR:-http://snapshots.ros.org}"
VERIFY_SCRIPT_NAME="${VERIFY_SCRIPT_NAME:-verify_ros2_humble_env.sh}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
banner() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

on_error() {
    local line="$1"
    local status="$2"
    echo -e "${RED}[ERROR]${NC} Failed at line ${line} with exit code ${status}." >&2
    echo -e "${YELLOW}[HINT]${NC}  Check the log file: ${LOG_FILE:-not initialized}" >&2
    if command -v apt-cache >/dev/null 2>&1; then
        echo -e "${YELLOW}[HINT]${NC}  Relevant apt candidates:" >&2
        apt-cache policy \
            "ros-${ROS_DISTRO:-humble}-desktop-full" \
            libignition-sensors6 \
            "ros-${ROS_DISTRO:-humble}-moveit" \
            "ros-${ROS_DISTRO:-humble}-navigation2" \
            2>/dev/null >&2 || true
    fi
}
trap 'on_error "$LINENO" "$?"' ERR

require_root() {
    [[ "${EUID}" -eq 0 ]] || die "Please run with sudo: sudo bash $0"
}

detect_real_user() {
    REAL_USER="${SUDO_USER:-${USER:-root}}"
    REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
    [[ -n "$REAL_HOME" && -d "$REAL_HOME" ]] || die "Could not resolve home directory for user: $REAL_USER"
    COLCON_WS="${COLCON_WS:-$REAL_HOME/ros2_ws}"
}

run_as_user() {
    sudo -H -u "$REAL_USER" bash -lc "$*"
}

apt_update() {
    apt-get update -y
}

apt_install_required() {
    local packages=("$@")
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        -o Dpkg::Options::=--force-confdef \
        -o Dpkg::Options::=--force-confold \
        "${packages[@]}"
}

version_candidate() {
    LC_ALL=C apt-cache policy "$1" | awk '/Candidate:/ {print $2; exit}'
}

apt_has_candidate() {
    local candidate
    candidate="$(version_candidate "$1")"
    [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

# Install the subset that has an apt candidate; skip (with a warning) the rest.
apt_install_available() {
    local keep=()
    local pkg
    for pkg in "$@"; do
        if apt_has_candidate "$pkg"; then
            keep+=("$pkg")
        else
            warn "Skipping $pkg (no apt candidate)"
        fi
    done
    if ((${#keep[@]} == 0)); then
        warn "No installable packages among: $*"
        return 0
    fi
    apt_install_required "${keep[@]}"
}

retry() {
    local attempt=1
    local delay=3
    until "$@"; do
        if (( attempt >= APT_RETRIES )); then
            return 1
        fi
        warn "Command failed, retrying in ${delay}s (${attempt}/${APT_RETRIES}): $*"
        sleep "$delay"
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done
}

write_file_as_user() {
    local target="$1"
    local mode="$2"
    local owner="$REAL_USER:$REAL_USER"
    install -o "$REAL_USER" -g "$REAL_USER" -m "$mode" /dev/stdin "$target"
    chown "$owner" "$target"
}

require_candidate_at_least() {
    local package="$1"
    local minimum="$2"
    local candidate
    candidate="$(version_candidate "$package")"

    if [[ -z "$candidate" || "$candidate" == "(none)" ]]; then
        apt-cache policy "$package" || true
        die "No apt candidate found for $package. Check Ubuntu/ROS/OSRF repositories."
    fi

    if ! dpkg --compare-versions "$candidate" ge "$minimum"; then
        apt-cache policy "$package" || true
        apt-cache madison "$package" || true
        die "$package candidate is $candidate, but >= $minimum is required. Your apt repositories still contain a stale Ignition/Gazebo version mix."
    fi

    info "$package candidate OK: $candidate >= $minimum"
}

# ROS-official snapshot mirror: on arm64 the live repo occasionally drops
# packages (desktop-full, ros-gz-sim, ...). The snapshot is pinned at 100 so it
# is only used when the live index has no candidate.
add_ros_snapshot_fallback() {
    [[ "$SNAPSHOT_FALLBACK" == "1" ]] || { warn "SNAPSHOT_FALLBACK=0; not adding ROS snapshot fallback"; return 0; }
    [[ "$ROS_DISTRO" == "humble" ]] || { warn "Snapshot fallback is only configured for Humble; skip"; return 0; }

    local list="/etc/apt/sources.list.d/ros2-snapshot.list"
    local pin="/etc/apt/preferences.d/ros2-snapshot-fallback"
    local url="${ROS_SNAPSHOT_MIRROR}/${ROS_DISTRO}/${ROS_SNAPSHOT_DATE}/ubuntu"
    local arch
    arch="$(dpkg --print-architecture)"

    info "Adding Humble ${ROS_SNAPSHOT_DATE} snapshot as low-priority fallback for arm64 packages dropped from the live repo"
    info "Snapshot: ${url}"

    cat > "$list" <<EOF
deb [arch=${arch} signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] ${url} ${CODENAME} main
EOF

    cat > "$pin" <<'EOF'
Package: *
Pin: origin snapshots.ros.org
Pin-Priority: 100
EOF

    if ! apt_update; then
        warn "Snapshot repo failed apt update with the live ROS key; retrying as trusted official snapshot host"
        cat > "$list" <<EOF
deb [arch=${arch} trusted=yes] ${url} ${CODENAME} main
EOF
        apt_update || die "apt update failed after adding ROS snapshot ${ROS_SNAPSHOT_DATE}"
    fi

    local pkg
    for pkg in \
        ros-${ROS_DISTRO}-desktop-full \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ros-gz-sim \
        ros-${ROS_DISTRO}-gz-ros2-control \
        ros-${ROS_DISTRO}-simulation
    do
        if apt_has_candidate "$pkg"; then
            info "Fallback candidate OK: $pkg $(version_candidate "$pkg")"
        else
            warn "Still no apt candidate for $pkg after enabling snapshot ${ROS_SNAPSHOT_DATE}"
        fi
    done
    ok "ROS snapshot fallback configured (${ROS_SNAPSHOT_DATE}, pin 100)"
}

# Remove leftover/foreign Gazebo apt sources so only the OSRF repo provides the
# Ignition/Gazebo stack (a mix of Ubuntu, ROS and OSRF candidates breaks it).
reset_gazebo_sources() {
    info "Resetting Gazebo/OSRF apt source to avoid mixed Ignition versions"
    rm -f /etc/apt/sources.list.d/gazebo-stable.list
    rm -f /etc/apt/sources.list.d/osrf*.list
    rm -f /etc/apt/sources.list.d/*gazebo*.list
    rm -f /etc/apt/preferences.d/gazebo-osrf-pin
    rm -f /etc/apt/keyrings/gazebo-archive-keyring.gpg
    rm -f /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    # apt-mark takes (unanchored) POSIX regexes, so anchor explicitly and keep
    # the trailing dash: a bare 'gz-*' also unheld gzip, bugzilla, node-gzip-size.
    apt-mark unhold '^ignition-' '^libignition-' '^gz-' '^libgz-' '^ros-humble-ros-gz' 2>/dev/null || true
}

add_osrf_gazebo_repo() {
    [[ "$FORCE_OSRF_GAZEBO" == "1" ]] || { warn "FORCE_OSRF_GAZEBO=0; not adding OSRF Gazebo repo"; return 0; }

    install -d -m 0755 /etc/apt/keyrings
    curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        | gpg --dearmor --batch --yes -o /etc/apt/keyrings/gazebo-archive-keyring.gpg
    chmod 0644 /etc/apt/keyrings/gazebo-archive-keyring.gpg

    cat > /etc/apt/sources.list.d/gazebo-stable.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable ${CODENAME} main
EOF

    # Keep all Ignition/Gazebo binary libraries on the OSRF repo together, so apt
    # cannot take libignition-gazebo6 from OSRF while keeping libignition-sensors6
    # at an older Ubuntu/ROS candidate.
    cat > /etc/apt/preferences.d/gazebo-osrf-pin <<'EOF'
Package: ignition-* libignition-* gz-* libgz-* gazebo* libgazebo* sdformat* libsdformat*
Pin: origin packages.osrfoundation.org
Pin-Priority: 1001
EOF
}

# Drop ROS shell blocks left by earlier versions, so exactly one managed block
# ends up in ~/.bashrc.
strip_legacy_ros_blocks() {
    local file="$1"
    local marker
    [[ -f "$file" ]] || return 0
    for marker in "ROS2 ARM64 setup" "ROS2 Humble official setup" "ROS2 Humble extras" \
                  "ROS2 Humble setup" \
                  "ROS2 Jazzy setup" "ROS2 Jazzy official setup" "ROS2 Jazzy extras"; do
        if grep -qF "# >>> ${marker} >>>" "$file"; then
            info "Removing legacy shell block: ${marker}"
            sed -i "\|^# >>> ${marker} >>>\$|,\|^# >>> end ${marker} >>>\$|d" "$file"
        fi
    done
}

# -----------------------------------------------------------------------------
banner "Pre-flight Checks"
require_root
detect_real_user

ARCH="$(uname -m)"
CODENAME="$(
    . /etc/os-release
    echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
)"
OS_PRETTY="$(
    . /etc/os-release
    echo "${PRETTY_NAME:-unknown}"
)"

[[ "$ARCH" == "aarch64" ]] || warn "Current architecture is '$ARCH'; this script is tuned for ARM64/aarch64."
[[ "$CODENAME" == "jammy" ]] || warn "Ubuntu 22.04 Jammy expected; detected codename '$CODENAME'."
[[ "$ROS_DISTRO" == "humble" ]] || die "This script is written for ROS_DISTRO=humble."

LOG_FILE="/var/log/ros2_humble_setup_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"
chmod 0644 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

info "OS: $OS_PRETTY"
info "Architecture: $ARCH"
info "Target user: $REAL_USER"
info "Workspace: $COLCON_WS"
info "ROS distro: $ROS_DISTRO"
info "Log file: $LOG_FILE"

retry apt_update || die "apt update failed"

# -----------------------------------------------------------------------------
banner "Step 1: Locale (official)"

apt_install_required locales

locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
ok "Locale configured (en_US.UTF-8)"

# -----------------------------------------------------------------------------
banner "Step 2: Enable universe (official)"

apt_install_required software-properties-common

if command -v add-apt-repository >/dev/null 2>&1; then
    add-apt-repository -y universe >/dev/null 2>&1 \
        && ok "universe component enabled" \
        || warn "Could not enable universe (add-apt-repository failed); continuing"
else
    warn "add-apt-repository not available; assuming universe is enabled"
fi

# -----------------------------------------------------------------------------
banner "Step 3: ROS 2 apt Source (official)"

apt_install_required curl ca-certificates gnupg lsb-release

install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor --batch --yes -o /etc/apt/keyrings/ros-archive-keyring.gpg
chmod 0644 /etc/apt/keyrings/ros-archive-keyring.gpg

cat > /etc/apt/sources.list.d/ros2.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${CODENAME} main
EOF

retry apt_update || die "apt update failed after adding ROS 2 repository"
ok "ROS 2 repository configured"

# -----------------------------------------------------------------------------
banner "Step 3a: ROS Snapshot Fallback (arm64 packages dropped from live repo)"
add_ros_snapshot_fallback

# -----------------------------------------------------------------------------
banner "Step 4: ROS 2 ${ROS_DISTRO} desktop-full (official, NO upgrade)"

# NOTE: the official instructions run `apt upgrade` before installing. It is
# intentionally not run here - see the header comment.
ROS_PACKAGES=(
    ros-dev-tools
    python3-argcomplete
    python3-colcon-common-extensions
    python3-colcon-mixin
    python3-rosdep
    python3-vcstool
)

if [[ "$INSTALL_DESKTOP_FULL" == "1" ]]; then
    if apt_has_candidate "ros-${ROS_DISTRO}-desktop-full"; then
        ROS_PACKAGES+=("ros-${ROS_DISTRO}-desktop-full")
        info "Installing ros-${ROS_DISTRO}-desktop-full ($(version_candidate "ros-${ROS_DISTRO}-desktop-full"))"
    else
        warn "ros-${ROS_DISTRO}-desktop-full has no candidate; installing ros-${ROS_DISTRO}-desktop instead"
        ROS_PACKAGES+=("ros-${ROS_DISTRO}-desktop")
    fi
else
    ROS_PACKAGES+=("ros-${ROS_DISTRO}-ros-base")
fi

apt_install_required "${ROS_PACKAGES[@]}"
ok "ROS 2 ${ROS_DISTRO} installed"

# -----------------------------------------------------------------------------
banner "Step 5: rosdep"

apt_install_required python3-rosdep

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    rosdep init
else
    info "rosdep already initialized"
fi
run_as_user "rosdep update" || warn "rosdep update failed; re-run 'rosdep update' manually"
ok "rosdep ready"

# -----------------------------------------------------------------------------
banner "Step 6: Gazebo/OSRF Repository"
if [[ "$INSTALL_GAZEBO" == "1" ]]; then
    reset_gazebo_sources
    add_osrf_gazebo_repo
    retry apt_update || die "apt update failed after adding Gazebo repository"
    ok "Gazebo/OSRF repository configured"
else
    warn "Skipping Gazebo repository because INSTALL_GAZEBO=0"
fi

# -----------------------------------------------------------------------------
banner "Step 7: Gazebo Fortress"
if [[ "$INSTALL_GAZEBO" == "1" ]]; then
    # Force a consistent Ignition Sensors 6 stack before installing Gazebo 6.
    require_candidate_at_least libignition-sensors6 6.8.1

    apt_install_required \
        libignition-sensors6 \
        libignition-sensors6-air-pressure \
        libignition-sensors6-altimeter \
        libignition-sensors6-boundingbox-camera \
        libignition-sensors6-camera \
        libignition-sensors6-depth-camera \
        libignition-sensors6-force-torque \
        libignition-sensors6-gpu-lidar \
        libignition-sensors6-imu \
        libignition-sensors6-logical-camera \
        libignition-sensors6-magnetometer \
        libignition-sensors6-navsat \
        libignition-sensors6-rendering \
        libignition-sensors6-rgbd-camera \
        libignition-sensors6-segmentation-camera \
        libignition-sensors6-thermal-camera

    apt_install_available \
        ignition-fortress \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ros-gz-bridge \
        ros-${ROS_DISTRO}-ros-gz-sim \
        ros-${ROS_DISTRO}-ros-gz-sim-demos \
        ros-${ROS_DISTRO}-simulation
    ok "Gazebo Fortress + ros_gz bridge/sim installed"
else
    warn "Skipping Gazebo because INSTALL_GAZEBO=0"
fi

# -----------------------------------------------------------------------------
banner "Step 8: MoveIt 2"
if [[ "$INSTALL_MOVEIT" == "1" ]]; then
    apt_install_required \
        ros-${ROS_DISTRO}-moveit \
        ros-${ROS_DISTRO}-moveit-planners \
        ros-${ROS_DISTRO}-moveit-ros-planning-interface \
        ros-${ROS_DISTRO}-moveit-servo \
        ros-${ROS_DISTRO}-moveit-setup-assistant \
        ros-${ROS_DISTRO}-moveit-visual-tools \
        ros-${ROS_DISTRO}-pilz-industrial-motion-planner
    # The meta package does NOT pull in the Python bindings, and unlike Jazzy
    # there is no apt package for them on Humble: 'import moveit' only works
    # after building moveit_py from source (see the cix-panda-vision-grasp
    # guide docs/ubuntu22.04-moveit2-source-build.md).
    ok "MoveIt 2 installed (Python bindings need a moveit_py source build on Humble)"
else
    warn "Skipping MoveIt 2 because INSTALL_MOVEIT=0"
fi

# -----------------------------------------------------------------------------
banner "Step 9: Nav2 and SLAM (Complete)"
if [[ "$INSTALL_NAV2" == "1" ]]; then
    apt_install_required \
        ros-${ROS_DISTRO}-cartographer-ros \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-nav2-costmap-2d \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-robot-localization \
        ros-${ROS_DISTRO}-slam-toolbox \
        ros-${ROS_DISTRO}-tf2-tools \
        ros-${ROS_DISTRO}-nav2-smac-planner \
        ros-${ROS_DISTRO}-nav2-bt-navigator \
        ros-${ROS_DISTRO}-nav2-smoother \
        ros-${ROS_DISTRO}-nav2-behaviors \
        ros-${ROS_DISTRO}-nav2-regulated-pure-pursuit-controller \
        ros-${ROS_DISTRO}-nav2-amcl \
        ros-${ROS_DISTRO}-nav2-map-server \
        ros-${ROS_DISTRO}-nav2-lifecycle-manager
    ok "Nav2 complete stack (SMAC, BT Navigator, smoother, costmap, cartographer) installed"
else
    warn "Skipping Nav2 because INSTALL_NAV2=0"
fi

# -----------------------------------------------------------------------------
banner "Step 10: Robot Tools and Sensor Packages"
ROBOT_PACKAGES=(
    ros-${ROS_DISTRO}-bondcpp
    ros-${ROS_DISTRO}-controller-manager
    ros-${ROS_DISTRO}-diagnostic-updater
    ros-${ROS_DISTRO}-diagnostic-msgs
    ros-${ROS_DISTRO}-diff-drive-controller
    ros-${ROS_DISTRO}-gz-ros2-control
    ros-${ROS_DISTRO}-gz-ros2-control-demos
    ros-${ROS_DISTRO}-image-transport
    ros-${ROS_DISTRO}-image-transport-plugins
    ros-${ROS_DISTRO}-joint-state-publisher
    ros-${ROS_DISTRO}-joint-state-publisher-gui
    ros-${ROS_DISTRO}-joy
    ros-${ROS_DISTRO}-joy-teleop
    ros-${ROS_DISTRO}-pluginlib
    ros-${ROS_DISTRO}-rclcpp-lifecycle
    ros-${ROS_DISTRO}-pcl-ros
    ros-${ROS_DISTRO}-robot-state-publisher
    ros-${ROS_DISTRO}-ros2-control
    ros-${ROS_DISTRO}-ros2-controllers
    ros-${ROS_DISTRO}-rqt
    ros-${ROS_DISTRO}-rqt-common-plugins
    ros-${ROS_DISTRO}-rviz2
    ros-${ROS_DISTRO}-sensor-msgs
    ros-${ROS_DISTRO}-teleop-twist-joy
    ros-${ROS_DISTRO}-teleop-twist-keyboard
    ros-${ROS_DISTRO}-tf-transformations
    ros-${ROS_DISTRO}-visualization-msgs
    ros-${ROS_DISTRO}-twist-mux
    ros-${ROS_DISTRO}-xacro
)
apt_install_available "${ROBOT_PACKAGES[@]}"
ok "Robot tools and sensor packages installed (joy-teleop + twist-mux + gz-ros2-control)"

# -----------------------------------------------------------------------------
banner "Step 11: DDS Middleware (FastDDS + CycloneDDS)"
# Both implementations are installed; no global RMW_IMPLEMENTATION is exported
# (the demos pick their own - see the note in the shell block below).
# FastDDS is the ROS default; CycloneDDS is required by the camera_handpose demo.
apt_install_required \
    ros-${ROS_DISTRO}-fastrtps \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp
apt_install_available \
    ros-${ROS_DISTRO}-cyclonedds \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp
ok "FastDDS + CycloneDDS installed (no global RMW default; demos select their own)"

# -----------------------------------------------------------------------------
banner "Step 12: Python Development Libraries"
if [[ "$INSTALL_PYTHON_EXTRAS" == "1" ]]; then
    # Install these via apt rather than pip: jammy is PEP 668 managed
    # (externally-managed-environment), so system pip installs are refused.
    apt_install_available \
        python3-numpy \
        python3-empy \
        python3-catkin-pkg \
        python3-lark \
        python3-matplotlib \
        python3-scipy \
        python3-serial \
        python3-transforms3d \
        python3-pynput
    ok "Python extras installed via apt"
else
    warn "Skipping Python extras because INSTALL_PYTHON_EXTRAS=0"
fi

# -----------------------------------------------------------------------------
banner "Step 13: ARM Hardware Support"
if [[ "$INSTALL_HARDWARE_TOOLS" == "1" ]]; then
    apt_install_available libraspberrypi-dev
    apt_install_required \
        i2c-tools \
        gpiod \
        libgpiod-dev \
        libi2c-dev \
        libserial-dev \
        minicom \
        python3-libgpiod \
        python3-smbus2
    ok "ARM hardware support installed"
else
    warn "Skipping hardware tools because INSTALL_HARDWARE_TOOLS=0"
fi

# -----------------------------------------------------------------------------
banner "Step 14: Development Tools"
apt_install_required \
    ccache \
    clang-format \
    clang-tidy \
    clangd \
    doxygen \
    gdb \
    graphviz \
    htop \
    iputils-ping \
    net-tools \
    ninja-build \
    tmux \
    tree \
    valgrind
ok "Development tools installed"

# -----------------------------------------------------------------------------
banner "Step 15: Colcon Workspace"
if [[ "$CREATE_WORKSPACE" == "1" ]]; then
    run_as_user "mkdir -p '$COLCON_WS/src'"
    run_as_user "cd '$COLCON_WS' && source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"
    ok "Workspace ready: $COLCON_WS"
else
    warn "Skipping workspace creation because CREATE_WORKSPACE=0"
fi

# -----------------------------------------------------------------------------
banner "Step 16: Shell Environment"
# One managed block holding both the ROS source line and the extras.
BASHRC="$REAL_HOME/.bashrc"
MARKER_BEGIN="# >>> ROS2 Humble setup >>>"
MARKER_END="# >>> end ROS2 Humble setup >>>"

strip_legacy_ros_blocks "$BASHRC"

ROS_ENV_BLOCK=$(cat <<EOF
source /opt/ros/${ROS_DISTRO}/setup.bash 2>/dev/null || true
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash 2>/dev/null || true

# Gazebo (Ignition Fortress) resource path
export IGN_GAZEBO_RESOURCE_PATH=\${IGN_GAZEBO_RESOURCE_PATH:-}:/opt/ros/${ROS_DISTRO}/share

# ccache for faster compilation
export CC='ccache gcc'
export CXX='ccache g++'
export CCACHE_DIR=\$HOME/.ccache

# Convenience aliases for ROS development
# Usage: cd your_workspace, then 'cs' to source the workspace
alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
alias cs='source install/setup.bash'
alias ros_env='env | grep -iE "ROS|RMW|IGN"'

# RMW_IMPLEMENTATION is intentionally NOT set: the camera_handpose demo selects
# rmw_cyclonedds_cpp itself, and the other demos expect the ROS default.
EOF
)

{
    echo ""
    echo "$MARKER_BEGIN"
    echo "$ROS_ENV_BLOCK"
    echo "$MARKER_END"
} >> "$BASHRC"
chown "$REAL_USER:$REAL_USER" "$BASHRC"
ok "Shell environment written: $BASHRC (one managed block)"

# -----------------------------------------------------------------------------
banner "Step 17: colcon mixin and metadata"
run_as_user "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml 2>/dev/null || true"
run_as_user "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon mixin update || true"
run_as_user "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon metadata add default https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml 2>/dev/null || true"
run_as_user "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon metadata update || true"
ok "colcon mixin and metadata configured"

# -----------------------------------------------------------------------------
banner "Step 18: Verification Script"
VERIFY_SCRIPT="$REAL_HOME/$VERIFY_SCRIPT_NAME"
cat <<VERIFY_EOF | write_file_as_user "$VERIFY_SCRIPT" 0755
#!/usr/bin/env bash
set -uo pipefail

D="\${ROS_DISTRO:-humble}"
# Resolve the workspace independently of who invokes this script: under 'sudo'
# \$HOME is /root, so fall back to the invoking user's home via SUDO_USER.
if [[ -n "\${SUDO_USER:-}" && "\${SUDO_USER}" != "root" ]]; then
    _home="\$(getent passwd "\$SUDO_USER" | cut -d: -f6)"
else
    _home="\${HOME:-}"
fi
WS="\${ROS_WS:-\${_home:-\$HOME}/ros2_ws}"
# 'set -u' must be off while sourcing: ROS's setup.bash dereferences variables
# it deliberately leaves unset, which would abort the script before any output.
set +u
source /opt/ros/\$D/setup.bash 2>/dev/null || true
source "\$WS/install/setup.bash" 2>/dev/null || true
set -u

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "\${GREEN}[PASS]\${NC} \$*"; PASS_COUNT=\$((PASS_COUNT + 1)); }
fail() { echo -e "\${RED}[FAIL]\${NC} \$*"; FAIL_COUNT=\$((FAIL_COUNT + 1)); }
warn() { echo -e "\${YELLOW}[WARN]\${NC} \$*"; }

PASS_COUNT=0
FAIL_COUNT=0

# One call into a variable: running 'ros2 pkg list | grep -q' per package makes
# ros2 die with SIGPIPE (exit 120), which 'pipefail' then reports as a failure.
ROS_PKGS="\$(ros2 pkg list 2>/dev/null || true)"
has_pkg() { grep -qx "\$1" <<<"\$ROS_PKGS"; }

echo "==============================="
echo "  ROS 2 Humble Environment Check (\$D)"
echo "==============================="

echo ""
echo "--- Base ROS 2 ---"
[[ -d "/opt/ros/\$D" ]] && pass "Install tree /opt/ros/\$D present" || fail "/opt/ros/\$D missing"
command -v ros2 >/dev/null 2>&1 && pass "ros2 command available" || fail "ros2 command not available"
[[ "\${ROS_DISTRO:-}" == "\$D" ]] && pass "ROS_DISTRO=\$D" || fail "ROS_DISTRO is '\${ROS_DISTRO:-unset}'"
command -v colcon >/dev/null 2>&1 && pass "colcon available" || fail "colcon not available"
command -v rosdep >/dev/null 2>&1 && pass "rosdep available" || warn "rosdep not available"

echo ""
echo "--- desktop-full ---"
has_pkg rviz2 && pass "rviz2 installed" || fail "rviz2 not found"
has_pkg rviz_default_plugins && pass "rviz_default_plugins installed" || fail "rviz_default_plugins not found"
has_pkg robot_state_publisher && pass "robot_state_publisher installed" || fail "robot_state_publisher not found"
has_pkg turtlesim && pass "turtlesim installed" || fail "turtlesim not found"

echo ""
echo "--- extras ---"
if command -v ign >/dev/null 2>&1 || command -v gz >/dev/null 2>&1; then
    pass "Ignition/Gazebo command available"
else
    warn "Ignition/Gazebo command not found"
fi
has_pkg moveit_core && pass "MoveIt 2 installed" || fail "MoveIt 2 (moveit_core) not found"
has_pkg nav2_bringup && pass "Nav2 installed" || fail "Nav2 (nav2_bringup) not found"
has_pkg nav2_smac_planner && pass "Nav2 SMAC Planner installed" || warn "Nav2 SMAC Planner not found"
has_pkg nav2_smoother && pass "Nav2 smoother installed" || warn "Nav2 smoother not found"
has_pkg nav2_costmap_2d && pass "Nav2 costmap_2d installed" || warn "Nav2 costmap_2d not found"
has_pkg cartographer_ros && pass "cartographer_ros installed" || warn "cartographer_ros not found"
has_pkg rmw_fastrtps_cpp && pass "FastDDS RMW installed" || fail "FastDDS RMW not found"
has_pkg rmw_cyclonedds_cpp && pass "CycloneDDS RMW installed (camera_handpose)" || warn "CycloneDDS RMW not found; camera_handpose needs it"
has_pkg twist_mux && pass "twist_mux installed" || warn "twist_mux not found"
has_pkg gz_ros2_control && pass "gz_ros2_control installed" || warn "gz_ros2_control not found"
has_pkg ros_gz_bridge && pass "ros_gz_bridge installed" || warn "ros_gz_bridge not found"

echo ""
echo "--- environment ---"
# RMW_IMPLEMENTATION is deliberately unset globally (demos choose their own),
# so it is only reported, never required.
if [[ -n "\${RMW_IMPLEMENTATION:-}" ]]; then
    warn "RMW_IMPLEMENTATION is set globally to '\$RMW_IMPLEMENTATION' (this installer leaves it unset)"
else
    pass "RMW_IMPLEMENTATION unset (ROS default; demos override as needed)"
fi
[[ -d "\$WS/src" ]] && pass "Workspace exists: \$WS" || fail "Workspace not found: \$WS"
command -v ccache >/dev/null 2>&1 && pass "ccache available" || warn "ccache not available"
python3 --version >/dev/null 2>&1 && pass "Python3: \$(python3 --version)" || fail "Python3 not found"
# empy 3.x (jammy's python3-empy) installs the module as 'em', not 'empy'.
python3 -c "import em" 2>/dev/null && pass "python3-empy importable (module em)" || warn "python3-empy not importable"

echo ""
echo "--- ROS graph quick check ---"
timeout 5 ros2 topic list >/dev/null 2>&1 \
    && pass "ros2 topic list responded" \
    || warn "ros2 topic list timed out; normal when no nodes are active"

echo ""
echo "==============================="
echo "  Result: \${PASS_COUNT} passed, \${FAIL_COUNT} failed"
echo "==============================="

exit "\$FAIL_COUNT"
VERIFY_EOF
ok "Verification script written: $VERIFY_SCRIPT"

# -----------------------------------------------------------------------------
banner "Step 19: VSCode Configuration"
VSCODE_DIR="$COLCON_WS/.vscode"
run_as_user "mkdir -p '$VSCODE_DIR'"

cat <<EOF | write_file_as_user "$VSCODE_DIR/c_cpp_properties.json" 0644
{
    "configurations": [
        {
            "name": "ROS2-Humble-ARM64",
            "includePath": [
                "\${workspaceFolder}/**",
                "/opt/ros/${ROS_DISTRO}/include/**",
                "/usr/include/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-arm64",
            "configurationProvider": "ms-vscode.cmake-tools"
        }
    ],
    "version": 4
}
EOF

cat <<EOF | write_file_as_user "$VSCODE_DIR/settings.json" 0644
{
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.configureOnOpen": true,
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "files.associations": {
        "*.launch.py": "python",
        "*.sdf": "xml",
        "*.urdf": "xml",
        "*.world": "xml",
        "*.xacro": "xml"
    },
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "ros.distro": "${ROS_DISTRO}"
}
EOF

cat <<'EOF' | write_file_as_user "$VSCODE_DIR/extensions.json" 0644
{
    "recommendations": [
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "ms-python.python",
        "ms-iot.vscode-ros",
        "twxs.cmake",
        "dotjoshjohnson.xml",
        "xaver.clang-format",
        "eamodio.gitlens"
    ]
}
EOF
chown -R "$REAL_USER:$REAL_USER" "$VSCODE_DIR"
ok "VSCode configuration written: $VSCODE_DIR"

# -----------------------------------------------------------------------------
banner "Step 20: Cleanup"
# 'apt autoremove' is intentionally not run: it can remove kernel and firmware
# packages that are still needed. 'autoclean' only drops expired .deb files.
apt-get autoclean
ok "apt cache cleaned (autoremove skipped by design)"

# -----------------------------------------------------------------------------
banner "Installation Complete"
cat <<EOF
ROS 2 ${ROS_DISTRO} setup finished.

Installed/configured:
  - ROS 2 ${ROS_DISTRO} desktop-full (official method, no system upgrade)
  - Gazebo Fortress / ros_gz: $([[ "$INSTALL_GAZEBO" == "1" ]] && echo installed || echo skipped)
  - MoveIt 2: $([[ "$INSTALL_MOVEIT" == "1" ]] && echo installed || echo skipped)
  - Nav2/SLAM: $([[ "$INSTALL_NAV2" == "1" ]] && echo installed || echo skipped) (Complete with SMAC, BT Navigator)
  - FastDDS + CycloneDDS middleware (no global RMW default)
  - twist_mux + joy-teleop + gz_ros2_control
  - Python libs, ARM tools, dev tools
  - Workspace: ${COLCON_WS}
  - Shell block (colcon-argcomplete, IGN_GAZEBO_RESOURCE_PATH, ccache, cb/cs/ros_env)
  - Verification script: ${VERIFY_SCRIPT}
  - Log: ${LOG_FILE}

Next steps:
  source ~/.bashrc
  bash ~/${VERIFY_SCRIPT_NAME}
  rviz2

Not installed by this script (by design):
  - cix-ros-${ROS_DISTRO}-camera / dmabuf / noe-handpose / noe-yolo
    (install from the CIX PPA in a separate step)
  - moveit_py Python bindings (Humble: requires a source build, see
    cix-panda-vision-grasp/docs/ubuntu22.04-moveit2-source-build.md)
EOF
