# camera_handpose demo

End-to-end zero-copy pipeline: **V4L2 ISP camera → dmabuf broker → NPU handpose
inference → debug image + finger-direction topic.** Image data is never memcpy'd
across process boundaries while the broker runs.

```
v4l2_dmabuf_camera (proc A)            handpose_dma_teleop (proc B)
  V4L2 EXPBUF → dmabuf fd                subscribes /cam0_stream1
  register fd → broker ──┐                parse ;dmabuf_handle=N from encoding
                         │  UDS + SCM_RIGHTS
  publish /cam0_stream1  │  (fd, no payload copy)
  (encoding carries ─────┘  fetch fd → dup of SAME physical memory
   ;dmabuf_handle=N)        → NPU infer directly on dmabuf (zero copy)
                            publish /handpose/debug_image, /handpose/finger_direction
```

## Packages used

| Package | Role | Built by you? |
|---------|------|---------------|
| `cix_ros_dmabuf` | `cix_ros_dmabuf_broker` — shared dmabuf zero-copy transport | yes (colcon) |
| `cix_ros_camera` | `v4l2_dmabuf_camera` — V4L2 ISP camera publisher | yes (colcon) |
| `cix_ros_noe_handpose` | `handpose_dmabuf_node` (NPU inference) | yes (colcon) |

This demo only orchestrates them — it does **not** build them. It also ships
the robot-control bridge scripts in `teleop/` (see [Teleop](#teleop)), which
are demo-level code and not part of any package.

## Prerequisites (on the board)

1. **All three packages built and sourced.** From a colcon workspace whose
   `src/` contains the packages under `ros_packages/`:
   ```bash
   cd ~/robotics_sdk
   colcon build --symlink-install --packages-select cix_ros_dmabuf cix_ros_camera cix_ros_noe_handpose \
     --base-paths ros_packages
   source install/setup.bash
   ```
2. **ISP stack up:** `armcb_isp_v4l2` loaded, `/dev/video9` present, `isp_app`
   running (managed by `isp-daemon.service`).
   ```bash
   sudo systemctl restart isp-daemon   # if /dev/video9 is missing
   ```
3. **(Optional, for preview)** A local graphical session on the board for
   `rqt_image_view`. Over SSH without a display, the pipeline still runs — just
   no preview window.

## Dependencies

The demo only orchestrates the three ROS packages; it adds **no build
dependencies** of its own. The DDS / shared-memory transport it relies on is
shipped by standard ROS 2 apt packages — no system files are modified,
no custom libraries are compiled.

### Core (always required)

Package names below are the ROS 2 Humble (Ubuntu 22.04) forms; on ROS 2 Jazzy
(Ubuntu 24.04) replace the `ros-humble-` prefix with `ros-jazzy-`. The demo
scripts detect the installed distro automatically (`ROS_DISTRO` overrides).

| Dependency | apt package | Role |
|------------|-------------|------|
| ROS 2 Humble | `ros-humble-ros-base` | rclpy / rclcpp, colcon, `ros2` CLI |
| Cyclone DDS | `ros-humble-cyclonedds` | DDS implementation (config via `cyclonedds_localhost.xml`) |
| rmw adapter | `ros-humble-rmw-cyclonedds-cpp` | binds ROS to Cyclone; set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |

### Shared-memory transport (optional but recommended)

Without it the `/handpose/debug_image` subscription rate is capped at ~17 fps by
per-frame deserialization of the 0.69 MB BGR frame. With it the rate matches the
~35 fps publish rate (Iceoryx passes a shared-memory pointer instead of
serializing the pixels). Enabled by `<SharedMemory><Enable>true</Enable>` in
`config/cyclonedds_localhost.xml` and auto-managed by `run_demo.sh`.

| Dependency | apt package | Role |
|------------|-------------|------|
| Iceoryx posh | `ros-humble-iceoryx-posh` | provides `iox-roudi` daemon + `roudi_config_example.toml` |
| Iceoryx hoofs | `ros-humble-iceoryx-hoofs` | Iceoryx support lib |
| Iceoryx binding-c | `ros-humble-iceoryx-binding-c` | C bindings libddsc links against |

> **Cyclone must be built with SHM (`DDS_HAS_SHM`).** The official ROS Humble
> apt build of `ros-humble-cyclonedds` links iceoryx by default (verify with
> `ldd $(find /opt/ros/humble -name 'libddsc.so*') | grep iceoryx`). If someone
> compiles Cyclone from source without SHM, the `<SharedMemory>` element is
> ignored and the demo silently falls back to socket transport.

### How `run_demo.sh` handles missing SHM deps

`ensure_roudi()` degrades gracefully — if `iox-roudi` is absent it only warns
and the demo keeps running on socket transport (no crash):
```
iox-roudi not found -- SHM disabled, debug_image may drop frames
```

### Known issues on this stack

- `ros2 topic hz /handpose/debug_image` occasionally crashes with
  `terminate called without an active exception` under SHM transport. This is a
  `rmw_cyclonedds_cpp` + iceoryx quirk affecting short-lived subscribers; the
  persistent `rqt_image_view` subscriber is unaffected.
- Iceoryx SHM requires the `iox-roudi` daemon running **before** any ROS node.
  `run_demo.sh` starts/stops it automatically; if you launch nodes manually,
  start `iox-roudi -c /opt/ros/humble/etc/roudi_config_example.toml` first.

### Portability notes (other platforms)

The demo runs on both supported distros — ROS 2 Humble (Ubuntu 22.04) and
ROS 2 Jazzy (Ubuntu 24.04). `run_demo.sh` / `env.sh` auto-detect the distro
under `/opt/ros` (Humble preferred when both are installed; override with
`ROS_DISTRO=<distro>`) and use it for `setup.bash`, `iox-roudi`, and
`roudi_config_example.toml`. On a non-standard install prefix these paths
still need adjusting. SHM also needs the Cyclone + iceoryx apt packages
present on the target; the `<SharedMemory>` element name is `<Enable>` in
Cyclone 0.10.x (later versions changed the schema).

## Run

```bash
bash ~/robotics_sdk/demos/camera_handpose/run_demo.sh
```

This starts broker → camera → handpose, verifies the zero-copy link, and opens
`rqt_image_view` on `/handpose/debug_image` (the downsampled overlay, smoothest).

Other previews:
```bash
bash run_demo.sh start /cam0_stream1       # raw camera frame
bash run_demo.sh start /handpose/image_out # inference output
```

Stop:
```bash
bash run_demo.sh stop
# or
bash stop_demo.sh
```

## Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/cam0_stream1` | `sensor_msgs/Image` | camera → handpose (encoding carries `;dmabuf_handle=N`) |
| `/handpose/debug_image` | `sensor_msgs/Image` | handpose → rqt (skeleton overlay) |
| `/handpose/image_out` | `sensor_msgs/Image` | handpose → rqt (clean output) |
| `/handpose/keypoints` | `cix_ros_noe_handpose/HandPoseKeypoints` | handpose → consumer (21 keypoints, only while subscribed) |
| `/handpose/finger_direction` | `geometry_msgs/Vector3` | handpose → consumer (±1 image-space direction) |

## Verify zero-copy

Run these from a second terminal with the demo env sourced first (Cyclone DDS
localhost-only, `ROS_DOMAIN_ID=0` — otherwise the CLI sees no topics):

```bash
source ~/robotics_sdk/demos/camera_handpose/env.sh
```

```bash
ros2 topic hz /cam0_stream1                         # ~35 fps
ros2 topic bw /cam0_stream1                         # tiny (header+encoding only, no pixel payload)
ros2 topic echo /handpose/finger_direction --once   # Vector3 from index-finger direction
grep Zero-copy /tmp/handpose.log                    # "Zero-copy: N (≈100%)"
```

The reliable zero-copy signal is the handpose log line
`dmabuf broker connected at ...; will use zero-copy` plus `Zero-copy: N (~100%)`.
The camera's own `zero-copy:` counter only checks `dmabuf_fd >= 0`, not whether
`register_fd` succeeded — not reliable alone.

## Teleop

The robot-control bridge scripts (moved here from the handpose package so the
inference package stays generic) live in `teleop/`:

| Script | Purpose |
|--------|---------|
| `finger_direction_to_cmd_vel.py` | `/handpose/finger_direction` (`Vector3`) → `Twist` on `key_vel` (e.g. for `twist_mux`) |
| `handpose_twist_teleop.py` | Standalone USB-camera variant: own Python inference + `Twist` output |

With the launch file, start the bridge alongside the pipeline via the `teleop`
argument (off by default — it drives a robot base):

```bash
ros2 launch camera_handpose_demo.launch.py teleop:=true
```

Or run each script's own launch file directly (see `teleop/*.launch.py`; the
`handpose_twist_teleop` launch exposes the same parameters as before the move).
Do **not** run `finger_direction_to_cmd_vel` twice in the same graph — `key_vel`
must have a single publisher.

## Configuration

Defaults live in `run_demo.sh` (device, resolution, fps, format, topics). The
broker socket is `/tmp/cix_ros_dmabuf_broker.sock` — this is the built-in
default of `cix_ros_dmabuf`'s broker and of `cix_ros_camera`'s publisher side,
so handpose is pointed at it via `broker_socket_path`. The handpose package
itself has **no** camera-package dependency; the socket path, image topic, and
encoding are all parameters.

Pixel format defaults to **RGB3** (`rgb8` on the wire). The handpose node
supports `rgb8`/`bgr8`/`nv12` dmabuf zero-copy; switch the camera to `NV12` by
editing `CAM_FMT` in `run_demo.sh`.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| handpose logs `broker not running; falling back to memcpy` | broker socket path mismatch or broker not started. Confirm `/tmp/cix_ros_dmabuf_broker.sock` is listening (`ss -xl \| grep cix_ros_dmabuf`). |
| `Zero-copy: 0 (0.0%)` | encoding reaching handpose has no `;dmabuf_handle=`. Check the camera exported a dmabuf fd (`/tmp/cam.log` `Buffer N: dmabuf fd=...`); if EXPBUF failed, camera falls back to MMAP and there is no fd to share. |
| rqt window blank | missing `CYCLONEDDS_URI` / not localhost-only. `run_demo.sh` sets it; if launching rqt manually, source the same env. |
| `ros2 pkg prefix cix_ros_camera` fails | workspace `install/setup.bash` not sourced. |
| `debug_image` stuck at ~17 fps despite ~35 fps publish | SHM transport inactive. Check `iox-roudi` is running (`pgrep -f iox-roudi`) and `<SharedMemory><Enable>true</Enable>` is in the Cyclone config; without SHM the 0.69 MB/frame deserialization caps the subscription rate. |
| `rmw handle is invalid` / `ROUDI_NOT_AVAILABLE` at node start | `<SharedMemory>` enabled but `iox-roudi` not running. `run_demo.sh` starts it automatically; if launching manually, start `iox-roudi` first. |
| `SharedMemory: unknown element` in Cyclone log | wrong element name for this Cyclone version. 0.10.x uses `<Enable>`, not `<Enabled>`. |

## Files

```
camera_handpose/
├── README.md
├── run_demo.sh                         # orchestrator (start/stop, preflight, verify)
├── env.sh                              # source in CLI terminals to match the demo DDS env
├── stop_demo.sh                        # = run_demo.sh stop
├── camera_handpose_demo.launch.py      # single-launch composition of the 3 processes
├── teleop/                             # robot-control bridges (moved from handpose pkg)
│   ├── finger_direction_to_cmd_vel.py  #   Vector3 -> Twist on key_vel
│   ├── finger_direction_to_cmd_vel.launch.py
│   ├── handpose_twist_teleop.py        #   standalone USB-camera inference + teleop
│   └── handpose_twist_teleop.launch.py
└── config/cyclonedds_localhost.xml     # DDS localhost-only (no LAN flooding)
```
