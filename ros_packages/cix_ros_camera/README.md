# cix_ros_camera

ROS 2 camera package for the **CIX ISP** pipeline. Captures
frames over V4L2 with **true zero-copy dmabuf** transport — image data is never
memcpy'd across process boundaries when the broker runs.

- Uses a standard ROS 2 Type Adapter (`DmaBufferImage ↔ sensor_msgs::msg::Image`).
- **V4L2 dmabuf only** — talks to the kernel ISP driver directly via V4L2 +
  EXPBUF.
- The cross-process zero-copy transport (dmabuf broker + `DmaBuffer` /
  `BrokerClient`) lives in the shared **`cix_ros_dmabuf`** package, so
  subscribers like `cix_ros_noe_handpose` get zero-copy without depending on
  this camera package.
---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  v4l2_dmabuf_camera   (standalone executable)                   │
│   - opens /dev/video9 (MPLANE), sets S_FMT, queues 4 dmabuf bufs│
│   - EXPBUF → dmabuf fd per buffer                                │
│   - publishes <topic>  via TypeAdapter<DmaBufferImage>           │
│     (sensor_msgs/msg/Image on the wire, QoS = SensorDataQoS)    │
└─────────────────────────────▲───────────────────────────────────┘
                              │ dmabuf fd
┌─────────────────────────────┴───────────────────────────────────┐
│  DmaBufferImage + Type Adapter  (zero-copy transport layer)     │
│   - DmaBuffer: RAII wrapper around a dmabuf fd                   │
│   - TypeAdapter: DmaBufferImage ↔ sensor_msgs::msg::Image        │
│   - Publisher registers fd → handle; encodes handle in encoding  │
└─────────────────────────────▲───────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│  cix_ros_dmabuf_broker   (cix_ros_dmabuf pkg, cross-proc fd)    │
│   - UDS server at /tmp/cix_ros_dmabuf_broker.sock                │
│   - register_fd / fetch_fd / unregister over SCM_RIGHTS         │
│   - hands subscribers a NEW fd aliasing the SAME physical memory│
└─────────────────────────────▲───────────────────────────────────┘
                              │ V4L2 EXPBUF
┌─────────────────────────────┴───────────────────────────────────┐
│  armcb_isp_v4l2.ko   (kernel driver, vb2_dma_contig, GPL-2.0)   │
└─────────────────────────────────────────────────────────────────┘
```

The dmabuf fd from EXPBUF flows untouched into the published message. When the
broker runs, a subscriber in another process gets a *new* fd that aliases the
*same* physical buffer — zero memcpy.

---

## Build

```bash
colcon build --symlink-install --packages-select cix_ros_dmabuf cix_ros_camera
```

The package links only `pthread dl m rt` + the ROS deps declared in
`package.xml` — no `isp_app`, no `yaml-cpp`.

---

## Run

> Prerequisites on the board: `armcb_isp_v4l2` kernel module loaded,
> `/dev/video9` present, `isp_app` running (managed by `isp-daemon.service`).

### Manual: V4L2 zero-copy pipeline

```bash
ros2 run cix_ros_dmabuf cix_ros_dmabuf_broker &       # start broker first

ros2 run cix_ros_camera v4l2_dmabuf_camera --ros-args \
  -p device:=/dev/video9 -p pixel_format:=RGB3 \
  -p width:=1280 -p height:=720 -p fps:=30.0 -p topic:=/cam0_stream1
```

Any `sensor_msgs/Image` subscriber now receives zero-copy frames. Verify:

```bash
ros2 topic hz /cam0_stream1        # ~20 fps
ros2 topic info -v /cam0_stream1   # Subscription count must be > 0
```

### Launch file

| File | What it brings up |
|------|-------------------|
| `launch/cix_camera_zero_copy_launch.py` | broker + `v4l2_dmabuf_camera` + `SubTestNode` subscriber (3-process zero-copy) |

### Topics

| Direction | Name | Type | QoS |
|-----------|------|------|-----|
| Publisher | `/cam0_stream1` (configurable) | `sensor_msgs/msg/Image` | `SensorDataQoS` (BEST_EFFORT, depth 5) |

> The wire type is plain `sensor_msgs::msg::Image`. Through the Type Adapter,
> intra-process subscribers receive a `DmaBufferImage::ConstSharedPtr` with no
> copy; inter-process subscribers get zero-copy via the broker, or memcpy
> fallback if the broker is absent.

---

## Zero-copy behavior

| Scenario | Copy? | Mechanism |
|----------|-------|-----------|
| Intra-process (same container) | None ✅ | ROS 2 intra-process `shared_ptr` pass-through |
| Inter-process, broker running | None ✅ | `SCM_RIGHTS` fd passing over Unix socket |
| Inter-process, no broker | memcpy ⚠️ | `DmaBuffer` mmapped, copied into `Image.data` |

### Cross-process zero-copy with the broker

```
v4l2 EXPBUF → fd → v4l2_dmabuf_camera (process A)
                      ↓ register(fd) → broker  (fd cached publisher-side: 4 stable fds)
                      ↓ handle encoded in Image.encoding ("rgb8;dmabuf_handle=N")
                      ↓
                Subscriber (process B)
                      ↓ fetch(handle)
                      ↓ new fd aliasing the SAME physical memory
                      ↓ zero-copy access to the frame
```

The broker is transparent to subscriber code — a plain `sensor_msgs/Image`
subscription just works. Socket path: `/tmp/cix_ros_dmabuf_broker.sock`. All
processes wanting zero-copy must reach that socket.

**Reliable zero-copy signal:** the subscriber's frame count + `Image.encoding`
containing `;dmabuf_handle=`. The camera's own `zero-copy:` log line only
checks `dmabuf_fd >= 0`, not whether `register_fd` succeeded — not reliable
alone.

---

## Parameters (`v4l2_dmabuf_camera`)

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `device` | string | `/dev/video9` | V4L2 MPLANE node |
| `width` | int | `1280` | Frame width |
| `height` | int | `720` | Frame height |
| `fps` | double | `30.0` | Requested frame rate |
| `pixel_format` | string | `RGB3` | `RGB3`, `BGR3`, or `NV12` |
| `topic` | string | `/cam0_stream1` | Output image topic |
| `frame_id` | string | `camera` | `Image.header.frame_id` |

### ISP driver quirks (V4L2)

- **RGB3 stride**: the driver reports `bytesperline = width` for packed RGB24;
  the node uses `stride = width*3`, `sizeimage = stride*height`.
- **S_FMT**: the driver rejects `S_FMT` unless `bytesperline`/`sizeimage` are
  set explicitly — the node always provides them.
- **EXPBUF**: required for dmabuf fds. Works on the rebuilt driver; on the
  stock driver it fails and the node falls back to MMAP'd buffer reuse (no
  zero-copy).
- **MPLANE**: ISP video nodes are `V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE`.

---

## Package contents

```
cix_ros_camera/
├── CMakeLists.txt
├── package.xml
├── include/cix_ros_camera/
│   └── dmabuf_type_adapter.hpp      # TypeAdapter<DmaBufferImage, Image>
├── launch/  cix_camera_zero_copy_launch.py
├── src/
│   └── v4l2_dmabuf_camera.cpp       # capture node (V4L2 + EXPBUF)
└── test/src/sub_test_node.cpp       # SubTestNode component (test subscriber)

cix_ros_dmabuf/                      # shared zero-copy transport package
├── include/cix_ros_dmabuf/
│   ├── dmabuf_image.hpp             # DmaBuffer (RAII fd) + DmaBufferImage
│   └── dmabuf_broker.hpp            # BrokerClient / BrokerServer (SCM_RIGHTS)
├── src/
│   ├── dmabuf_broker.cpp            # broker impl (register/fetch/unregister)
│   └── dmabuf_broker_main.cpp       # cix_ros_dmabuf_broker executable
└── cix_ros_dmabuf/dma_image.py      # Python BrokerClient / DmaBuffer
```

---

## Deb packaging

Yes — **the package builds into a self-contained `.deb` with no `isp_app`
dependency.** Build it with
[`scripts/build-ros-debs-board.sh`](../../scripts/build-ros-debs-board.sh)
at the repo root (run on the board, Ubuntu 22.04 arm64 + ROS 2 humble; it
colcon-builds the package and packs the deb in one step):

```bash
# on the board (arm64), from the repo root
bash scripts/build-ros-debs-board.sh cix-ros-humble-camera
sudo dpkg -i out/cix-ros-humble-camera_<ver>_arm64.deb
# binaries land under /opt/ros/humble/lib/cix_ros_camera/
```

With no unit argument the script builds all `ros_packages/` units; see
[docs/RELEASE.md](../../docs/RELEASE.md) for the full per-package deb flow.

Both in-package targets (`v4l2_dmabuf_camera`, `cix_ros_camera_sub_test`) link
only `pthread dl m rt` + the ROS deps in `package.xml` (the broker executable
now ships in the `cix-ros-<distro>-dmabuf` deb).

Runtime deps still required on the target board: `armcb_isp_v4l2.ko` loaded,
`/dev/video9` present, and (for the demo) `isp_app`/`isp-daemon` running.

---

## License

BSD-3-Clause for this package's source. The `isp_app` and the
`armcb_isp_v4l2` kernel driver retain their original (confidential / GPL-2.0)
licenses respectively.
