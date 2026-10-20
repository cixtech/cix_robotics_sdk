# cix_ros_noe_handpose

[中文说明 / Chinese](README.cn.md)

## Overview

ROS 2 package for **NPU–based hand-pose inference**. It is a **hybrid
ament_cmake + ament_cmake_python** package:

- **`handpose_dmabuf_node`** (C++, primary) — zero-copy NPU inference. Subscribes
  a `sensor_msgs/Image` whose encoding carries a `;dmabuf_handle=N` suffix,
  fetches the dmabuf fd from a broker, and runs the full inference pipeline on
  shared physical memory: **CME** does the crop/resize/cvtcolor + quantization
  normalize **directly into the NPU input dmabuf**, which is bound to the NPU
  job via `noe_specify_iobuf` (`NOE_SHARE_BUF_ATTACH_DMABUF`) — **no pixel
  memcpy** on the inference path. Publishes the full **21 keypoints**
  (`HandPoseKeypoints`), a discretized **index-finger direction** as
  `geometry_msgs/Vector3`, and optional `bgr8` image streams.
- **`handpose_dma_teleop`** (Python, alternative) — a slower Python inference
  node (same topics minus the keypoints message); useful for debugging.

The package is **camera-agnostic**: the image topic, broker socket path, model
path, and all output topics are parameters. Any camera that publishes
`;dmabuf_handle=N` encodings through a dmabuf broker works — the only package
dependency is `cix_ros_dmabuf` (the shared zero-copy transport layer), never a
camera package.

Robot-control glue that used to live here (`handpose_twist_teleop`,
`finger_direction_to_cmd_vel`) has moved to the
**camera_handpose demo** (`demos/camera_handpose/teleop/`): inference packages
stay generic, demo-level robot control stays with the demo.

---

## `handpose_dmabuf_node` — I/O and parameters (C++, primary)

- **Executable:** `handpose_dmabuf_node`
- **Node name:** `handpose_dmabuf_node`

### Publishes

| Topic (parameter) | Type | Condition | Notes |
|-------------------|------|-----------|-------|
| `direction_topic` | `geometry_msgs/Vector3` | always | Discretized index-finger direction (see convention below). |
| `keypoints_topic` | `cix_ros_noe_handpose/HandPoseKeypoints` | on subscription | All 21 keypoints in input-image pixel coordinates (`geometry_msgs/Point[]`); published when at least one subscriber is attached. Build your own gesture semantics on this. |
| `output_topic` | `sensor_msgs/Image` `bgr8` | `publish_output_image` = true | NPU input frame. QoS: sensor data. |
| `debug_topic` | `sensor_msgs/Image` `bgr8` | `publish_debug_image` = true | Skeleton + FPS overlay, downscaled by `debug_image_scale`. QoS: sensor data. |

### Subscribes

| Topic (parameter) | Type | Notes |
|-------------------|------|-------|
| `image_topic` | `sensor_msgs/Image` | Encoding carries `;dmabuf_handle=N` for zero-copy. QoS: sensor data. |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/cam0_stream1` | Input image topic. |
| `broker_socket_path` | string | `""` | Broker UDS path. Empty = built-in default from `cix_ros_dmabuf`. Mismatch → memcpy fallback. |
| `npu_model_path` | string | `""` | Path to hand-pose `.cix`. Empty = `share/cix_ros_noe_handpose/models/handpose_resnet50_size256.cix`. |
| `output_topic` | string | `/handpose/image_out` | Raw bgr8 per frame. |
| `debug_topic` | string | `/handpose/debug_image` | Debug bgr8 overlay. |
| `direction_topic` | string | `/handpose/finger_direction` | `Vector3` output. |
| `keypoints_topic` | string | `/handpose/keypoints` | `HandPoseKeypoints` output. |
| `publish_output_image` | bool | true | Enable `output_topic`. |
| `publish_debug_image` | bool | true | Enable `debug_topic`. |
| `debug_image_scale` | double | 0.5 | Downscale factor for the debug image (bandwidth optimization). |
| `min_finger_len_px` | float | 6.0 | Minimum index segment length (px) to emit a direction. |
| `require_index_dominant` | bool | false | If true, require index over other fingers; see `index_vs_other_ratio`. |
| `index_vs_other_ratio` | float | 1.15 | Used when `require_index_dominant` = true. |
| `direction_axis_ratio` | float | 1.4 | Four-way split: horizontal if \|dx\| ≥ ratio·\|dy\| (1.4 ≈ ±54° from vertical). |

### Zero-copy contract

Zero-copy requires **both**:
1. the camera publishes `sensor_msgs/Image` with an encoding carrying
   `;dmabuf_handle=N` (fd registered with the broker), and
2. `broker_socket_path` matches the path the broker listens on.

When both hold, the inference path is fully zero-copy:

```
camera dmabuf fd -> CME import (EXTERNAL_DMA)
   -> CME crfrc_ext (resize 256x256 + cvtcolor -> BGR_888)
   -> CME normalize_ext (quantize -> NCHW U8, written into NPU input dmabuf)
   -> noe_specify_iobuf (bound ONCE at init) -> noe_job_infer_sync
   -> NPU output dmabuf -> mmap + CPU dequant (S8 -> F32) -> postprocess
```

The output is only 42 int8 values, so it is dequantized on the CPU (CME has no
INT8->F32 dequant task; at this size a CME pass would cost more than it saves).

The NPU input/output dmabufs are allocated once from `/dev/dma_heap/system` and
bound to the job at init; each frame CME overwrites the input dmabuf and the NPU
reads it directly. No `memcpy` of pixel data occurs.

If the broker is unavailable, the node falls back to importing `msg->data` as
CME external memory (no explicit host memcpy in our code, though CME may use an
internal shadow buffer). Visualization (debug/output image) is the only path
that touches CPU pixels: the camera dmabuf is mmap'd read-only to draw the
keypoint overlay. It is skipped entirely when both `publish_*_image` are false.

The node logs `dmabuf broker connected - zero-copy mode enabled` and a periodic
`Frames: N (zero-copy: M)` counter.

### Model requirements

The zero-copy CME path writes quantized `UINT8 NCHW` directly into the NPU
input dmabuf, so the model input must be **UINT8 NCHW** (the bundled
`handpose_resnet50_size256.cix` is `1×3×256×256` UINT8, BGR channel order). At
init the node queries the input tensor descriptor and refuses to start if the
dtype is not UINT8 (a different dtype would need a different preprocess path).

### `geometry_msgs/Vector3` (finger direction)

Values are in **image** coordinates (remap as needed):

- `x`: **−1** left, **+1** right, **0** neutral.
- `y`: **−1** up in image, **+1** down, **0** neutral.
- `z`: **0** in the current implementation.

A zero vector indicates no / neutral output.

---

## Robot-control helpers moved to the demo

`handpose_twist_teleop` (USB-camera Python inference + teleop) and
`finger_direction_to_cmd_vel` (`Vector3` → `Twist` on `key_vel` for
`twist_mux`) now live in the camera_handpose demo
(`demos/camera_handpose/teleop/`), together with their launch files. Their
parameters are unchanged; see the demo README for usage.

---

## Requirements

- ROS 2 Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04), **ament_cmake** + **ament_cmake_python**
- `cix_ros_dmabuf` — the shared dmabuf zero-copy transport package (provides
  the `dmabuf_broker.hpp` / `dmabuf_image.hpp` headers and the broker client
  library this node links)
- CIX (target) **NOE** runtime: `libnoe.so` (cix-noe-umd, low-level
  `cix_noe_standard_api.h`) under `/usr/share/cix` (board-only; not present on
  a host mount)
- CIX (target) **CME** runtime: `libcme` (cix-libcme-dev, the 2D hardware
  accelerator with `cme_2d_crop_resize_flip_rotation_cvtcolor_ext` +
  `cme_2d_normalize_ext`) — install on the board with
  `sudo dpkg -i cix-libcme-dev_1.1.0_arm64.deb` (provides
  `/usr/include/cme/*.h` and `/usr/lib/aarch64-linux-gnu/libcme.so`). The repo's
  in-tree `libcme` component is an older simplified API and **cannot** be used.
- OpenCV, `cv_bridge`
- A camera publishing `;dmabuf_handle=N` encodings through a dmabuf broker, or
  any `bgr8`/`rgb8`/`nv12` `sensor_msgs/Image` stream (external-memory fallback)

## Build

Build on the **target board** (the NPU SDK headers/libs are not on the host).
`cix_ros_dmabuf` must be built first because this package does
`find_package(cix_ros_dmabuf)` (and its message interfaces are generated from
`msg/HandPoseKeypoints.msg` at build time).

```bash
cd <workspace>
source /opt/ros/humble/setup.bash   # or /opt/ros/jazzy on Ubuntu 24.04
colcon build --symlink-install --packages-select cix_ros_dmabuf
source install/setup.bash   # so find_package(cix_ros_dmabuf) resolves
colcon build --symlink-install --packages-select cix_ros_noe_handpose
source install/setup.bash
```

Run the C++ node directly:

```bash
ros2 run cix_ros_noe_handpose handpose_dmabuf_node \
  --ros-args -p image_topic:=/cam0_stream1 \
             -p broker_socket_path:=/tmp/cix_ros_dmabuf_broker.sock
```

## Launch files

| Launch file | Node started | Function |
|-------------|--------------|----------|
| `handpose_dma_teleop.launch.py` | `handpose_dmabuf_node` (C++) | Zero-copy NPU inference: `image_topic` → `keypoints` + `finger_direction` + optional bgr8 topics. Camera/broker provided separately. |

```bash
ros2 launch cix_ros_noe_handpose handpose_dma_teleop.launch.py \
  image_topic:=/cam0_stream1 broker_socket_path:=/tmp/cix_ros_dmabuf_broker.sock
```

For the end-to-end camera → NPU pipeline and the teleop bridge, use the
**camera_handpose demo** (`demos/camera_handpose/`), which orchestrates this
package together with `cix_ros_camera` and `cix_ros_dmabuf`. When a bridge is
already running elsewhere (e.g. an integrated simulation launch), do **not**
start a second `finger_direction_to_cmd_vel` — multiple publishers on `key_vel`
would conflict.

## License

Apache-2.0. See `package.xml`.
