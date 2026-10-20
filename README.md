# CIX P1 Robotics SDK

Robotics SDK for the **CIX P1** SoC. Wraps P1's ISP / NPU / GPU capabilities into reusable ROS 2 packages, with reference demos.

- **Target board**: CIX P1 CD8180, aarch64.
- **Supported platforms**: ROS 2 Humble (Ubuntu 22.04) and ROS 2 Jazzy (Ubuntu 24.04), aarch64.
- **Runtime deps** (board-side system libs, not in this repo): CIX NOE SDK, libcme, CIX ISP camera stack.

## Layout

### `ros_packages/` - reusable ROS 2 packages (one arm64 deb each)

| Package | Capability |
|---|---|
| [`cix_ros_dmabuf`](ros_packages/cix_ros_dmabuf/) | Shared dmabuf zero-copy transport: UDS broker (fd shared across processes, SCM_RIGHTS) + broker client lib; defines the `;dmabuf_handle=N` wire convention |
| [`cix_ros_camera`](ros_packages/cix_ros_camera/) | V4L2 dmabuf zero-copy camera publisher (CIX ISP), standard `sensor_msgs/Image` on the wire |
| [`cix_ros_noe_yolo`](ros_packages/cix_ros_noe_yolo/) | NPU YOLO inference (YOLOv8/11/12/13, `.cix`), publishes `vision_msgs/Detection2DArray` |
| [`cix_ros_noe_handpose`](ros_packages/cix_ros_noe_handpose/) | NPU hand keypoints (ResNet50, `.cix`), C++ zero-copy inference node + Python fallback |

Inter-package deps go through `<depend>` in `package.xml` - no source vendoring. Models live in each package's `models/` and are resolved at runtime via `get_package_share_directory`. See each package's README.

### `demos/` - reference demos (no deb, source only)

- [`camera_handpose`](demos/camera_handpose/) - **official reference demo**: orchestrates `cix_ros_dmabuf` + `cix_ros_camera` + `cix_ros_noe_handpose` for an end-to-end "ISP camera -> NPU hand pose" pipeline (pure orchestration: `run_demo.sh` / `stop_demo.sh` + launch + config + teleop bridges).

Larger demo applications built on this SDK (e.g. `cix-panda-vision-grasp` and `cix-autonombot`) are developed and released as standalone repositories, separate from this SDK repo. Those standalone demos currently support **Ubuntu 22.04 + ROS 2 Humble only**; Ubuntu 24.04 / Jazzy applies to the SDK packages and the in-repo `camera_handpose` reference demo.

## Build

```bash
# Board-side (CIX P1) colcon build of ros_packages
# ROS 2 Humble (Ubuntu 22.04) or Jazzy (Ubuntu 24.04)
source /opt/ros/humble/setup.bash        # or: source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --paths ros_packages/*
```

For the per-package arm64 deb build, see [docs/RELEASE.md](docs/RELEASE.md) §4.

## Docs

- [docs/RELEASE.md](docs/RELEASE.md) - release model (monorepo + per-package deb; deb<->source-locked tag flow).

> Note: this repository is published as source reference; external
> contributions are not accepted at this time.

## License

- Repo as a whole (unless noted otherwise): **Apache-2.0**, see [LICENSE](LICENSE); third-party assets are listed in [NOTICE.md](NOTICE.md).
- `cix_ros_noe_yolo` / `cix_ros_noe_handpose`: Apache-2.0 (see each `package.xml`).
- `cix_ros_camera` / `cix_ros_dmabuf`: **BSD-3-Clause** (see each `package.xml` and the SPDX headers in their sources).
- `.cix` models: the YOLO models in `cix_ros_noe_yolo` are derived from **Ultralytics AGPL-3.0** checkpoints (AGPL-3.0 applies to those model files). Per-model provenance and licensing: see [NOTICE.md](NOTICE.md).
