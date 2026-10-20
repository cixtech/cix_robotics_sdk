# cix_ros_noe_handpose

[English README](README.md)

## 概述

本 **ROS 2 功能包** 在 **NPU** 上执行手部姿态推理：发布完整 **21 个手部关键点**（自定义消息 `HandPoseKeypoints`）、将**食指四向**离散为 `geometry_msgs/Vector3`，并可发布 **bgr8** 的 `sensor_msgs/Image`。包结构为 **ament_cmake + ament_cmake_python** 混合：

- **`handpose_dmabuf_node`**（C++，主节点）—— 零拷贝 NPU 推理：订阅带 `;dmabuf_handle=N` 的 `sensor_msgs/Image`，经 broker 取得 dmabuf fd，由 **CME** 硬件完成 resize+cvtcolor+量化归一化**直接写入 NPU 输入 dmabuf**，并通过 `noe_specify_iobuf(ATTACH_DMABUF)` 绑定给 NPU——**推理路径无像素 memcpy**。
- **`handpose_dma_teleop`**（Python，备选）—— 较慢的 Python 推理节点（输出同主节点但不含 keypoints 消息），便于调试。

本包**与相机解耦**：图像话题、broker socket 路径、模型路径与所有输出话题均为参数。任何经 dmabuf broker 发布 `;dmabuf_handle=N` 编码的相机均可接入——唯一的包依赖是 `cix_ros_dmabuf`（共享零拷贝传输层），**不依赖任何相机包**。

原先在本包内的机器人遥控节点（`handpose_twist_teleop`、`finger_direction_to_cmd_vel`）已移至 **camera_handpose demo**（`demos/camera_handpose/teleop/`）：推理包保持通用，demo 级的机器人控制逻辑归 demo。

**在其他工程中集成：** 将本包加入工作空间、满足[环境要求](#环境要求)后，配置相机或 `sensor_msgs/Image` 话题即可；如需遥控桥接（`Vector3` → `key_vel` 上的 `Twist`），使用 demo 中的脚本。

## `handpose_dmabuf_node` — 发布、订阅与参数

- **可执行文件：** `handpose_dmabuf_node`
- **节点名：** `handpose_dmabuf_node`

完整参数表见 [English README](README.md)。要点：

### 发布

| 话题名（由参数决定） | 类型 | 条件 | 说明 |
|----------------------|------|------|------|
| `direction_topic` | `geometry_msgs/Vector3` | 每帧 | 食指四向离散方向（见下文约定）。 |
| `keypoints_topic` | `cix_ros_noe_handpose/HandPoseKeypoints` | 有订阅者时 | 全部 21 个关键点，输入图像像素坐标（`geometry_msgs/Point[]`）；下游可在其上构建自定义手势语义。 |
| `output_topic` | `sensor_msgs/Image` `bgr8` | `publish_output_image` 为 true | 送入 NPU 的原始帧。**QoS：** 传感器数据（sensor data）。 |
| `debug_topic` | `sensor_msgs/Image` `bgr8` | `publish_debug_image` 为 true | 骨架 + FPS 叠加图，按 `debug_image_scale` 缩放。**QoS：** 传感器数据（sensor data）。 |

### 订阅

| 话题名（由参数决定） | 类型 | 说明 |
|----------------------|------|------|
| `image_topic` | `sensor_msgs/Image` | encoding 带 `;dmabuf_handle=N` 时零拷贝。**QoS：** 传感器数据（sensor data）。 |

关键参数默认值：`image_topic=/cam0_stream1`、`direction_topic=/handpose/finger_direction`、`keypoints_topic=/handpose/keypoints`、`broker_socket_path=""`（空 = `cix_ros_dmabuf` 内置默认路径，不匹配则退化为 memcpy）。

### `geometry_msgs/Vector3`（食指方向）约定

数值定义在**图像**坐标下（与机器人/光学系的关系由上层坐标变换处理）：

- `x`：**−1** 左、**+1** 右、**0** 为中性。
- `y`：**−1** 为图像中向上、**+1** 为向下、**0** 为中性。
- `z`：当前实现恒为 **0**。

全零表示无有效方向 / 中性输出。该约定同样适用于 demo 中的遥控脚本。

---

## 机器人控制节点（已移至 demo）

`handpose_twist_teleop`（USB 相机 Python 推理 + 遥控）与 `finger_direction_to_cmd_vel`（`Vector3` → `key_vel` 上的 `Twist`，供 `twist_mux`）现位于 **camera_handpose demo**（`demos/camera_handpose/teleop/`），连同各自的 launch 文件；参数不变，用法见 demo 的 README。

**请勿**在已有桥接时重复启动 `finger_direction_to_cmd_vel`，以免 `key_vel` 上出现多路发布者。

---

## 环境要求

- ROS 2 Humble（Ubuntu 22.04）或 Jazzy（Ubuntu 24.04）（**ament_cmake + ament_cmake_python**）
- `cix_ros_dmabuf` —— 共享 dmabuf 零拷贝传输包（提供 `dmabuf_broker.hpp` / `dmabuf_image.hpp` 头文件与 broker 客户端库）
- 板端 **NOE** 运行环境：`libnoe.so`（`cix-noe-umd`）提供低层 `noe_*` API，及与模型匹配的 **`.cix`**
- 板端 **CME**：`libcme`（由 `cix-libcme-dev_1.1.0_arm64.deb` 安装，提供 `cme/cme.h` 等）。**需板端构建**，因 libnoe/libcme 为板侧专有。`libcme.pc` 的 `-I/usr/include/cme` 与代码所用 `#include <cme/cme.h>` 不一致，故 `CMakeLists.txt` 硬编码 `include_directories("/usr/include")` + `link_directories("/usr/lib/aarch64-linux-gnu")`（与 `cix_yolo_demo` 的 Makefile 一致），未使用 pkg-config
- 模型输入须为 **UINT8 NCHW 1×3×256×256**，通道顺序 **BGR**（与训练一致）；节点在 `init()` 处断言 dtype 为 UINT8，否则报错退出
- 任何发布带 `;dmabuf_handle=N` 的 rgb8/bgr8/nv12 图像的相机（例如经 `cix_ros_camera`），或已存在的 **`sensor_msgs/Image`**（无 fd 时退化为 CME `EXTERNAL_MEMORY` 模式，使用消息 payload）

## 编译

在**目标板**上构建（NPU SDK 头文件/库不在宿主机上）。本包 `find_package(cix_ros_dmabuf)`，且消息接口（`msg/HandPoseKeypoints.msg`）在构建时生成，需先编译 `cix_ros_dmabuf`：

```bash
cd <workspace>
source /opt/ros/humble/setup.bash   # Ubuntu 24.04 上为 /opt/ros/jazzy
colcon build --symlink-install --packages-select cix_ros_dmabuf
source install/setup.bash   # 使 find_package(cix_ros_dmabuf) 可解析
colcon build --symlink-install --packages-select cix_ros_noe_handpose
source install/setup.bash
```

## 启动文件

| 启动文件 | 所启动节点 | 功能 |
|----------|------------|------|
| `handpose_dma_teleop.launch.py` | `handpose_dmabuf_node`（C++） | 零拷贝 NPU 推理：`image_topic` → `keypoints` + `finger_direction` 与可选 bgr8 输出；相机/broker 由外部提供。 |

```bash
ros2 launch cix_ros_noe_handpose handpose_dma_teleop.launch.py \
  image_topic:=/cam0_stream1 broker_socket_path:=/tmp/cix_ros_dmabuf_broker.sock
```

端到端「相机 → NPU」流水线与遥控桥接请使用 **camera_handpose demo**（`demos/camera_handpose/`），由它编排本包与 `cix_ros_camera`、`cix_ros_dmabuf`。

## 许可证

Apache-2.0。详见 `package.xml`。
