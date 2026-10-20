# cix_ros_noe_yolo

[English README](README.md)

在 CIX NPU 上运行 YOLOv8 的 ROS 2 功能包。四个独立节点各加载一个 `.cix` 模型，订阅相机图像，发布检测结果。

## 目录结构

```
cix_ros_noe_yolo/
├── src/                          # ROS 节点与引擎（
│   ├── npu_yolo_coco_*           # COCO 检测：inference 节点 + engine
│   ├── npu_yolo_pose_*           # 人体姿态
│   ├── npu_yolo_seg_*            # 实例分割
│   ├── npu_yolo_obb_*            # 螺栓 OBB（单类旋转框检测）
│   └── detection_output.py       # OBB 消息组装
├── yolo_postprocess/             # NPU 后处理
├── model/                        # .cix 模型（需下载，见"模型下载"）
├── launch/                       # 四个 npu_yolo_*_inference.launch.py
├── config/                       # 可选 FastDDS 配置
├── scripts/download_models.sh   # 从 ModelScope 下载 .cix 模型
├── resource/cix_ros_noe_yolo         # ament 包注册标记（空文件）
├── setup.py / package.xml        # ament_python 构建与依赖
└── README.md / README.cn.md
```

- `src/*_inference.py`：ROS 节点（订阅图像、发布话题）
- `src/*_engine.py`：无 ROS 的 NPU 推理封装（`EngineInfer` + 后处理）
- 新增模型：放入 `model/`，通过 `npu_model_name` 或 `npu_model_path` 指定

---

## 速览

#### `npu_yolo_coco_inference`
- 任务：COCO 目标检测
- 默认模型：`yolov8n.cix`
- 主要输出：`/vision/detections_2d`

#### `npu_yolo_pose_inference`
- 任务：COCO-17 人体姿态
- 默认模型：`yolov8s-pose.cix`
- 主要输出：`/vision/pose_detections_2d` + 关键点

#### `npu_yolo_seg_inference`
- 任务：实例分割
- 默认模型：`yolov8l_seg.cix`
- 主要输出：`/vision/seg_detections_2d`

#### `npu_yolo_obb_inference`
- 任务：螺栓 OBB
- 默认模型：`yolo_obb.cix`
- 主要输出：`/Yolov8_Inference`

---

## 快速开始

### 编译

```bash
colcon build --packages-select cix_ros_noe_yolo
source install/setup.bash
```

> 可选的 OBB 节点额外依赖 `yolov8_msgs`（本仓库与 ROS apt 仓库均不含其源码，需源码编译）：
> 源码在 [cix-panda-vision-grasp](https://github.com/cixtech/cix-panda-vision-grasp) demo 仓库
> （`ros2_project/yolo_msgs/`），或上游 [yolov8_ros](https://github.com/mgonzs13/yolov8_ros)；
> 拷贝/克隆到同一 colcon 工作空间后，用
> `colcon build --packages-select yolov8_msgs cix_ros_noe_yolo` 编译。不装时 OBB 节点需
> `output_format:=vision_msgs`，其余节点不受影响。

### 模型下载

`.cix` NPU 权重**不随本仓库和 deb 分发**（派生自 Ultralytics 权重，许可证为 AGPL-3.0），从 ModelScope 上的 CIX 模型库下载：

```bash
# robotics_sdk 源码目录内（下载到 ros_packages/cix_ros_noe_yolo/model/）
bash scripts/download_models.sh

# 已安装 deb 的环境（下载到 share/cix_ros_noe_yolo/model/）
sudo bash $(ros2 pkg prefix cix_ros_noe_yolo)/share/cix_ros_noe_yolo/scripts/download_models.sh
```

`MODELS="yolov8n" bash scripts/download_models.sh` 可只下载部分模型。各模型的来源与许可证见仓库根 [NOTICE.md](../../NOTICE.md)。

### 启动

```bash
# OBB
ros2 launch cix_ros_noe_yolo npu_yolo_obb_inference.launch.py

# 其余节点
ros2 launch cix_ros_noe_yolo npu_yolo_coco_inference.launch.py
ros2 launch cix_ros_noe_yolo npu_yolo_pose_inference.launch.py
ros2 launch cix_ros_noe_yolo npu_yolo_seg_inference.launch.py
```

### 直接运行

```bash
ros2 run cix_ros_noe_yolo npu_yolo_obb_inference --ros-args -p image_topic:=/image_raw
```

---

## 通用约定

以下四个节点共享相同模式，仅在默认值和专有话题上有差异。

### 数据流

```
sensor_msgs/Image (bgr8)  →  EngineInfer (.cix)  →  检测话题 + 可选调试图
```

### 模型路径

- `npu_model_path` — 非空时为 `.cix` 绝对路径，忽略 `npu_model_name`
- `npu_model_name` — 为空时从 `share/cix_ros_noe_yolo/model/{name}.cix` 加载
- `onnx_yolov8_root` — 后处理根目录（含 `yolo_postprocess/`）；空则用包内 share

### 共用参数

- `image_topic`
  - 默认：见各节点
  - 说明：输入图像，`bgr8`，sensor data QoS
- `use_sim_time`
  - 默认：launch 常为 `true`
  - 说明：仿真须提供 `/clock`
- `conf_thr` / `nms_thr`
  - 默认：见各节点
  - 说明：置信度 / NMS 阈值
- `publish_debug_image`
  - 默认：见各节点
  - 说明：是否发布叠画调试图
- `debug_image_topic`
  - 默认：见各节点
  - 说明：调试图话题
- `npu_active_topic`
  - 默认：见各节点
  - 说明：NPU 占用 Bool，启动发 `true`
- `publish_npu_active_heartbeat`
  - 默认：`true`
  - 说明：是否周期心跳
- `npu_active_heartbeat_hz`
  - 默认：`2.0`
  - 说明：心跳频率
- `reenable_ultralytics_on_exit`
  - 默认：`true`
  - 说明：退出时发 `false`（便于与 CPU 侧切换）

### 启动文件对照

- `npu_yolo_coco_inference.launch.py` → `npu_yolo_coco_inference`
- `npu_yolo_pose_inference.launch.py` → `npu_yolo_pose_inference`
- `npu_yolo_seg_inference.launch.py` → `npu_yolo_seg_inference`
- `npu_yolo_obb_inference.launch.py` → `npu_yolo_obb_inference`

---

## 节点详情

### 1. COCO 检测 — `npu_yolo_coco_inference`

#### 发布

- `detections_topic`
  - 类型：`vision_msgs/Detection2DArray`
  - 默认：`/vision/detections_2d`
- `debug_image_topic`
  - 类型：`sensor_msgs/Image`
  - 默认：`/vision/yolo_debug_image`
- `npu_active_topic`
  - 类型：`std_msgs/Bool`
  - 默认：`/npu_vision_active`

专有参数：`npu_model_name=yolov8n`，`conf_thr=0.25`，`nms_thr=0.45`

---

### 2. 人体姿态 — `npu_yolo_pose_inference`

#### 发布

- `detections_topic`
  - 类型：`vision_msgs/Detection2DArray`
  - 默认：`/vision/pose_detections_2d`
- `pose_keypoints_topic`
  - 类型：`std_msgs/Float32MultiArray`
  - 默认：`/vision/pose_keypoints_flat`
- `debug_image_topic`
  - 类型：`sensor_msgs/Image`
  - 默认：`/vision/yolo_pose_debug_image`
- `npu_active_topic`
  - 类型：`std_msgs/Bool`
  - 默认：`/npu_pose_active`

专有参数：`npu_model_name=yolov8s-pose`，`kpt_vis_thr=0.25`

后处理：letterbox + BGR→RGB，头 `(C, 8400)`，`C ∈ {56, 135}`。详见 `yolo_postprocess/yolov8_pose_npu_decode.py`。

---

### 3. 实例分割 — `npu_yolo_seg_inference`

#### 发布

- `detections_topic`
  - 类型：`vision_msgs/Detection2DArray`
  - 默认：`/vision/seg_detections_2d`
- `debug_image_topic`
  - 类型：`sensor_msgs/Image`
  - 默认：`/vision/yolo_seg_debug_image`
- `npu_active_topic`
  - 类型：`std_msgs/Bool`
  - 默认：`/npu_seg_active`

专有参数

- `npu_model_name`
  - 默认：`yolov8l_seg`
- `conf_thr`
  - 默认：`0.35`
- `iou_thr`
  - 默认：`0.45`
  - 说明：分割 NMS
- `seg_nm`
  - 默认：`32`
  - 说明：掩码系数通道
- `seg_proto_hw`
  - 默认：`160`
  - 说明：原型特征图尺寸
- `seg_num_classes`
  - 默认：`80`
  - 说明：须满足 `4 + seg_num_classes + seg_nm = 首路 C`

后处理：双路输出 `(1, C, 8400)` + `(1, nm, H, W)`，与 CIX `seg_inference_npu.py` 一致。

---

### 4. 螺栓 OBB — `npu_yolo_obb_inference`

#### 发布

- `yolov8_inference_topic`
  - 类型：`yolov8_msgs/Yolov8Inference`
  - 默认：`/Yolov8_Inference`
  - 条件：`output_format` 含 `yolov8_obb`
- `detections_topic`
  - 类型：`vision_msgs/Detection2DArray`
  - 默认：—
  - 条件：`output_format` 含 `vision_msgs`
- `debug_image_topic`
  - 类型：`sensor_msgs/Image`
  - 默认：`/inference_result`
  - 条件：`publish_debug_image=true`
- `/npu_inference_fps`
  - 类型：`std_msgs/Float32`
  - 默认：—
  - 条件：`profile_timing` 时
- `npu_active_topic`
  - 类型：`std_msgs/Bool`
  - 默认：`/npu_vision_active`

专有参数

- `npu_model_name`
  - 默认：`yolo_obb`
- `class_names`
  - 默认：`bolt`
  - 说明：逗号分隔
- `num_classes`
  - 默认：`1`
  - 说明：OBB 头类别数
- `conf_thr` / `nms_thr`
  - 默认：`0.90` / `0.40`
- `output_format`
  - 默认：`yolov8_obb`
  - 说明：`vision_msgs` | `yolov8_obb` | `both`
- `class_filter`
  - 默认：`bolt`
  - 说明：空串不过滤
- `profile_timing`
  - 默认：`false`
  - 说明：记录各阶段耗时
- `image_topic`
  - 默认：`/image_raw`

后处理：`yolo_postprocess/npu_yolo_obb_postprocess.py`，头 `(4+nc+1, 8400)`，单类螺栓为 `(6, 8400)`。

节点名：ROS 节点注册为 `cix_ros_noe_yolo`（与可执行文件名不同，属历史命名）。

---

## 环境要求

- ROS 2 ament_python
- 板端 NOE 与 `NOE_Engine.EngineInfer`
- 与后处理布局匹配的 `.cix` 模型
- 输入图像编码 bgr8

依赖见 `package.xml`。

---

## 集成到其他包

```xml
<exec_depend>cix_ros_noe_yolo</exec_depend>
```

- 检测 / 分割 / OBB：订阅各节点的 `detections_topic`
- 姿态：额外订阅 `pose_keypoints_topic`
- OBB：订阅 `/Yolov8_Inference`（`yolov8_msgs`）
- 可与 CPU 侧视觉通过 `/npu_*_active` Bool 互斥切换

本包不提供 ROS 服务，仅通过话题与参数集成。

---

## 许可证

Apache-2.0，详见 `package.xml`。`scripts/download_models.sh` 下载的 `.cix`
模型派生自 Ultralytics 权重，许可证为 **AGPL-3.0**（见
[NOTICE.md](../../NOTICE.md)），不随本包分发。
