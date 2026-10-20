# cix_ros_noe_yolo

[中文说明 / Chinese](README.cn.md)

ROS 2 package for YOLOv8 on the CIX NPU. Four nodes each load a `.cix` model, subscribe to camera images, and publish detections.

## Directory layout

```
cix_ros_noe_yolo/
├── src/                          # ROS nodes and engines (Python package cix_ros_noe_yolo)
│   ├── npu_yolo_coco_*           # COCO detection: inference node + engine
│   ├── npu_yolo_pose_*           # human pose
│   ├── npu_yolo_seg_*            # instance segmentation
│   ├── npu_yolo_obb_*            # bolt OBB (single-class oriented bounding boxes)
│   └── detection_output.py       # OBB message builders (yolov8_msgs / vision_msgs)
├── yolo_postprocess/             # NPU postprocess (letterbox, decode, NMS, labels)
├── model/                        # .cix models (downloaded, see Models; installed to share/.../model/)
├── launch/                       # four npu_yolo_*_inference.launch.py files
├── config/                       # optional FastDDS config
├── scripts/download_models.sh   # fetch the .cix models from ModelScope
├── resource/cix_ros_noe_yolo         # ament package marker (empty file)
├── setup.py / package.xml        # ament_python build and dependencies
└── README.md / README.cn.md
```

- `src/*_inference.py`: ROS nodes (image in, topics out)
- `src/*_engine.py`: non-ROS NPU wrapper (`EngineInfer` + postprocess)
- New models: place under `model/`, select via `npu_model_name` or `npu_model_path`

---

## At a glance

#### `npu_yolo_coco_inference`
- Task: COCO object detection
- Default model: `yolov8n.cix`
- Main output: `/vision/detections_2d`

#### `npu_yolo_pose_inference`
- Task: COCO-17 human pose
- Default model: `yolov8s-pose.cix`
- Main output: `/vision/pose_detections_2d` + keypoints

#### `npu_yolo_seg_inference`
- Task: instance segmentation
- Default model: `yolov8l_seg.cix`
- Main output: `/vision/seg_detections_2d`

#### `npu_yolo_obb_inference`
- Task: bolt OBB
- Default model: `yolo_obb.cix`
- Main output: `/Yolov8_Inference`

---

## Quick start

### Build

```bash
colcon build --packages-select cix_ros_noe_yolo
source install/setup.bash
```

> The optional OBB node additionally requires `yolov8_msgs` (not in this repository and not in the
> ROS apt repo). Sources live in the
> [cix-panda-vision-grasp](https://github.com/cixtech/cix-panda-vision-grasp) demo repo
> (`ros2_project/yolo_msgs/`) or upstream [yolov8_ros](https://github.com/mgonzs13/yolov8_ros);
> copy/clone the package into the same colcon workspace, then build with
> `colcon build --packages-select yolov8_msgs cix_ros_noe_yolo`. Without it, the OBB node needs
> `output_format:=vision_msgs`; all other nodes are unaffected.

### Models

The `.cix` NPU models are **not part of this repository or the deb** (they derive from Ultralytics checkpoints, license AGPL-3.0). Fetch them from the CIX model hub on ModelScope:

```bash
# from a robotics_sdk checkout (into ros_packages/cix_ros_noe_yolo/model/)
bash scripts/download_models.sh

# for an installed package (into share/cix_ros_noe_yolo/model/)
sudo bash $(ros2 pkg prefix cix_ros_noe_yolo)/share/cix_ros_noe_yolo/scripts/download_models.sh
```

`MODELS="yolov8n" bash scripts/download_models.sh` downloads a subset. Provenance and licensing per model: root [NOTICE.md](../../NOTICE.md).

### Launch

```bash
# OBB
ros2 launch cix_ros_noe_yolo npu_yolo_obb_inference.launch.py

# Other nodes
ros2 launch cix_ros_noe_yolo npu_yolo_coco_inference.launch.py
ros2 launch cix_ros_noe_yolo npu_yolo_pose_inference.launch.py
ros2 launch cix_ros_noe_yolo npu_yolo_seg_inference.launch.py
```

### Direct run

```bash
ros2 run cix_ros_noe_yolo npu_yolo_obb_inference --ros-args -p image_topic:=/image_raw
```

---

## Common conventions

All four nodes share the same pattern; only defaults and node-specific topics differ.

### Data flow

```
sensor_msgs/Image (bgr8)  →  EngineInfer (.cix)  →  detection topics + optional debug image
```

### Model paths

- `npu_model_path` — when set: absolute path to `.cix`; ignores `npu_model_name`
- `npu_model_name` — when path empty: load `share/cix_ros_noe_yolo/model/{name}.cix`
- `onnx_yolov8_root` — root containing `yolo_postprocess/`; empty uses package share

### Shared parameters

- `image_topic`
  - Default: see per-node section
  - Notes: input image, `bgr8`, sensor data QoS
- `use_sim_time`
  - Default: often `true` in launch
  - Notes: simulation must provide `/clock`
- `conf_thr` / `nms_thr`
  - Default: see per-node section
  - Notes: score / NMS thresholds
- `publish_debug_image`
  - Default: see per-node section
  - Notes: publish annotated debug image
- `debug_image_topic`
  - Default: see per-node section
  - Notes: debug image topic
- `npu_active_topic`
  - Default: see per-node section
  - Notes: NPU-active Bool; publishes `true` on start
- `publish_npu_active_heartbeat`
  - Default: `true`
  - Notes: periodic heartbeat
- `npu_active_heartbeat_hz`
  - Default: `2.0`
  - Notes: heartbeat rate
- `reenable_ultralytics_on_exit`
  - Default: `true`
  - Notes: publish `false` on exit (handoff to CPU-side vision)

### Launch file map

- `npu_yolo_coco_inference.launch.py` → `npu_yolo_coco_inference`
- `npu_yolo_pose_inference.launch.py` → `npu_yolo_pose_inference`
- `npu_yolo_seg_inference.launch.py` → `npu_yolo_seg_inference`
- `npu_yolo_obb_inference.launch.py` → `npu_yolo_obb_inference`

---

## Node details

### 1. COCO detection — `npu_yolo_coco_inference`

#### Publishes

- `detections_topic`
  - Type: `vision_msgs/Detection2DArray`
  - Default: `/vision/detections_2d`
- `debug_image_topic`
  - Type: `sensor_msgs/Image`
  - Default: `/vision/yolo_debug_image`
- `npu_active_topic`
  - Type: `std_msgs/Bool`
  - Default: `/npu_vision_active`

Node-specific params: `npu_model_name=yolov8n`, `conf_thr=0.25`, `nms_thr=0.45`

---

### 2. Human pose — `npu_yolo_pose_inference`

#### Publishes

- `detections_topic`
  - Type: `vision_msgs/Detection2DArray`
  - Default: `/vision/pose_detections_2d`
- `pose_keypoints_topic`
  - Type: `std_msgs/Float32MultiArray`
  - Default: `/vision/pose_keypoints_flat`
- `debug_image_topic`
  - Type: `sensor_msgs/Image`
  - Default: `/vision/yolo_pose_debug_image`
- `npu_active_topic`
  - Type: `std_msgs/Bool`
  - Default: `/npu_pose_active`

Node-specific params: `npu_model_name=yolov8s-pose`, `kpt_vis_thr=0.25`

Postprocess: letterbox + BGR→RGB, head `(C, 8400)`, `C ∈ {56, 135}`. See `yolo_postprocess/yolov8_pose_npu_decode.py`.

---

### 3. Instance segmentation — `npu_yolo_seg_inference`

#### Publishes

- `detections_topic`
  - Type: `vision_msgs/Detection2DArray`
  - Default: `/vision/seg_detections_2d`
- `debug_image_topic`
  - Type: `sensor_msgs/Image`
  - Default: `/vision/yolo_seg_debug_image`
- `npu_active_topic`
  - Type: `std_msgs/Bool`
  - Default: `/npu_seg_active`


Node-specific params

- `npu_model_name`
  - Default: `yolov8l_seg`
- `conf_thr`
  - Default: `0.35`
- `iou_thr`
  - Default: `0.45`
  - Notes: segmentation NMS
- `seg_nm`
  - Default: `32`
  - Notes: mask coefficient channels
- `seg_proto_hw`
  - Default: `160`
  - Notes: prototype feature map size
- `seg_num_classes`
  - Default: `80`
  - Notes: must satisfy `4 + seg_num_classes + seg_nm = first output C`

Postprocess: dual outputs `(1, C, 8400)` + `(1, nm, H, W)`, aligned with CIX `seg_inference_npu.py`.

---

### 4. Bolt OBB — `npu_yolo_obb_inference`

#### Publishes

- `yolov8_inference_topic`
  - Type: `yolov8_msgs/Yolov8Inference`
  - Default: `/Yolov8_Inference`
  - When: `output_format` includes `yolov8_obb`
- `detections_topic`
  - Type: `vision_msgs/Detection2DArray`
  - Default: —
  - When: `output_format` includes `vision_msgs`
- `debug_image_topic`
  - Type: `sensor_msgs/Image`
  - Default: `/inference_result`
  - When: `publish_debug_image=true`
- `/npu_inference_fps`
  - Type: `std_msgs/Float32`
  - Default: —
  - When: `profile_timing` enabled
- `npu_active_topic`
  - Type: `std_msgs/Bool`
  - Default: `/npu_vision_active`

Node-specific params

- `npu_model_name`
  - Default: `yolo_obb`
- `class_names`
  - Default: `bolt`
  - Notes: comma-separated
- `num_classes`
  - Default: `1`
  - Notes: OBB head class count
- `conf_thr` / `nms_thr`
  - Default: `0.90` / `0.40`
- `output_format`
  - Default: `yolov8_obb`
  - Notes: `vision_msgs` | `yolov8_obb` | `both`
- `class_filter`
  - Default: `bolt`
  - Notes: empty string = no filter
- `profile_timing`
  - Default: `false`
  - Notes: log per-stage timing (ms)
- `image_topic`
  - Default: `/image_raw`

Postprocess: `yolo_postprocess/npu_yolo_obb_postprocess.py`, head `(4+nc+1, 8400)`; single-class bolt uses `(6, 8400)`.

ROS node name is registered as `cix_ros_noe_yolo` (differs from the executable name; legacy naming).

---

## Requirements

- ROS 2 ament_python
- Board-side NOE and `NOE_Engine.EngineInfer`
- `.cix` models matching the postprocess tensor layout
- Input images encoded as bgr8

See `package.xml` for dependencies. 

---

## Integration

```xml
<exec_depend>cix_ros_noe_yolo</exec_depend>
```

- Detection / seg / OBB: subscribe to each node’s `detections_topic`
- Pose: also subscribe to `pose_keypoints_topic`
- OBB: subscribe to `/Yolov8_Inference` (`yolov8_msgs`)
- Use `/npu_*_active` Bool to arbitrate with CPU-side vision if needed

This package provides no ROS services; integrate via topics and parameters only.

---

## License

Apache-2.0. See `package.xml`. The `.cix` models fetched by
`scripts/download_models.sh` are derived from Ultralytics checkpoints and are
licensed **AGPL-3.0** (see [NOTICE.md](../../NOTICE.md)); they are not
distributed with this package.
