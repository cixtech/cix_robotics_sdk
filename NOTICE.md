# NOTICE

This repository ("CIX P1 Robotics SDK") is released by CIX. Unless stated
otherwise, source code is licensed as declared in each package's `package.xml`
(Apache-2.0; `cix_ros_camera` is BSD-3-Clause). See [LICENSE](LICENSE) and
each package's README.

This file lists third-party assets and their licenses.

## Precompiled NPU models (`.cix`)

`.cix` files are NPU models compiled from upstream neural-network checkpoints
by the CIX NOE toolchain. The MIT / Apache-2.0 models ship inside the packages
in this repository and inside the built debs; the AGPL YOLO models do not
(see below).

### `ros_packages/cix_ros_noe_yolo/model/` — AGPL-3.0, NOT distributed

The YOLO models for this package are derived from **Ultralytics** YOLOv8 /
YOLO11 / YOLO12 / YOLO13 checkpoints, whose weights are distributed under
**AGPL-3.0** (see
[ultralytics/LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)).
They are therefore **not committed for the SDK package and not bundled in the
`cix-ros-$ROS_DISTRO-noe-yolo` deb** (e.g. `cix-ros-humble-noe-yolo`,
`cix-ros-jazzy-noe-yolo`). Fetch them from the CIX model hub on ModelScope
([cix/ai_model_hub](https://modelscope.cn/models/cix/ai_model_hub)) with:

```bash
bash ros_packages/cix_ros_noe_yolo/scripts/download_models.sh
# or, for an installed package:
sudo bash $(ros2 pkg prefix cix_ros_noe_yolo)/share/cix_ros_noe_yolo/scripts/download_models.sh
```

| File | Upstream checkpoint | ModelScope location |
|---|---|---|
| `yolov8n.cix` | Ultralytics `yolov8n.pt` (COCO) | `models/ComputeVision/Object_Detection/onnx_yolov8_n/` |
| `yolov8s.cix` | Ultralytics `yolov8s.pt` (COCO) | `models/ComputeVision/Object_Detection/onnx_yolov8_s/` |
| `yolov8s-pose.cix` | Ultralytics `yolov8s-pose.pt` (COCO keypoints) | `models/ComputeVision/Pose_Estimation/onnx_yolov8s_pose/` |
| `yolov8l_seg.cix` | Ultralytics `yolov8l-seg.pt` (COCO segmentation) | `models/ComputeVision/Semantic_Segmentation/onnx_yolov8_segmentation/` |
| `yolov11_n.cix` | Ultralytics `yolo11n.pt` (COCO) | `models/ComputeVision/Object_Detection/onnx_yolov11_n/` |
| `yolov12_n.cix` | Ultralytics `yolo12n.pt` (COCO) | `models/ComputeVision/Object_Detection/onnx_yolov12_n/` |
| `yolov13_n.cix` | Ultralytics `yolo13n.pt` (COCO) | `models/ComputeVision/Object_Detection/onnx_yolov13_n/` |

**AGPL-3.0 applies to the downloaded model files**, on top of the package's
Apache-2.0 code license. Distributing the models, or offering them as a
service, carries AGPL source-disclosure obligations for the model weights. If
that does not fit your use case, obtain an Ultralytics enterprise license or
replace the models with your own AGPL-compatible weights.

### `ros_packages/cix_ros_noe_handpose/models/`

| File | Provenance | License |
|---|---|---|
| `handpose_resnet50_size256.cix` | ResNet50-based hand-keypoint model (256x256, BGR, UINT8) provided by CIX | Apache-2.0 |

## Standalone demo repositories

The larger demo applications built on this SDK (e.g. `cix-panda-vision-grasp`,
`cix-autonombot`) are developed and released as standalone repositories,
separate from this SDK repo; their licenses and third-party notices live in
those repositories.
