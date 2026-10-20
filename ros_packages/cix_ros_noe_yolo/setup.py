# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import os
from glob import glob

from setuptools import setup

package_name = "cix_ros_noe_yolo"
here = os.path.abspath(os.path.dirname(__file__))
model_rel = "model"
postprocess_rel = "yolo_postprocess"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]

for root, _dirs, files in os.walk(os.path.join(here, model_rel)):
    rel = os.path.relpath(root, here)
    for fname in files:
        abs_src = os.path.join(root, fname)
        rel_src = os.path.relpath(abs_src, here)
        dest_dir = os.path.join("share", package_name, rel)
        data_files.append((dest_dir, [rel_src]))

postprocess_dir = os.path.join(here, postprocess_rel)
if os.path.isdir(postprocess_dir):
    for root, _dirs, files in os.walk(postprocess_dir):
        rel = os.path.relpath(root, here)
        for fname in files:
            abs_src = os.path.join(root, fname)
            rel_src = os.path.relpath(abs_src, here)
            dest_dir = os.path.join("share", package_name, rel)
            data_files.append((dest_dir, [rel_src]))

launch_glob = glob(os.path.join(here, "launch", "*.launch.py"))
if launch_glob:
    launch_rel = sorted(os.path.relpath(p, here) for p in launch_glob)
    data_files.append((os.path.join("share", package_name, "launch"), launch_rel))

config_dir = os.path.join(here, "config")
if os.path.isdir(config_dir):
    for root, _dirs, files in os.walk(config_dir):
        rel = os.path.relpath(root, here)
        for fname in files:
            abs_src = os.path.join(root, fname)
            rel_src = os.path.relpath(abs_src, here)
            dest_dir = os.path.join("share", package_name, rel)
            data_files.append((dest_dir, [rel_src]))

scripts_dir = os.path.join(here, "scripts")
if os.path.isdir(scripts_dir):
    for fname in sorted(os.listdir(scripts_dir)):
        if os.path.isfile(os.path.join(scripts_dir, fname)):
            data_files.append(
                (os.path.join("share", package_name, "scripts"), [os.path.join("scripts", fname)])
            )

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_dir={package_name: "src"},
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CIX Robotics team",
    description="NPU YOLO inference (CIX NOE) for CIX P1",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "npu_yolo_coco_inference = cix_ros_noe_yolo.npu_yolo_coco_inference:main",
            "npu_yolo_pose_inference = cix_ros_noe_yolo.npu_yolo_pose_inference:main",
            "npu_yolo_seg_inference = cix_ros_noe_yolo.npu_yolo_seg_inference:main",
            "npu_yolo_obb_inference = cix_ros_noe_yolo.npu_yolo_obb_inference:main",
        ],
    },
)
