# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""Helpers for the .cix NPU model files (distributed out-of-band)."""

import os


def model_missing_hint(path: str) -> str:
    """Error text for a missing .cix model, pointing at the download script.

    The YOLO .cix models are not bundled with this package, its repo, or its
    deb (Ultralytics-derived weights, AGPL-3.0); they are fetched from the CIX
    model hub on ModelScope by scripts/download_models.sh, which lives next to
    the package share dir (repo checkout) or inside it (installed package).
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("cix_ros_noe_yolo")
    except Exception:
        share = ""
    script = os.path.join(share, "scripts", "download_models.sh") if share else \
        "<share>/cix_ros_noe_yolo/scripts/download_models.sh"
    return (
        f"model file not found: {path}. YOLO .cix models are not bundled with "
        "cix_ros_noe_yolo (Ultralytics-derived weights, AGPL-3.0); fetch them "
        f"from the CIX model hub on ModelScope by running: bash {script} "
        "(prefix with sudo when the package is installed under /opt)"
    )
