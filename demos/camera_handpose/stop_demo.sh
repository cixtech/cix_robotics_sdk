#!/bin/bash
# ---------------------------------------------------------------------
# Copyright 2026 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
# Stop the camera_handpose demo: kill broker + camera + handpose + rqt.
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DEMO_DIR/run_demo.sh" stop
