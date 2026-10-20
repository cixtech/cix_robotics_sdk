# Release Model

> Scope: robotics_sdk (CIX P1 Robotics SDK).

## 1. Two-layer structure

- **`ros_packages/`** - reusable ROS 2 packages wrapping P1's hardware capabilities. **Each package is built and released as an arm64 deb.**
- **`demos/`** - reference examples that orchestrate packages from `ros_packages/` to run an end-to-end scenario. **Demos ship source only, no deb.**

## 2. Deb granularity: one deb per ros_package

The board-side build that turns this monorepo into per-package arm64 debs is driven by `scripts/build-ros-debs-board.sh`, a self-contained script (sources no other script) meant to run on the board under either supported platform: Ubuntu 22.04 jammy + ROS 2 Humble, or Ubuntu 24.04 noble + ROS 2 Jazzy. It takes optional unit names as arguments; with no arguments it builds all units. The distro is selected with `ROS_DISTRO` (or derived from `ROS_PREFIX`) and drives the deb names (`cix-ros-<distro>-*`) and the control `Depends` (`ros-<distro>-*`), so the same script produces the `cix-ros-humble-*` and `cix-ros-jazzy-*` deb sets.

Source is a **monorepo** (this repo), binaries are **per-package debs**. Board-side `colcon build` produces arm64 debs. One monorepo -> N debs is the standard ROS ecosystem mapping (navigation2, turtlebot4, vision_opencv all follow this model).

Why not split source into one repo per package: these packages form a tightly coupled stack (dmabuf transport -> camera -> noe_* perception -> demo), so cross-package contract changes need atomic commits; demos must track package API; one colcon workspace builds all debs at once. Splitting repos only adds coordination overhead with zero benefit.

## 3. External fork demos: standalone repositories

The larger demo applications are forks of upstream external projects with CIX
integration changes (NPU acceleration, P1 bring-up). They are developed and
released as standalone repositories, separate from this SDK repo, so their
upstream licenses and third-party assets stay out of the SDK distribution:

| Demo | Upstream | Upstream License | Supported platform |
|---|---|---|---|
| `cix-panda-vision-grasp` | `laoxue888/Moveit2YoloObb` | MIT | Ubuntu 22.04 + ROS 2 Humble only |
| `cix-autonombot` | `houimliraed/autonomous_ai_rbt` | Apache-2.0 | Ubuntu 22.04 + ROS 2 Humble only |

The Robotics SDK packages (`ros_packages/` + in-repo `demos/camera_handpose`) support both Ubuntu 22.04 / Humble and Ubuntu 24.04 / Jazzy. The two standalone fork demos above currently target Ubuntu 22.04 / Humble only; Ubuntu 24.04 / Jazzy is out of scope for them until a dedicated port lands.

Conventions each standalone demo repo must hold:

- Keep the upstream `LICENSE` + copyright notice; for Apache-2.0 upstreams, add a `NOTICE` describing the CIX changes.
- **Integration delta = `exec_depend` on SDK packages + launch/DDS wiring**, **without copying SDK package sources or `.cix`** (compile the SDK debs, or set `ROBOTICS_SDK_ROOT` to a robotics_sdk checkout, before `colcon build`).
- README/NOTICE records which upstream commit/tag the fork is based on + the list of CIX changes.
- `*.pt` weights are never committed; each demo's README documents the download URL. YOLO `.cix` models (AGPL-3.0 Ultralytics) are likewise not committed for the SDK packages, with one deliberate exception: `yolo_obb.cix` ships inside the standalone `cix-panda-vision-grasp` repository — that repo's NOTICE declares the AGPL license for it.
- **No deb**: after installing the SDK, the user clones the standalone fork repo and runs `colcon build` themselves.

## 4. Deb <-> source-locked release flow

```
git tag vX.Y.Z
  -> board-side colcon build at that tag produces per-package arm64 debs
  -> deb changelog version = git tag
  -> published to the CIX apt repo
  -> source at that tag exported to GitHub as a fresh single-commit repo
     (git archive + fresh git init, so only a single initial commit exists).
     The internal Gerrit history (ticket IDs, personal emails, since-deleted
     AGPL weight blobs) is never pushed.
```

A user installs a deb -> traces the exact source commit from the changelog version. Each package's README notes "this deb was built from robotics_sdk `<tag>`".

## 5. Official reference demo

`demos/camera_handpose/` is the official reference demo (pure orchestration: `run_demo.sh` / `stop_demo.sh` + launch + config), released with the SDK repo at the same source/tag, as the template for "how to orchestrate ros_packages". New official reference demos follow the same pattern; external fork demos live in their own standalone repositories (§3).

## 6. Model assets and licensing

- The Apache-2.0 `.cix` NPU model (`handpose_resnet50_size256.cix`) ships in this repo and in the debs. Per-model provenance and licensing is listed in the root [NOTICE.md](../NOTICE.md).
- The YOLO `.cix` models are derived from Ultralytics checkpoints and carry **AGPL-3.0**. They are **not committed for the SDK packages and not bundled in the `cix-ros-$ROS_DISTRO-noe-yolo` deb** (e.g. `cix-ros-humble-noe-yolo`, `cix-ros-jazzy-noe-yolo`); users fetch them from the CIX model hub on ModelScope with `ros_packages/cix_ros_noe_yolo/scripts/download_models.sh` (the script is installed into the package share dir, so it works for deb installs too). Exception: `yolo_obb.cix` ships inside the standalone `cix-panda-vision-grasp` repository (see §3). Users who cannot accept AGPL must replace the YOLO models with their own weights.
- The deb build (`scripts/build-ros-debs-board.sh`) strips the YOLO models from the staged package even if untracked copies are present in the build checkout; the deb metadata notes the AGPL-3.0 model license where applicable.
