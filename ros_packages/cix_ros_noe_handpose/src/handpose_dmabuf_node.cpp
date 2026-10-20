/*
 * Copyright 2026 Cix Technology Group Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * handpose_dmabuf_node.cpp - Zero-copy NPU handpose inference node.
 *
 * Zero-copy chain (no pixel memcpy on the inference path):
 *
 *   camera dmabuf fd (broker)
 *      |  CME import_from_external_buffer (EXTERNAL_DMA: NV12 / RGB / BGR)
 *      v
 *   src cme_img -> CME cme_2d_crop_resize_flip_rotation_cvtcolor_task
 *      |           (resize 256x256 + cvtcolor -> lb_img RGB_888)
 *      v
 *   CME cme_2d_normalize_task (multi=scale/255, add=zp)
 *      |                      (HWC RGB -> NCHW U8, written into NOE input dmabuf)
 *      v
 *   NOE input dmabuf --- noe_specify_iobuf(ATTACH_DMABUF, bound ONCE at init)
 *      |  noe_job_infer_sync
 *      v
 *   NOE output dmabuf --- noe_specify_iobuf(ATTACH_DMABUF, bound ONCE at init)
 *      |  mmap + DMA_BUF_SYNC (cache invalidate) + CPU dequant (S8 -> F32)
 *      v
 *   42 floats -> postprocess -> keypoints -> direction -> publish
 *
 * The NOE input/output dmabufs are allocated once from /dev/dma_heap/system and
 * bound to the NPU job via noe_specify_iobuf; every frame CME overwrites the
 * input dmabuf and the NPU reads it directly. Modeled on
 * the CIX NOE sample
cix_yolo_demo.cpp (EngineInfer), but uses the
 * non-ext CME API (board libcme 1.1.0 has no _ext tasks) and does output
 * dequant on CPU (output is only 42 bytes — too small to justify a CME pass).
 *
 * Visualization (debug/output image) is the only path that touches CPU pixels:
 * the camera dmabuf is mmap'd read-only to draw the keypoint overlay. It is
 * skipped entirely when both publish_*_image are false.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>
#include <linux/dma-buf.h>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3.hpp>
// cv_bridge >= 4.1 (Iron+; Kilted/Jazzy) removed the .h header; Humble has
// only .h. Support both distros.
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "cix_ros_dmabuf/dmabuf_broker.hpp"
#include "cix_ros_dmabuf/dmabuf_image.hpp"
#include "cix_ros_noe_handpose/msg/hand_pose_keypoints.hpp"

#include <cme/cme.h>
#include "cix_noe_standard_api.h"

// case4 shared_case_type: dma_buf managed by another module (the dma_heap
// allocator here), attached to the NPU job. Not enumerated in the public header.
#ifndef NOE_SHARE_BUF_ATTACH_DMABUF
#define NOE_SHARE_BUF_ATTACH_DMABUF 0x3
#endif

using namespace cix_ros_dmabuf;

namespace cix_ros_noe_handpose
{

// ---------------------------------------------------------------------------
// Constants & utilities
// ---------------------------------------------------------------------------

// Hand skeleton edges (for visualization)
const std::vector<std::pair<int, int>> HAND_SKELETON_EDGES = {
  {0, 1}, {1, 2}, {2, 3}, {3, 4},         // thumb
  {0, 5}, {5, 6}, {6, 7}, {7, 8},         // index
  {0, 9}, {9, 10}, {10, 11}, {11, 12},     // middle
  {0, 13}, {13, 14}, {14, 15}, {15, 16},   // ring
  {0, 17}, {17, 18}, {18, 19}, {19, 20},   // pinky
  {5, 9}, {9, 13}, {13, 17}                // palm
};

// Broker client wrapper (thread-safe)
struct BrokerWrapper
{
  broker::BrokerClient client;
  bool connected{false};
  std::mutex mutex;

  bool ensure_connected(const std::string & socket_path)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!connected) {
      if (socket_path.empty()) {
        connected = client.connect();
      } else {
        connected = client.connect(socket_path);
      }
    }
    return connected;
  }
};

// ---------------------------------------------------------------------------
// HandposeNoeEngine - low-level NOE + CME zero-copy inference engine
//
// Replaces the high-level NoeEngine (libnoe_engine.so) with direct
// cix_noe_standard_api.h calls so the NPU input/output tensors live in
// dma_heap dmabufs bound via noe_specify_iobuf -- no host->device memcpy.
// ---------------------------------------------------------------------------

class HandposeNoeEngine
{
public:
  explicit HandposeNoeEngine(rclcpp::Logger logger) : logger_(logger) {}

  ~HandposeNoeEngine() { clean(); }

  bool init(const std::string & model_path)
  {
    cme_initialize();
    cme_set_hwaccel_mode(CME_HWACCEL_MODE_PERFORMANCE);

    if (noe_init_context(&context_, NOE_DEVICE_AIPU) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_init_context failed");
      return false;
    }
    if (noe_load_graph(context_, model_path.c_str(), &graph_id_) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_load_graph failed for %s", model_path.c_str());
      return false;
    }

    noe_get_tensor_descriptor(context_, graph_id_, NOE_TENSOR_TYPE_INPUT, 0, &in_desc_);
    noe_get_tensor_descriptor(context_, graph_id_, NOE_TENSOR_TYPE_OUTPUT, 0, &out_desc_);

    if (!derive_shapes(model_path)) {
      return false;
    }

    // Model input must be UINT8 NCHW: CME writes quantized U8 NCHW straight
    // into the NPU input dmabuf. Other dtypes would need a different path.
    if (in_desc_.data_type != NOE_DATA_TYPE_U8) {
      RCLCPP_ERROR(logger_,
        "NOE: model input dtype=%d is not UINT8; this zero-copy path only "
        "supports UINT8 NCHW models (see cix_yolo_demo.cpp).", in_desc_.data_type);
      return false;
    }

    job_config_npu_t npu_cfg = {};
    job_config_t job_cfg = {};
    job_cfg.conf_j_npu = &npu_cfg;
    if (noe_create_job(context_, graph_id_, &job_id_, &job_cfg) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_create_job failed");
      return false;
    }

    if (!setup_input_dmabuf()) {
      return false;
    }
    if (!setup_output_dmabuf()) {
      return false;
    }

    RCLCPP_INFO(logger_,
      "NOE engine ready: input %ux%u UINT8 NCHW RGB (fd=%d), output %zu elems dtype=%d (zero-copy: bound dmabuf I/O)",
      in_w_, in_h_, in_dmabuf_fd_, out_elems_, out_desc_.data_type);
    return true;
  }

  // Preprocess src (camera frame as a cme_img) into the NPU input dmabuf.
  // src may be EXTERNAL_DMA (camera dmabuf fd) or EXTERNAL_MEMORY (msg->data).
  bool preprocess(cme_img * src)
  {
    cme_job_handle h = cme_begin_task();
    if (h < 0) {
      RCLCPP_ERROR(logger_, "CME: cme_begin_task failed");
      return false;
    }

    // Board libcme (non-ext API): cme_crfrc_ctrl has no format/letterbox/resize
    // fields — output format is set on lb_img_ (alloc_cme_img), and the resize
    // target is lb_img_ size (256x256 stretch, matches cv::resize).
    cme_crfrc_ctrl crfrc = {};
    crfrc.enable_crop = 0;
    crfrc.rotate = CME_ROTATE_0;
    crfrc.flip_mode = CME_HAL_TRANSFORM_FLIP_NONE;

    CME_RET ret = cme_2d_crop_resize_flip_rotation_cvtcolor_task(
      h, src, lb_img_, &crfrc);
    if (ret != CME_RET_SUCCESS) {
      RCLCPP_ERROR(logger_, "CME: crfrc_task failed: %d", ret);
      return false;
    }

    // CME normalize: dst = clamp(src * multi + add, min, max).
    // Model quantization: q = float_value * scale + zp, float_value = src/255,
    // so multi = scale/255, add = zp. Single-channel ctrl applies to all pixels.
    cme_normalize_ctrl nctrl = {};
    const float in_multi = in_desc_.scale / 255.0f;
    const float in_add = static_cast<float>(in_desc_.zero_point);
    nctrl.multi_factor = in_multi;
    nctrl.add_factor = in_add;
    nctrl.min = 0.0f;
    nctrl.max = 255.0f;
    nctrl.enable_clamp = 1;

    ret = cme_2d_normalize_task(h, lb_img_, nchw_img_, &nctrl);
    if (ret != CME_RET_SUCCESS) {
      RCLCPP_ERROR(logger_, "CME: normalize_task failed: %d", ret);
      return false;
    }

    if (cme_end_task(h, 1, nullptr) != CME_RET_SUCCESS) {  // sync
      RCLCPP_ERROR(logger_, "CME: cme_end_task failed");
      return false;
    }
    return true;
  }

  bool infer()
  {
    // The input dmabuf is bound to the NPU job via noe_specify_iobuf, so the
    // NPU reads the CME-written NCHW U8 input directly — no noe_load_tensor
    // host->device copy. Just kick inference and wait for completion.
    if (noe_job_infer_sync(context_, job_id_, -1) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_job_infer_sync failed");
      return false;
    }
    return true;
  }

  // Read NPU output directly from the bound output dmabuf and dequantize to
  // out_f32_. The output dmabuf is bound via noe_specify_iobuf, so the NPU
  // writes it directly — no noe_get_tensor device->host copy. Output is only
  // ~42 elems, so the mmap + cache invalidate + CPU dequant is negligible.
  const float * read_output()
  {
    if (out_dmabuf_fd_ < 0) {
      RCLCPP_ERROR(logger_, "read_output: no output dmabuf bound");
      return nullptr;
    }
    struct dma_buf_sync sync0 = {};
    sync0.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    ioctl(out_dmabuf_fd_, DMA_BUF_IOCTL_SYNC, &sync0);
    void * va = mmap(nullptr, out_desc_.size, PROT_READ, MAP_SHARED, out_dmabuf_fd_, 0);
    if (va == MAP_FAILED) {
      RCLCPP_ERROR(logger_, "read_output: mmap output dmabuf failed");
      struct dma_buf_sync synce = {};
      synce.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
      ioctl(out_dmabuf_fd_, DMA_BUF_IOCTL_SYNC, &synce);
      return nullptr;
    }
    const uint8_t * raw_ptr = static_cast<const uint8_t *>(va);
    const float scale = out_desc_.scale > 0.0f ? out_desc_.scale : 1.0f;
    const float zp = static_cast<float>(out_desc_.zero_point);
    const size_t n = out_elems_;
    if (out_f32_.size() < n) { out_f32_.resize(n); }
    switch (out_desc_.data_type) {
      case NOE_DATA_TYPE_F32: {
        const auto * p = reinterpret_cast<const float *>(raw_ptr);
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = p[i]; }
        break;
      }
      case NOE_DATA_TYPE_F16: {
        const auto * p = reinterpret_cast<const __fp16 *>(raw_ptr);
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = static_cast<float>(p[i]); }
        break;
      }
      case NOE_DATA_TYPE_S8: {
        const auto * p = reinterpret_cast<const int8_t *>(raw_ptr);
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = (static_cast<float>(p[i]) - zp) / scale; }
        break;
      }
      case NOE_DATA_TYPE_U8: {
        const auto * p = raw_ptr;
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = (static_cast<float>(p[i]) - zp) / scale; }
        break;
      }
      case NOE_DATA_TYPE_S16: {
        const auto * p = reinterpret_cast<const int16_t *>(raw_ptr);
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = (static_cast<float>(p[i]) - zp) / scale; }
        break;
      }
      case NOE_DATA_TYPE_U16: {
        const auto * p = reinterpret_cast<const uint16_t *>(raw_ptr);
        for (size_t i = 0; i < n; ++i) { out_f32_[i] = (static_cast<float>(p[i]) - zp) / scale; }
        break;
      }
      default:
        RCLCPP_ERROR(logger_, "read_output: unsupported output dtype=%u", out_desc_.data_type);
        munmap(va, out_desc_.size);
        struct dma_buf_sync syncx = {};
        syncx.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
        ioctl(out_dmabuf_fd_, DMA_BUF_IOCTL_SYNC, &syncx);
        return nullptr;
    }
    munmap(va, out_desc_.size);
    struct dma_buf_sync sync3 = {};
    sync3.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    ioctl(out_dmabuf_fd_, DMA_BUF_IOCTL_SYNC, &sync3);
    return out_f32_.data();
  }

  uint32_t in_w() const { return in_w_; }
  uint32_t in_h() const { return in_h_; }
  size_t out_elems() const { return out_elems_; }

  void clean()
  {
    if (lb_img_) { free_cme_img(lb_img_); lb_img_ = nullptr; }
    if (nchw_img_) { free_cme_img(nchw_img_); nchw_img_ = nullptr; }

    if (in_dmabuf_fd_ >= 0) { close(in_dmabuf_fd_); in_dmabuf_fd_ = -1; }
    if (out_dmabuf_fd_ >= 0) { close(out_dmabuf_fd_); out_dmabuf_fd_ = -1; }

    if (job_id_) {
      noe_clean_job(context_, job_id_);
      job_id_ = 0;
    }
    if (graph_id_) {
      noe_unload_graph(context_, graph_id_);
      graph_id_ = 0;
    }
    if (context_) {
      noe_deinit_context(context_);
      context_ = nullptr;
    }
    cme_destroy();
  }

private:
  static int noe_dtype_bytes(uint32_t dt)
  {
    switch (dt) {
      case NOE_DATA_TYPE_U8: case NOE_DATA_TYPE_S8: return 1;
      case NOE_DATA_TYPE_U16: case NOE_DATA_TYPE_S16: case NOE_DATA_TYPE_F16: return 2;
      case NOE_DATA_TYPE_U32: case NOE_DATA_TYPE_S32: case NOE_DATA_TYPE_F32: return 4;
      default: return 2;
    }
  }

  // Allocate a dma_heap buffer and return its fd (-1 on failure).
  static int alloc_dmabuf(size_t size)
  {
    int heap_fd = open("/dev/dma_heap/system", O_RDWR);
    if (heap_fd < 0) {
      return -1;
    }
    struct dma_heap_allocation_data data = {};
    data.len = size;
    data.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
      close(heap_fd);
      return -1;
    }
    close(heap_fd);
    return data.fd;
  }

  // One-time zero fill so padding regions are not garbage.
  static bool zero_fill_dmabuf(int fd, size_t size)
  {
    void * va = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (va == MAP_FAILED) {
      return false;
    }
    memset(va, 0, size);
    munmap(va, size);
    return true;
  }

  bool derive_shapes(const std::string & model_path)
  {
    int32_t in_shape[8] = {};
    uint32_t in_dims = 8;
    char in_name[128] = {};
    uint32_t in_name_len = sizeof(in_name);
    noe_status_t sret = noe_get_tensor_shape(model_path.c_str(), NOE_TENSOR_TYPE_INPUT, 0,
        in_name, &in_name_len, in_shape, &in_dims);
    if (sret == NOE_STATUS_SUCCESS && in_dims >= 4) {
      in_h_ = static_cast<uint32_t>(in_shape[in_dims - 2]);
      in_w_ = static_cast<uint32_t>(in_shape[in_dims - 1]);
    } else {
      // Fallback: the handpose .cix does not embed tensor shape metadata that
      // noe_get_tensor_shape can read, but the descriptor (size/dtype) is
      // always available. Derive square HxW from size / dtype_bytes / 3 ch
      // (input is 1x3xHxW NCHW U8). Mirrors cix_yolo_demo.cpp _derive_model_shape.
      int in_bpe = noe_dtype_bytes(in_desc_.data_type);
      size_t in_elements = in_desc_.size / static_cast<size_t>(in_bpe);
      size_t hw = in_elements / 3;
      in_w_ = static_cast<uint32_t>(std::sqrt(static_cast<double>(hw)));
      in_h_ = static_cast<uint32_t>(hw / in_w_);
      if (static_cast<size_t>(in_w_) * in_h_ != hw) {
        RCLCPP_ERROR(logger_,
          "NOE: cannot derive input shape (get_tensor_shape ret=%d dims=%u; "
          "descriptor size=%u dtype=%d -> hw=%zu non-square)",
          static_cast<int>(sret), in_dims, in_desc_.size, in_desc_.data_type, hw);
        return false;
      }
      RCLCPP_WARN(logger_,
        "NOE: noe_get_tensor_shape ret=%d (dims=%u); fell back to descriptor "
        "size=%u -> %ux%u (3ch U8 NCHW)",
        static_cast<int>(sret), in_dims, in_desc_.size, in_w_, in_h_);
    }

    int32_t out_shape[8] = {};
    uint32_t out_dims = 8;
    char out_name[128] = {};
    uint32_t out_name_len = sizeof(out_name);
    noe_status_t oret = noe_get_tensor_shape(model_path.c_str(), NOE_TENSOR_TYPE_OUTPUT, 0,
        out_name, &out_name_len, out_shape, &out_dims);
    if (oret == NOE_STATUS_SUCCESS && out_dims >= 1) {
      out_elems_ = 1;
      for (uint32_t i = 0; i < out_dims; ++i) {
        out_elems_ *= static_cast<size_t>(out_shape[i]);
      }
    } else {
      // Fallback: derive from descriptor size.
      out_elems_ = out_desc_.size / noe_dtype_bytes(out_desc_.data_type);
      RCLCPP_WARN(logger_,
        "NOE: noe_get_tensor_shape(OUTPUT) ret=%d (dims=%u); fell back to "
        "descriptor size=%u dtype=%d -> %zu elems",
        static_cast<int>(oret), out_dims, out_desc_.size, out_desc_.data_type, out_elems_);
    }

    RCLCPP_INFO(logger_,
      "NOE: input  %u bytes dtype=%d scale=%.4f zp=%d -> %ux%u NCHW",
      in_desc_.size, in_desc_.data_type, in_desc_.scale, in_desc_.zero_point, in_w_, in_h_);
    RCLCPP_INFO(logger_,
      "NOE: output %u bytes dtype=%d scale=%.4f zp=%d -> %zu elems",
      out_desc_.size, out_desc_.data_type, out_desc_.scale, out_desc_.zero_point, out_elems_);
    return true;
  }

  bool setup_input_dmabuf()
  {
    in_dmabuf_fd_ = alloc_dmabuf(in_desc_.size + 0x800);
    if (in_dmabuf_fd_ < 0) {
      RCLCPP_ERROR(logger_, "NOE: alloc input dmabuf failed");
      return false;
    }
    if (!zero_fill_dmabuf(in_dmabuf_fd_, in_desc_.size)) {
      RCLCPP_ERROR(logger_, "NOE: zero-fill input dmabuf failed");
      return false;
    }

    // Bind the input dmabuf to the NPU job once at init. Every frame CME
    // overwrites it with the preprocessed NCHW U8 input and the NPU reads it
    // directly (no per-frame noe_load_tensor host->device copy).
    shared_tensor_info_t in_share = {};
    in_share.type = NOE_TENSOR_TYPE_INPUT;
    in_share.tensor_idx = 0;
    in_share.dmabuf_fd = in_dmabuf_fd_;
    in_share.offset_in_dmabuf = 0;
    in_share.shared_case_type = NOE_SHARE_BUF_ATTACH_DMABUF;
    if (noe_specify_iobuf(context_, job_id_, &in_share) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_specify_iobuf INPUT failed");
      return false;
    }

    // Letterbox intermediate: 256x256. The CME normalize CL path only accepts
    // RGB_888 src (BGR_888 is absent from the normalize hwaccel format mask and
    // is rejected with INVALID_PARAM), so lb_img_ is always RGB_888. The NPU
    // input is therefore RGB NCHW planes (R,G,B), matching the reference
    // cix_yolo_demo.cpp.
    alloc_img_ctrl lb_ctrl = {};
    lb_ctrl.width = static_cast<int>(in_w_);
    lb_ctrl.height = static_cast<int>(in_h_);
    lb_ctrl.format = CME_FORMAT_RGB_888;
    lb_ctrl.use_clmem = 0;
    lb_ctrl.stride[0][0] = static_cast<int>(in_w_) * 3;
    lb_ctrl.stride[0][1] = static_cast<int>(in_h_);
    int err = 0;
    lb_img_ = alloc_cme_img(&lb_ctrl, &err);
    if (!lb_img_) {
      RCLCPP_ERROR(logger_, "CME: alloc lb_img failed: %d", err);
      return false;
    }

    // NPU input dmabuf as 3-plane NCHW U8 (3 planes share one fd).
    const int plane = static_cast<int>(in_w_) * static_cast<int>(in_h_);
    external_buffer_ctrl dst_ctrl = {};
    dst_ctrl.nplanes = 3;
    dst_ctrl.format = CME_FORMAT_RGB_NCHW_U8;
    dst_ctrl.dma_fd[0] = in_dmabuf_fd_;
    dst_ctrl.dma_fd[1] = in_dmabuf_fd_;
    dst_ctrl.dma_fd[2] = in_dmabuf_fd_;
    dst_ctrl.stride[0][0] = static_cast<int>(in_w_);
    dst_ctrl.stride[0][1] = static_cast<int>(in_h_);
    dst_ctrl.stride[1][0] = static_cast<int>(in_w_);
    dst_ctrl.stride[1][1] = static_cast<int>(in_h_);
    dst_ctrl.stride[2][0] = static_cast<int>(in_w_);
    dst_ctrl.stride[2][1] = static_cast<int>(in_h_);
    dst_ctrl.offset[0] = 0;
    dst_ctrl.offset[1] = plane;
    dst_ctrl.offset[2] = 2 * plane;
    dst_ctrl.width = static_cast<int>(in_w_);
    dst_ctrl.height = static_cast<int>(in_h_);
    nchw_img_ = import_from_external_buffer(&dst_ctrl, &err);
    if (!nchw_img_) {
      RCLCPP_ERROR(logger_, "CME: import NCHW input dmabuf failed: %d", err);
      return false;
    }
    return true;
  }

  bool setup_output_dmabuf()
  {
    // Allocate the output dmabuf once and bind it to the NPU job. The NPU
    // writes its output directly into this dmabuf; read_output() mmaps it
    // (with a cache invalidate) and dequantizes — no per-frame noe_get_tensor
    // device->host copy.
    out_dmabuf_fd_ = alloc_dmabuf(out_desc_.size + 0x800);
    if (out_dmabuf_fd_ < 0) {
      RCLCPP_ERROR(logger_, "NOE: alloc output dmabuf failed");
      return false;
    }
    shared_tensor_info_t share = {};
    share.type = NOE_TENSOR_TYPE_OUTPUT;
    share.tensor_idx = 0;
    share.dmabuf_fd = out_dmabuf_fd_;
    share.offset_in_dmabuf = 0;
    share.shared_case_type = NOE_SHARE_BUF_ATTACH_DMABUF;
    if (noe_specify_iobuf(context_, job_id_, &share) != NOE_STATUS_SUCCESS) {
      RCLCPP_ERROR(logger_, "NOE: noe_specify_iobuf OUTPUT failed");
      return false;
    }
    out_f32_.resize(out_elems_);
    return true;
  }

  rclcpp::Logger logger_;

  noe_context_t context_{nullptr};
  uint64_t graph_id_{0};
  uint64_t job_id_{0};
  tensor_desc_t in_desc_{};
  tensor_desc_t out_desc_{};

  int in_dmabuf_fd_{-1};
  int out_dmabuf_fd_{-1};

  uint32_t in_w_{0}, in_h_{0};
  size_t out_elems_{0};

  cme_img * lb_img_{nullptr};       // 256x256 RGB_888 letterbox
  cme_img * nchw_img_{nullptr};     // NPU input dmabuf as NCHW U8
  std::vector<float> out_f32_;      // CPU-dequantized output (out_elems_ floats)
};

// ---------------------------------------------------------------------------
// Postprocess: NPU raw output -> pixel-space keypoints
// ---------------------------------------------------------------------------

std::vector<cv::Point2f> raw_output_to_keypoints_pixels(
  const float * raw, size_t count, int img_w, int img_h)
{
  std::vector<cv::Point2f> kpts;
  const size_t n = count / 2;
  kpts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    float x = raw[i * 2] * static_cast<float>(img_w);
    float y = raw[i * 2 + 1] * static_cast<float>(img_h);
    kpts.emplace_back(x, y);
  }
  return kpts;
}

// ---------------------------------------------------------------------------
// Gesture direction classification
// ---------------------------------------------------------------------------

bool is_index_mostly_extended(
  const std::vector<cv::Point2f> & kpts, float index_vs_other_ratio)
{
  if (kpts.size() < 21) {
    return false;
  }
  float dx = kpts[8].x - kpts[5].x;
  float dy = kpts[8].y - kpts[5].y;
  float index_len = std::sqrt(dx * dx + dy * dy);
  if (index_len < 1e-3f) {
    return false;
  }
  float mid_len = std::sqrt(
    std::pow(kpts[12].x - kpts[9].x, 2) + std::pow(kpts[12].y - kpts[9].y, 2));
  float ring_len = std::sqrt(
    std::pow(kpts[16].x - kpts[13].x, 2) + std::pow(kpts[16].y - kpts[13].y, 2));
  float pinky_len = std::sqrt(
    std::pow(kpts[20].x - kpts[17].x, 2) + std::pow(kpts[20].y - kpts[17].y, 2));
  float max_other = std::max(std::max(mid_len, ring_len), pinky_len);
  return index_len >= max_other * index_vs_other_ratio;
}

geometry_msgs::msg::Vector3 classify_index_finger_direction(
  const std::vector<cv::Point2f> & kpts,
  float min_finger_len_px,
  float direction_axis_ratio)
{
  geometry_msgs::msg::Vector3 result;
  if (kpts.size() < 9) {
    return result;
  }
  cv::Point2f mcp = kpts[5];
  cv::Point2f tip = kpts[8];
  float dx = tip.x - mcp.x;
  float dy = tip.y - mcp.y;
  float sq = dx * dx + dy * dy;
  if (sq < min_finger_len_px * min_finger_len_px) {
    return result;  // all-zero = NONE
  }
  float adx = std::abs(dx);
  float ady = std::abs(dy);
  if (adx >= direction_axis_ratio * ady) {
    result.x = (dx < 0) ? -1.0 : 1.0;  // LEFT = -1, RIGHT = 1
  } else {
    result.y = (dy < 0) ? -1.0 : 1.0;  // UP = -1, DOWN = 1
  }
  return result;
}

// ---------------------------------------------------------------------------
// Visualization: draw keypoints & skeleton
// ---------------------------------------------------------------------------

cv::Mat render_handpose_debug(
  const cv::Mat & bgr,
  const std::vector<cv::Point2f> & keypoints_px,
  const geometry_msgs::msg::Vector3 & direction,
  bool is_zero_copy,
  double fps = 0.0)
{
  cv::Mat out = bgr.clone();

  cv::putText(out, is_zero_copy ? "ZERO-COPY" : "COPY", cv::Point(20, 40),
              cv::FONT_HERSHEY_SIMPLEX, 1.0,
              is_zero_copy ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 3);

  if (fps > 0) {
    char fps_text[64];
    snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
    cv::putText(out, fps_text, cv::Point(out.cols - 180, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);
  }

  std::string dir_text = "DIR: ";
  if (direction.x > 0.5) {
    dir_text += "RIGHT";
  } else if (direction.x < -0.5) {
    dir_text += "LEFT";
  } else if (direction.y > 0.5) {
    dir_text += "DOWN";
  } else if (direction.y < -0.5) {
    dir_text += "UP";
  } else {
    dir_text += "NONE";
  }
  cv::putText(out, dir_text, cv::Point(20, 80),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 128, 0), 2);

  if (keypoints_px.size() >= 21) {
    for (const auto & edge : HAND_SKELETON_EDGES) {
      cv::Point p1(static_cast<int>(keypoints_px[edge.first].x),
                   static_cast<int>(keypoints_px[edge.first].y));
      cv::Point p2(static_cast<int>(keypoints_px[edge.second].x),
                   static_cast<int>(keypoints_px[edge.second].y));
      cv::line(out, p1, p2, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }
    cv::Point mcp_pt(static_cast<int>(keypoints_px[5].x),
                     static_cast<int>(keypoints_px[5].y));
    cv::Point tip_pt(static_cast<int>(keypoints_px[8].x),
                     static_cast<int>(keypoints_px[8].y));
    cv::line(out, mcp_pt, tip_pt, cv::Scalar(0, 220, 255), 3, cv::LINE_AA);
    cv::circle(out, mcp_pt, 6, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    cv::circle(out, tip_pt, 6, cv::Scalar(0, 128, 255), -1, cv::LINE_AA);
    for (size_t i = 0; i < keypoints_px.size(); ++i) {
      if (i != 5 && i != 8) {
        cv::Point p(static_cast<int>(keypoints_px[i].x),
                    static_cast<int>(keypoints_px[i].y));
        cv::circle(out, p, 3, cv::Scalar(180, 180, 180), -1, cv::LINE_AA);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// HandPoseDmaBufNode - zero-copy hand pose inference node
// ---------------------------------------------------------------------------

class HandPoseDmaBufNode : public rclcpp::Node
{
public:
  explicit HandPoseDmaBufNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("handpose_dmabuf_node", options)
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/cam0_stream1");
    output_topic_ = declare_parameter<std::string>("output_topic", "/handpose/image_out");
    debug_topic_ = declare_parameter<std::string>("debug_topic", "/handpose/debug_image");
    direction_topic_ = declare_parameter<std::string>("direction_topic", "/handpose/finger_direction");
    keypoints_topic_ = declare_parameter<std::string>("keypoints_topic", "/handpose/keypoints");
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    publish_output_image_ = declare_parameter<bool>("publish_output_image", true);
    npu_model_path_ = declare_parameter<std::string>("npu_model_path", "");
    broker_socket_path_ = declare_parameter<std::string>("broker_socket_path", "");
    // Preview downscale: 1280x720 bgr8 = 2.76MB/frame floods DDS; downscale before publish.
    debug_image_scale_ = declare_parameter<double>("debug_image_scale", 0.5);

    min_finger_len_px_ = declare_parameter<float>("min_finger_len_px", 6.0f);
    require_index_dominant_ = declare_parameter<bool>("require_index_dominant", false);
    index_vs_other_ratio_ = declare_parameter<float>("index_vs_other_ratio", 1.15f);
    // 1.4 => LEFT/RIGHT only past ~54 degrees from vertical (UP/DOWN sticky).
    direction_axis_ratio_ = declare_parameter<float>("direction_axis_ratio", 1.4f);

    RCLCPP_INFO(get_logger(), "========================================");
    RCLCPP_INFO(get_logger(), "  HandPose DMA-Buf Zero-Copy Node");
    RCLCPP_INFO(get_logger(), "========================================");
    RCLCPP_INFO(get_logger(), "  Input topic: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  Direction output: %s", direction_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  Zero-copy: V4L2 dmabuf -> CME -> NPU (no pixel memcpy)");
    RCLCPP_INFO(get_logger(), "========================================");

    init_npu_engine();

    const rclcpp::QoS image_qos = rclcpp::SensorDataQoS();
    if (publish_output_image_) {
      out_img_pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, image_qos);
      RCLCPP_INFO(get_logger(), "Output image -> %s", output_topic_.c_str());
    }
    if (publish_debug_image_) {
      debug_img_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_topic_, image_qos);
      RCLCPP_INFO(get_logger(), "Debug image -> %s", debug_topic_.c_str());
    }
    dir_pub_ = create_publisher<geometry_msgs::msg::Vector3>(direction_topic_, 10);
    RCLCPP_INFO(get_logger(), "Finger direction -> %s", direction_topic_.c_str());
    kpts_pub_ = create_publisher<cix_ros_noe_handpose::msg::HandPoseKeypoints>(keypoints_topic_, 10);
    RCLCPP_INFO(get_logger(), "Keypoints -> %s", keypoints_topic_.c_str());

    broker_.ensure_connected(broker_socket_path_);
    if (broker_.connected) {
      RCLCPP_INFO(get_logger(), "dmabuf broker connected - zero-copy mode enabled");
    } else {
      RCLCPP_WARN(get_logger(), "dmabuf broker not running - will use msg->data (external-memory) mode");
    }

    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, image_qos,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) { process_frame(msg); });

    RCLCPP_INFO(get_logger(), "========================================");
    RCLCPP_INFO(get_logger(), "  HandPose DMA-Buf Node started!");
    RCLCPP_INFO(get_logger(), "========================================");
  }

  ~HandPoseDmaBufNode()
  {
    if (npu_engine_) {
      npu_engine_->clean();
      npu_engine_.reset();
    }
  }

private:
  void init_npu_engine()
  {
    std::string model_path = npu_model_path_;
    if (model_path.empty()) {
      try {
        const std::string share =
          ament_index_cpp::get_package_share_directory("cix_ros_noe_handpose");
        model_path = share + "/models/handpose_resnet50_size256.cix";
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "package share dir not found: %s", e.what());
        throw std::runtime_error("cannot resolve handpose model path");
      }
    }
    if (model_path.empty() || access(model_path.c_str(), F_OK) != 0) {
      RCLCPP_ERROR(get_logger(), "NPU model not found: %s", model_path.c_str());
      throw std::runtime_error("NPU model file not found");
    }
    RCLCPP_INFO(get_logger(), "Loading NPU model: %s", model_path.c_str());

    npu_engine_ = std::make_unique<HandposeNoeEngine>(get_logger());
    if (!npu_engine_->init(model_path)) {
      npu_engine_->clean();
      npu_engine_.reset();
      throw std::runtime_error("NPU engine init failed");
    }
  }

  // Extract dmabuf_handle from the encoding string: "nv12;dmabuf_handle=0x.."
  uint64_t parse_handle(const std::string & encoding, std::string & out_plain_encoding)
  {
    const size_t semi_pos = encoding.find(';');
    if (semi_pos == std::string::npos) {
      out_plain_encoding = encoding;
      return 0;
    }
    out_plain_encoding = encoding.substr(0, semi_pos);
    const std::string key = "dmabuf_handle=";
    const size_t handle_pos = encoding.find(key, semi_pos + 1);
    if (handle_pos == std::string::npos) {
      return 0;
    }
    try {
      return std::stoull(encoding.substr(handle_pos + key.size()));
    } catch (...) {
      return 0;
    }
  }

  // Build a BGR cv::Mat for visualization from the camera dmabuf (mmap) or msg->data.
  // Only called when a debug/output image is requested.
  bool build_bgr_for_viz(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const std::string & encoding,
    const std::shared_ptr<DmaBuffer> & dmabuf,
    cv::Mat & bgr_out)
  {
    const uint32_t w = msg->width;
    const uint32_t h = msg->height;
    const uint32_t stride = msg->step;
    const uint8_t * base = nullptr;
    std::vector<uint8_t> copy_buf;  // only if we must copy out of msg->data

    if (dmabuf && dmabuf->fd() >= 0) {
      base = dmabuf->map(PROT_READ);
      if (!base) {
        return false;
      }
    } else if (!msg->data.empty()) {
      // msg->data is contiguous enough for a cv::Mat view when stride==w*bpp.
      base = msg->data.data();
    } else {
      return false;
    }

    try {
      if (encoding == "rgb8" || encoding == "RGB8") {
        cv::Mat rgb(static_cast<int>(h), static_cast<int>(w), CV_8UC3,
                    const_cast<uint8_t *>(base), stride);
        cv::cvtColor(rgb, bgr_out, cv::COLOR_RGB2BGR);
      } else if (encoding == "bgr8" || encoding == "BGR8") {
        cv::Mat bgr(static_cast<int>(h), static_cast<int>(w), CV_8UC3,
                   const_cast<uint8_t *>(base), stride);
        bgr_out = bgr.clone();
      } else {
        // NV12: single buffer, Y at 0, UV at stride*h.
        cv::Mat nv12(static_cast<int>(h + h / 2), static_cast<int>(stride), CV_8UC1,
                     const_cast<uint8_t *>(base));
        cv::cvtColor(nv12, bgr_out, cv::COLOR_YUV2BGR_NV12);
      }
    } catch (const cv::Exception & e) {
      RCLCPP_ERROR(get_logger(), "viz cvtColor error: %s", e.what());
      return false;
    }
    return !bgr_out.empty();
  }

  // Import the camera frame into a CME src cme_img.
  // dmabuf path: EXTERNAL_DMA (zero-copy). Else: EXTERNAL_MEMORY (msg->data, no memcpy).
  // Returns a cme_img the caller must free_cme_img(), or nullptr on failure.
  cme_img * import_src(
    const sensor_msgs::msg::Image::SharedPtr msg,
    const std::string & encoding,
    const std::shared_ptr<DmaBuffer> & dmabuf)
  {
    const int w = static_cast<int>(msg->width);
    const int h = static_cast<int>(msg->height);
    const int stride = static_cast<int>(msg->step);
    external_buffer_ctrl ctrl = {};
    int err = 0;

    const bool have_dmabuf = dmabuf && dmabuf->fd() >= 0;
    const bool is_nv12 = (encoding == "nv12" || encoding == "NV12");

    if (have_dmabuf) {
      const int fd = dmabuf->fd();
      if (is_nv12) {
        // Single-buffer NV12: Y at offset 0, UV interleaved at stride*h.
        ctrl.nplanes = 2;
        ctrl.format = CME_FORMAT_NV12;
        ctrl.dma_fd[0] = fd;
        ctrl.dma_fd[1] = fd;
        ctrl.offset[0] = 0;
        ctrl.offset[1] = stride * h;
        ctrl.stride[0][0] = stride;
        ctrl.stride[0][1] = h;
        ctrl.stride[1][0] = stride;
        ctrl.stride[1][1] = h / 2;
      } else if (encoding == "bgr8" || encoding == "BGR8") {
        ctrl.nplanes = 1;
        ctrl.format = CME_FORMAT_BGR_888;
        ctrl.dma_fd[0] = fd;
        ctrl.stride[0][0] = stride;
        ctrl.stride[0][1] = h;
      } else {  // rgb8 default
        ctrl.nplanes = 1;
        ctrl.format = CME_FORMAT_BGR_888;
        ctrl.dma_fd[0] = fd;
        ctrl.stride[0][0] = stride;
        ctrl.stride[0][1] = h;
      }
      ctrl.width = w;
      ctrl.height = h;
      return import_from_external_buffer(&ctrl, &err);
    }

    // Fallback: msg->data as external (non-dma) memory. CME may use a shadow
    // buffer internally, but no explicit host memcpy in our code.
    ctrl.nplanes = 1;
    if (is_nv12) {
      ctrl.format = CME_FORMAT_NV12;
      ctrl.nplanes = 2;
      ctrl.vir_addr[0] = const_cast<uint8_t *>(msg->data.data());
      ctrl.vir_addr[1] = const_cast<uint8_t *>(msg->data.data() + stride * h);
      ctrl.stride[0][0] = stride;
      ctrl.stride[0][1] = h;
      ctrl.stride[1][0] = stride;
      ctrl.stride[1][1] = h / 2;
    } else if (encoding == "bgr8" || encoding == "BGR8") {
      ctrl.format = CME_FORMAT_BGR_888;
      ctrl.vir_addr[0] = const_cast<uint8_t *>(msg->data.data());
      ctrl.stride[0][0] = stride;
      ctrl.stride[0][1] = h;
    } else {
      ctrl.format = CME_FORMAT_RGB_888;
      ctrl.vir_addr[0] = const_cast<uint8_t *>(msg->data.data());
      ctrl.stride[0][0] = stride;
      ctrl.stride[0][1] = h;
    }
    ctrl.width = w;
    ctrl.height = h;
    return import_from_external_buffer(&ctrl, &err);
  }

  void process_frame(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    const auto start_time = std::chrono::high_resolution_clock::now();

    const std::string & full_encoding = msg->encoding;
    std::string encoding;
    const uint64_t handle = parse_handle(full_encoding, encoding);

    // Acquire the camera dmabuf fd (zero-copy) if a broker handle is present.
    std::shared_ptr<DmaBuffer> dmabuf;
    bool use_zero_copy = false;
    if (handle != 0 && broker_.ensure_connected(broker_socket_path_)) {
      int fd = -1;
      {
        std::lock_guard<std::mutex> lock(broker_.mutex);
        fd = broker_.client.fetch_fd(handle);
      }
      if (fd >= 0) {
        const size_t size = static_cast<size_t>(msg->height) * msg->step;
        dmabuf = std::make_shared<DmaBuffer>(fd, size, DmaBuffer::Ownership::OWN);
        use_zero_copy = true;
      }
    }

    // Stats (unconditional so the periodic latency log fires regardless of
    // whether the debug image is published).
    frame_count_++;
    if (use_zero_copy) {
      zero_copy_count_++;
    }
    // Import the camera frame into CME (EXTERNAL_DMA or EXTERNAL_MEMORY).
    cme_img * src_raw = import_src(msg, encoding, dmabuf);
    if (!src_raw) {
      RCLCPP_ERROR(get_logger(), "CME: import src image failed");
      return;
    }
    // Scope guard: free src cme_img on every exit. free_cme_img does not close
    // our fd (DmaBuffer owns it), matching the reference's behavior.
    auto src_guard = std::unique_ptr<cme_img, void (*)(cme_img *)>(
      src_raw, [](cme_img * p) { if (p) { free_cme_img(p); } });

    // Inference path: CME preprocess -> NPU input dmabuf -> infer -> dequant.
    if (!npu_engine_->preprocess(src_raw)) {
      return;
    }
    if (!npu_engine_->infer()) {
      return;
    }
    const float * out = npu_engine_->read_output();
    if (!out) {
      return;
    }

    // Visualization base (only if a debug/output image is requested).
    cv::Mat bgr_mat;
    const bool need_viz = (publish_output_image_ && out_img_pub_) ||
                          (publish_debug_image_ && debug_img_pub_);
    if (need_viz && !build_bgr_for_viz(msg, encoding, dmabuf, bgr_mat)) {
      RCLCPP_WARN(get_logger(), "viz: could not build BGR frame");
    }

    // Postprocess: 42 floats -> keypoints -> direction.
    std::vector<cv::Point2f> keypoints = raw_output_to_keypoints_pixels(
      out, npu_engine_->out_elems(),
      bgr_mat.empty() ? static_cast<int>(msg->width) : bgr_mat.cols,
      bgr_mat.empty() ? static_cast<int>(msg->height) : bgr_mat.rows);

    bool index_ok = !require_index_dominant_ ||
                   is_index_mostly_extended(keypoints, index_vs_other_ratio_);
    geometry_msgs::msg::Vector3 direction = index_ok
      ? classify_index_finger_direction(keypoints, min_finger_len_px_, direction_axis_ratio_)
      : geometry_msgs::msg::Vector3();
    dir_pub_->publish(direction);

    // Full 21-point output: downstream consumers build their own gesture
    // semantics on top of the raw keypoints instead of the coarse direction.
    if (kpts_pub_->get_subscription_count() > 0) {
      cix_ros_noe_handpose::msg::HandPoseKeypoints kpts_msg;
      kpts_msg.header.stamp = get_clock()->now();
      kpts_msg.header.frame_id = msg->header.frame_id;
      kpts_msg.keypoints.reserve(keypoints.size());
      for (const auto & p : keypoints) {
        geometry_msgs::msg::Point pt;
        pt.x = p.x;
        pt.y = p.y;
        kpts_msg.keypoints.push_back(pt);
      }
      kpts_pub_->publish(kpts_msg);
    }

    if (publish_output_image_ && out_img_pub_ && !bgr_mat.empty()) {
      auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr_mat).toImageMsg();
      out_msg->header.stamp = get_clock()->now();
      out_img_pub_->publish(*out_msg);
    }

    if (publish_debug_image_ && debug_img_pub_ && !bgr_mat.empty()) {
      // frame_count_ is already incremented unconditionally above (Stats block);
      // do NOT double-count here, or the reported zero-copy ratio is halved.
      const auto now = std::chrono::high_resolution_clock::now();
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_fps_time_).count();
      double fps = 0.0;
      if (elapsed_ms >= 1000) {
        fps = static_cast<double>(frame_count_ - last_fps_count_) * 1000.0 / elapsed_ms;
        last_fps_count_ = frame_count_;
        last_fps_time_ = now;
        last_fps_ = fps;
      } else {
        fps = last_fps_;
      }
      cv::Mat debug = render_handpose_debug(bgr_mat, keypoints, direction, use_zero_copy, fps);
      cv::Mat debug_pub = debug;
      if (debug_image_scale_ > 0.0 && debug_image_scale_ < 1.0) {
        cv::resize(debug, debug_pub, cv::Size(), debug_image_scale_, debug_image_scale_, cv::INTER_AREA);
      }
      auto debug_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_pub).toImageMsg();
      debug_msg->header.stamp = get_clock()->now();
      debug_img_pub_->publish(*debug_msg);
    }

    if (frame_count_ % 100 == 0) {
      const auto end_time = std::chrono::high_resolution_clock::now();
      const auto avg_latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
      RCLCPP_INFO(get_logger(),
        "Frames: %lu (zero-copy: %lu), last latency: %.1f us",
        frame_count_, zero_copy_count_, static_cast<double>(avg_latency_us));
    }
  }

  // Parameters
  std::string image_topic_;
  std::string output_topic_;
  std::string debug_topic_;
  std::string direction_topic_;
  std::string keypoints_topic_;
  std::string npu_model_path_;
  std::string broker_socket_path_;
  bool publish_debug_image_;
  bool publish_output_image_;
  double debug_image_scale_;

  float min_finger_len_px_;
  bool require_index_dominant_;
  float index_vs_other_ratio_;
  float direction_axis_ratio_;

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr out_img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr dir_pub_;
  rclcpp::Publisher<cix_ros_noe_handpose::msg::HandPoseKeypoints>::SharedPtr kpts_pub_;

  BrokerWrapper broker_;
  std::unique_ptr<HandposeNoeEngine> npu_engine_;

  // Stats
  size_t frame_count_{0};
  size_t zero_copy_count_{0};
  size_t last_fps_count_{0};
  std::chrono::high_resolution_clock::time_point last_fps_time_{
    std::chrono::high_resolution_clock::now()};
  double last_fps_{0.0};
};

}  // namespace cix_ros_noe_handpose

// ROS 2 component registration
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cix_ros_noe_handpose::HandPoseDmaBufNode)

// Standalone main
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cix_ros_noe_handpose::HandPoseDmaBufNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
