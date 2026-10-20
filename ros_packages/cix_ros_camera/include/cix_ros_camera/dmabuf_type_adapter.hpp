/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ROS 2 Type Adapter: DmaBufferImage <-> sensor_msgs::msg::Image
 *
 * Behaviour:
 *   +---------------------------------------------------------------+
 *   | Intra-process                                                  |
 *   |   Publisher publishes DmaBufferImage::SharedPtr                |
 *   |   -> ROS 2 intra-process manager hands over the shared_ptr     |
 *   |   Subscriber receives DmaBufferImage::ConstSharedPtr           |
 *   |   -> zero-copy, no conversion at all                          |
 *   +---------------------------------------------------------------+
 *
 *   +---------------------------------------------------------------+
 *   | Cross-process + broker running                                |
 *   |   Publisher:  register_fd -> broker -> handle                 |
 *   |                encoding = "nv12;dmabuf_handle=0x1234"         |
 *   |   Subscriber: detect handle -> fetch_fd -> DmaBufferImage     |
 *   |   -> zero-copy!                                               |
 *   +---------------------------------------------------------------+
 *
 *   +---------------------------------------------------------------+
 *   | Cross-process + broker NOT running                            |
 *   |   Publisher:  DmaBufferImage -> mmap + memcpy -> Image.data   |
 *   |   Subscriber: Image.data -> memcpy -> new DmaBufferImage + fd |
 *   |   Copies memory, but stays compatible with all standard ROS   |
 *   |   tools (rviz, image_view, etc.)                              |
 *   +---------------------------------------------------------------+
 */

#ifndef CIX_ROS_CAMERA__DMABUF_TYPE_ADAPTER_HPP_
#define CIX_ROS_CAMERA__DMABUF_TYPE_ADAPTER_HPP_

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#include "cix_ros_dmabuf/dmabuf_broker.hpp"
#include "cix_ros_dmabuf/dmabuf_image.hpp"

namespace cix_ros_camera
{

// The zero-copy transport types live in cix_ros_dmabuf; pull them in here so the
// adapter below can keep using the unqualified names.
using ::cix_ros_dmabuf::DmaBuffer;
using ::cix_ros_dmabuf::DmaBufferImage;
namespace broker = ::cix_ros_dmabuf::broker;

namespace
{

// memfd_create wrapper (used by the fallback path)
inline int memfd_create_wrapper(const char * name, unsigned int flags)
{
#ifdef SYS_memfd_create
  return static_cast<int>(syscall(SYS_memfd_create, name, flags));
#else
  (void)name; (void)flags;
  errno = ENOSYS;
  return -1;
#endif
}

// thread_local broker client -- at most one connection per process
// Lazy: connects to the broker on first use
inline broker::BrokerClient & get_broker_client()
{
  static thread_local broker::BrokerClient client;
  static thread_local bool tried_connect = false;
  static thread_local bool is_connected = false;

  if (!tried_connect) {
    tried_connect = true;
    is_connected = client.connect();
  }
  return client;
}

inline bool broker_is_available()
{
  return get_broker_client().is_connected();
}

// Extract the dmabuf_handle from an encoding string; returns 0 if absent
inline uint64_t parse_handle(const std::string & encoding, std::string & out_plain_encoding)
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

}  // namespace

// ---------------------------------------------------------------------------
// Type Adapter core
// ---------------------------------------------------------------------------
struct DmaBufferImageAdapter
{
  using custom_type = DmaBufferImage;
  using ros_message_type = sensor_msgs::msg::Image;

  // -------------------------------------------------------------------------
  // DmaBufferImage -> sensor_msgs::msg::Image
  // Called by ROS only for cross-process serialization
  // -------------------------------------------------------------------------
  static ros_message_type convert_to_ros_message(const custom_type & src)
  {
    ros_message_type dst;

    // Header
    dst.header.stamp.sec = static_cast<int32_t>(src.timestamp_ns / 1000000000ULL);
    dst.header.stamp.nanosec = static_cast<uint32_t>(src.timestamp_ns % 1000000000ULL);
    dst.header.frame_id = src.frame_id;

    // Image metadata
    dst.width = src.width;
    dst.height = src.height;
    dst.step = src.stride;
    dst.encoding = src.encoding;
    dst.is_bigendian = 0;

    uint64_t handle = 0;

    // Try the broker zero-copy path
    if (src.dmabuf && src.dmabuf->fd() >= 0 && broker_is_available()) {
      // fd -> handle cache: the camera's V4L2 buffer fds (4 fixed, rotating)
      // are registered with the broker only once; later frames reuse the same
      // handle. Otherwise every frame re-registers, and on the broker side
      // SCM_RIGHTS delivers a different fd number (same underlying dmabuf)
      // -> no dedup possible -> the handle table grows without bound -> the
      // broker hits its fd limit -> all registers fail -> degrades to COPY.
      static thread_local std::unordered_map<int, uint64_t> fd_cache;
      auto it = fd_cache.find(src.dmabuf->fd());
      if (it != fd_cache.end()) {
        handle = it->second;
      } else {
        handle = get_broker_client().register_fd(src.dmabuf->fd());
        if (handle != 0) {
          fd_cache[src.dmabuf->fd()] = handle;
        }
      }
      if (handle != 0) {
        // Success: stash the handle in encoding, no data copy needed
        dst.encoding += ";dmabuf_handle=" + std::to_string(handle);
      }
    }

    // If the broker is unavailable or registration failed, fall back to memcpy.
    // (Also covers external-memory mode: the dmabuf wraps a raw MMAP pointer
    // rather than an fd.)
    if (handle == 0 && src.dmabuf) {
      const size_t data_size = src.height * src.stride;
      dst.data.resize(data_size);

      const uint8_t * src_ptr = src.dmabuf->map();
      if (src_ptr) {
        std::memcpy(dst.data.data(), src_ptr, data_size);
        src.dmabuf->unmap();
      } else {
        // mmap failed: fill with black NV12
        std::memset(dst.data.data(), 0, data_size);
      }
    }

    return dst;
  }

  // -------------------------------------------------------------------------
  // sensor_msgs::msg::Image -> DmaBufferImage
  // Called during cross-process deserialization
  // -------------------------------------------------------------------------
  static custom_type convert_to_custom(const ros_message_type & src)
  {
    custom_type dst;

    // Header
    dst.timestamp_ns = static_cast<uint64_t>(src.header.stamp.sec) * 1000000000ULL +
        static_cast<uint64_t>(src.header.stamp.nanosec);
    dst.frame_id = src.header.frame_id;

    // Image metadata
    dst.width = src.width;
    dst.height = src.height;
    dst.stride = src.step;

    // Parse encoding and extract any dmabuf handle
    uint64_t handle = parse_handle(src.encoding, dst.encoding);

    // Prefer the zero-copy path: fetch an fd from the broker using the handle
    if (handle != 0 && broker_is_available()) {
      const int fd = get_broker_client().fetch_fd(handle);
      if (fd >= 0) {
        // Zero-copy success! We own this fd, need Ownership::OWN to close it
        const size_t size = src.height * src.step;
        dst.dmabuf = std::make_shared<DmaBuffer>(fd, size, DmaBuffer::Ownership::OWN);
        dst.transport_handle = handle;
        return dst;
      }
    }

    // Fallback: copy from src.data into a new memfd-backed buffer
    if (!src.data.empty()) {
      const size_t data_size = src.data.size();

      void * ptr = mmap(nullptr, data_size, PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (ptr != MAP_FAILED) {
        std::memcpy(ptr, src.data.data(), data_size);

        int mem_fd = memfd_create_wrapper("ros_image_copy", MFD_CLOEXEC);
        if (mem_fd >= 0) {
          ftruncate(mem_fd, data_size);
          write(mem_fd, ptr, data_size);
          dst.dmabuf = std::make_shared<DmaBuffer>(mem_fd, data_size,
              DmaBuffer::Ownership::OWN);
        }
        munmap(ptr, data_size);
      }
    }

    return dst;
  }
};

}  // namespace cix_ros_camera

// Register the Type Adapter with ROS (standard ROS 2 Humble+ form)
namespace rclcpp
{

template<>
struct TypeAdapter<::cix_ros_dmabuf::DmaBufferImage, sensor_msgs::msg::Image>
{
  using is_specialized = std::true_type;
  using custom_type = ::cix_ros_dmabuf::DmaBufferImage;
  using ros_message_type = sensor_msgs::msg::Image;

  static void convert_to_ros_message(const custom_type & src, ros_message_type & dst)
  {
    dst = ::cix_ros_camera::DmaBufferImageAdapter::convert_to_ros_message(src);
  }

  static ros_message_type convert_to_ros_message(const custom_type & src)
  {
    return ::cix_ros_camera::DmaBufferImageAdapter::convert_to_ros_message(src);
  }

  static void convert_to_custom(const ros_message_type & src, custom_type & dst)
  {
    dst = ::cix_ros_camera::DmaBufferImageAdapter::convert_to_custom(src);
  }

  static custom_type convert_to_custom(const ros_message_type & src)
  {
    return ::cix_ros_camera::DmaBufferImageAdapter::convert_to_custom(src);
  }
};

}  // namespace rclcpp

#endif  // CIX_ROS_CAMERA__DMABUF_TYPE_ADAPTER_HPP_
