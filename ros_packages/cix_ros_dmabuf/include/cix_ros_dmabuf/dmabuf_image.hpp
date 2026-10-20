/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * DmaBuffer - self-contained dmabuf fd wrapper
 *
 * Design principles:
 *   1. RAII: takes an fd on construction, optionally closes on destruction
 *   2. Reference counted: lifetime managed via shared_ptr
 *   3. Zero-copy: fd can be passed across processes (SCM_RIGHTS)
 *   4. Mappable: mmap into user space to access the data
 */

#ifndef CIX_ROS_DMABUF__DMABUF_IMAGE_HPP_
#define CIX_ROS_DMABUF__DMABUF_IMAGE_HPP_

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace cix_ros_dmabuf
{

// ---------------------------------------------------------------------------
// DmaBuffer - RAII wrapper around a single dmabuf fd
// ---------------------------------------------------------------------------
class DmaBuffer
{
public:
  // fd ownership mode
  enum class Ownership
  {
    BORROW,  // does not own fd, destructor does not close
    OWN,     // owns fd, destructor closes it
  };

  DmaBuffer(int fd, size_t size, Ownership ownership = Ownership::BORROW)
  : fd_(fd), size_(size), ownership_(ownership), mapped_addr_(nullptr),
    external_ptr_(nullptr)
  {
  }

  // External-memory mode: wraps an already-mapped raw memory region (no fd).
  // Used when V4L2 EXPBUF is unsupported, to reuse the MMAP buffer directly
  // and avoid an extra copy. map() returns external_ptr; destructor does not
  // free it.
  DmaBuffer(void * external_ptr, size_t size)
  : fd_(-1), size_(size), ownership_(Ownership::BORROW), mapped_addr_(nullptr),
    external_ptr_(external_ptr)
  {
  }

  ~DmaBuffer()
  {
    unmap();
    if (ownership_ == Ownership::OWN && fd_ >= 0) {
      close(fd_);
    }
  }

  // Non-copyable
  DmaBuffer(const DmaBuffer &) = delete;
  DmaBuffer & operator=(const DmaBuffer &) = delete;

  // Movable
  DmaBuffer(DmaBuffer && other) noexcept
  : fd_(other.fd_), size_(other.size_), ownership_(other.ownership_),
    mapped_addr_(other.mapped_addr_), external_ptr_(other.external_ptr_)
  {
    other.fd_ = -1;
    other.size_ = 0;
    other.mapped_addr_ = nullptr;
    other.external_ptr_ = nullptr;
  }

  int fd() const { return fd_; }
  size_t size() const { return size_; }

  // mmap into user space
  const uint8_t * map(int prot = PROT_READ)
  {
    if (mapped_addr_) {
      return static_cast<const uint8_t *>(mapped_addr_);
    }
    // External-memory mode: return the external pointer directly, no mmap needed
    if (external_ptr_) {
      mapped_addr_ = external_ptr_;
      return static_cast<const uint8_t *>(mapped_addr_);
    }
    if (fd_ < 0) {
      return nullptr;
    }
    mapped_addr_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (mapped_addr_ == MAP_FAILED) {
      mapped_addr_ = nullptr;
      return nullptr;
    }
    return static_cast<const uint8_t *>(mapped_addr_);
  }

  uint8_t * map_rw()
  {
    return const_cast<uint8_t *>(map(PROT_READ | PROT_WRITE));
  }

  void unmap()
  {
    // External-memory mode: do not free the external pointer
    if (external_ptr_) {
      mapped_addr_ = nullptr;
      return;
    }
    if (mapped_addr_ && mapped_addr_ != MAP_FAILED) {
      munmap(mapped_addr_, size_);
      mapped_addr_ = nullptr;
    }
  }

  // Explicitly relinquish ownership (call before passing to another process)
  void release_ownership() { ownership_ = Ownership::BORROW; }

private:
  int fd_{-1};
  size_t size_{0};
  Ownership ownership_{Ownership::BORROW};
  void * mapped_addr_{nullptr};
  void * external_ptr_{nullptr};  // external-memory mode (no fd)
};

// ---------------------------------------------------------------------------
// DmaBufferImage - a DmaBuffer wrapper carrying image metadata
//
// This is the actual C++ type passed between publisher and subscriber.
// The ROS Type Adapter handles conversion to/from sensor_msgs::msg::Image.
// ---------------------------------------------------------------------------
struct DmaBufferImage
{
  using SharedPtr = std::shared_ptr<DmaBufferImage>;
  using ConstSharedPtr = std::shared_ptr<const DmaBufferImage>;

  // Header
  uint32_t seq{0};
  uint64_t timestamp_ns{0};
  std::string frame_id;

  // Image metadata
  uint32_t width{0};
  uint32_t height{0};
  uint32_t stride{0};      // bytes per row
  std::string encoding{"nv12"};

  // DMA buffer
  std::shared_ptr<DmaBuffer> dmabuf;

  // Cross-process transport metadata (used only with the UDS broker).
  // When crossing a process boundary, the publisher registers its fd with the
  // broker and gets back a handle; the subscriber uses that handle to fetch
  // the fd from the broker.
  uint64_t transport_handle{0};
};

}  // namespace cix_ros_dmabuf

#endif  // CIX_ROS_DMABUF__DMABUF_IMAGE_HPP_
