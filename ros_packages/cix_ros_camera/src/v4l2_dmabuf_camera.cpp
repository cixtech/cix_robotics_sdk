/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * v4l2_dmabuf_camera.cpp - V4L2 dmabuf capture node
 *
 * Captures NV12/RGB frames from a V4L2 device, exports dmabuf fds, and publishes
 * them zero-copy via the ROS 2 Type Adapter.
 */

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cix_ros_dmabuf/dmabuf_image.hpp"

#include "cix_ros_camera/dmabuf_type_adapter.hpp"
#include <poll.h>

namespace cix_ros_camera
{

using cix_ros_dmabuf::DmaBuffer;
using cix_ros_dmabuf::DmaBufferImage;

using DmaBufferImageAdapted = rclcpp::TypeAdapter<DmaBufferImage, sensor_msgs::msg::Image>;

struct Buffer
{
  void * start{MAP_FAILED};
  size_t length{0};
  int dmabuf_fd{-1};
};

// ISP's video nodes are multiplanar (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, see
// isp_app isp_core_v4l.c). RGB24 is a single plane; NV12 is also exposed as a
// single plane by this driver, so num_planes_ is typically 1.
static constexpr __u32 kBufType = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

class V4L2DmaBufCamera : public rclcpp::Node
{
public:
  explicit V4L2DmaBufCamera(const rclcpp::NodeOptions & options)
  : rclcpp::Node("v4l2_dmabuf_camera", options)
  {
    // Declare parameters
    device_ = declare_parameter<std::string>("device", "/dev/video9");
    width_ = declare_parameter<int>("width", 1280);
    height_ = declare_parameter<int>("height", 720);
    pixel_format_ = declare_parameter<std::string>("pixel_format", "RGB3");
    topic_name_ = declare_parameter<std::string>("topic", "/cam0_stream1");
    fps_ = declare_parameter<double>("fps", 30.0);
    frame_id_ = declare_parameter<std::string>("frame_id", "camera");

    RCLCPP_INFO(get_logger(), "V4L2 dmabuf camera starting...");
    RCLCPP_INFO(get_logger(), "  Device: %s", device_.c_str());
    RCLCPP_INFO(get_logger(), "  Resolution: %dx%d", width_, height_);
    RCLCPP_INFO(get_logger(), "  Pixel format: %s", pixel_format_.c_str());
    RCLCPP_INFO(get_logger(), "  Topic: %s", topic_name_.c_str());

    // Open the V4L2 device (NONBLOCK matches the working cix_v4l2_rgb3_shm producer)
    fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to open %s: %s", device_.c_str(), strerror(errno));
      throw std::runtime_error("Failed to open V4L2 device");
    }

    // Set the format
    if (!set_format()) {
      close(fd_);
      throw std::runtime_error("Failed to set V4L2 format");
    }

    // Request buffers
    if (!request_buffers()) {
      close(fd_);
      throw std::runtime_error("Failed to request V4L2 buffers");
    }

    // Create the publisher -- SensorDataQoS (BEST_EFFORT, KEEP_LAST 5): large
    // image frames under RELIABLE would stall on retransmission; matches the QoS
    // of the downstream handpose/rqt image topics to avoid frame drops.
    pub_ = create_publisher<DmaBufferImageAdapted>(topic_name_, rclcpp::SensorDataQoS());

    // Start streaming
    if (!start_streaming()) {
      close(fd_);
      throw std::runtime_error("Failed to start streaming");
    }

    // Timer: poll V4L2
    timer_ = create_wall_timer(
      std::chrono::microseconds(static_cast<int64_t>(1000000.0 / fps_ / 2.0)),
      [this]() { poll_and_publish(); });

    RCLCPP_INFO(get_logger(), "V4L2 dmabuf camera started successfully!");
  }

  ~V4L2DmaBufCamera()
  {
    stop_streaming();
    for (auto & buf : buffers_) {
      if (buf.dmabuf_fd >= 0) {
        close(buf.dmabuf_fd);
      }
      if (buf.start != MAP_FAILED) {
        munmap(buf.start, buf.length);
      }
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

private:
  bool set_format()
  {
    struct v4l2_format fmt{};
    fmt.type = kBufType;
    fmt.fmt.pix_mp.width = width_;
    fmt.fmt.pix_mp.height = height_;

    uint32_t bytespp = 0;  // bytes per pixel for single-plane packed formats
    // Convert the FourCC string
    if (pixel_format_ == "NV12") {
      fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
      bytespp = 1;  // Y plane stride = width; size handled below
    } else if (pixel_format_ == "RGB3" || pixel_format_ == "BGR3") {
      fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_RGB24;
      bytespp = 3;
    } else {
      RCLCPP_ERROR(get_logger(), "Unsupported pixel format: %s", pixel_format_.c_str());
      return false;
    }

    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;
    // The ISP driver rejects S_FMT unless bytesperline/sizeimage are provided
    // (matches cix_v4l2_rgb3_shm.cpp behaviour).
    uint32_t stride = 0;
    uint32_t sizeimage = 0;
    if (bytespp == 3) {  // RGB24 packed
      stride = width_ * 3;
      sizeimage = stride * height_;
    } else {  // NV12: Y plane stride=width, total = w*h*3/2
      stride = width_;
      sizeimage = width_ * height_ * 3 / 2;
    }
    fmt.fmt.pix_mp.plane_fmt[0].bytesperline = stride;
    fmt.fmt.pix_mp.plane_fmt[0].sizeimage = sizeimage;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
      RCLCPP_ERROR(get_logger(), "VIDIOC_S_FMT failed: %s", strerror(errno));
      return false;
    }

    // Verify the actual configuration applied
    num_planes_ = fmt.fmt.pix_mp.num_planes;
    if (num_planes_ < 1) {
      RCLCPP_ERROR(get_logger(), "driver reported 0 planes");
      return false;
    }
    actual_width_ = fmt.fmt.pix_mp.width;
    actual_height_ = fmt.fmt.pix_mp.height;
    // ISP driver quirk: for packed RGB24 it returns bytesperline = width (1280)
    // instead of width*3 (3840), even though the buffer is laid out with a
    // width*3 stride (sizeimage = width*3*height). The reference producer
    // (cix_v4l2_rgb3_shm) ignores the reported bytesperline and uses
    // width*bytes_per_pixel. Do the same so subscribers get the true stride.
    const uint32_t driver_stride = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
    if (bytespp == 3) {
      stride_ = actual_width_ * 3;
    } else {
      // NV12: Y plane stride = width (driver reports this correctly)
      stride_ = actual_width_;
    }

    RCLCPP_INFO(get_logger(), "V4L2 format set: %dx%d, planes=%u, driver_stride=%d, using stride=%d",
                actual_width_, actual_height_, num_planes_, driver_stride, stride_);
    return true;
  }

  bool request_buffers()
  {
    struct v4l2_requestbuffers req{};
    req.count = 4;
    req.type = kBufType;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
      RCLCPP_ERROR(get_logger(), "VIDIOC_REQBUFS failed: %s", strerror(errno));
      return false;
    }

    RCLCPP_INFO(get_logger(), "Got %d V4L2 buffers", req.count);

    // Map each buffer and export its dmabuf
    for (uint32_t i = 0; i < req.count; ++i) {
      struct v4l2_buffer buf{};
      struct v4l2_plane planes[VIDEO_MAX_PLANES]{};
      struct v4l2_exportbuffer expbuf{};

      buf.type = kBufType;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      buf.length = num_planes_;        // number of planes (mplane)
      buf.m.planes = planes;

      if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
        RCLCPP_ERROR(get_logger(), "VIDIOC_QUERYBUF %d failed: %s", i, strerror(errno));
        return false;
      }

      // For multiplane, offset/length live in planes[0]
      const size_t buf_len = planes[0].length;
      void * start = mmap(NULL, buf_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                          fd_, planes[0].m.mem_offset);
      if (start == MAP_FAILED) {
        RCLCPP_ERROR(get_logger(), "mmap buffer %d failed: %s", i, strerror(errno));
        return false;
      }

      // Export the dmabuf fd
      expbuf.type = kBufType;
      expbuf.index = i;
      expbuf.plane = 0;
      expbuf.flags = O_RDWR;

      if (ioctl(fd_, VIDIOC_EXPBUF, &expbuf) < 0) {
        RCLCPP_WARN(get_logger(), "VIDIOC_EXPBUF %d failed: %s (fallback to copy)",
                    i, strerror(errno));
        buffers_.push_back({start, buf_len, -1});
      } else {
        RCLCPP_INFO(get_logger(), "Buffer %d: dmabuf fd=%d", i, expbuf.fd);
        buffers_.push_back({start, buf_len, expbuf.fd});
      }

      // Queue the buffer (mplane: must reset the planes pointer)
      struct v4l2_buffer qbuf{};
      struct v4l2_plane qplanes[VIDEO_MAX_PLANES]{};
      qbuf.type = kBufType;
      qbuf.memory = V4L2_MEMORY_MMAP;
      qbuf.index = i;
      qbuf.length = num_planes_;
      qbuf.m.planes = qplanes;

      if (ioctl(fd_, VIDIOC_QBUF, &qbuf) < 0) {
        RCLCPP_ERROR(get_logger(), "VIDIOC_QBUF %d failed: %s", i, strerror(errno));
        return false;
      }
    }

    // Count buffers that support dmabuf
    int dmabuf_count = 0;
    for (auto & buf : buffers_) {
      if (buf.dmabuf_fd >= 0) dmabuf_count++;
    }
    RCLCPP_INFO(get_logger(), "%d/%d buffers support dmabuf", dmabuf_count, (int)buffers_.size());

    return true;
  }

  bool start_streaming()
  {
    enum v4l2_buf_type type = static_cast<enum v4l2_buf_type>(kBufType);
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
      RCLCPP_ERROR(get_logger(), "VIDIOC_STREAMON failed: %s", strerror(errno));
      return false;
    }
    return true;
  }

  void stop_streaming()
  {
    enum v4l2_buf_type type = static_cast<enum v4l2_buf_type>(kBufType);
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
  }

  void poll_and_publish()
  {
    struct pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;

    const int ret = poll(&pfd, 1, 10); // 10ms timeout
    if (ret <= 0) {
      return; // no data
    }

    if (!(pfd.revents & POLLIN)) {
      return;
    }

    // Dequeue a buffer (mplane)
    struct v4l2_buffer buf{};
    struct v4l2_plane planes[VIDEO_MAX_PLANES]{};
    buf.type = kBufType;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = num_planes_;
    buf.m.planes = planes;

    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
      RCLCPP_ERROR(get_logger(), "VIDIOC_DQBUF failed: %s", strerror(errno));
      return;
    }

    const size_t idx = buf.index;
    if (idx >= buffers_.size()) {
      RCLCPP_ERROR(get_logger(), "Buffer index out of range: %zu", idx);
      return;
    }

    // Publish the frame (mplane: bytesused is in planes[0])
    const size_t bytesused = planes[0].bytesused;
    publish_frame(idx, bytesused);

    // Re-queue the buffer
    struct v4l2_buffer qbuf{};
    struct v4l2_plane qplanes[VIDEO_MAX_PLANES]{};
    qbuf.type = kBufType;
    qbuf.memory = V4L2_MEMORY_MMAP;
    qbuf.index = idx;
    qbuf.length = num_planes_;
    qbuf.m.planes = qplanes;

    if (ioctl(fd_, VIDIOC_QBUF, &qbuf) < 0) {
      RCLCPP_ERROR(get_logger(), "VIDIOC_QBUF failed: %s", strerror(errno));
    }
  }

  void publish_frame(size_t idx, size_t bytesused)
  {
    auto & buffer = buffers_[idx];
    DmaBufferImage msg{};

    // Use rclcpp::Time to get a nanosecond timestamp
    auto stamp = get_clock()->now();
    msg.timestamp_ns = static_cast<uint64_t>(stamp.nanoseconds());
    msg.frame_id = frame_id_;
    msg.width = actual_width_;
    msg.height = actual_height_;
    msg.stride = stride_;
    // Map the V4L2 pixel_format token to a ROS-standard encoding string so
    // subscribers (e.g. handpose_dmabuf_node) interpret the bytes correctly.
    // Anything else is passed through verbatim.
    if (pixel_format_ == "RGB3") {
      msg.encoding = "rgb8";
    } else if (pixel_format_ == "BGR3") {
      msg.encoding = "bgr8";
    } else if (pixel_format_ == "NV12") {
      msg.encoding = "nv12";
    } else {
      msg.encoding = pixel_format_;
    }

    if (buffer.dmabuf_fd >= 0) {
      // Zero-copy path: wrap the dmabuf fd
      msg.dmabuf = std::make_shared<DmaBuffer>(
        buffer.dmabuf_fd, bytesused, DmaBuffer::Ownership::BORROW);
      zero_copy_count_++;
    } else {
      // Fallback path: the ISP driver doesn't support EXPBUF, so reuse the
      // already-mapped buffer memory directly (external-memory DmaBuffer,
      // fd=-1). The Type Adapter will memcpy it into Image.data.
      msg.dmabuf = std::make_shared<DmaBuffer>(buffer.start, bytesused);
    }

    pub_->publish(msg);

    frame_count_++;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time_).count();
    if (elapsed >= 5) {
      const double actual_fps = static_cast<double>(frame_count_) / static_cast<double>(elapsed);
      RCLCPP_INFO(get_logger(), "Published %zu frames (%.1f fps), zero-copy: %zu",
                  frame_count_, actual_fps, zero_copy_count_);
      frame_count_ = 0;
      zero_copy_count_ = 0;
      last_log_time_ = now;
    }
  }

  std::string device_;
  int width_;
  int height_;
  int actual_width_;
  int actual_height_;
  int stride_;
  std::string pixel_format_;
  std::string topic_name_;
  std::string frame_id_;
  double fps_;

  int fd_{-1};
  std::vector<Buffer> buffers_;
  uint32_t num_planes_{0};

  rclcpp::Publisher<DmaBufferImageAdapted>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  size_t frame_count_{0};
  size_t zero_copy_count_{0};
  std::chrono::steady_clock::time_point last_log_time_{std::chrono::steady_clock::now()};
};

}  // namespace cix_ros_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(cix_ros_camera::V4L2DmaBufCamera)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cix_ros_camera::V4L2DmaBufCamera>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
