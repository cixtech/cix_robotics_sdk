/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SubTestNode - zero-copy subscription test node
 *
 * Subscribes to DmaBufferImage (auto-converted to/from Image via the Type Adapter)
 *
 * Behaviour:
 *   - Intra-process: receives DmaBufferImage::ConstSharedPtr directly, zero-copy
 *   - Cross-process: automatically degrades to a copy
 *   - With dump=true, dumps each frame to /tmp
 */

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "cix_ros_dmabuf/dmabuf_image.hpp"
#include "cix_ros_camera/dmabuf_type_adapter.hpp"

namespace cix_ros_camera
{

using cix_ros_dmabuf::DmaBufferImage;

class SubTestNode : public rclcpp::Node
{
public:
  using ImageSubscription = rclcpp::Subscription<
      rclcpp::TypeAdapter<DmaBufferImage, sensor_msgs::msg::Image>>;

  explicit SubTestNode(const rclcpp::NodeOptions & options)
    : rclcpp::Node("cix_camera_sub_test", options)
  {
    dump_ = declare_parameter<bool>("dump", false);
    dump_camera_info_ = declare_parameter<bool>("dump_camera_info", false);

    image_sub_ = create_subscription<
        rclcpp::TypeAdapter<DmaBufferImage, sensor_msgs::msg::Image>>(
        "image", 10,
        [this](const DmaBufferImage::ConstSharedPtr & msg) {
          on_image(msg);
        });

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", 10,
        [this](const sensor_msgs::msg::CameraInfo & msg) {
          on_camera_info(msg);
        });

    RCLCPP_INFO(get_logger(), "SubTestNode ready (dump=%s)",
        dump_ ? "true" : "false");
  }

private:
  void on_image(const DmaBufferImage::ConstSharedPtr & msg)
  {
    const uint64_t ts_ns = static_cast<uint64_t>(msg->timestamp_ns);
    const uint64_t now_ns = static_cast<uint64_t>(get_clock()->now().nanoseconds());

    {
      std::lock_guard<std::mutex> lock(stats_mu_);
      ++frame_count_;
      if (last_log_ns_ == 0) {
        last_log_ns_ = now_ns;
      }

      const uint64_t window_ns = 5ULL * 1000000000ULL;
      if (now_ns - last_log_ns_ >= window_ns) {
        const double fps = static_cast<double>(frame_count_) * 1e9 /
            static_cast<double>(now_ns - last_log_ns_);
        RCLCPP_INFO(get_logger(),
            "[sub] frame_id='%s' enc=%s %ux%u stride=%u count=%lu fps=%.2f",
            msg->frame_id.c_str(), msg->encoding.c_str(),
            msg->width, msg->height, msg->stride,
            static_cast<unsigned long>(frame_count_), fps);
        frame_count_ = 0;
        last_log_ns_ = now_ns;
      }
    }

    if (dump_ && msg->dmabuf && msg->dmabuf->fd() >= 0) {
      const size_t size = msg->height * msg->stride;
      const uint8_t * data = msg->dmabuf->map();
      if (data) {
        char path[512];
        std::snprintf(path, sizeof(path), "/tmp/sub_%s_%ux%u_%u.nv12",
            msg->frame_id.c_str(), msg->width, msg->height, ++dump_seq_);
        FILE * fp = std::fopen(path, "wb");
        if (fp) {
          std::fwrite(data, 1, size, fp);
          std::fclose(fp);
        }
        msg->dmabuf->unmap();
      }
    }
  }

  void on_camera_info(const sensor_msgs::msg::CameraInfo & info)
  {
    if (!dump_camera_info_) {
      return;
    }
    RCLCPP_INFO(get_logger(),
        "[sub] CameraInfo %ux%u fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        info.width, info.height,
        info.k[0], info.k[4], info.k[2], info.k[5]);
  }

  bool dump_{false};
  bool dump_camera_info_{false};
  uint32_t dump_seq_{0};

  ImageSubscription::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

  std::mutex stats_mu_;
  uint64_t frame_count_{0};
  uint64_t last_log_ns_{0};
};

}  // namespace cix_ros_camera

RCLCPP_COMPONENTS_REGISTER_NODE(cix_ros_camera::SubTestNode)
