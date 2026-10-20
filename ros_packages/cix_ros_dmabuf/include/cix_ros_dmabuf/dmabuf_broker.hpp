/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * dmabuf_broker - cross-process dmabuf fd sharing service
 *
 * Zero-copy fd passing over Unix Domain Sockets + SCM_RIGHTS
 *
 * Flow:
 *
 *   1. Start the broker: `cix_ros_dmabuf_broker`
 *      - Listens on UDS: `/tmp/cix_ros_dmabuf_broker.sock`
 *
 *   2. Publisher (e.g. the V4L2 dmabuf camera node):
 *      - Connects to the broker
 *      - register_fd(fd) -> handle (uint64_t)
 *      - Appends to the Image encoding field: "nv12;dmabuf_handle=0x12345678"
 *      - Publishes a standard Image (converted from DmaBufferImage)
 *
 *   3. Subscriber (e.g. an NPU inference node):
 *      - Receives the Image, detects the dmabuf_handle in encoding
 *      - Connects to the broker, fetch_fd(handle) -> gets an fd in this process
 *      - Wraps the fd in a DmaBuffer for zero-copy data access
 *
 *   4. Buffer reclaim:
 *      - When the publisher's DmaBuffer refcount reaches zero, it tells the
 *        broker to unregister(handle)
 *      - isp_app reclaims the buffer
 *
 * Protocol (binary):
 *   Command (2 bytes BE) + Handle (8 bytes BE) + Payload (optional)
 *
 *   CMD_REGISTER:  register fd -> handle
 *     Send: CMD_REGISTER + no payload
 *     Recv: CMD_RESPONSE + handle
 *     Note: fd is passed via SCM_RIGHTS ancillary message
 *
 *   CMD_FETCH:  fetch fd by handle
 *     Send: CMD_FETCH + handle
 *     Recv: CMD_RESPONSE + fd via SCM_RIGHTS
 *
 *   CMD_UNREGISTER:  unregister handle
 *     Send: CMD_UNREGISTER + handle
 *     Recv: CMD_RESPONSE + ok
 *
 *   CMD_PING:  health check
 *     Send: CMD_PING
 *     Recv: CMD_RESPONSE
 */

#ifndef CIX_ROS_DMABUF__DMABUF_BROKER_HPP_
#define CIX_ROS_DMABUF__DMABUF_BROKER_HPP_

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unordered_map>
#include <vector>

namespace cix_ros_dmabuf
{
namespace broker
{

// ---------------------------------------------------------------------------
// Protocol definitions
// ---------------------------------------------------------------------------
enum Command : uint16_t {
  CMD_PING       = 0x0001,
  CMD_REGISTER   = 0x0002,
  CMD_FETCH      = 0x0003,
  CMD_UNREGISTER = 0x0004,
  CMD_RESPONSE   = 0x0010,
  CMD_ERROR      = 0x00FF,
};

constexpr const char * DEFAULT_SOCKET_PATH = "/tmp/cix_ros_dmabuf_broker.sock";
constexpr int MAX_CLIENTS = 64;

// ---------------------------------------------------------------------------
// BrokerClient - used by both publisher and subscriber
// ---------------------------------------------------------------------------
class BrokerClient
{
public:
  BrokerClient();
  ~BrokerClient();

  // Connect to the broker
  bool connect(const std::string & socket_path = DEFAULT_SOCKET_PATH);
  bool is_connected() const { return sock_fd_ >= 0; }
  void disconnect();

  // Publisher: register an fd -> handle
  // Note: fd must be a dmabuf (memfd or any other dup-able fd)
  uint64_t register_fd(int fd);

  // Subscriber: obtain an fd for a handle
  // Returns a new fd in the current process; caller must close it
  int fetch_fd(uint64_t handle);

  // Publisher: unregister (called when a buffer is reclaimed)
  bool unregister(uint64_t handle);

  // Health check
  bool ping();

private:
  int sock_fd_{-1};
  std::mutex mu_;

  // Send a command + optional fd, receive the response
  ssize_t send_cmd(Command cmd, uint64_t handle = 0, int * send_fd = nullptr);
  ssize_t recv_response(Command & out_cmd, uint64_t & out_handle, int * out_fd = nullptr);
};

// ---------------------------------------------------------------------------
// BrokerServer - the broker service
// ---------------------------------------------------------------------------
class BrokerServer
{
public:
  explicit BrokerServer(const std::string & socket_path = DEFAULT_SOCKET_PATH);
  ~BrokerServer();

  bool start();
  void stop();
  void run();  // Blocks until stop()

  bool is_running() const { return running_.load(); }

  // Statistics
  size_t registered_count() const { return handle_to_fd_.size(); }
  size_t client_count() const { return client_count_; }

private:
  // Returns false when the peer closed/errored; the main loop should then
  // close and drop that fd.
  bool handle_client(int client_fd);

  // Called when a client connection drops: unregisters every handle that
  // client registered (a crashed publisher can't send CMD_UNREGISTER itself).
  void drop_client_handles(int client_fd);

  std::string socket_path_;
  int listen_fd_{-1};
  std::atomic<bool> running_{false};

  mutable std::mutex mu_;
  uint64_t next_handle_{1};
  std::unordered_map<uint64_t, int> handle_to_fd_;  // handle -> original fd (owned by us)
  std::unordered_map<int, uint64_t> fd_to_handle_;  // fd -> handle (dedup:
                                                    // repeated register of the
                                                    // same fd returns the same
                                                    // handle, so the 4 rotating
                                                    // V4L2 buffers don't accrue
                                                    // a new fd each frame -> leak)
  std::unordered_map<int, std::vector<uint64_t>> client_handles_;  // client fd -> handles it registered
  std::atomic<size_t> client_count_{0};
};

}  // namespace broker
}  // namespace cix_ros_dmabuf

#endif  // CIX_ROS_DMABUF__DMABUF_BROKER_HPP_
