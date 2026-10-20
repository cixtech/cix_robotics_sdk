/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "cix_ros_dmabuf/dmabuf_broker.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <atomic>
#include <thread>
#include <vector>

namespace cix_ros_dmabuf
{
namespace broker
{

namespace
{

// Helper: big-endian conversion
#if !defined(htobe16)
inline uint16_t htobe16(uint16_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return static_cast<uint16_t>(((x & 0xFF00) >> 8) | ((x & 0x00FF) << 8));
#else
  return x;
#endif
}
#endif

#if !defined(be16toh)
inline uint16_t be16toh(uint16_t x) { return htobe16(x); }
#endif

#if !defined(htobe64)
inline uint64_t htobe64(uint64_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return ((x & 0xFF00000000000000ULL) >> 56) |
         ((x & 0x00FF000000000000ULL) >> 40) |
         ((x & 0x0000FF0000000000ULL) >> 24) |
         ((x & 0x000000FF00000000ULL) >>  8) |
         ((x & 0x00000000FF000000ULL) <<  8) |
         ((x & 0x0000000000FF0000ULL) << 24) |
         ((x & 0x000000000000FF00ULL) << 40) |
         ((x & 0x00000000000000FFULL) << 56);
#else
  return x;
#endif
}
#endif

#if !defined(be64toh)
inline uint64_t be64toh(uint64_t x) { return htobe64(x); }
#endif

// sendmsg + SCM_RIGHTS to pass an fd
ssize_t send_with_fd(int sock, const void * data, size_t len, int fd)
{
  struct msghdr msg = {0};
  struct iovec iov = {.iov_base = const_cast<void*>(data), .iov_len = len};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))] = {0};

  if (fd >= 0) {
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    *reinterpret_cast<int*>(CMSG_DATA(cmsg)) = fd;
  }

  return sendmsg(sock, &msg, 0);
}

// recvmsg + SCM_RIGHTS to receive an fd
ssize_t recv_with_fd(int sock, void * data, size_t len, int * out_fd)
{
  struct msghdr msg = {0};
  struct iovec iov = {.iov_base = data, .iov_len = len};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))] = {0};
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  const ssize_t n = recvmsg(sock, &msg, 0);
  if (n <= 0) {
    return n;
  }

  if (out_fd) {
    *out_fd = -1;
    for (struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
         cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&msg, cmsg)) {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
        *out_fd = *reinterpret_cast<int*>(CMSG_DATA(cmsg));
        break;
      }
    }
  }

  return n;
}

}  // namespace

// ---------------------------------------------------------------------------
// BrokerClient
// ---------------------------------------------------------------------------
BrokerClient::BrokerClient() = default;

BrokerClient::~BrokerClient()
{
  disconnect();
}

bool BrokerClient::connect(const std::string & socket_path)
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ >= 0) {
    ::close(sock_fd_);
  }

  sock_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock_fd_ < 0) {
    return false;
  }

  struct sockaddr_un addr = {0};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (::connect(sock_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(sock_fd_);
    sock_fd_ = -1;
    return false;
  }

  // Set non-blocking receive/send timeout
  struct timeval tv = {1, 0}; // 1 second
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  return true;
}

void BrokerClient::disconnect()
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ >= 0) {
    ::close(sock_fd_);
    sock_fd_ = -1;
  }
}

ssize_t BrokerClient::send_cmd(Command cmd, uint64_t handle, int * send_fd)
{
  uint8_t buf[10];
  *reinterpret_cast<uint16_t*>(buf) = htobe16(static_cast<uint16_t>(cmd));
  *reinterpret_cast<uint64_t*>(buf + 2) = htobe64(handle);
  return send_with_fd(sock_fd_, buf, sizeof(buf), send_fd ? *send_fd : -1);
}

ssize_t BrokerClient::recv_response(Command & out_cmd, uint64_t & out_handle, int * out_fd)
{
  uint8_t buf[10];
  ssize_t n = recv_with_fd(sock_fd_, buf, sizeof(buf), out_fd);
  if (n < 2) {
    return n;
  }
  out_cmd = static_cast<Command>(be16toh(*reinterpret_cast<uint16_t*>(buf)));
  out_handle = (n >= 10) ? be64toh(*reinterpret_cast<uint64_t*>(buf + 2)) : 0;
  return n;
}

uint64_t BrokerClient::register_fd(int fd)
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ < 0) {
    return 0;
  }

  if (send_cmd(CMD_REGISTER, 0, &fd) < 0) {
    return 0;
  }

  Command cmd;
  uint64_t handle;
  if (recv_response(cmd, handle) < 0 || cmd != CMD_RESPONSE) {
    return 0;
  }
  return handle;
}

int BrokerClient::fetch_fd(uint64_t handle)
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ < 0) {
    return -1;
  }

  if (send_cmd(CMD_FETCH, handle) < 0) {
    return -1;
  }

  Command cmd;
  uint64_t dummy;
  int fd = -1;
  if (recv_response(cmd, dummy, &fd) < 0 || cmd != CMD_RESPONSE || fd < 0) {
    return -1;
  }
  return fd;
}

bool BrokerClient::unregister(uint64_t handle)
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ < 0) {
    return false;
  }

  if (send_cmd(CMD_UNREGISTER, handle) < 0) {
    return false;
  }

  Command cmd;
  uint64_t dummy;
  if (recv_response(cmd, dummy) < 0 || cmd != CMD_RESPONSE) {
    return false;
  }
  return true;
}

bool BrokerClient::ping()
{
  std::lock_guard<std::mutex> lock(mu_);
  if (sock_fd_ < 0) {
    return false;
  }

  if (send_cmd(CMD_PING, 0) < 0) {
    return false;
  }

  Command cmd;
  uint64_t dummy;
  if (recv_response(cmd, dummy) < 0 || cmd != CMD_RESPONSE) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// BrokerServer
// ---------------------------------------------------------------------------
BrokerServer::BrokerServer(const std::string & socket_path)
: socket_path_(socket_path)
{}

BrokerServer::~BrokerServer()
{
  stop();
}

bool BrokerServer::start()
{
  // Remove an existing socket file before binding
  unlink(socket_path_.c_str());

  listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    return false;
  }

  struct sockaddr_un addr = {0};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  if (listen(listen_fd_, MAX_CLIENTS) < 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    return false;
  }

  running_.store(true);
  return true;
}

void BrokerServer::stop()
{
  running_.store(false);
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  unlink(socket_path_.c_str());
}

void BrokerServer::run()
{
  if (!running_.load()) {
    return;
  }

  std::vector<pollfd> pfds;
  pfds.push_back({.fd = listen_fd_, .events = POLLIN, .revents = 0});

  while (running_.load()) {
    const int n = poll(pfds.data(), pfds.size(), 1000); // 1s timeout
    if (n < 0) {
      break;
    }
    if (n == 0) {
      continue;
    }

    // Check for new connections
    if (pfds[0].revents & POLLIN) {
      const int client_fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
      if (client_fd >= 0) {
        pfds.push_back({.fd = client_fd, .events = POLLIN, .revents = 0});
        client_count_.fetch_add(1, std::memory_order_relaxed);
      }
      pfds[0].revents = 0;
    }

    // Handle clients
    for (size_t i = 1; i < pfds.size(); ) {
      if (pfds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
        // handle_client returns false when the peer closed/errored; we must
        // close and remove that fd, otherwise poll keeps reporting POLLIN while
        // recvmsg keeps returning 0 -> a 100% CPU busy loop.
        const bool alive = handle_client(pfds[i].fd);
        pfds[i].revents = 0;
        if (!alive) {
          // Drop everything this client registered: if a publisher crashed
          // without sending CMD_UNREGISTER, its handles (and the fds we hold
          // for them) would otherwise stay in the tables forever.
          drop_client_handles(pfds[i].fd);
          ::close(pfds[i].fd);
          pfds.erase(pfds.begin() + i);
          client_count_.fetch_sub(1, std::memory_order_relaxed);
          continue;
        }
        ++i;
      } else {
        ++i;
      }
    }
  }

  // Close all remaining client connections
  for (size_t i = 1; i < pfds.size(); ++i) {
    ::close(pfds[i].fd);
  }
}

bool BrokerServer::handle_client(int client_fd)
{
  uint8_t buf[10];
  int recv_fd = -1;
  const ssize_t n = recv_with_fd(client_fd, buf, sizeof(buf), &recv_fd);
  if (n <= 0) {
    // Peer closed (0) or errored (<0): tell the main loop to close and drop this fd.
    return false;
  }
  if (n < 2) {
    return true;
  }

  const Command cmd =
      static_cast<Command>(be16toh(*reinterpret_cast<uint16_t*>(buf)));
  const uint64_t handle = (n >= 10) ?
      be64toh(*reinterpret_cast<uint64_t*>(buf + 2)) : 0;

  Command resp_cmd = CMD_RESPONSE;
  uint64_t resp_handle = 0;
  int send_fd = -1;

  switch (cmd) {
    case CMD_PING:
      // OK, just respond
      break;

    case CMD_REGISTER:
      if (recv_fd >= 0) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          // Dedup: when the same fd (one of the 4 rotating V4L2 buffers) is
          // registered again, return the existing handle instead of holding
          // another fd. Otherwise a new fd accumulates every frame and soon
          // exhausts the process 1024 fd limit, after which all register calls
          // fail -> encoding carries no handle -> subscribers degrade to COPY.
          auto it = fd_to_handle_.find(recv_fd);
          if (it != fd_to_handle_.end()) {
            resp_handle = it->second;
            // Existing record for this fd: the received recv_fd is a redundant
            // dup, close it to avoid a leak.
            ::close(recv_fd);
          } else {
            resp_handle = next_handle_++;
            handle_to_fd_[resp_handle] = recv_fd; // server owns this fd
            fd_to_handle_[recv_fd] = resp_handle;
            // Track ownership so a crashed client's buffers can be reclaimed
            // when its connection drops (see drop_client_handles).
            client_handles_[client_fd].push_back(resp_handle);
            // Do NOT close recv_fd: we must keep it to dup for subscribers later.
          }
        }
      } else {
        resp_cmd = CMD_ERROR;
      }
      break;

    case CMD_FETCH:
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = handle_to_fd_.find(handle);
        if (it != handle_to_fd_.end()) {
          send_fd = dup(it->second); // dup for the requesting process
        } else {
          resp_cmd = CMD_ERROR;
        }
      }
      break;

    case CMD_UNREGISTER:
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = handle_to_fd_.find(handle);
        if (it != handle_to_fd_.end()) {
          fd_to_handle_.erase(it->second);  // keep reverse map in sync
          ::close(it->second);
          handle_to_fd_.erase(it);
        }
      }
      break;

    default:
      resp_cmd = CMD_ERROR;
      break;
  }

  uint8_t resp_buf[10];
  *reinterpret_cast<uint16_t*>(resp_buf) = htobe16(static_cast<uint16_t>(resp_cmd));
  *reinterpret_cast<uint64_t*>(resp_buf + 2) = htobe64(resp_handle);
  send_with_fd(client_fd, resp_buf, sizeof(resp_buf), send_fd);

  if (send_fd >= 0) {
    ::close(send_fd); // our local dup can be closed; the peer gets its own copy
  }
  return true;
}

void BrokerServer::drop_client_handles(int client_fd)
{
  std::lock_guard<std::mutex> lock(mu_);
  const auto cit = client_handles_.find(client_fd);
  if (cit == client_handles_.end()) {
    return;  // subscriber or a client that never registered anything
  }
  for (const uint64_t handle : cit->second) {
    // A handle may already be gone if the client cleanly unregistered it
    // before disconnecting - skip those.
    const auto it = handle_to_fd_.find(handle);
    if (it == handle_to_fd_.end()) {
      continue;
    }
    fd_to_handle_.erase(it->second);
    ::close(it->second);
    handle_to_fd_.erase(it);
  }
  client_handles_.erase(cit);
}

}  // namespace broker
}  // namespace cix_ros_dmabuf
