/*
 * Copyright 2026 Cix Technology Group Co., Ltd
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * cix_ros_dmabuf_broker - standalone broker executable
 *
 * Usage:
 *   cix_ros_dmabuf_broker [socket_path]
 *     - Default listen path: /tmp/cix_ros_dmabuf_broker.sock
 *
 * As long as the broker is running, publishers and subscribers auto-detect
 * and enable cross-process zero-copy.
 */

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <thread>

#include <iostream>
#include <string>

#include "cix_ros_dmabuf/dmabuf_broker.hpp"

using namespace cix_ros_dmabuf::broker;

int main(int argc, char * argv[])
{
  std::string socket_path = DEFAULT_SOCKET_PATH;
  if (argc > 1) {
    socket_path = argv[1];
  }

  // Block shutdown signals in every thread and handle them synchronously in a
  // dedicated sigwait() thread. This keeps signal context out of an async
  // signal handler, where calling into C++ (BrokerServer::stop -> close,
  // unlink, atomics) is undefined behavior.
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGINT);
  sigaddset(&sigs, SIGTERM);
  sigaddset(&sigs, SIGHUP);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  BrokerServer broker(socket_path);

  if (!broker.start()) {
    std::cerr << "Failed to start broker at " << socket_path << std::endl;
    return 1;
  }

  std::thread signal_thread([&broker, sigs]() {
      int sig = 0;
      sigwait(&sigs, &sig);
      broker.stop();
    });
  signal_thread.detach();

  std::cout << "========================================" << std::endl;
  std::cout << "  cix_ros_dmabuf_broker running" << std::endl;
  std::cout << "  socket: " << socket_path << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  press Ctrl+C to stop" << std::endl;
  std::cout << std::endl;

  // Monitor thread: prints periodic statistics
  std::thread stats_thread([&broker]() {
    while (broker.is_running()) {
      sleep(5);
      printf("[stats] clients=%zu, registered_buffers=%zu\n",
          broker.client_count(), broker.registered_count());
    }
  });

  broker.run();

  if (stats_thread.joinable()) {
    stats_thread.join();
  }

  std::cout << std::endl;
  std::cout << "Broker stopped, cleaning up..." << std::endl;
  return 0;
}
