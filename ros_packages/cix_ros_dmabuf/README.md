# cix_ros_dmabuf

Cross-process **dmabuf zero-copy transport** for ROS 2 on CIX boards.

Large `sensor_msgs/Image` payloads normally get serialized through DDS even
between processes on the same SoC. This package moves the pixels out of that
path entirely: the publisher keeps the frame in a DMA buffer (dmabuf fd) and
publishes a nearly-empty Image message; the buffer itself is passed
out-of-band over a Unix domain socket using `SCM_RIGHTS`, and the subscriber
maps the **same physical memory** for inference — zero pixel memcpy.

It is deliberately free of any camera- or NPU-specific logic so that generic
packages can share it: `cix_ros_camera` (publisher side) and
`cix_ros_noe_handpose` (subscriber side) both depend on it, and any future
NPU inference package can do the same.

## Contents

| Piece | Language | Role |
|---|---|---|
| `cix_ros_dmabuf_broker` | C++ (executable) | UDS fd relay: publishers register fds, subscribers fetch them by handle |
| `include/cix_ros_dmabuf/dmabuf_broker.hpp` | C++ | `BrokerClient` (publisher + subscriber side) and `BrokerServer` |
| `include/cix_ros_dmabuf/dmabuf_image.hpp` | C++ | `DmaBuffer` (RAII fd wrapper) and `DmaBufferImage` (image metadata + dmabuf) |
| `cix_ros_dmabuf/dma_image.py` | Python | Python counterpart: `BrokerClient`, `DmaBuffer`, `get_broker_client()` |

The library is pure POSIX C++ (no rclcpp dependency), so both C++ nodes and
plain Python nodes can use it.

## The `;dmabuf_handle=N` encoding convention

The wire contract between publisher and subscriber lives in the Image
message's `encoding` field:

```
encoding = "<pixel_format>;dmabuf_handle=<N>"
```

- `<pixel_format>` — the plain encoding (`nv12`, `rgb8`, `bgr8`, ...), exactly
  what a subscriber that ignores the suffix would expect.
- `<N>` — a broker handle (decimal). The subscriber connects to the broker and
  calls `fetch_fd(N)` to obtain its own fd for the same underlying dmabuf.
- **No suffix / no broker** — the message is a normal full Image: subscribers
  fall back to reading `msg->data` (memcpy path). Zero-copy is always an
  optimization, never a correctness requirement.

Example: `nv12;dmabuf_handle=7` means "an NV12 frame whose pixels live in the
dmabuf registered with the broker as handle 7".

Protocol details (REGISTER / FETCH / UNREGISTER / PING, big-endian
`command(2) + handle(8)` framing) are documented in
`include/cix_ros_dmabuf/dmabuf_broker.hpp`.

## Usage

Start the broker first (both sides auto-connect to its default socket):

```bash
ros2 run cix_ros_dmabuf cix_ros_dmabuf_broker           # default /tmp/cix_ros_dmabuf_broker.sock
ros2 run cix_ros_dmabuf cix_ros_dmabuf_broker /tmp/other.sock   # custom path
```

Publisher side (C++): build a `DmaBufferImage`, publish via the type adapter
in `cix_ros_camera` (it registers the fd and appends the handle to the
encoding automatically).

Subscriber side (C++): parse the handle out of `msg->encoding`, then

```cpp
cix_ros_dmabuf::broker::BrokerClient client;   // or the thread-safe helper
client.connect();                              // default socket path
int fd = client.fetch_fd(handle);              // same physical memory
```

Subscriber side (Python):

```python
from cix_ros_dmabuf.dma_image import get_broker_client
fd, _ = get_broker_client().fetch_fd(handle)   # then mmap it
```

The Python client resolves the socket path from the `DMABUF_BROKER_SOCK`
environment variable when set, falling back to the same built-in default as
the C++ side.

## Build

```bash
colcon build --symlink-install --packages-select cix_ros_dmabuf
```

No board-specific dependencies — builds on any Linux host.
