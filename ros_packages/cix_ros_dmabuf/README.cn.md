# cix_ros_dmabuf

CIX 板卡上 ROS 2 的跨进程 **dmabuf 零拷贝传输** 包。

`sensor_msgs/Image` 的大负载在进程间通常要经 DDS 序列化传输，即使双方在
同一颗 SoC 上。本包把像素彻底移出这条路径：发布方把帧留在 DMA buffer
（dmabuf fd）里，只发布一个几乎为空的 Image 消息；buffer 本身经 Unix
domain socket 以 `SCM_RIGHTS` 旁路传递，订阅方映射**同一块物理内存**做
推理——全程无像素 memcpy。

包内刻意不包含任何相机或 NPU 专属逻辑，便于通用包共享：
`cix_ros_camera`（发布方）和 `cix_ros_noe_handpose`（订阅方）都依赖它，
后续任何 NPU 推理包同样可以直接受益。

## 内容

| 组件 | 语言 | 作用 |
|---|---|---|
| `cix_ros_dmabuf_broker` | C++（可执行） | UDS fd 中继：发布方注册 fd，订阅方按 handle 取回 |
| `include/cix_ros_dmabuf/dmabuf_broker.hpp` | C++ | `BrokerClient`（发布/订阅两侧）与 `BrokerServer` |
| `include/cix_ros_dmabuf/dmabuf_image.hpp` | C++ | `DmaBuffer`（fd 的 RAII 包装）与 `DmaBufferImage`（图像元数据 + dmabuf） |
| `cix_ros_dmabuf/dma_image.py` | Python | Python 侧对应实现：`BrokerClient`、`DmaBuffer`、`get_broker_client()` |

库本身是纯 POSIX C++（不依赖 rclcpp），C++ 节点和纯 Python 节点都能使用。

## `;dmabuf_handle=N` encoding 约定

发布方与订阅方之间的线上契约写在 Image 消息的 `encoding` 字段里：

```
encoding = "<像素格式>;dmabuf_handle=<N>"
```

- `<像素格式>` —— 原始 encoding（`nv12`、`rgb8`、`bgr8`……），忽略后缀的
  订阅方看到的就是普通 Image。
- `<N>` —— broker handle（十进制）。订阅方连接 broker，调用
  `fetch_fd(N)` 取得指向同一底层 dmabuf 的本进程 fd。
- **无后缀 / broker 未运行** —— 消息就是普通完整 Image：订阅方回落到读
  `msg->data`（memcpy 路径）。零拷贝永远是优化项，不是正确性前提。

示例：`nv12;dmabuf_handle=7` 表示"一帧 NV12，像素存放在 broker 中注册为
handle 7 的 dmabuf 里"。

协议细节（REGISTER / FETCH / UNREGISTER / PING，大端
`命令(2字节) + handle(8字节)` 帧格式）见
`include/cix_ros_dmabuf/dmabuf_broker.hpp` 的头部注释。

## 使用

先启动 broker（两侧默认自动连接其默认 socket）：

```bash
ros2 run cix_ros_dmabuf cix_ros_dmabuf_broker           # 默认 /tmp/cix_ros_dmabuf_broker.sock
ros2 run cix_ros_dmabuf cix_ros_dmabuf_broker /tmp/other.sock   # 自定义路径
```

发布方（C++）：构造 `DmaBufferImage`，经 `cix_ros_camera` 中的 type
adapter 发布（它会自动注册 fd 并把 handle 追加到 encoding）。

订阅方（C++）：从 `msg->encoding` 解析出 handle，然后

```cpp
cix_ros_dmabuf::broker::BrokerClient client;   // 或线程安全的辅助封装
client.connect();                              // 默认 socket 路径
int fd = client.fetch_fd(handle);              // 同一块物理内存
```

订阅方（Python）：

```python
from cix_ros_dmabuf.dma_image import get_broker_client
fd, _ = get_broker_client().fetch_fd(handle)   # 再 mmap 使用
```

Python 客户端优先从 `DMABUF_BROKER_SOCK` 环境变量解析 socket 路径，未设
时回落到与 C++ 侧相同的内置默认值。

## 编译

```bash
colcon build --symlink-install --packages-select cix_ros_dmabuf
```

无板卡专属依赖，任意 Linux 主机均可编译。
