# SPDX-License-Identifier: BSD-3-Clause
"""dmabuf image Python wrapper for the cix_ros_dmabuf broker.

Zero-copy image transfer between camera/ISP drivers, ROS 2, and NPU —
the Python counterpart of the C++ BrokerClient / DmaBuffer headers.

Provides:
- DmaBuffer: RAII wrapper for dmabuf fd
- BrokerClient: Unix domain socket client to register/fetch fd handles
- Image message encoding/decoding with optional dmabuf handle embedding
"""

from __future__ import annotations

import ctypes
import enum
import os
import socket
import struct
import threading
from dataclasses import dataclass


_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _MSGHDR(ctypes.Structure):
    """struct msghdr for SCM_RIGHTS ancillary data transfer."""
    _fields_ = [
        ("msg_name", ctypes.c_void_p),
        ("msg_namelen", ctypes.c_uint),
        ("msg_iov", ctypes.c_void_p),
        ("msg_iovlen", ctypes.c_size_t),
        ("msg_control", ctypes.c_void_p),
        ("msg_controllen", ctypes.c_size_t),
        ("msg_flags", ctypes.c_int),
    ]


class _CMSGHDR(ctypes.Structure):
    _fields_ = [
        ("cmsg_len", ctypes.c_size_t),
        ("cmsg_level", ctypes.c_int),
        ("cmsg_type", ctypes.c_int),
    ]


def _CMSG_DATA(cmsg: ctypes.c_void_p) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.cast(cmsg, ctypes.c_uint).value +
                            ctypes.sizeof(_CMSGHDR))


# Mmap flags
PROT_READ = 1
PROT_WRITE = 2
MAP_SHARED = 0x01


class DmaBufferOwnership(enum.IntEnum):
    BORROW = 0  # don't close fd
    OWN = 1     # own fd, close on delete


class DmaBuffer:
    """RAII wrapper for dmabuf file descriptor."""

    def __init__(self, fd: int, size: int, ownership: DmaBufferOwnership = DmaBufferOwnership.OWN):
        self._fd = fd
        self._size = size
        self._ownership = ownership
        self._mapped: ctypes.c_void_p | None = None

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def size(self) -> int:
        return self._size

    def map(self, prot: int = PROT_READ) -> memoryview:
        """Map buffer into process address space."""
        if self._mapped is not None:
            return memoryview(ctypes.cast(self._mapped,
                ctypes.POINTER(ctypes.c_ubyte * self._size)).contents)

        _libc.mmap.restype = ctypes.c_void_p
        _libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_size_t,
        ]

        addr = _libc.mmap(None, self._size, prot, MAP_SHARED, self._fd, 0)
        if ctypes.cast(addr, ctypes.c_int).value == -1:
            errno = ctypes.get_errno()
            raise OSError(errno, f"mmap failed: {os.strerror(errno)}")

        self._mapped = ctypes.c_void_p(addr)
        return memoryview(ctypes.cast(self._mapped,
            ctypes.POINTER(ctypes.c_ubyte * self._size)).contents)

    def unmap(self) -> None:
        """Unmap buffer from process address space."""
        if self._mapped is not None:
            _libc.munmap(self._mapped, self._size)
            self._mapped = None

    def __del__(self) -> None:
        self.unmap()
        if self._ownership == DmaBufferOwnership.OWN and self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


@dataclass
class DmaBufferImage:
    """Image backed by dmabuf buffer."""
    width: int
    height: int
    stride: int
    encoding: str  # "nv12", "bgr8", etc.
    dmabuf: DmaBuffer | None = None
    timestamp_ns: int = 0
    frame_id: str = ""
    transport_handle: int = 0  # broker handle for cross-process

    @property
    def size(self) -> int:
        """Total bytes in buffer."""
        return self.dmabuf.size if self.dmabuf else 0


# ---------------------------------------------------------------------------
# Broker Protocol
# ---------------------------------------------------------------------------
class Command(enum.IntEnum):
    PING = 0x0001
    REGISTER = 0x0002
    FETCH = 0x0003
    UNREGISTER = 0x0004
    RESPONSE = 0x0010
    ERROR = 0x00FF


def _pack(cmd: Command, handle: int = 0) -> bytes:
    return struct.pack("!HQ", cmd, handle)  # 2 + 8 bytes


def _unpack(data: bytes) -> tuple[Command, int]:
    if len(data) >= 10:
        cmd, handle = struct.unpack("!HQ", data[:10])
        return Command(cmd), handle
    return Command.ERROR, 0


# Default broker socket path. Resolved at import time from the
# DMABUF_BROKER_SOCK environment variable so this package stays decoupled from
# any specific camera package — whatever broker the user runs, they point both
# sides at the same path (env var, constructor arg, or ROS param). Falls back to
# a generic name that contains no package identifier.
DEFAULT_SOCKET_PATH = os.environ.get(
    "DMABUF_BROKER_SOCK", "/tmp/cix_ros_dmabuf_broker.sock"
)


class BrokerClient:
    """Client for dmabuf broker service.

    Allows cross-process fd sharing via Unix domain socket SCM_RIGHTS. The
    socket path is fully parameterizable; no camera-package-specific default.
    """

    # Module-level singleton (see get_broker_client).
    _instance: BrokerClient | None = None
    _lock = threading.Lock()

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH) -> None:
        self._socket_path = socket_path
        self._sock: socket.socket | None = None
        self._connected = False
        self._connect()

    @classmethod
    def _ensure(cls, socket_path: str | None) -> BrokerClient:
        """Return the singleton.

        - ``socket_path=None``: keep whatever instance is already bound (and
          reconnect to its own path if the connection dropped). This matters
          because callers that already bound an explicit path at init later
          call ``get_broker_client()`` with no arg -- they must NOT be silently
          re-bound to the default.
        - ``socket_path`` given and different from the current instance: rebind.
        """
        with cls._lock:
            inst = cls._instance
            if socket_path is not None:
                if inst is None or inst._socket_path != socket_path:
                    if inst is not None and inst._sock is not None:
                        try:
                            inst._sock.close()
                        except Exception:
                            pass
                    inst = cls(socket_path)
                    cls._instance = inst
                return inst
            # No explicit path: use the existing instance if there is one.
            if inst is None:
                inst = cls(DEFAULT_SOCKET_PATH)
                cls._instance = inst
                return inst
            if not inst._connected:
                inst._connect()
            return inst

    def _connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self._socket_path)
            self._connected = True
            return True
        except Exception:
            self._sock = None
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def ping(self) -> bool:
        if not self._connected:
            return False
        try:
            self._sock.sendall(_pack(Command.PING))
            resp = self._sock.recv(2048)
            cmd, _ = _unpack(resp)
            return cmd == Command.RESPONSE
        except Exception:
            self._connected = False
            return False

    def register_fd(self, fd: int) -> int:
        """Register fd with broker, returning a handle for cross-process sharing."""
        if not self._connected:
            return 0
        try:
            # Send REGISTER with fd via SCM_RIGHTS
            fd_list = [fd]
            ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                          struct.pack("@i" * len(fd_list), *fd_list))]
            self._sock.sendmsg([_pack(Command.REGISTER)], ancillary)
            resp = self._sock.recv(2048)
            cmd, handle = _unpack(resp)
            return handle if cmd == Command.RESPONSE else 0
        except Exception:
            self._connected = False
            return 0

    def fetch_fd(self, handle: int) -> tuple[int, int]:
        """Fetch an fd from broker by handle. Returns (fd, size_bytes)."""
        if not self._connected or handle == 0:
            return -1, 0
        try:
            self._sock.sendall(_pack(Command.FETCH, handle))
            resp, anc, _, _ = self._sock.recvmsg(2048, 4096)
            cmd, _ = _unpack(resp)
            if cmd != Command.RESPONSE:
                return -1, 0
            # Extract fd from SCM_RIGHTS ancillary data
            for cmsg_level, cmsg_type, cmsg_data in anc:
                if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                    fds = struct.unpack("@i", cmsg_data[:struct.calcsize("@i")])
                    if fds:
                        return fds[0], 0  # Size not returned from current broker
            return -1, 0
        except Exception:
            self._connected = False
            return -1, 0

    def unregister(self, handle: int) -> bool:
        """Unregister handle from broker."""
        if not self._connected or handle == 0:
            return True
        try:
            self._sock.sendall(_pack(Command.UNREGISTER, handle))
            resp = self._sock.recv(2048)
            cmd, _ = _unpack(resp)
            return cmd == Command.RESPONSE
        except Exception:
            self._connected = False
            return False


# Convenience: module-level singleton. Pass a socket_path to (re)bind to a
# different broker; omit it to use DEFAULT_SOCKET_PATH (env DMABUF_BROKER_SOCK).
def get_broker_client(socket_path: str | None = None) -> BrokerClient:
    return BrokerClient._ensure(socket_path)
