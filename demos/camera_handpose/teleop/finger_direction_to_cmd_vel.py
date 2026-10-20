#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Subscribe finger direction (geometry_msgs/Vector3) and publish geometry_msgs/Twist for diff-drive (twist_mux)."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cix_ros_noe_handpose.handpose_gesture import RobotCommand


def _as_float(val, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _vector3_to_command(v: Vector3, th: float) -> RobotCommand:
    """Map discrete -1,0,1 unit vectors back to RobotCommand (inverse of robot_command_to_discrete_xyz)."""
    x, y, _z = float(v.x), float(v.y), float(v.z)
    if abs(x) < th and abs(y) < th:
        return RobotCommand.NONE
    if abs(x) >= abs(y) and abs(x) >= th:
        return RobotCommand.LEFT if x < 0.0 else RobotCommand.RIGHT
    if abs(y) >= th:
        return RobotCommand.FORWARD if y < 0.0 else RobotCommand.BACK
    return RobotCommand.NONE


class FingerDirectionToCmdVel(Node):
    def __init__(self) -> None:
        super().__init__("finger_direction_to_cmd_vel")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.declare_parameter("input_topic", "/handpose/finger_direction")
        self.declare_parameter("output_topic", "key_vel")
        self.declare_parameter("linear_speed", 0.25)
        self.declare_parameter("angular_speed", 0.8)
        self.declare_parameter("cmd_vel_timeout_sec", 0.35)
        self.declare_parameter("component_threshold", 0.45)

        if self.get_parameter("use_sim_time").get_parameter_value().bool_value:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self._linear = _as_float(self.get_parameter("linear_speed").value, 0.25)
        self._angular = _as_float(self.get_parameter("angular_speed").value, 0.8)
        self._timeout = max(0.0, _as_float(self.get_parameter("cmd_vel_timeout_sec").value, 0.35))
        self._cth = _as_float(self.get_parameter("component_threshold").value, 0.45)

        in_t = self.get_parameter("input_topic").get_parameter_value().string_value.strip() or (
            "/handpose/finger_direction"
        )
        out_t = self.get_parameter("output_topic").get_parameter_value().string_value.strip() or "key_vel"

        self._pub = self.create_publisher(Twist, out_t, 10)
        # RELIABLE: match typical teleop; Vector3 is small so depth 1 ok
        vqos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Vector3, in_t, self._on_dir, vqos)
        self.get_logger().info(f"subscribe {in_t} -> publish Twist on {out_t} (twist_mux)")

        self._last_cmd = RobotCommand.NONE
        self._last_msg_time = self.get_clock().now()
        if self._timeout > 0.0:
            self._zero_timer = self.create_timer(0.05, self._maybe_stop)
        else:
            self._zero_timer = None

    def _command_to_twist(self, cmd: RobotCommand) -> Twist:
        t = Twist()
        if cmd == RobotCommand.FORWARD:
            t.linear.x = self._linear
        elif cmd == RobotCommand.BACK:
            t.linear.x = -self._linear
        elif cmd == RobotCommand.LEFT:
            t.angular.z = self._angular
        elif cmd == RobotCommand.RIGHT:
            t.angular.z = -self._angular
        return t

    def _on_dir(self, msg: Vector3) -> None:
        cmd = _vector3_to_command(msg, self._cth)
        self._last_cmd = cmd
        self._last_msg_time = self.get_clock().now()
        # publish immediately so the robot feels responsive
        self._pub.publish(self._command_to_twist(cmd))

    def _maybe_stop(self) -> None:
        if self._timeout <= 0.0:
            return
        now = self.get_clock().now()
        if (now - self._last_msg_time).nanoseconds * 1e-9 > self._timeout:
            if self._last_cmd != RobotCommand.NONE:
                self._pub.publish(Twist())
                self._last_cmd = RobotCommand.NONE


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FingerDirectionToCmdVel()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"finger_direction_to_cmd_vel: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
