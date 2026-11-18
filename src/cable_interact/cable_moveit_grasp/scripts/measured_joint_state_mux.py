#!/usr/bin/env python3
"""Merge only actual hardware measurements, never URDF default positions."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from joint_state_store import MeasuredJointStore


class MeasuredJointStateMux(Node):
    def __init__(self):
        super().__init__('dual_joint_states')
        self.declare_parameter('stale_timeout', 1.0)
        self.declare_parameter('required_arms', ['a', 'b'])
        self.declare_parameter('arm_a_namespace', '/NS_2')
        self.declare_parameter('arm_b_namespace', '/NS_1')
        self.declare_parameter('output_topic', '/dual_arm/joint_states')
        arms = self.get_parameter('required_arms').value
        self.store = MeasuredJointStore(self.get_parameter('stale_timeout').value, arms)
        self.output = self.create_publisher(JointState, self.get_parameter('output_topic').value, 10)
        topics = {}
        for arm in arms:
            ns = self.get_parameter('arm_' + arm + '_namespace').value.rstrip('/')
            topics[arm] = ns + '/franka/joint_states'
            topics[arm + '_gripper'] = ns + '/franka_gripper/joint_states'
        self.input_subscriptions = [self.create_subscription(
            JointState, topic, lambda msg, source=source: self.receive(source, msg), qos_profile_sensor_data)
            for source, topic in topics.items()]
        self.timer = self.create_timer(1./30., self.publish)

    def receive(self, source, message):
        try:
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            ignored = self.store.update(source, list(message.name), list(message.position),
                                        list(message.velocity), list(message.effort), stamp, time.monotonic())
            if ignored:
                self.get_logger().warning(f'{source}: ignoring joints owned by another source: {ignored}',
                                          throttle_duration_sec=3.)
        except ValueError as error:
            self.get_logger().warning(f'{source}: dropped malformed measurement: {error}', throttle_duration_sec=3.)

    def publish(self):
        snapshot = self.store.snapshot(self.get_clock().now().nanoseconds, time.monotonic())
        if snapshot is None:
            self.get_logger().warning('Waiting for fresh raw joint positions from BOTH arms; no default poses published',
                                      throttle_duration_sec=3.)
            return
        msg = JointState()
        msg.name, msg.position, msg.velocity, msg.effort, stamp = snapshot
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        self.output.publish(msg)


def main():
    rclpy.init()
    node = MeasuredJointStateMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
