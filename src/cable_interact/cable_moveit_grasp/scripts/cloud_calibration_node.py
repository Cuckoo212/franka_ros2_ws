#!/usr/bin/python3
"""Republish calibrated XYZ while preserving original cloud layout and colors."""
import copy
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from calibration_math import load_calibration, transform_xyz_buffer, rotation_quaternion


class CalibratedCloud(Node):
    def __init__(self):
        super().__init__('cloud_calibration')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('input_topic', '/cable_camera/cable_points')
        self.declare_parameter('output_topic', '/cable_camera/cable_points_base')
        self.declare_parameter('input_frame', 'cable_camera_color_optical_frame')
        self.declare_parameter('input_is_optical', True)
        self.declare_parameter('expected_base_frame', 'fr3_link0')
        self.declare_parameter('publish_calibration_tf', True)
        parameter = lambda name: self.get_parameter(name).value
        self.input_frame = parameter('input_frame')
        self.rotation, self.translation, source, self.target = load_calibration(
            Path(parameter('calibration_file')).expanduser(), parameter('input_is_optical'))
        if self.target != parameter('expected_base_frame'):
            raise ValueError('Calibration robot_base_frame differs from expected_base_frame')
        if not self.input_frame or self.input_frame == self.target:
            raise ValueError('Invalid input_frame; refusing possible double transformation')
        if parameter('input_topic') == parameter('output_topic'):
            raise ValueError('Input/output topics must differ')
        cloud_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(PointCloud2, parameter('output_topic'), cloud_qos)
        if parameter('publish_calibration_tf'):
            # Quaternion of the composed base<-input transform; no SciPy dependency.
            q = rotation_quaternion(self.rotation)
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = self.target
            tf.child_frame_id = self.input_frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, self.translation)
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
            self.broadcaster = StaticTransformBroadcaster(self)
            self.broadcaster.sendTransform(tf)
        self.subscription = self.create_subscription(PointCloud2, parameter('input_topic'), self.receive, cloud_qos)
        self.get_logger().info(f'Calibration {source} -> {self.target}; input alias {self.input_frame}; output {parameter("output_topic")}')

    def receive(self, message):
        try:
            if message.header.frame_id != self.input_frame:
                raise ValueError(f'Expected frame {self.input_frame}, received {message.header.frame_id}')
            fields = {f.name: f for f in message.fields}
            for key in ('x', 'y', 'z'):
                if key not in fields or fields[key].datatype != PointField.FLOAT32 or fields[key].count != 1:
                    raise ValueError('Expected scalar FLOAT32 x/y/z fields')
            data = transform_xyz_buffer(message.data, message.width, message.height, message.point_step,
                                        message.row_step, [fields[k].offset for k in ('x', 'y', 'z')],
                                        message.is_bigendian, self.rotation, self.translation)
            output = copy.deepcopy(message)
            output.header.frame_id = self.target
            output.data = bytes(data)
            self.publisher.publish(output)
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f'Dropped cloud: {error}', throttle_duration_sec=3.)


def main():
    rclpy.init()
    node = None
    try:
        node = CalibratedCloud()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
